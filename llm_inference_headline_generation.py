# %% [markdown]
# # UdaciHeadline: LLM Inference Optimization Project
# 
# ## Project Introduction
# Large Language Models (LLMs) are transforming content creation, but deploying them efficiently remains a major hurdle. Automatically generate catchy headlines from article summaries using an LLM. In this project, UdaciHeadline, we accelerate the headline generation pipeline significantly by applying state-of-the-art LLM inference optimization techniques.

# %% [markdown]
# ## Project Summary
# This project provides hands-on experience in optimizing the inference performance of a pre-trained Large Language Model (like Llama-3.2-1B) for news headline generation. You will bring together concepts of LLM architecture, optimization techniques, and deployment frameworks. Specifically, you will:
# 
# 1.  **Establish a baseline** inference pipeline and profile its performance.
# 2.  Implement and evaluate architectural optimizations like **KV-caching**.
# 3.  Apply model compression techniques like **quantization** and **pruning**.
# 4.  Configure and benchmark **distributed inference** using Tensor and Pipeline Parallelism.
# 5.  Apply advanced decoding mechanisms like **speculative decoding**.
# 6.  Perform comprehensive **benchmarking and analysis** across all stages.
# 7.  Produce a **final report** summarizing findings and trade-offs.

# ## Imports and Global Configuration
# 
# Let's import the libraries we'll use throughout the project and define some constants like the model name and the prompt template.

import os
import torch
import pandas as pd
import numpy as np
import kagglehub
import gc
import evaluate
from rouge_score import rouge_scorer
print("kagglehub imported successfully")
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GPTQConfig
from evaluate import load as load_metric
import time
from pprint import pprint
import torch.profiler
import torch.nn.utils.prune as prune
#import deepspeed
import subprocess
import pyarrow as pa
print(pa.__version__)
import numpy as np
print(np.__version__)


try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

os.environ["HF_HUB_OFFLINE"] = "1" #Set the Hugging face in offline mode.
# ---- Constants ----
MODEL_NAME = "/voc/shared/models/llama/Llama-3.2-1B"
MAX_NEW_TOKENS = 20 # Max length for the generated headline
print(os.listdir("/voc/shared/models/llama"))
PROMPT = \
"""
Write a concise and factual news headline for the article below.
The headline should be one sentence and contain no extra commentary.

Article:
{article}

Headline:
"""

import torch
print("torch cuda available:", torch.cuda.is_available())
print("torch cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))


# ## Data Loading
# 
# We will use the "News Category Dataset" from Kaggle. The `kagglehub` library makes it easy to download and access. Your task is to implement the function to load and preprocess the data according to the docstring.

def load_news_dataset(path):
    """
    Download and load the News Category Dataset from KaggleHub,
    then preprocess it for headline generation.
    """

    # Download dataset
    dataset_dir = kagglehub.dataset_download(path)

    # Load the JSON dataset
    dataset = load_dataset(
        "json",
        data_files=f"{dataset_dir}/News_Category_Dataset_v3.json",
        split="train"
    )

    # Preprocess dataset
    def preprocess(example):
        return {
            "text": example["short_description"],
            "headline": example["headline"]
        }

    dataset = dataset.map(preprocess)

    # Remove unused columns
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ["text", "headline"]]
    )
    print("dataset has been loaded")
    return dataset
    
# # 2. Baseline Performance
# 
# Before we can optimize, we need a starting point. Here, you'll establish the baseline performance of the `Llama-3.2-1B` model without any specific optimizations. We will measure latency, throughput, and the quality of the generated headlines using the ROUGE score.
# 
# ### Your Task: Implement the Evaluation Pipeline
# You need to implement the core functions for loading a model, generating a headline, and evaluating performance. These functions will be reused for every optimization technique.

def load_model(model_name, quantization_config=None, device_map=None):
    """TODO: Implement the logic for loading a tokenizer and model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model_kwargs = dict(
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        if device_map is not None:
            model_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        #The model's config and the tokenizer can get out of sync — the tokenizer might now know the pad token ID (from block 1), but the model's internal config still doesn't. This syncs them so the model knows which token to ignore during attention masking.


        #Batched inference/training requires padding shorter sequences to match the longest one in the batch. Without a pad token, this breaks.
        #The model uses pad_token_id to build the attention mask — telling the model to ignore padded positions. If it's None, you can get incorrect attention or loss calculations.
        #This is essentially a safety/compatibility shim for decoder-only models that were originally designed for single-sequence generation, not batched training.

        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        print("eos_token_id:", tokenizer.eos_token_id)
        print("pad_token_id:", tokenizer.pad_token_id)
    except Exception as e:
        print(f"Error loading model {model_name}. Make sure you have internet connection "
          f"and the model name is correct. Error: {e}")        
    return tokenizer, model


def load_model_tensor_parallel(model_name, quantization_config=None):
    """Load model with real tensor parallelism (`tp_plan="auto"`) for multi-process runs.

    Run with:
      torchrun --nproc_per_node=<num_gpus> llm_inference_headline_generation.py
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    model_kwargs = dict(
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        tp_plan="auto"
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print("Tensor parallel plan:", getattr(model, "_tp_plan", None))
    print("Tensor parallel world size:", int(os.environ.get("WORLD_SIZE", "1")))
    return tokenizer, model
    
def describe_parallelism_support(model):
    """Explain what `device_map="auto"` actually did for the loaded model."""
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        print("No hf_device_map found. Model is likely on a single device, so no model parallelism is active.")
        return

    gpus_used = sorted({v for v in device_map.values() if isinstance(v, int)})
    if len(gpus_used) <= 1:
        print("`device_map=\"auto\"` did not shard layers across multiple GPUs.")
        print("Reason: only one visible GPU (or the model fits on one GPU).")
        print("Note: this is layer/model sharding, not tensor-parallel kernel splitting.")
    else:
        print(f"Model layers were sharded across {len(gpus_used)} GPUs: {gpus_used}")
        print("This is model/layer parallelism via Accelerate device mapping.")


def generate_headline(model, tokenizer, summary, generation_args):

    print("Entered generate_headline")
    prompt = PROMPT.format(article=summary)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    print("Tokenization done")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print("Moved inputs to device")
    print({k: v.shape for k, v in inputs.items()})

    with torch.no_grad():
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        print("About to call model.generate")
        outputs = model.generate(
            **inputs,
            **generation_args,
            pad_token_id=tokenizer.pad_token_id,
        )
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        print("model.generate finished")

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_headline = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return generated_headline, generated_tokens

def report_metrics(results, latencies, max_new_tokens):
    """TODO: Implement the logic for calculating and reporting all performance metrics."""
    """Measure average generation speed and get a sample output."""
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = max_new_tokens / avg_latency if avg_latency > 0 else 0

    print("Average latency:", avg_latency)
    print("Throughput (tokens/sec):", throughput)

    if "rouge" in results:
        print("ROUGE scores:", results["rouge"])

    if "sample_output" in results:
        print("Sample output:", results["sample_output"])

def evaluate_model(dataset, model, tokenizer, generation_args, n=20):
    """TODO: Implement the model evaluation loop."""
    num_samples = len(dataset)
    model.eval()
    for i in range(num_samples):
        sample = dataset[i]
        summary = sample["text"]
        reference = sample["headline"]

        try:
            generated_headline, actual_new_tokens = generate_headline(
                model=model,
                tokenizer=tokenizer,
                summary=summary,
                generation_args=generation_args
            )

            results.append({
                "input": summary,
                "reference": reference,
                "prediction": generated_headline,
                "actual_new_tokens": actual_new_tokens
            })
            latencies.append(latency)

        except Exception as e:
            print(f"Error during generation for sample {i}: {e}") 

def benchmark_model(model, tokenizer, dataset, generation_args,
                    n_samples=25, label=""):
    """
    Scores ONE loaded model over the same n_samples articles.
    Returns mean latency, throughput, and averaged ROUGE.
    Run this once per technique (baseline, KV, pruned, quantized).
    """
    device = next(model.parameters()).device
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    latencies, tok_counts, r1, r2, rl = [], [], [], [], []
    n = min(n_samples, len(dataset))

    for i in range(n):
        article   = dataset[i]["text"]
        reference = dataset[i]["headline"]

        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen_text, gen_tokens = generate_headline(model, tokenizer, article, generation_args)
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append(t1 - t0)
        tok_counts.append(gen_tokens.shape[0])

        s = scorer.score(reference, gen_text)   # reference first
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    mean_latency    = np.mean(latencies)
    mean_throughput = np.sum(tok_counts) / np.sum(latencies)   # total tokens / total time

    return {
        "Technique": label,
        "Mean Latency (s)": round(mean_latency, 4),
        "Throughput (tok/s)": round(mean_throughput, 4),
        "ROUGE-1": round(np.mean(r1), 4),
        "ROUGE-2": round(np.mean(r2), 4),
        "ROUGE-L": round(np.mean(rl), 4),
        "N": n,
    }

# TODO: Establish your baseline performance.
def get_model_memory_footprint(model):
    """Calculates and returns the model's memory footprint in MB."""
    mem_params = sum(param.nelement() * param.element_size() for param in model.parameters())
    mem_bufs = sum(buf.nelement() * buf.element_size() for buf in model.buffers())
    total_mem_bytes = mem_params + mem_bufs
    return total_mem_bytes / (1024 ** 2) # Convert bytes to MB

gpu_wall_time = -1.0 # Initialize in case GPU is not available
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#gpu_device = torch.device("cuda")
tokenizer, model = load_model(MODEL_NAME, quantization_config=None)
#model.to(gpu_device)
dataset = load_news_dataset("rmisra/news-category-dataset")
memory_mb_32bit = get_model_memory_footprint(model)
rows = []
print(f"4-bit Memory Footprint: {memory_mb_32bit:.2f} MB")

if torch.cuda.is_available():
    print("\n--- Profiling on GPU ---")

    print("Performing GPU warm-up run...")
    
    
    print(dataset[0])
    # TODO: Call run_gpu_inference for warm-up
    generation_args = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": False
    }
    model.eval()
    if torch.cuda.is_available():
        print("allocated GB:", torch.cuda.memory_allocated() / 1e9)
        print("reserved GB:", torch.cuda.memory_reserved() / 1e9)
    generated_headline, generated_tokens = generate_headline(model, tokenizer, dataset[0]["text"], generation_args)
    print("Warm-up complete.")

    print("Running inference on GPU and capturing profile...")
    start_time_gpu_wall = time.perf_counter()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False
    ) as prof_gpu:
        with torch.profiler.record_function("model_inference_gpu"):
            generate_headline(model, tokenizer, dataset[0]["text"], generation_args)

    end_time_gpu_wall = time.perf_counter()
    gpu_wall_time = end_time_gpu_wall - start_time_gpu_wall
    throughput = len(generated_tokens)/(gpu_wall_time)
    print(f"GPU Wall clock time: {gpu_wall_time:.4f} seconds ")

    print(f"Latency: {gpu_wall_time:.4f} s Throughput: {throughput:.4f} tokens/s")

    print("GPU Profiler Analysis (Top 5 Operators by Self CUDA Time):")
    print(prof_gpu.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    # --- ROUGE evaluation ---

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    reference = dataset[0]["headline"]
    s = scorer.score(reference, generated_headline)

    print(f"ROUGE-1: {s['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2: {s['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L: {s['rougeL'].fmeasure:.4f}")

else:
    print("\nCUDA not available on this system. Skipping GPU profiling.")

# Baseline (no cache): set use_cache=False in generation_args
gen_args_nocache = {"max_new_tokens": MAX_NEW_TOKENS, "use_cache": False}
rows.append(benchmark_model(model, tokenizer, dataset, gen_args_nocache,
                            n_samples=25, label="Baseline (No Cache)"))
df = pd.DataFrame(rows)
print(df.to_string(index=False))


# Evaluate the model with KV Caching enabled.

generation_args = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "use_cache": True
}

if torch.cuda.is_available():
    print("\n--- Profiling on GPU ---")

    print("Performing GPU warm-up run...")
    
    
    print(dataset[0])
    # TODO: Call run_gpu_inference for warm-up

    generated_headline, generated_tokens = generate_headline(model, tokenizer, dataset[0]["text"], generation_args)
    print("Warm-up complete.")

    print("Running inference on GPU and capturing profile...")
    start_time_gpu_wall = time.perf_counter()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False
    ) as prof_gpu:
        with torch.profiler.record_function("model_inference_gpu"):
            generated_headline, generated_tokens =  generate_headline(model, tokenizer, dataset[0]["text"], generation_args)

    end_time_gpu_wall = time.perf_counter()
    gpu_wall_time = end_time_gpu_wall - start_time_gpu_wall

    throughput = len(generated_tokens)/(gpu_wall_time)
    print(f"GPU Wall clock time: {gpu_wall_time:.4f} seconds ")

    print(f"Latency: {gpu_wall_time:.4f} s Throughput: {throughput:.4f} tokens/s")

    print("GPU Profiler Analysis (Top 5 Operators by Self CUDA Time):")
    print(prof_gpu.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    # --- ROUGE evaluation ---

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    reference = dataset[0]["headline"]
    s = scorer.score(reference, generated_headline)

    print(f"ROUGE-1: {s['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2: {s['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L: {s['rougeL'].fmeasure:.4f}")

    print("\nGPU Profiler Analysis (Top 5 Operators by Self CUDA Time):")
    print(prof_gpu.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

else:
    print("\nCUDA not available on this system. Skipping GPU profiling.")

# KV caching: same model, use_cache=True
gen_args_cache = {"max_new_tokens": MAX_NEW_TOKENS, "use_cache": True}
rows.append(benchmark_model(model, tokenizer, dataset, gen_args_cache,
                            n_samples=25, label="KV Caching"))
df = pd.DataFrame(rows)
print(df.to_string(index=False))    

# TODO: Evaluate the model with pruning.

def prune_model_weights(model, amount=0.01):
    if not 0 <= amount <= 1:
        raise ValueError("amount must be between 0 and 1.")

    pruned_layers = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            pruned_layers += 1
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
            break

    print(f"Pruned {pruned_layers} linear layer(s)")
    return model

# TODO: Evaluate the pruned model.

def run_performance_test(model, tokenizer, prompt, max_tokens, num_runs):
    """Measure average generation speed and get a sample output."""
    total_time = 0
    sample_output = "Error during generation."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    total_tokens=0
    with torch.no_grad():
        for i in range(num_runs):
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            #outputs = model.generate(**inputs, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
            generated_headline, generated_tokens = generate_headline(model, tokenizer, dataset[0]["text"], generation_args)
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time += (end_time - start_time)
            total_tokens += generated_tokens.shape[0]
            if i == 0: # Get sample from the first run
                #sample_output = tokenizer.decode(generated_headline, skip_special_tokens=True)
                sample_output = generated_headline
    avg_time = total_time / num_runs

    # --- ROUGE on the sample output ---
    rouge_scores = None
    if reference is not None:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        s = scorer.score(reference, sample_output)
        rouge_scores = {
            "rouge1": s["rouge1"].fmeasure,
            "rouge2": s["rouge2"].fmeasure,
            "rougeL": s["rougeL"].fmeasure,
        }

    return avg_time, sample_output, total_tokens, rouge_scores

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

print(torch.cuda.get_device_name(0))
free_mem, total_mem = torch.cuda.mem_get_info()
print(f"Free GPU memory: {free_mem/1e9:.2f} GB / {total_mem/1e9:.2f} GB")

#tokenizer, model = load_model(MODEL_NAME, quantization_config=None)
torch.cuda.empty_cache()
gc.collect()
pruned_model = prune_model_weights(model, amount=0.3)

#pruned_model = pruned_model.to(device)

print(f"After load - Free GPU memory: {free_mem/1e9:.2f} GB / {total_mem/1e9:.2f} GB")
# Move model to CPU before pruning
#model = model.cpu()
torch.cuda.empty_cache()
gc.collect()

#pruned_model = prune_model_weights(model, amount=0.05)

# Move back to GPU after pruning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#pruned_model = pruned_model.to(device)

#pruned_model = prune_model_weights(model, amount=0.3)
NUM_SPEED_RUNS = 3

#tokenizer, model = load_model(MODEL_NAME, quantization_config=None)
free_mem, total_mem = torch.cuda.mem_get_info()
print(f"After load - Free GPU memory: {free_mem/1e9:.2f} GB / {total_mem/1e9:.2f} GB")

avg_time, output, total_tokens, rouge_scores = run_performance_test(pruned_model, tokenizer, dataset[0]["text"], MAX_NEW_TOKENS, NUM_SPEED_RUNS)

throughput = total_tokens/(avg_time)
print(f"Avg. Latency: {avg_time:.4f} s  Throughput: {throughput:.4f} tokens/s (over {NUM_SPEED_RUNS} runs)")
print(f"Result:\n  - Avg Time: {avg_time:.4f}s\n  - Output: '{output}'\n  - Throughput: {throughput:.4f} tok/s")
if rouge_scores:
    print(f"  - ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  - ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  - ROUGE-L: {rouge_scores['rougeL']:.4f}")
print(f"Avg. Latency: {avg_time:.4f} s Throughput: {throughput: .4f} s (over {NUM_SPEED_RUNS} runs)")
#Log results
#results_log.append({
#        "Configuration": config_name,
#        "Avg Inference Time (s)": f"{avg_time:.4f}",
#        "Generated Output": output
#    })
    
print(f"Result:\n  - Avg Time: {avg_time:.4f}s\n  - Output: '{output}'s\n  - throughput: {throughput:.4f}")

# Pruned model
rows.append(benchmark_model(pruned_model, tokenizer, dataset, gen_args_cache,
                            n_samples=25, label="Pruning (30%)"))

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# Clean up to save memory
#del pruned_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# TODO: Implement and evaluate 4-bit quantization.

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


tokenizer_4bit, model_4bit = load_model(MODEL_NAME, quantization_config=quant_config, device_map="auto")

memory_mb_4bit = get_model_memory_footprint(model_4bit)
print(f"4-bit Memory Footprint: {memory_mb_4bit:.2f} MB")

model_4bit.eval()
latencies_4bit = []

avg_time, output, total_tokens, rouge_scores = run_performance_test(model_4bit, tokenizer, dataset[0]["text"], MAX_NEW_TOKENS, NUM_SPEED_RUNS)

throughput_4bit = total_tokens/(avg_time)

print(f"4-bit Avg. Latency: {avg_time:.4f} s 4-bit Throughput: {throughput_4bit:.4f} s (over {NUM_SPEED_RUNS} runs)")
if rouge_scores:
    print(f"  - ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  - ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  - ROUGE-L: {rouge_scores['rougeL']:.4f}")
print(output)

# Quantized model
rows.append(benchmark_model(model_4bit, tokenizer, dataset, gen_args_cache,
                            n_samples=25, label="Quantization (4-bit)"))

df = pd.DataFrame(rows)
print(df.to_string(index=False))



# %% [markdown]
# # 6. Distributed Inference (Multi-GPU)
# 
# **Your Task:** If you have multiple GPUs, you can split the model across them to reduce the memory burden on a single GPU and potentially improve latency. We will explore two common techniques: Tensor Parallelism and Pipeline Parallelism.
# 
# *Note: This section requires a multi-GPU environment.*
# 
# ### Tensor Parallelism
# Tensor parallelism splits individual model layers (the tensors) across multiple GPUs. Operations like matrix multiplications are executed in parallel on different GPUs, and the results are aggregated. This is highly effective for reducing the memory footprint of very large layers. The `accelerate` library can handle this automatically via `device_map="auto"`.
# 
# ### Pipeline Parallelism
# Pipeline parallelism assigns entire layers or blocks of layers to different GPUs, creating a sequence or "pipeline" that the data flows through. For example, layers 1-10 run on GPU 0, layers 11-20 run on GPU 1, and so on. This is useful for very deep models where even a single layer might be too large for one GPU after tensor parallelism.

# %%
# TODO: Check for multi-GPU environment and evaluate with Tensor Parallelism.
# The `device_map="auto"` in your `load_model` function should automatically apply this.

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

print(model_4bit.hf_device_map)
# e.g. {'model.embed_tokens': 0, 'model.layers.0': 0, ..., 'model.layers.25': 1, 'lm_head': 1}

# 2. How many unique GPUs were used
gpus_used = set(
    v for v in model_4bit.hf_device_map.values()
    if isinstance(v, int)  # filters out "cpu" or "disk" if offloading kicked in
)
print(f"GPUs used: {len(gpus_used)} — devices {sorted(gpus_used)}")

if num_gpus > 1:
    print("Multi-GPU environment detected.")
else:
    print("Single-GPU environment detected.Tensor and Pipeline parallelism will not be performed.")

# %%
# TODO: Evaluate with Pipeline Parallelism.
# This is more advanced and may require manually defining a device_map to assign
# different layers of the model to different GPUs.

# %% [markdown]
# # 7. Advanced Decoding: Speculative Decoding
# 
# **Your Task:** Speculative decoding uses a smaller, faster "draft" model to generate several candidate tokens. A larger, more accurate "target" model then verifies these tokens in a single forward pass. This can significantly speed up generation if the draft model is a good predictor. You will load a larger target model and a smaller draft model, benchmark the target model alone, and then benchmark it with assistance from the draft model.

def run_speculative_decoding(draft_model, target_model, tokenizer, prompt_text, max_tokens, k):
    """Runs a speculative decoding loop for a given k and measures performance."""
    device = next(target_model.parameters()).device
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    target_passes = 0

    start_time = time.time()
    with torch.no_grad():
        while input_ids.shape[1] < (prompt_len + max_tokens):
            ctx_len = input_ids.shape[1]

            # 1. Draft Phase: small model proposes k tokens
            draft_ids = draft_model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=k,
                pad_token_id=tokenizer.eos_token_id,
            )
            draft_candidates = draft_ids[:, ctx_len:]
            num_drafted = draft_candidates.shape[1]
            if num_drafted == 0:
                break

            # 2. Verification Phase: target sees context + draft
            verification_input = torch.cat([input_ids, draft_candidates], dim=1)
            target_logits = target_model(verification_input).logits
            target_passes += 1

            # 3. Acceptance: compare draft vs target greedy preds
            num_matched = 0
            for i in range(num_drafted):
                # logit at ctx_len + i - 1 predicts the i-th draft token
                logit_idx = ctx_len + i - 1
                target_pred_id = torch.argmax(target_logits[:, logit_idx, :], dim=-1)
                if draft_candidates[0, i] == target_pred_id.item():
                    num_matched += 1
                else:
                    break

            # Accept all matched tokens
            accepted = draft_candidates[:, :num_matched]
            input_ids = torch.cat([input_ids, accepted], dim=1)

            # On mismatch, accept the target's correction
            if num_matched < num_drafted:
                corr_idx = ctx_len + num_matched - 1
                correction_id = torch.argmax(
                    target_logits[:, corr_idx, :], dim=-1, keepdim=True
                )
                input_ids = torch.cat([input_ids, correction_id], dim=1)

            if tokenizer.eos_token_id in input_ids[0, prompt_len:]:
                break
    end_time = time.time()

    total_accepted = input_ids.shape[1] - prompt_len
    avg_accepted_per_pass = total_accepted / target_passes if target_passes > 0 else 0

        # Decode only the generated portion (exclude the prompt)
    generated_text = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

    return end_time - start_time, target_passes, avg_accepted_per_pass, generated_text

#DRAFT_MODEL_NAME = "/voc/shared/models/llama/Llama-3.2-1B"
#TARGET_MODEL_NAME = "/voc/shared/models/llama/Llama-3.2-3B"

# --- Config ---
TARGET_MODEL_NAME = "gpt2-medium"
DRAFT_MODEL_NAME = "gpt2"
MAX_TOTAL_TOKENS = 50
K_VALUES_TO_TEST = [1, 2, 3, 4, 5, 8, 10]
spec_device = "cuda" if torch.cuda.is_available() else "cpu"
spec_dtype = torch.float16 if spec_device == "cuda" else torch.float32

def benchmark_spec_decoding(draft_model, target_model, tokenizer, dataset,
                            max_tokens, k, n_samples=25, label="Speculative Decoding"):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    times, tok_counts, r1, r2, rl = [], [], [], [], []
    n = min(n_samples, len(dataset))

    for i in range(n):
        prompt    = dataset[i]["text"][:500]   # same truncation you used before
        reference = dataset[i]["headline"]

        spec_time, passes, avg_acc, gen_text = run_speculative_decoding(
            draft_model, target_model, tokenizer, prompt, max_tokens, k
        )
        times.append(spec_time)
        tok_counts.append(len(tokenizer.encode(gen_text)))

        s = scorer.score(reference, gen_text)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    return {
        "Technique": label,
        "Mean Latency (s)": round(np.mean(times), 4),
        "Throughput (tok/s)": round(np.sum(tok_counts) / np.sum(times), 4),
        "ROUGE-1": round(np.mean(r1), 4),
        "ROUGE-2": round(np.mean(r2), 4),
        "ROUGE-L": round(np.mean(rl), 4),
        "N": n,
    }

# --- Load Target ---
print(f"Loading Target Model: {TARGET_MODEL_NAME} on {spec_device} ({spec_dtype})...")
target_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, local_files_only=True)
target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL_NAME,
    local_files_only=True,
    torch_dtype=spec_dtype,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
if target_tokenizer.pad_token_id is None and target_tokenizer.eos_token_id is not None:
    target_tokenizer.pad_token = target_tokenizer.eos_token
if target_model.config.pad_token_id is None:
    target_model.config.pad_token_id = target_tokenizer.pad_token_id
target_model.to(spec_device)          # FIX: actually move to GPU
target_model.eval()

# --- Load Draft ---
print(f"Loading Draft Model: {DRAFT_MODEL_NAME} on {spec_device} ({spec_dtype})...")
draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_NAME, local_files_only=True)
draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL_NAME,
    local_files_only=True,
    torch_dtype=spec_dtype,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
if draft_tokenizer.pad_token_id is None and draft_tokenizer.eos_token_id is not None:
    draft_tokenizer.pad_token = draft_tokenizer.eos_token
if draft_model.config.pad_token_id is None:
    draft_model.config.pad_token_id = draft_tokenizer.pad_token_id
draft_model.to(spec_device)
draft_model.eval()

# --- Prompt (truncated to avoid long-doc OOM) ---
PROMPT_TEXT = dataset[0]["text"][:500]
REFERENCE = dataset[0]["headline"]
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# --- Run experiment ---
results_log = []
print("--- Running Speculative Decoding Experiment ---")
for k in K_VALUES_TO_TEST:
    print(f"Testing with K = {k}...")
    spec_time, spec_passes, avg_accepted, gen_text = run_speculative_decoding(
        draft_model, target_model, target_tokenizer,   # FIX: correct tokenizer
        PROMPT_TEXT, MAX_TOTAL_TOKENS, k
    )
    s = scorer.score(REFERENCE, gen_text)
    results_log.append({
        "K": k,
        "Time (s)": spec_time,
        "Target Passes": spec_passes,
        "Avg. Accepted Tokens": avg_accepted,
        "ROUGE-1": s["rouge1"].fmeasure,
        "ROUGE-2": s["rouge2"].fmeasure,
        "ROUGE-L": s["rougeL"].fmeasure,
    })
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

df_results = pd.DataFrame(results_log)
print("--- Speculative Decoding Experiment Results Summary ---")
print(df_results.to_string())

rows.append(benchmark_spec_decoding(
    draft_model, target_model, target_tokenizer, dataset,
    max_tokens=MAX_TOTAL_TOKENS, k=3, n_samples=25,
    label="Speculative Decoding (K=3)"
))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
