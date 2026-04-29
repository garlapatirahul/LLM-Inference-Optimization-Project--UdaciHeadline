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
print("kagglehub imported successfully")
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from evaluate import load as load_metric
import time
from pprint import pprint
import torch.profiler
import torch.nn.utils.prune as prune
import subprocess

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
    
    pass

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
    

# %%
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
    print(f"GPU Wall clock time: {gpu_wall_time:.4f} seconds")

    print("GPU Profiler Analysis (Top 5 Operators by Self CUDA Time):")
    print(prof_gpu.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
else:
    print("\nCUDA not available on this system. Skipping GPU profiling.")

# # 3. Architectural Optimization: KV Caching
# 
# **Your Task:** One of the most effective ways to speed up token generation is using a Key-Value (KV) cache. This avoids re-computing attention scores for tokens that are already part of the sequence. Enable the `use_cache` flag in the generation arguments and re-run the evaluation. Observe the impact on latency and throughput.

# %%
# TODO: Evaluate the model with KV Caching enabled.

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
            generate_headline(model, tokenizer, dataset[0]["text"], generation_args)

    end_time_gpu_wall = time.perf_counter()
    gpu_wall_time = end_time_gpu_wall - start_time_gpu_wall
    print(f"GPU Wall clock time: {gpu_wall_time:.4f} seconds")

    print("GPU Profiler Analysis (Top 5 Operators by Self CUDA Time):")
    print(prof_gpu.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
else:
    print("\nCUDA not available on this system. Skipping GPU profiling.")

# # 4. Model Compression: Pruning
# 
# **Your Task:** Pruning removes redundant model weights, which can reduce model size and potentially speed up inference. Here, you will implement unstructured, magnitude-based pruning by creating a function that applies it to the model's linear layers and then evaluating the result.


def prune_model_weights(model, amount=0.01):
    if not 0 <= amount <= 1:
        raise ValueError("amount must be between 0 and 1.")

    pruned_layers = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            pruned_layers += 1
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")

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
    return avg_time, sample_output, total_tokens


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

avg_time, output, total_tokens = run_performance_test(pruned_model, tokenizer, dataset[0]["text"], MAX_NEW_TOKENS, NUM_SPEED_RUNS)

#Log results
#results_log.append({
#        "Configuration": config_name,
#        "Avg Inference Time (s)": f"{avg_time:.4f}",
#        "Generated Output": output
#    })
    
print(f"Result:\n  - Avg Time: {avg_time:.4f}s\n  - Output: '{output}'s\n  - Total Tokens: {total_tokens:.4f}")

# Clean up to save memory
#del pruned_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# # 5. Model Compression: Quantization
# 
#:** Quantization reduces the precision of model weights (e.g., from 16-bit to 4-bit), significantly cutting down memory usage and often speeding up inference. You will define a 4-bit quantization configuration and use it to load and evaluate a new model.

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

avg_time, output, total_tokens = run_performance_test(model_4bit, tokenizer, dataset[0]["text"], MAX_NEW_TOKENS, NUM_SPEED_RUNS)

throughput_4bit = total_tokens/(avg_time)

print(f"4-bit Avg. Latency: {avg_time:.4f} s 4-bit Throughput: {throughput_4bit:.4f} s (over {NUM_SPEED_RUNS} runs)")

print(output)




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
# NOTE: `device_map="auto"` is model/layer sharding, not true tensor parallelism.
# For true TP, launch this script with torchrun and call load_model_tensor_parallel(...).

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

print(getattr(model_4bit, "hf_device_map", "No device map available"))
describe_parallelism_support(model_4bit)

if num_gpus > 1:
    print("Multi-GPU environment detected.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        print("WORLD_SIZE > 1, enabling true tensor parallel load...")
        tp_tokenizer, tp_model = load_model_tensor_parallel(MODEL_NAME, quantization_config=quant_config)
        print("Loaded TP model successfully.")
    else:
        print("To run true tensor parallelism, launch with torchrun (multi-process).")
else:
    print("Single-GPU environment detected.Tensor and Pipeline parallelism will not be performed.")

# %%
# TODO: Evaluate with Pipeline Parallelism.
# This is more advanced and may require manually defining a device_map to assign
# different layers of the model to different GPUs.



def benchmark_vllm_inference(model_name, prompts, max_new_tokens=20):
    """Optional benchmark path for vLLM serving-style inference."""
    if not VLLM_AVAILABLE:
        print("vLLM is not installed. Install with: pip install vllm")
        return None

    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    llm = LLM(model=model_name, tensor_parallel_size=max(1, torch.cuda.device_count()))

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end = time.perf_counter()

    generated_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    latency = end - start
    throughput = generated_tokens / latency if latency > 0 else 0.0

    print(f"vLLM latency: {latency:.4f}s | throughput: {throughput:.2f} tok/s")
    return {"latency": latency, "throughput": throughput, "outputs": outputs}


def print_nsight_profile_command(script_path="llm_inference_headline_generation.py", nproc_per_node=1):
    """Print a ready-to-run Nsight Systems profiling command."""
    cmd = (
        f"nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true "
        f"--output=nsys_llm_profile torchrun --nproc_per_node={nproc_per_node} {script_path}"
    )
    print("Run this command to collect Nsight Systems traces:")
    print(cmd)
    return cmd



# Optional benchmarking hooks
print_nsight_profile_command(nproc_per_node=max(1, num_gpus))
# Example vLLM run (uncomment when vLLM is installed):
# benchmark_vllm_inference(MODEL_NAME, [PROMPT.format(article=dataset[0]["text"])], MAX_NEW_TOKENS)

# %% [markdown]
# # 7. Advanced Decoding: Speculative Decoding
# 
# **Your Task:** Speculative decoding uses a smaller, faster "draft" model to generate several candidate tokens. A larger, more accurate "target" model then verifies these tokens in a single forward pass. This can significantly speed up generation if the draft model is a good predictor. You will load a larger target model and a smaller draft model, benchmark the target model alone, and then benchmark it with assistance from the draft model.

# %%
# TODO: Implement and evaluate speculative decoding.
DRAFT_MODEL_NAME = "/voc/shared/models/llama/Llama-3.2-1B"
TARGET_MODEL_NAME = "/voc/shared/models/llama/Llama-3.2-3B"
K_DRAFT_TOKENS = 5 # Let's have the draft model propose 5 tokens
target_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, local_files_only=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME, local_files_only=True,
                    #torch_dtype = torch.float16,
                    #device_map = "auto",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager")

if target_tokenizer.pad_token_id is None and target_tokenizer.eos_token_id is not None:
    target_tokenizer_.pad_token = target_tokenizer.eos_token

if target_model.config.pad_token_id is None:
    target_model.config.pad_token_id = target_tokenizer.pad_token_id

print(f"Loaded Target Model ('The Scout'): {TARGET_MODEL_NAME}...")
INITIAL_CONTEXT_TEXT = dataset[0]["text"]

target_model.eval()

print(f"Loading Draft Model ('The Scout'): {DRAFT_MODEL_NAME}...")
draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_NAME, local_files_only=True,
                    #torch_dtype = torch.float16,
                    #device_map = "auto",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager")

if draft_tokenizer.pad_token_id is None and draft_tokenizer.eos_token_id is not None:
    draft_tokenizer_.pad_token = draft_tokenizer.eos_token

if draft_model.config.pad_token_id is None:
    draft_model.config.pad_token_id = draft_tokenizer.pad_token_id

draft_model.eval() # Set to evaluation mode

print(f"--- Step 1: Draft Model generates {K_DRAFT_TOKENS} candidate tokens ---")
current_context_ids = draft_tokenizer.encode(INITIAL_CONTEXT_TEXT, return_tensors="pt")

with torch.no_grad():
    # Use the draft model's generate function for simplicity
    draft_output_ids = draft_model.generate(
        current_context_ids,
        max_new_tokens=K_DRAFT_TOKENS,
        pad_token_id=tokenizer.eos_token_id
    )

# Isolate just the newly generated draft tokens
draft_candidate_ids = draft_output_ids[:, current_context_ids.shape[1]:]

print(f"Initial Context: '{INITIAL_CONTEXT_TEXT}'")
print(f"Draft Model's Proposal: '{draft_tokenizer.decode(draft_candidate_ids[0], skip_special_tokens=True)}'")


print(f"--- Step 2: Target Model verifies the draft in a single pass ---")
# Combine context and draft for the verification input
verification_input_ids = torch.cat([current_context_ids, draft_candidate_ids], dim=1)

with torch.no_grad():
    target_verification_logits = target_model(verification_input_ids).logits

# Get the target model's top-1 prediction for each position
# Note: The logits at index `t-1` predict the token at index `t`
# So we look at logits from the start of the draft sequence onwards
start_of_draft_in_logits = current_context_ids.shape[1] - 1
end_of_draft_in_logits = verification_input_ids.shape[1] - 1

target_preferred_ids = torch.argmax(target_verification_logits[:, start_of_draft_in_logits:end_of_draft_in_logits, :], dim=-1)

print(f"Verification Input: '{target_tokenizer.decode(verification_input_ids[0])}'")
print(f"Target Model's Preferences: '{target_tokenizer.decode(target_preferred_ids[0], skip_special_tokens=True)}'")

print("--- Step 3: Comparing draft against target preferences ---")
num_matched_tokens = 0
for i in range(draft_candidate_ids.shape[1]):
    draft_token = draft_candidate_ids[0, i]
    target_token = target_preferred_ids[0, i]
    
    print(f"Pos {i+1}: Draft ('{draft_tokenizer.decode(draft_token)}') vs Target ('{target_tokenizer.decode(target_token)}')")
    
    if draft_token == target_token:
        print("  ✅ Match!")
        num_matched_tokens += 1
    else:
        print("  ❌ Mismatch! Halting comparison.")
        break

print(f"\nNumber of matched tokens: {num_matched_tokens}")


print("--- Step 4: Constructing the final accepted sequence for this step ---")

# 1. Take all the tokens that matched
accepted_ids = draft_candidate_ids[0, :num_matched_tokens]

# 2. Take the target's token at the next position 
# (This is either the correction at the mismatch point, or the next token if all matched)
if num_matched_tokens < target_preferred_ids.shape[1]:
    next_token = target_preferred_ids[0, num_matched_tokens].unsqueeze(0)
    final_accepted_ids = torch.cat([accepted_ids, next_token], dim=0)
else: # This case is rare, means we ran out of target preferences
    final_accepted_ids = accepted_ids

print(f"Matched Tokens Accepted: '{target_tokenizer.decode(accepted_ids)}'")
if num_matched_tokens < target_preferred_ids.shape[1]:
    print(f"Correction/Extension Token: '{target_tokenizer.decode(next_token)}'")

# Update our full context
new_context_ids = torch.cat([current_context_ids, final_accepted_ids.unsqueeze(0)], dim=1)

print("\n--- SUMMARY OF THIS STEP ---")
print(f"Tokens generated this step: {len(final_accepted_ids)} -> '{target_tokenizer.decode(final_accepted_ids)}'")
print(f"Target Model expensive calls: 1")
print(f"New Context: '{target_tokenizer.decode(new_context_ids[0])}'")



