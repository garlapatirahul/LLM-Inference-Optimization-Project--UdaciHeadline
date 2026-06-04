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

os.environ["HF_HUB_OFFLINE"] = "1" #Set the Hugging face in offline mode.
# ---- Constants ----
MODEL_NAME = "/voc/shared/models/llama/Llama-3.2-1B"
MAX_NEW_TOKENS = 20 # Max length for the generated headline
EVAL_SAMPLE_COUNT = 20 # Number of samples for each comparable benchmark
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

def load_model(model_name, quantization_config=None):
    """Load a tokenizer/model pair and place the model on the best available device."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model_kwargs = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if quantization_config is None:
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        print("eos_token_id:", tokenizer.eos_token_id)
        print("pad_token_id:", tokenizer.pad_token_id)
    except Exception as e:
        print(f"Error loading model {model_name}. Make sure you have internet connection "
          f"and the model name is correct. Error: {e}")
        raise
    return tokenizer, model


def get_model_device(model):
    """Return the primary device for regular and accelerate-dispatched models."""
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def synchronize_if_cuda(device):
    """Synchronize CUDA work before reading timers or memory counters."""
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def reset_gpu_peak_memory(device):
    """Reset CUDA peak memory stats when evaluating on a GPU."""
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_gpu_memory_mb(device):
    """Return peak allocated CUDA memory in MiB for the current evaluation window."""
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def generate_headline(model, tokenizer, summary, generation_args):
    prompt = PROMPT.format(article=summary)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    device = get_model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        synchronize_if_cuda(device)
        outputs = model.generate(
            **inputs,
            **generation_args,
            pad_token_id=tokenizer.pad_token_id,
        )
        synchronize_if_cuda(device)

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_headline = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return generated_headline, generated_tokens


def report_metrics(results, latencies, max_new_tokens):
    """Calculate latency, throughput, GPU memory, and ROUGE metrics consistently."""
    if not results:
        raise ValueError("No successful generations were collected; cannot report metrics.")

    latency_array = np.asarray(latencies, dtype=np.float64)
    total_generated_tokens = sum(int(row["actual_new_tokens"]) for row in results)
    total_latency = float(latency_array.sum())
    gpu_memory_values = [float(row.get("gpu_memory_mb", 0.0)) for row in results]

    rouge = load_metric("rouge")
    rouge_scores = rouge.compute(
        predictions=[row["prediction"] for row in results],
        references=[row["reference"] for row in results],
        use_stemmer=True,
    )

    metrics = {
        "samples": len(results),
        "max_new_tokens": max_new_tokens,
        "avg_latency_s": float(latency_array.mean()),
        "p99_latency_s": float(np.percentile(latency_array, 99)),
        "throughput_tokens_per_s": total_generated_tokens / total_latency if total_latency else 0.0,
        "peak_gpu_memory_mb": max(gpu_memory_values) if gpu_memory_values else 0.0,
        "total_generated_tokens": total_generated_tokens,
        "rouge1": float(rouge_scores["rouge1"]),
        "rouge2": float(rouge_scores["rouge2"]),
        "rougeL": float(rouge_scores["rougeL"]),
        "rougeLsum": float(rouge_scores["rougeLsum"]),
    }

    print("Evaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")
    return metrics


def evaluate_model(dataset, model, tokenizer, generation_args, n=20):
    """Run a bounded evaluation loop and return per-sample results plus aggregate metrics."""
    sample_count = min(n, len(dataset))
    model.eval()
    results = []
    latencies = []
    device = get_model_device(model)

    reset_gpu_peak_memory(device)
    for i in range(sample_count):
        sample = dataset[i]
        summary = sample["text"]
        reference = sample["headline"]

        try:
            synchronize_if_cuda(device)
            start_time = time.perf_counter()
            generated_headline, generated_tokens = generate_headline(
                model=model,
                tokenizer=tokenizer,
                summary=summary,
                generation_args=generation_args
            )
            synchronize_if_cuda(device)
            latency = time.perf_counter() - start_time

            results.append({
                "input": summary,
                "reference": reference,
                "prediction": generated_headline,
                "actual_new_tokens": int(generated_tokens.numel()),
                "latency_s": latency,
                "gpu_memory_mb": get_peak_gpu_memory_mb(device),
            })
            latencies.append(latency)

        except Exception as e:
            print(f"Error during generation for sample {i}: {e}")

    metrics = report_metrics(results, latencies, generation_args.get("max_new_tokens", MAX_NEW_TOKENS))
    return results, metrics


def benchmark_configuration(name, dataset, model, tokenizer, generation_args, n=20):
    """Evaluate one optimization configuration and label its metrics."""
    print(f"\n--- Evaluating {name} ---")
    results, metrics = evaluate_model(dataset, model, tokenizer, generation_args, n=n)
    metrics["configuration"] = name
    return results, metrics


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
print(f"Baseline Model Memory Footprint: {memory_mb_32bit:.2f} MB")

benchmark_summary = []
baseline_generation_args = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "use_cache": False
}
_, baseline_metrics = benchmark_configuration(
    "baseline_no_kv_cache",
    dataset,
    model,
    tokenizer,
    baseline_generation_args,
    n=EVAL_SAMPLE_COUNT,
)
benchmark_summary.append(baseline_metrics)

if torch.cuda.is_available():
    print("\n--- Profiling on GPU ---")

    print("Performing GPU warm-up run...")
    
    
    print(dataset[0])
    # TODO: Call run_gpu_inference for warm-up
    generation_args = baseline_generation_args
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
_, kv_cache_metrics = benchmark_configuration(
    "kv_cache",
    dataset,
    model,
    tokenizer,
    generation_args,
    n=EVAL_SAMPLE_COUNT,
)
benchmark_summary.append(kv_cache_metrics)

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
    total_time = 0.0
    total_tokens = 0
    sample_output = "Error during generation."
    local_generation_args = {"max_new_tokens": max_tokens, "use_cache": True}
    device = get_model_device(model)

    with torch.no_grad():
        for i in range(num_runs):
            synchronize_if_cuda(device)
            start_time = time.perf_counter()
            generated_headline, generated_tokens = generate_headline(
                model, tokenizer, prompt, local_generation_args
            )
            synchronize_if_cuda(device)
            total_time += time.perf_counter() - start_time
            total_tokens += int(generated_tokens.numel())
            if i == 0:
                sample_output = generated_headline

    avg_time = total_time / num_runs
    return avg_time, sample_output, total_tokens


print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"Free GPU memory: {free_mem/1e9:.2f} GB / {total_mem/1e9:.2f} GB")
else:
    free_mem, total_mem = 0, 0

#tokenizer, model = load_model(MODEL_NAME, quantization_config=None)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
pruned_model = prune_model_weights(model, amount=0.3)

#pruned_model = pruned_model.to(device)

if torch.cuda.is_available():
    print(f"After load - Free GPU memory: {free_mem/1e9:.2f} GB / {total_mem/1e9:.2f} GB")
# Move model to CPU before pruning
#model = model.cpu()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

#pruned_model = prune_model_weights(model, amount=0.05)

# Move back to GPU after pruning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#pruned_model = pruned_model.to(device)

#pruned_model = prune_model_weights(model, amount=0.3)
NUM_SPEED_RUNS = 3

#tokenizer, model = load_model(MODEL_NAME, quantization_config=None)
if torch.cuda.is_available():
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"After load - Free GPU memory: {free_mem/1e9:.2f} GB / {total_mem/1e9:.2f} GB")

_, pruned_metrics = benchmark_configuration(
    "pruned_30_percent",
    dataset,
    pruned_model,
    tokenizer,
    generation_args,
    n=EVAL_SAMPLE_COUNT,
)
benchmark_summary.append(pruned_metrics)

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

tokenizer_4bit, model_4bit = load_model(MODEL_NAME, quantization_config=quant_config)

memory_mb_4bit = get_model_memory_footprint(model_4bit)
print(f"4-bit Memory Footprint: {memory_mb_4bit:.2f} MB")

model_4bit.eval()
_, quantized_metrics = benchmark_configuration(
    "quantized_4bit",
    dataset,
    model_4bit,
    tokenizer_4bit,
    generation_args,
    n=EVAL_SAMPLE_COUNT,
)
benchmark_summary.append(quantized_metrics)

print("--- Comparable Benchmark Summary ---")
summary_columns = [
    "configuration",
    "avg_latency_s",
    "p99_latency_s",
    "throughput_tokens_per_s",
    "peak_gpu_memory_mb",
    "rouge1",
    "rouge2",
    "rougeL",
    "rougeLsum",
]
print(pd.DataFrame(benchmark_summary)[summary_columns].to_string(index=False))




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
    return end_time - start_time, target_passes, avg_accepted_per_pass

#DRAFT_MODEL_NAME = "/voc/shared/models/llama/Llama-3.2-1B"
#TARGET_MODEL_NAME = "/voc/shared/models/llama/Llama-3.2-3B"

# --- Config ---
TARGET_MODEL_NAME = "gpt2-medium"
DRAFT_MODEL_NAME = "gpt2"
MAX_TOTAL_TOKENS = 50
K_VALUES_TO_TEST = [1, 2, 3, 4, 5, 8, 10]
spec_device = "cuda" if torch.cuda.is_available() else "cpu"
spec_dtype = torch.float16 if spec_device == "cuda" else torch.float32

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

# --- Run experiment ---
results_log = []
print("--- Running Speculative Decoding Experiment ---")
for k in K_VALUES_TO_TEST:
    print(f"Testing with K = {k}...")
    spec_time, spec_passes, avg_accepted = run_speculative_decoding(
        draft_model, target_model, target_tokenizer,   # FIX: correct tokenizer
        PROMPT_TEXT, MAX_TOTAL_TOKENS, k
    )
    results_log.append({
        "K": k,
        "Time (s)": spec_time,
        "Target Passes": spec_passes,
        "Avg. Accepted Tokens": avg_accepted,
    })
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

df_results = pd.DataFrame(results_log)
print("--- Speculative Decoding Experiment Results Summary ---")
print(df_results.to_string())
