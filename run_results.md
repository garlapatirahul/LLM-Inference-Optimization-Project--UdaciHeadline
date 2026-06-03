#Below are outputs when run using a cuda device and comparing different optimization techniques.


#Step 1 Loading the required models and libs.

kagglehub imported successfully
['Llama-4-Scout-17B-16E-Instruct', 'Llama-3.2-3B', 'Llama-3.3-70B-Instruct', 'Llama-3.2-1B']


#Step 2 Cheking Cuda device availability

torch cuda available: True
torch cuda device count: 1
device 0: Tesla T4

#step 3 Checking on Baseline Performance

eos_token_id: 128001
pad_token_id: 128001
dataset has been loaded
4-bit Memory Footprint: 4714.26 MB

--- Profiling on GPU ---
Performing GPU warm-up run...
{'headline': 'Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters', 'text': 'Health experts said it is too early to predict whether demand would match up with the 171 million doses of the new boosters the U.S. ordered for the fall.'}
allocated GB: 0.0
reserved GB: 0.0
Entered generate_headline
Tokenization done
Moved inputs to device
{'input_ids': torch.Size([1, 65]), 'attention_mask': torch.Size([1, 65])}
About to call model.generate
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
model.generate finished
Warm-up complete.
Running inference on GPU and capturing profile...
Entered generate_headline
Tokenization done
Moved inputs to device
{'input_ids': torch.Size([1, 65]), 'attention_mask': torch.Size([1, 65])}
About to call model.generate
model.generate finished
GPU Wall clock time: 26.3895 seconds
GPU Profiler Analysis (Top 5 Operators by Self CUDA Time):
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
          model_inference_gpu         1.73%     449.020ms        99.18%       25.714s       25.714s             1  
                  aten::empty         0.02%       4.798ms         0.02%       4.798ms       3.673us          1306  
                     aten::to         0.03%       6.809ms         0.13%      33.711ms       7.004us          4813  
             aten::lift_fresh         0.00%      16.180us         0.00%      16.180us       0.647us            25  
                aten::detach_         0.00%       9.800us         0.00%      16.625us       3.325us             5  
                      detach_         0.00%       6.825us         0.00%       6.825us       1.365us             5  
              aten::unsqueeze         0.03%       6.902ms         0.03%       8.974ms       6.059us          1481  
             aten::as_strided         0.09%      24.599ms         0.09%      24.599ms       1.727us         14245  
                   aten::isin         0.00%     345.516us         0.00%     414.100us      19.719us            21  
                  aten::fill_         0.01%       3.043ms         0.01%       3.140ms       3.639us           863  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 25.926s


#step 4 KV Caching Enabled

--- Profiling on GPU ---
Performing GPU warm-up run...
{'headline': 'Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters', 'text': 'Health experts said it is too early to predict whether demand would match up with the 171 million doses of the new boosters the U.S. ordered for the fall.'}
Entered generate_headline
Tokenization done
Moved inputs to device
{'input_ids': torch.Size([1, 65]), 'attention_mask': torch.Size([1, 65])}
About to call model.generate
model.generate finished
Warm-up complete.
Running inference on GPU and capturing profile...
Entered generate_headline
Tokenization done
Moved inputs to device
{'input_ids': torch.Size([1, 65]), 'attention_mask': torch.Size([1, 65])}
About to call model.generate
model.generate finished
GPU Wall clock time: 3.6024 seconds
GPU Profiler Analysis (Top 5 Operators by Self CUDA Time):
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
          model_inference_gpu         4.84%     164.039ms       100.00%        3.392s        3.392s             1  
                  aten::empty         0.04%       1.408ms         0.04%       1.408ms       2.751us           512  
...
                  aten::fill_         0.01%     490.809us         0.02%     530.810us       1.226us           433  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.392s

#step 5 Applying Pruning and compressing the model

CUDA available: True
CUDA device count: 1
Tesla T4
Free GPU memory: 15.52 GB / 15.64 GB
Pruning first linear layer only
After load - Free GPU memory: 15.52 GB / 15.64 GB
After load - Free GPU memory: 15.52 GB / 15.64 GB
Entered generate_headline
Tokenization done
Moved inputs to device
{'input_ids': torch.Size([1, 65]), 'attention_mask': torch.Size([1, 65])}
About to call model.generate
model.generate finished
Entered generate_headline
Tokenization done
Moved inputs to device
{'input_ids': torch.Size([1, 65]), 'attention_mask': torch.Size([1, 65])}
About to call model.generate
model.generate finished
Entered generate_headline
Tokenization done
Moved inputs to device
{'input_ids': torch.Size([1, 65]), 'attention_mask': torch.Size([1, 65])}
About to call model.generate
model.generate finished
Result:
  - Avg Time: 5.4826s
  - Output: 'Doses of new COVID boosters will be needed to match demand for the fall's
  - Total Tokens: 56.0000

#step 6 Applying Quantization

`low_cpu_mem_usage` was None, now set to True since model is quantized.
4-bit Memory Footprint: 965.13 MB
Entered generate_headline
Tokenization done
Moved inputs to device
{'input_ids': torch.Size([1, 65]), 'attention_mask': torch.Size([1, 65])}
About to call model.generate
model.generate finished
Entered generate_headline
Tokenization done
Moved inputs to device
{'input_ids': torch.Size([1, 65]), 'attention_mask': torch.Size([1, 65])}
About to call model.generate
model.generate finished
Entered generate_headline
Tokenization done
Moved inputs to device
{'input_ids': torch.Size([1, 65]), 'attention_mask': torch.Size([1, 65])}
About to call model.generate
model.generate finished
4-bit Avg. Latency: 0.8381 s 4-bit Throughput: 71.5913 s (over 3 runs)
Demand for COVID-19 vaccines likely to exceed supply

Commentary:
The demand for COVID-19





Self CPU time total: 25.926s



#step7 Advanced Decoding: Speculative Decoding

Reading the results:
 At K=1 you get 50 target passes (one per token, no speedup — basic
decoding). As K grows, target passes drop and avg accepted tokens climb,
which is exactly the win: fewer expensive target forward passes because
the cheap draft model is correctly predicting multiple tokens at a time.



Loading Target Model: gpt2-medium on cuda (torch.float16)...
Loading weights: 100%
 292/292 [00:03<00:00, 126.02it/s]
Loading Draft Model: gpt2 on cuda (torch.float16)...
Loading weights: 100%
 148/148 [00:01<00:00, 198.35it/s]
[transformers] The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
--- Running Speculative Decoding Experiment ---
Testing with K = 1...
Testing with K = 2...
Testing with K = 3...
Testing with K = 4...
Testing with K = 5...
Testing with K = 8...
Testing with K = 10


--- Speculative Decoding Experiment Results Summary ---
    K  Time (s)  Target Passes  Avg. Accepted Tokens
0   1  2.788328             50              1.000000
1   2  1.565935             27              1.851852
2   3  1.260171             19              2.631579
3   4  1.356594             17              3.058824
4   5  1.462748             16              3.250000
5   8  1.549432             12              4.333333
6  10  1.844262             12              4.333333

Sweet spot: K=3 at 1.26s — the fastest of the batch. After that, time
creeps back up even though passes keep falling. Classic spec-decoding
tradeoff: past a certain K, the draft model's predictions diverge from
the target, so you spend draft compute generating tokens that get
rejected. The wasted draft work outweighs the saved target passes.

K=8 and K=10 produce identical results (12 passes, 4.33 accepted). That
means generation hits the MAX_TOTAL_TOKENS=50 cap or EOS before the larger
K budget is fully used — the extra draft headroom is unused, so K=10 just
costs slightly more draft time (1.55s → 1.84s) for no benefit.

