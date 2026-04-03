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
