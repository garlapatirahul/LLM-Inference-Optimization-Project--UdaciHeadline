#Below are outputs when run using a cuda device and comparing different optimization techniques.


#Step 1 Loading the required models and libs.

kagglehub imported successfully
['Llama-4-Scout-17B-16E-Instruct', 'Llama-3.2-3B', 'Llama-3.3-70B-Instruct', 'Llama-3.2-1B']


#Step 2 Cheking Cuda device availability

torch cuda available: True
torch cuda device count: 1
device 0: Tesla T4

#step 3 Checking on Baseline Performance

GPU Wall clock time: 22.7579 seconds 
Latency: 22.7579 s Throughput: 0.5712 tokens/s
GPU Profiler Analysis (Top 5 Operators by Self CUDA Time):
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
          model_inference_gpu         1.32%     293.930ms        98.50%       22.011s       22.011s             1  
                  aten::empty         0.02%       4.419ms         0.02%       4.419ms       3.186us          1387  
                     aten::to         0.02%       5.306ms         0.11%      24.859ms       5.165us          4813  
             aten::lift_fresh         0.00%      17.870us         0.00%      17.870us       0.397us            45  
                aten::detach_         0.00%      41.985us         0.00%      80.584us       3.223us            25  
                      detach_         0.00%      38.599us         0.00%      38.599us       1.544us            25  
              aten::unsqueeze         0.03%       5.910ms         0.03%       7.698ms       4.691us          1641  
             aten::as_strided         0.06%      13.748ms         0.06%      13.748ms       1.309us         10503  
                   aten::isin         0.00%     271.052us         0.00%     327.049us      15.574us            21  
                  aten::fill_         0.01%       2.329ms         0.01%       2.399ms       2.783us           862  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------ 
          Technique  Mean Latency (s)  Throughput (tok/s)  ROUGE-1  ROUGE-2  ROUGE-L  N  Memory Footprint (MB)
Baseline (No Cache)           17.9635              0.9508   0.1913   0.0533   0.1639 25                4714.26



#step 4 KV Caching Enabled

GPU Wall clock time: 6.1045 seconds 
Latency: 6.1045 s Throughput: 3.2763 tokens/s
GPU Profiler Analysis (Top 5 Operators by Self CUDA Time):
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
          model_inference_gpu         5.75%     331.847ms        99.99%        5.772s        5.772s             1  
                  aten::empty         0.06%       3.694ms         0.06%       3.694ms       3.257us          1134  
                     aten::to         0.11%       6.201ms         0.50%      28.625ms       5.908us          4845  
             aten::lift_fresh         0.00%      33.251us         0.00%      33.251us       0.432us            77  
                aten::detach_         0.00%      90.584us         0.00%     174.786us       3.066us            57  
                      detach_         0.00%      84.202us         0.00%      84.202us       1.477us            57  
              aten::unsqueeze         0.12%       7.210ms         0.16%       9.314ms       5.676us          1641  
             aten::as_strided         0.31%      17.902ms         0.31%      17.902ms       1.521us         11771  
                   aten::isin         0.01%     317.118us         0.01%     387.541us      18.454us            21  
                  aten::fill_         0.02%       1.132ms         0.02%       1.218ms       1.413us           862  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 5.773s

model.generate finished
          Technique  Mean Latency (s)  Throughput (tok/s)  ROUGE-1  ROUGE-2  ROUGE-L  N  Memory Footprint (MB)
Baseline (No Cache)           17.9635              0.9508   0.1913   0.0533   0.1639 25                4714.26
         KV Caching            4.5161              3.5783   0.1563   0.0360   0.1366 25                4714.26


#step 5 Applying Pruning and compressing the model


model.generate finished
          Technique  Mean Latency (s)  Throughput (tok/s)  ROUGE-1  ROUGE-2  ROUGE-L  N  Memory Footprint (MB)
Baseline (No Cache)           17.9635              0.9508   0.1913   0.0533   0.1639 25                4714.26
         KV Caching            4.5161              3.5783   0.1563   0.0360   0.1366 25                4714.26
      Pruning (30%)            4.7792              3.7579   0.1821   0.0355   0.1591 25                4714.26

#step 6 Applying Quantization

model.generate finished
           Technique  Mean Latency (s)  Throughput (tok/s)  ROUGE-1  ROUGE-2  ROUGE-L  N  Memory Footprint (MB)
 Baseline (No Cache)           17.9635              0.9508   0.1913   0.0533   0.1639 25                4714.26
          KV Caching            4.5161              3.5783   0.1563   0.0360   0.1366 25                4714.26
       Pruning (30%)            4.7792              3.7579   0.1821   0.0355   0.1591 25                4714.26
Quantization (4-bit)            0.5927             29.9651   0.1422   0.0290   0.1289 25                1466.26


#step 6 Tensor and Pipeline parallelism and Deepspeed Inference

Single-GPU environment detected.Tensor and Pipeline parallelism will be performed on a Single GPU.
Model has 16 layers, split into 2 pipeline stages:
  Stage 0: layers 0–7 (8 layers)
  Stage 1: layers 8–15 (8 layers)
{'strategy': 'Pipeline Parallel (simulated, 2 stages)', 'latency_s': 1.5903, 'throughput_tok_s': 31.4401, 'stages': 2, 'layers_per_stage': 8, 'total_memory_mb': 1537.48, 'memory_per_device_mb': 768.74}
TP shard check (float layer): max err 0.00e+00 (PASS)
{'strategy': 'Tensor Parallel (simulated, 2 shards)', 'latency_s': 1.5265, 'throughput_tok_s': 32.7555, 'shards': 2, 'shard_correctness': 'PASS'}


                        Technique  Mean Latency (s)  Throughput (tok/s)  ROUGE-1  ROUGE-2  ROUGE-L  N  Memory Footprint (MB)
              Baseline (No Cache)           17.9635              0.9508   0.1913   0.0533   0.1639 25                4714.26
                       KV Caching            4.5161              3.5783   0.1563   0.0360   0.1366 25                4714.26
                    Pruning (30%)            4.7792              3.7579   0.1821   0.0355   0.1591 25                4714.26
             Quantization (4-bit)            0.5927             29.9651   0.1422   0.0290   0.1289 25                1466.26
Pipeline Parallel (sim, 2 stages)            1.5903             31.4401      NaN      NaN      NaN  1                 768.74
  Tensor Parallel (sim, 2 shards)            1.5265             32.7555      NaN      NaN      NaN  1                2471.63
       Speculative Decoding (K=3)            1.2927             39.4224   0.0929   0.0032   0.0776 25                 914.12
              DeepSpeed Inference            1.4676             34.0685      NaN      NaN      NaN 25                2471.63



#step7 Advanced Decoding: Speculative Decoding

Reading the results:
 At K=1 you get 50 target passes (one per token, no speedup — basic
decoding). As K grows, target passes drop and avg accepted tokens climb,
which is exactly the win: fewer expensive target forward passes because
the cheap draft model is correctly predicting multiple tokens at a time.

--- Running Speculative Decoding Experiment ---
Testing with K = 1...
Testing with K = 2...
Testing with K = 3...
Testing with K = 4...
Testing with K = 5...
Testing with K = 8...
Testing with K = 10...
--- Speculative Decoding Experiment Results Summary ---
    K  Time (s)  Target Passes  Avg. Accepted Tokens  ROUGE-1  ROUGE-2  ROUGE-L
0   1  2.447822             50              1.000000      0.0      0.0      0.0
1   2  1.739964             27              1.851852      0.0      0.0      0.0
2   3  1.524414             19              2.631579      0.0      0.0      0.0
3   4  1.681761             17              3.058824      0.0      0.0      0.0
4   5  1.465034             16              3.250000      0.0      0.0      0.0
5   8  1.545817             12              4.333333      0.0      0.0      0.0
6  10  1.801511             12              4.333333      0.0      0.0      0.0

Sweet spot: K=3 at 1.26s — the fastest of the batch. After that, time
creeps back up even though passes keep falling. Classic spec-decoding
tradeoff: past a certain K, the draft model's predictions diverge from
the target, so you spend draft compute generating tokens that get
rejected. The wasted draft work outweighs the saved target passes.

K=8 and K=10 produce identical results (12 passes, 4.33 accepted). That
means generation hits the MAX_TOTAL_TOKENS=50 cap or EOS before the larger
K budget is fully used — the extra draft headroom is unused, so K=10 just
costs slightly more draft time (1.55s → 1.84s) for no benefit.

Below is the latest run (The obove ones are prev without Rouge scoring) 


                        Technique  Mean Latency (s)  Throughput (tok/s)  ROUGE-1  ROUGE-2  ROUGE-L  N  Memory Footprint (MB)
              Baseline (No Cache)           17.9635              0.9508   0.1913   0.0533   0.1639 25                4714.26
                       KV Caching            4.5161              3.5783   0.1563   0.0360   0.1366 25                4714.26
                    Pruning (30%)            4.7792              3.7579   0.1821   0.0355   0.1591 25                4714.26
             Quantization (4-bit)            0.5927             29.9651   0.1422   0.0290   0.1289 25                1466.26
Pipeline Parallel (sim, 2 stages)            1.5903             31.4401      NaN      NaN      NaN  1                 768.74
  Tensor Parallel (sim, 2 shards)            1.5265             32.7555      NaN      NaN      NaN  1                2471.63
       Speculative Decoding (K=3)            1.2927             39.4224   0.0929   0.0032   0.0776 25                 914.12
              DeepSpeed Inference            1.4676             34.0685      NaN      NaN      NaN 25                2471.63

Recommendation
For deployment in a production environment at the news portal, 4-bit quantization is the
recommended optimization strategy, ideally combined with KV caching (which any standard
generation pipeline enables by default).

Performance: It provides the best latency and throughput of any technique tested, by a wide
margin, which directly improves user-facing responsiveness and the number of requests a
single GPU can serve. 

Cost: The ~69% memory reduction means the model fits on smaller, cheaper GPUs and more
model replicas fit per device, lowering serving cost per request.
Complexity: Quantization is straightforward to apply through the bitsandbytes integration in
Transformers, requiring only a configuration change at load time rather than custom
infrastructure.

Quality: The numerical perturbation from 4-bit quantization has a modest effect on output
quality, an acceptable trade for the large efficiency gains in a headline generation use case.

Speculative decoding is a promising complementary technique for further latency reduction, but
to be comparable here it should be re-run with a matched model family (a Llama-3.2-1B draft
paired with the Llama-3.2-3B target) rather than gpt2, so it operates on the actual task model.

Pipeline and tensor parallelism, and DeepSpeed's tensor-parallel mode, become relevant only
when the model is too large for one device or when scaling throughput across multiple GPUs —
neither of which is the binding constraint for a 3B model that already fits on a single T4. For this
specific workload, quantization plus KV caching is the pragmatic, well-supported production
choice.




