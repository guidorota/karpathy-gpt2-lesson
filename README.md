# About

Code created while following Karpathy's "Let's reproduce GPT-2 (124M)" lesson.

## Optimising training runtime

* 16 batches @ 1024 tokens doesn't fit in 24GB of memory on 4090, 10 batches is the max (22676MiB used), afterwards we get an OOM.
* 10 batches @ 1024 tokens FP32: mean per iter: 388.93ms, mean tps: 26329.16
* 8 batches @ 1024 tokens FP32: mean per iter: 310.01ms, mean tps: 26425.05
* 4 batches @ 1024 tokens FP32: mean per iter: 163.68ms, mean tps: 25025.09

With FP32 iteration time scales linearly with batch size. Different from what Karpathy mentioned, but that might be due to the fact that we're not using TF32 yet. I'll run again later.

### With TF32:

* 10 batches @ 1024 tokens: mean per iter: 328.97ms, mean tps: 31127.51
* 8 batches @ 1024 tokens: mean per iter: 262.71ms, mean tps: 31182.58
* 4 batches @ 1024 tokens: mean per iter: 139.62ms, mean tps: 29335.79

The low TPS at 4 batches could be due to under-utilization: memory utilization (read/write, not allocation) is at 86% at 4 batches, 96% at 8 batches, and 100% at 10 batches. This also indicates that we're memory bound at 10 batches.

### With BF16 (forward and loss only)

* 10 batches @ 1024 tokens: mean per iter: 253.65ms, mean tps: 40370.40
* 8 batches @ 1024 tokens: mean per iter: 203.31ms, mean tps: 40292.21
* 4 batches @ 1024 tokens: mean per iter: 108.48ms, mean tps: 37756.75

Similarly to what Karpathy's experiencing on his (A100?) GPU, we're also bound by memory bandwidth, as we're not quite getting 2x of TF32 when using BF16 (datasheet mentions peak TFLOPS for FT32 = 82.6, and BF16 = 165.2).

### With torch.compile and BF16 (forward and loss only)

* 17 batches @ 1024 tokens: mean per iter: 201.52ms, mean tps: 86381.87
* 16 batches @ 1024 tokens: mean per iter: 193.99ms, mean tps: 84460.04
* 10 batches @ 1024 tokens: mean per iter: 125.90ms, mean tps: 81352.11
* 8 batches @ 1024 tokens: mean per iter: 100.90ms, mean tps: 81187.24
* 4 batches @ 1024 tokens: mean per iter: 57.52ms, mean tps: 71215.55

Interestingly with torch.compile memory utilisation at 10 batches dropped to about 15GB, giving us more headroom to run up to 17 batches. 10 -> 16 shows 3.82% increase, and 16 -> 17 shows 2.27% increase.

It's likely there is more room for improvement since memory utilization is not maxed out at 17 batches, but at this point we're limited by the amount of available memory. This seems to be corroborated by plotting memory utilization across time when training for 1000 iteration: with 17 batches, memory utilization is consistently higher than 16 batches. Moreover, neither is reaching 100% (tops at 92% for 17 batches), meaning that there is likely some more room for improvement.

### Flash attention + all previous optimisations

* 17 batches @ 1024 tokens: mean per iter: 137.12ms, mean tps: 126958.62
* 16 batches @ 1024 tokens: mean per iter: 133.59ms, mean tps: 122648.53

TPS is still higher with 17 batches even when introducing flash attention, even if it's a prime number instead of being a power of 2.

### Rounding numbers up to power of twos

vocab_size 50257 -> 50304

* 17 batches @ 1024 tokens: mean per iter: 136.33ms, mean tps: 127701.01
* 16 batches @ 1024 tokens: mean per iter: 128.66ms, mean tps: 127341.65

Using numbers that are close to power of twos allows us to make the best of most cuda kernels, which are optimised for running operations on data sizes that are power of twos.