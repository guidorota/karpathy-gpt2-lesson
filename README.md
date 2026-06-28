# About

Code created while following Karpathy's "Let's reproduce GPT-2 (124M)" lesson.

## Optimising training runtime

* 16 batches @ 1024 tokens doesn't fit in 24GB of memory on 4090, 10 batches is the max (22676MiB used), afterwards we get an OOM.
* 10 batches @ 1024 tokens FP32: mean per iter: 388.93ms, mean tps: 26329.16
* 8 batches @ 1024 tokens FP32: mean per iter: 310.01ms, mean tps: 26425.05
* 4 batches @ 1024 tokens FP32: mean per iter: 163.68ms, mean tps: 25025.09

With FP32 iteration time scales linearly with batch size. Different from what Karpathy mentioned, but that might be due to the fact that we're not using TF32 yet. I'll run again later.

With TF32:
* 10 batches @ 1024 tokens TF32: mean per iter: 328.97ms, mean tps: 31127.51
* 8 batches @ 1024 tokens TF32: mean per iter: 262.71ms, mean tps: 31182.58
* 4 batches @ 1024 tokens TF32: mean per iter: 139.62ms, mean tps: 29335.79