import os
import numpy as np
import torch
import random
import tiktoken

split = "val"
n_shard_samples = 3
sequence_len = 500

enc = tiktoken.get_encoding("gpt2")

data_root = "edu_fineweb10B"
shards = os.listdir(data_root)
shards = [s for s in shards if split in s]
shards = sorted(shards)
shards = [os.path.join(data_root, s) for s in shards]

assert len(shards) > 0, f"no shards found for split {split}"
print(f"found {len(shards)} shards for split {split}")

max_tokens = len(shards) * int(1e8)

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

for i in range(len(shards)):
    tokens = load_tokens(shards[i])
    for _ in range(n_shard_samples):
        start = random.randint(0, len(tokens) - sequence_len)
        data = tokens[start:start+sequence_len]
        print(enc.decode(data.tolist()))
        print("------------")
