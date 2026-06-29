from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import tiktoken
import time
import inspect
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # q, k, v in a single layer
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # Get all 3 q, k, v in a single operation
        qkv = self.c_attn(x)
        # Split each
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Split each head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Perform attention for each head
        # att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Concat all heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Sharing weights between input embedding and output linear network
        self.transformer.wte.weight = self.lm_head.weight

        # init params by iterating over all sub-modules
        self.apply(self._init_weights)

    # There's a subtle bug that Karpathy doesn't fix, wte and lm_head
    # weights are initialized twice, but it's harmless
    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            # 2 because each transformer layer has 2 residual layers
            # that add to the residual path, attn and mlp
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, self.config.vocab_size), targets.view(B*T))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('loading weights from pretrained gpt: %s' % model_type)

        # Hyperparams
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Init HF model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # Ignore causal mask
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # Ignore causal mask
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizer(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Usually we only weight decay that participate in matmul, and the embeddings
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # Common not to weight decay 1-dimension params (biases, layernorm, scales, etc.)
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # Weight decay regularises the network by making sure that no weight
        # grows too large, hence helps distribute probabilities more
        # uniformely across different options
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")

        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer
    
class DataLoaderLite:

    def __init__(self, B, T, device):
        self.B = B
        self.T = T
        self.device = device

        with open("tiny-shakespeare.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        x = x.to(device)
        y = y.to(device)
        return x, y

num_return_sequences = 5
max_length = 30

device = "cuda"
print(f"using device: {device}")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

n_batch = 17
batch_size = 1024

train_loader = DataLoaderLite(B=n_batch, T=batch_size, device=device)

torch.set_float32_matmul_precision("high") # Set FT32 precision

# model = GPT.from_pretrained('gpt2')
# Override vocab_size for a number that can be
# divided by more power of twos.
#
# We're essentially adding tokens that are never
# going to be used, and that will be driven to prob=0
# during training.
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
print("compiling model...")
model = torch.compile(model)
print("finished compiling model")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(step):
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps # +1 because lr = 0 would be useless
    # Min learning rate if over decay threshold
    if step > max_steps:
        return min_lr
    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=3e-4)
dts = []
tpss = []
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    optimizer.zero_grad()

    # BFLOAT16 means we don't need to use gradient scalers,
    # which would otherwise be required if using FLOAT26, which reduces
    # the number of bits reserved for the exponent
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize() # Ensure GPU work is complete
    t1 = time.time()
    dt = (t1 - t0)*1000
    tps = (n_batch * batch_size)/(t1 - t0)
    if step != 0: # Runtime of the first iteration is an outlier
        dts.append(dt)
        tpss.append(tps)
    print(f"step {step:4d}, loss: {loss.item()}, norm: {norm:.4f}, lr: {lr:.4e}, dt: {dt:.2f}ms, tps: {tps:.2f}")

mean_time = sum(dts) / len(dts)
mean_tps = sum(tpss) / len(tpss)
print(f"mean per iter: {mean_time:.2f}ms, mean tps: {mean_tps:.2f}")

import sys; sys.exit(0)