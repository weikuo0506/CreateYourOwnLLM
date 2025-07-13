import torch
import torch.nn as nn
import sentencepiece as spm

LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16
}

tokenizer_file = "Llama-2-7b/tokenizer.model"
weights_file = "Llama-2-7b/consolidated.00.pth"

class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode_as_ids(text)

    def decode(self, ids):
        return self.tokenizer.decode_pieces(ids)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        # Learnable scaling parameter (gamma)
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        # Compute root mean square (RMS)
        means = x.pow(2).mean(dim=-1, keepdim=True)
        # Normalize input by RMS
        x_normed = x * torch.rsqrt(means + self.eps)
        # Apply scaling and restore original dtype
        return (x_normed * self.weight).to(dtype=x.dtype)

def precompute_rope_params(seq_len, head_dim):
    """
    Precompute sin and cos tensors for RoPE.

    Args:
        seq_len: sequence length
        head_dim: embedding dimension (must be even)

    Returns:
        sin, cos: tensors of shape (seq_len, dim//2)
    """
    half_dim = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(half_dim).float() / half_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", positions, inv_freq)  # (seq_len, half_dim)
    return torch.sin(angles), torch.cos(angles)

def rotary_pos_emb(x, sin, cos):
    """
    Apply Rotary Positional Embedding on input tensor x using precomputed sin and cos.

    Args:
        x: tensor of shape (batch, seq_len, dim)
        sin: precomputed sin tensor of shape (seq_len, dim//2)
        cos: precomputed cos tensor of shape (seq_len, dim//2)

    Returns:
        tensor same shape as x with RoPE applied.
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch, num_heads, seq_len, head_dim = x.shape

    # x: (batch, seq_len, dim) -> (batch, seq_len, half_dim, 2)
    x_ = x.view(batch, num_heads, seq_len, head_dim // 2, 2)

    # ➤ Crop sin/cos to match actual seq_len
    sin = sin[:seq_len, :]
    cos = cos[:seq_len, :]

    x_rotated = torch.zeros_like(x_)
    x_rotated[..., 0] = x_[..., 0] * cos - x_[..., 1] * sin
    x_rotated[..., 1] = x_[..., 0] * sin + x_[..., 1] * cos

    return x_rotated.view_as(x)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Use nn.Linear with shared kwargs to reduce repetition
        linear_kwargs = dict(bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, d_out, **linear_kwargs)
        self.W_key = nn.Linear(d_in, d_out, **linear_kwargs)
        self.W_value = nn.Linear(d_in, d_out, **linear_kwargs)
        self.out_proj = nn.Linear(d_out, d_out, **linear_kwargs)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

        cos, sin = precompute_rope_params(seq_len=context_length, head_dim=self.head_dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Project inputs
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        keys = rotary_pos_emb(keys, self.cos, self.sin)
        queries = rotary_pos_emb(queries, self.cos, self.sin)

        # Attention scores with causal mask
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], float('-inf'))

        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)

        context_vec = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)

        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x

class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# updated to v2 to support temperature scaling and top_k sampling
def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None,device=torch.device("cpu")):
    for _ in range(max_new_tokens):
        idx = idx.to(device)
        idx_cond = idx[:, -context_size:]

        # Get logits from model
        with torch.no_grad():
            logits = model(idx_cond)

        # Take logits for the last time step
        # (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k, dim=-1)  # (batch, top_k)
            threshold = top_logits[:, -1].unsqueeze(-1)  # (batch, ) -> (batch, 1)
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float('-inf')),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Sample from distribution
            idx_next = torch.multinomial(probas, num_samples=1)  # (batch, 1)
        else:
            # Greedy sampling
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        if eos_id is not None and idx_next == eos_id:
            break

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach()) if isinstance(right, torch.Tensor) else torch.nn.Parameter(torch.tensor(right))


def load_weights_into_llama(model, param_config, params):
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]

        # map of attribute path (relative to block) -> param name
        attr_param_map = {
            f"att.W_query.weight": f"layers.{l}.attention.wq.weight",
            f"att.W_key.weight": f"layers.{l}.attention.wk.weight",
            f"att.W_value.weight": f"layers.{l}.attention.wv.weight",
            f"att.out_proj.weight": f"layers.{l}.attention.wo.weight",
            f"norm1.weight": f"layers.{l}.attention_norm.weight",
            f"ff.fc1.weight": f"layers.{l}.feed_forward.w1.weight",
            f"ff.fc2.weight": f"layers.{l}.feed_forward.w3.weight",  # swapped order
            f"ff.fc3.weight": f"layers.{l}.feed_forward.w2.weight",
            f"norm2.weight": f"layers.{l}.ffn_norm.weight",
        }

        for attr_path, param_name in attr_param_map.items():
            obj = block
            *parents, attr = attr_path.split('.')
            for p in parents:
                obj = getattr(obj, p)
            old_tensor = getattr(obj, attr)
            setattr(obj, attr, assign(old_tensor, params[param_name]))

    model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
    model.out_head.weight = assign(model.out_head.weight, params["output.weight"])


def text_to_tensor(text, tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def tensor_to_text(tensor, tokenizer):
    return tokenizer.decode(tensor.squeeze(0).tolist())


def build_tokenizer():
    return LlamaTokenizer(tokenizer_file)

def load_model(device=torch.device("cpu")):
    torch.manual_seed(123)
    model = Llama2Model(LLAMA2_CONFIG_7B).to(device)
    model.eval()
    return model

def load_pretrained_weights(model,device=torch.device("cpu")):
    weights = torch.load(weights_file, weights_only=True)
    load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)
    return model.to(device=device)

def complete_text(input_text, model, max_new_tokens=20, config=LLAMA2_CONFIG_7B,device=torch.device("cpu")):
    tokenizer = build_tokenizer()
    encoded_tensor = text_to_tensor(input_text, tokenizer)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=max_new_tokens,
        context_size=config["context_length"],
        device=device
    )

    decoded_text = tensor_to_text(out, tokenizer)
    return decoded_text

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    with torch.no_grad():
        result = complete_text(start_context, model, 20,device=device)
        print(result)
    model.train()


def main():
    start_context = "Once upon a time there"
    device = torch.device("cpu")
    model = load_model(device)
    model = load_pretrained_weights(model,device)
    result = complete_text(start_context, model, 10,device=device)
    print(result)


if __name__ == "__main__":
    main()
