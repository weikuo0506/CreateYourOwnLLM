import torch
import torch.nn as nn
from typing import Optional
import logging
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe


LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,   # Increased vocabulary size for broader language coverage
    "context_length": 8192,  # Extended context window for handling longer sequences
    "emb_dim": 4096,         # Embedding dimension for token representations
    "n_heads": 32,           # Number of attention heads in each self-attention layer
    "n_layers": 32,          # Total number of transformer blocks
    "hidden_dim": 14_336,    # Expanded feedforward network dimension (MLP inner size)
    "n_kv_groups": 8,        # Number of key-value groups for grouped-query attention (GQA)
    "rope_base": 500_000.0,  # Higher RoPE base to better encode longer positions
    "rope_freq": None,       # Optional override for RoPE frequency scaling
    "dtype": torch.bfloat16  # Use bfloat16 for lower memory usage and faster compute
}

# Set up logging only once
log = logging.getLogger("Llama3_v1")

class Llama3Tokenizer:
    """
    Tokenizer wrapper for LLaMA 3 using custom tiktoken BPE files.

    Automatically loads custom merge rules, special tokens, and regex-based tokenization pattern.
    """

    def __init__(self, model_path: str):
        """
        Initialize the tokenizer with a given BPE model file.

        Args:
            model_path (str): Path to the .tiktoken file used by LLaMA 3.
        """
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Tokenizer model file not found: {model_path}")

        # Load mergeable BPE ranks from file
        mergeable_ranks = load_tiktoken_bpe(str(model_path))

        # Define special token IDs
        special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }

        # Add reserved special tokens from 128002 to 128257 (excluding used IDs)
        special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i
            for i in range(256)
            if (128002 + i) not in special_tokens.values()
        })

        # Regex pattern string used for LLaMA-style tokenization
        pat_str = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
            r"[^\r\n\p{L}\p{N}]?\p{L}+|"
            r"\p{N}{1,3}|"
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
            r"\s*[\r\n]+|"
            r"\s+(?!\S)|"
            r"\s+"
        )

        self.special_tokens = special_tokens

        # Create the tiktoken Encoding instance
        self.model = tiktoken.Encoding(
            name=model_path.name,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

    def encode(self, text: str, bos: bool = False, eos: bool = False,
               allowed_special: set = set(), disallowed_special=()) -> list[int]:
        """
        Encode a text string into token IDs.

        Args:
            text (str): Input string to tokenize.
            bos (bool): Whether to prepend <|begin_of_text|> token.
            eos (bool): Whether to append <|end_of_text|> token.
            allowed_special (set): Set of allowed special token strings.
            disallowed_special: Set or policy for disallowed tokens.

        Returns:
            List[int]: Token ID list.
        """
        tokens = []
        if bos:
            tokens.append(self.special_tokens["<|begin_of_text|>"])

        tokens += self.model.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special
        )

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            tokens (List[int]): Token ID list.

        Returns:
            str: Decoded string.
        """
        return self.model.decode(tokens)

def download_tokenizer_if_needed(repo_id: str, filename: str, local_dir: str) -> str:
    local_path = Path(local_dir) / filename
    if local_path.exists():
        print(f"Tokenizer file {local_path} already exists, skipping.")
        return str(local_path)

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir
    )


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

def precompute_rope_params(
    context_length: int,
    head_dim: int,
    theta_base: float = 500000.0,  # Default base for LLaMA 3
    freq_config: Optional[dict] = None,
):
    """
    Precompute sin and cos tensors for RoPE with optional frequency scaling/smoothing.

    Args:
        context_length: Sequence length
        head_dim: Embedding dimension (must be even)
        theta_base: Base for inverse frequency calculation (default 500000)
        freq_config: Optional dict with keys:
            - original_context_length: int, original training context length
            - low_freq_factor: float, low frequency threshold factor (>1)
            - high_freq_factor: float, high frequency threshold factor (>1)
            - factor: float, scaling factor (>1)

    Returns:
        sin, cos: Tensors of shape (seq_len, half_dim)
    """
    assert head_dim % 2 == 0, "head_dim must be even"

    half_dim = head_dim // 2
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))

    if freq_config is not None:
        # Extract frequency config parameters
        orig_len = freq_config["original_context_length"]
        low_factor = freq_config["low_freq_factor"]
        high_factor = freq_config["high_freq_factor"]
        scale_factor = freq_config["factor"]

        # Compute wavelength
        wavelen = 2 * torch.pi / inv_freq
        low_wavelen = orig_len / low_factor
        high_wavelen = orig_len / high_factor

        # Scale inverse frequencies for low frequency bands
        condition = wavelen > low_wavelen
        inv_freq_scaled = torch.where(condition, inv_freq / scale_factor, inv_freq)

        # Compute smooth factor for medium frequency band
        smooth_factor = (orig_len / wavelen - low_factor) / (high_factor - low_factor)
        smooth_factor = smooth_factor.clamp(0.0, 1.0)
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / scale_factor) + smooth_factor * inv_freq

        # Apply smoothed frequencies for medium band
        is_medium = (wavelen <= low_wavelen) & (wavelen >= high_wavelen)
        inv_freq = torch.where(is_medium, smoothed_inv_freq, inv_freq_scaled)

    # Compute position angles
    positions = torch.arange(context_length, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", positions, inv_freq)  # Shape: (seq_len, half_dim)
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

class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
            cos, sin = precompute_rope_params(context_length, head_dim, rope_base, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]

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

class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads,
            num_kv_groups,
            rope_base=10_000,
            rope_config=None,
            dtype=None
        ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_out // num_heads
        self.group_size = num_heads // num_kv_groups
        log.debug(f"d_out={self.d_out}, num_heads={self.num_heads}, num_kv_groups={self.num_kv_groups}, head_dim={self.head_dim}, group_size={self.group_size}")

        linear_kwargs = dict(bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, d_out, **linear_kwargs)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, **linear_kwargs)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, **linear_kwargs)
        self.out_proj = nn.Linear(d_out, d_out, **linear_kwargs)

        mask, cos, sin = SharedBuffers.get_buffers(
            context_length, self.head_dim, rope_base, rope_config, dtype
        )
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, seq_len, _ = x.shape
        log.debug("shape of x: %s", x.shape)

        queries = self.W_query(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        log.debug("shape of queries: %s", queries.shape)
        log.debug("shape of keys: %s", keys.shape)

        # Apply rotary positional embeddings
        queries = rotary_pos_emb(queries, self.cos, self.sin)
        keys = rotary_pos_emb(keys, self.cos, self.sin)
        log.debug("shape of queries: %s", queries.shape)

        # Repeat keys and values to match num_heads
        keys = keys.repeat_interleave(self.group_size, dim=1)  # (b, num_heads, seq_len, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)
        log.debug("shape of keys: %s", keys.shape)
        log.debug("shape of values: %s", values.shape)

        # Compute attention scores with causal mask
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        log.debug("shape of attn_scores: %s", attn_scores.shape)

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        log.debug("shape of attn_weights: %s", attn_weights.shape)
        assert keys.shape[-1] == self.head_dim

        context = torch.matmul(attn_weights, values)  # (b, num_heads, seq_len, head_dim)
        log.debug("shape of context: %s", context.shape)
        context = context.transpose(1, 2).reshape(b, seq_len, self.d_out)
        log.debug("shape of context: %s", context.shape)

        out = self.out_proj(context)
        log.debug("shape of out: %s", out.shape)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],  # NEW
            rope_base=cfg["rope_base"],        # NEW
            rope_config=cfg["rope_freq"],      # NEW
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)

        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-5)
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

class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
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
    # Embedding
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"])

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]

        # map of attribute path (relative to block) -> param name
        attr_param_map = {
            f"att.W_query.weight": f"model.layers.{l}.self_attn.q_proj.weight",
            f"att.W_key.weight": f"model.layers.{l}.self_attn.k_proj.weight",
            f"att.W_value.weight": f"model.layers.{l}.self_attn.v_proj.weight",
            f"att.out_proj.weight": f"model.layers.{l}.self_attn.o_proj.weight",
            f"norm1.weight": f"model.layers.{l}.input_layernorm.weight",
            f"ff.fc1.weight": f"model.layers.{l}.mlp.gate_proj.weight",
            f"ff.fc2.weight": f"model.layers.{l}.mlp.up_proj.weight",
            f"ff.fc3.weight": f"model.layers.{l}.mlp.down_proj.weight",
            f"norm2.weight": f"model.layers.{l}.post_attention_layernorm.weight",
        }

        for attr_path, param_name in attr_param_map.items():
            obj = block
            *parents, attr = attr_path.split('.')
            for p in parents:
                obj = getattr(obj, p)
            old_tensor = getattr(obj, attr)
            setattr(obj, attr, assign(old_tensor, params[param_name]))

    # Final normalization
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"])

    # Output head with fallback (for weight tying)
    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"])
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"])
        print("Model uses weight tying.")


def text_to_tensor(text, tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def tensor_to_text(tensor, tokenizer):
    return tokenizer.decode(tensor.squeeze(0).tolist())


def build_tokenizer():
    tokenizer_file_path = download_tokenizer_if_needed(
        repo_id="meta-llama/Meta-Llama-3-8B",
        filename="original/tokenizer.model",
        local_dir="Llama-3-8B"
    )
    tokenizer = Llama3Tokenizer(tokenizer_file_path)
    return tokenizer

def load_model(device=torch.device("cpu")):
    torch.manual_seed(123)
    model = Llama3Model(LLAMA3_CONFIG_8B).to(device)
    model.eval()
    return model

def load_combined_weights(repo_id, filenames, local_dir):
    combined = {}
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for name in filenames:
        local_path = local_dir / name
        if not local_path.exists():
            # download if not already present
            hf_hub_download(
                repo_id=repo_id,
                filename=name,
                local_dir=str(local_dir)
            )
        else:
            print(f"Using existing file: {local_path}")
        weights = load_file(str(local_path))
        combined.update(weights)

    return combined

def load_pretrained_weights(model,device=torch.device("cpu")):
    filenames = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
    combined_weights = load_combined_weights(
        repo_id="meta-llama/Meta-Llama-3-8B",
        filenames=filenames,
        local_dir="Llama-3-8B"
    )
    load_weights_into_llama(model, LLAMA3_CONFIG_8B, combined_weights)
    return model.to(device=device)

def complete_text(input_text, model, max_new_tokens=20, config=LLAMA3_CONFIG_8B,device=torch.device("cpu")):
    tokenizer = build_tokenizer()
    encoded_tensor = text_to_tensor(input_text, tokenizer)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=30,
        context_size=LLAMA3_CONFIG_8B["context_length"],
        top_k=1,
        temperature=0.
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
    log.setLevel(logging.INFO)

    start_context = "Once upon a time there"
    device = torch.device("cpu")
    model = load_model(device)
    model = load_pretrained_weights(model,device)
    result = complete_text(start_context, model, 10,device=device)
    print(result)


if __name__ == "__main__":
    main()
