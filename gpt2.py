import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}

class GPTDatasetV1(Dataset):
    def __init__(self, txt,tokenizer, context_size, stride):
        token_ids = tokenizer.encode(txt)
        assert len(token_ids) > context_size, "Text is too short"

        self.input_ids = [torch.tensor(token_ids[i:i+context_size])
                          for i in range(0, len(token_ids)-context_size, stride)]
        self.target_ids = [torch.tensor(token_ids[i+1:i+context_size+1])
                          for i in range(0, len(token_ids)-context_size, stride)]
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def dataloader_v1(txt,batch_size=3,context_size=5,stride=2,shuffle=False,drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,tokenizer,context_size,stride)
    return DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention by splitting the attention matrix into multiple heads.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, d_in)
        """
        batch_size, seq_len, _ = x.size()

        # Split Q, K, V into multiple heads
        # (batch_size, seq_len, d_in) -> (batch_size, seq_len, d_out) ->
        # -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / (self.d_out ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], -torch.inf)

        # Compute softmax weights and apply dropout
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Compute output
        output = weights @ V  # (batch_size, num_heads, seq_len, head_dim)
        # Concatenate heads and project to output dimension
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # ->   (batch_size, seq_len, d_out)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Should be helpful, but not strictly necessary.
        output = self.out_proj(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), nn.GELU(), nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]))

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg["emb_dim"])
        self.ln2 = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        token_emb = self.token_emb(token_ids)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=token_ids.device))
        x = token_emb + pos_emb
        x = self.drop(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        # Get logits from model
        with torch.no_grad():
            logits = model(idx_cond)

        # Take logits for the last time step
        # (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def text_to_tensor(text,tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)

def tensor_to_text(tensor,tokenizer):
    return tokenizer.decode(tensor.squeeze(0).tolist())

def build_tokenizer():
    return tiktoken.get_encoding("gpt2")

def load_model():
    torch.manual_seed(123)
    model = GPT2Model(GPT_CONFIG_124M)
    model.eval()
    return model

def complete_text(input_text, model, max_new_tokens=20, config=GPT_CONFIG_124M):
    tokenizer = build_tokenizer()
    encoded_tensor = text_to_tensor(input_text,tokenizer)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=max_new_tokens,
        context_size=config["context_length"]
    )

    decoded_text = tensor_to_text(out, tokenizer)
    return decoded_text

def main():
    start_context = "Once upon a time there"
    model = load_model()
    result = complete_text(start_context, model,10)
    print(result)


if __name__ == "__main__":
    main()