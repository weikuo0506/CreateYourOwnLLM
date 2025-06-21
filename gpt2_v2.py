import torch
import torch.nn as nn
import tiktoken
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    def __init__(self, txt, tokenizer, context_size, stride):
        token_ids = tokenizer.encode(txt)
        assert len(token_ids) > context_size, "Text is too short"

        self.input_ids = [torch.tensor(token_ids[i:i + context_size])
                          for i in range(0, len(token_ids) - context_size, stride)]
        self.target_ids = [torch.tensor(token_ids[i + 1:i + context_size + 1])
                           for i in range(0, len(token_ids) - context_size, stride)]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def dataloader_v1(txt, batch_size=3, context_size=5, stride=2, shuffle=False, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, context_size, stride)
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
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

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
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), nn.GELU(),
                                    nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], cfg["drop_rate"],
                                       cfg["n_heads"], cfg["qkv_bias"])
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
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        token_emb = self.tok_emb(token_ids)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=token_ids.device))
        x = token_emb + pos_emb
        x = self.drop(x)
        x = self.blocks(x)
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


def text_to_tensor(text, tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def tensor_to_text(tensor, tokenizer):
    return tokenizer.decode(tensor.squeeze(0).tolist())


def build_tokenizer():
    return tiktoken.get_encoding("gpt2")


def load_model(device=torch.device("cpu")):
    torch.manual_seed(123)
    model = GPT2Model(GPT_CONFIG_124M).to(device)
    model.eval()
    return model


def complete_text(input_text, model, max_new_tokens=20, config=GPT_CONFIG_124M,device=torch.device("cpu")):
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


def loss_batch(inputs, targets, model, device):
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(0))
    return loss


def loss_loader(loader, model, device, num_batches=None):
    if len(loader) == 0:
        return float('nan')

    total_loss = 0.0
    # num_batches no more than len(loader), default to len(loader)
    num_batches = min(num_batches or len(loader), len(loader))

    for i, (inputs, targets) in enumerate(loader):
        if i >= num_batches:
            break
        loss = loss_batch(inputs, targets, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


import os
import torch


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       save_dir="checkpoints", resume_path=None):
    os.makedirs(save_dir, exist_ok=True)

    # === Initialize training state ===
    start_epoch = 0
    step = 0
    tokens_seen = 0
    train_losses, val_losses, tokens_seen_track = [], [], []

    # === Resume from checkpoint if provided ===
    if resume_path and os.path.exists(resume_path):
        print(f"🔁 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        tokens_seen = checkpoint.get('tokens_seen', 0)
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        tokens_seen_track = checkpoint.get('tokens_seen_track', [])
        print(f"✅ Resumed at epoch {start_epoch}, step {step}, tokens seen: {tokens_seen}")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss = loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()

                tokens_seen += input_batch.numel()
                step += 1

                if step % eval_freq == 0:
                    train_loss = loss_loader(train_loader, model, device, eval_iter)
                    val_loss = loss_loader(val_loader, model, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    tokens_seen_track.append(tokens_seen)
                    print(
                        f"Ep {epoch + 1} (Step {step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, Tokens seen: {tokens_seen}")

            # generate sample after each epoch
            generate_and_print_sample(model, tokenizer, device, start_context)

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'step': step,
                'tokens_seen': tokens_seen,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'tokens_seen_track': tokens_seen_track,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

    except KeyboardInterrupt:
        print("🛑 Training interrupted. Saving checkpoint...")
        interrupted_path = os.path.join(save_dir, f"checkpoint_interrupted_epoch{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'step': step,
            'tokens_seen': tokens_seen,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'tokens_seen_track': tokens_seen_track,
        }, interrupted_path)
        print(f"✅ Checkpoint saved: {interrupted_path}")
        return train_losses, val_losses, tokens_seen_track

    # Save final model
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save({
        'epoch': num_epochs,
        'step': step,
        'tokens_seen': tokens_seen,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'tokens_seen_track': tokens_seen_track,
    }, final_path)
    print(f"🎉 Training complete. Final model saved: {final_path}")

    return train_losses, val_losses, tokens_seen_track


def plot_losses(epochs, tokens, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.plot(epochs, val_losses, linestyle="--", label="Val loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    # plt.savefig("loss-plot.pdf")
    plt.show()


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    with torch.no_grad():
        result = complete_text(start_context, model, 20,device)
        print(result)
    model.train()


def main():
    start_context = "Once upon a time there"
    device = torch.device("mps")
    model = load_model(device)
    result = complete_text(start_context, model, 10,device=device)
    print(result)


if __name__ == "__main__":
    main()
