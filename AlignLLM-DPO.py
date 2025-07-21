#!/usr/bin/env python3

# Import libraries
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from torch.amp import autocast  # Updated to torch.amp.autocast
import os
import random

def setup_environment():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Set device (use MPS for Apple Silicon if available, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load gpt2-medium model and tokenizer
    model_name = "openai-community/gpt2-medium"
    try:
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
            device_map="auto"
        )
        reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise
    tokenizer.pad_token = tokenizer.eos_token
    return device, policy_model, reference_model, tokenizer

def load_and_preprocess_dataset(tokenizer):
    # Load dataset from local disk if exists, otherwise download and sample
    local_path = "./orca_dpo_sample_1000"
    if os.path.exists(local_path):
        print("Loading dataset from local disk...")
        dataset = load_from_disk(local_path)
    else:
        print("Downloading dataset from HuggingFace...")
        full_dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
        sampled_dataset = full_dataset.shuffle(seed=42).select(range(1000))
        train_size = int(0.8 * len(sampled_dataset))
        val_size = int(0.1 * len(sampled_dataset))
        train_dataset = sampled_dataset.select(range(train_size))
        val_dataset = sampled_dataset.select(range(train_size, train_size + val_size))
        test_dataset = sampled_dataset.select(range(train_size + val_size, len(sampled_dataset)))
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        dataset.save_to_disk(local_path)
        print(f"Dataset saved locally at: {local_path}")

    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")
    print("\nExample sample:")
    print(dataset["train"][0])

    # Preprocess data: format and tokenize
    def format_dpo_data(example):
        # Map 'question' to 'prompt' and include 'system' in the prompt for context
        prompt = f"{example['system']}\n\n{example['question']}"
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    def tokenize_data(example):
        # Handle empty or invalid inputs
        prompt = example["prompt"] if example["prompt"] and isinstance(example["prompt"], str) else ""
        chosen = example["chosen"] if example["chosen"] and isinstance(example["chosen"], str) else ""
        rejected = example["rejected"] if example["rejected"] and isinstance(example["rejected"], str) else ""

        # Tokenize with padding and truncation, keeping as tensors
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"][0]  # Shape: [256]
        chosen_tokens = tokenizer(
            chosen,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"][0]  # Shape: [512]
        rejected_tokens = tokenizer(
            rejected,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"][0]  # Shape: [512]

        return {
            "prompt_ids": prompt_tokens,
            "chosen_ids": chosen_tokens,
            "rejected_ids": rejected_tokens
        }

    # Apply preprocessing
    dataset = dataset.map(format_dpo_data)
    dataset = dataset.map(tokenize_data)
    dataset.set_format(type="torch", columns=["prompt_ids", "chosen_ids", "rejected_ids"])

    # Custom collate function to ensure consistent tensor shapes
    def collate_fn(batch):
        prompt_ids = torch.stack([item["prompt_ids"] for item in batch])
        chosen_ids = torch.stack([item["chosen_ids"] for item in batch])
        rejected_ids = torch.stack([item["rejected_ids"] for item in batch])
        return {
            "prompt_ids": prompt_ids,
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids
        }

    # Create DataLoader with custom collate function
    train_loader = DataLoader(dataset["train"], batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["validation"], batch_size=2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset["test"], batch_size=2, shuffle=False, collate_fn=collate_fn)

    return dataset, train_loader, val_loader, test_loader

def compute_log_probs(logits, labels, mask):
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)

def compute_dpo_loss_batch(batch, policy_model, reference_model, beta, tokenizer, device):
    prompt_ids = batch["prompt_ids"].to(device)
    chosen_ids = batch["chosen_ids"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)

    chosen_mask = (chosen_ids != tokenizer.pad_token_id).to(device)
    rejected_mask = (rejected_ids != tokenizer.pad_token_id).to(device)

    with torch.no_grad():
        ref_chosen_outputs = reference_model(chosen_ids, attention_mask=chosen_mask)
        ref_rejected_outputs = reference_model(rejected_ids, attention_mask=rejected_mask)

    policy_chosen_outputs = policy_model(chosen_ids, attention_mask=chosen_mask)
    policy_rejected_outputs = policy_model(rejected_ids, attention_mask=rejected_mask)

    chosen_logprobs = compute_log_probs(policy_chosen_outputs.logits, chosen_ids, chosen_mask)
    rejected_logprobs = compute_log_probs(policy_rejected_outputs.logits, rejected_ids, rejected_mask)
    ref_chosen_logprobs = compute_log_probs(ref_chosen_outputs.logits, chosen_ids, chosen_mask)
    ref_rejected_logprobs = compute_log_probs(ref_rejected_outputs.logits, rejected_ids, rejected_mask)

    log_ratio_chosen = chosen_logprobs - ref_chosen_logprobs
    log_ratio_rejected = rejected_logprobs - ref_rejected_logprobs
    logits = beta * (log_ratio_chosen - log_ratio_rejected)

    loss = -F.logsigmoid(logits).mean()
    chosen_rewards = beta * log_ratio_chosen
    rejected_rewards = beta * log_ratio_rejected

    return loss, chosen_rewards.mean(), rejected_rewards.mean()

def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, tokenizer, device, num_batches=None):
    # Initialize accumulators
    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    if len(data_loader) == 0:
        return float("nan"), float("nan"), float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    # Set model to evaluation mode
    policy_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch, policy_model, reference_model, beta, tokenizer, device
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()

    # Compute averages
    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards

def train_and_evaluate(policy_model, reference_model, tokenizer, train_loader, val_loader, test_loader, device):
    # Initialize optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    num_epochs = 3
    beta = 0.1

    for epoch in range(num_epochs):
        # Set model to training mode
        policy_model.train()
        total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
        for batch in train_loader:
            optimizer.zero_grad()
            with autocast(device_type="mps" if device.type == "mps" else "cpu"):  # Updated autocast
                loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                    batch, policy_model, reference_model, beta, tokenizer, device
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()

        # Compute average metrics
        avg_loss = total_loss / len(train_loader)
        avg_chosen_rewards = total_chosen_rewards / len(train_loader)
        avg_rejected_rewards = total_rejected_rewards / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Chosen Rewards: {avg_chosen_rewards:.4f}, Rejected Rewards: {avg_rejected_rewards:.4f}")

        # Validation
        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            val_loader, policy_model, reference_model, beta, tokenizer, device
        )
        print(f"Validation Loss: {val_loss:.4f}, Chosen Rewards: {val_chosen_rewards:.4f}, Rejected Rewards: {val_rejected_rewards:.4f}")

    # Save model and tokenizer
    policy_model.save_pretrained("./gpt2_medium_dpo_final")
    tokenizer.save_pretrained("./gpt2_medium_dpo_final")

    # Test set evaluation
    test_loss, test_chosen_rewards, test_rejected_rewards = compute_dpo_loss_loader(
        test_loader, policy_model, reference_model, beta, tokenizer, device
    )
    print(f"Test Loss: {test_loss:.4f}, Chosen Rewards: {test_chosen_rewards:.4f}, Rejected Rewards: {test_rejected_rewards:.4f}")

    # Example inference
    policy_model.eval()
    prompt = dataset["test"][0]["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = policy_model.generate(**inputs, max_length=512)
    print("\nExample generation:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    # Setup environment
    device, policy_model, reference_model, tokenizer = setup_environment()

    # Load and preprocess dataset
    dataset, train_loader, val_loader, test_loader = load_and_preprocess_dataset(tokenizer)

    # Train and evaluate
    train_and_evaluate(policy_model, reference_model, tokenizer, train_loader, val_loader, test_loader, device)