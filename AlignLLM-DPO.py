#!/usr/bin/env python3

# Import libraries
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from torch.amp import autocast
import os
import random
import numpy as np
import psutil  # For memory monitoring

def setup_environment():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Define device: Set USE_MPS to True for MPS
    USE_MPS = True  # Set to False to use CPU
    device = torch.device("mps" if USE_MPS and torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "mps" else torch.float32
    print(f"Using device: {device}, torch_dtype: {torch_dtype}")
    if device.type == "mps":
        print("MPS is enabled. Ensure PyTorch version supports MPS (2.0+).")

    # Load smaller gpt2 model for faster training
    model_name = "openai-community/gpt2"
    try:
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=None
        ).to(device)
        reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=None
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise
    tokenizer.pad_token = tokenizer.eos_token
    return device, torch_dtype, policy_model, reference_model, tokenizer

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

    # Filter out invalid samples (e.g., missing or non-string fields)
    def filter_valid_examples(example):
        system_valid = example["system"] and isinstance(example["system"], str)
        question_valid = example["question"] and isinstance(example["question"], str)
        chosen_valid = example["chosen"] and isinstance(example["chosen"], str)
        rejected_valid = example["rejected"] and isinstance(example["rejected"], str)
        if not (system_valid and question_valid and chosen_valid and rejected_valid):
            print(f"Filtered out invalid sample: {example}")
        return system_valid and question_valid and chosen_valid and rejected_valid

    # Apply initial filtering
    dataset = dataset.filter(filter_valid_examples)
    print(f"After initial filtering - Train size: {len(dataset['train'])}")
    print(f"After initial filtering - Validation size: {len(dataset['validation'])}")
    print(f"After initial filtering - Test size: {len(dataset['test'])}")

    # Preprocess data: format and tokenize
    def format_dpo_data(example):
        # Map 'question' to 'prompt' and include 'system' in the prompt for context
        prompt = f"{example['system']}\n\n{example['question']}"
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    # Apply formatting
    dataset = dataset.map(format_dpo_data)

    # Filter out samples with invalid tokenized outputs
    def filter_tokenized_examples(example):
        prompt_ids = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"][0]
        chosen_ids = tokenizer(
            example["chosen"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"][0]
        rejected_ids = tokenizer(
            example["rejected"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"][0]
        # Ensure at least one non-pad token in each field
        prompt_valid = (prompt_ids != tokenizer.pad_token_id).sum() > 0
        chosen_valid = (chosen_ids != tokenizer.pad_token_id).sum() > 0
        rejected_valid = (rejected_ids != tokenizer.pad_token_id).sum() > 0
        if not (prompt_valid and chosen_valid and rejected_valid):
            print(f"Filtered out invalid tokenized sample: prompt={example['prompt'][:50]}, chosen={example['chosen'][:50]}, rejected={example['rejected'][:50]}")
        return prompt_valid and chosen_valid and rejected_valid

    # Apply tokenization and secondary filtering
    dataset = dataset.filter(filter_tokenized_examples)
    dataset = dataset.map(tokenize_data)
    dataset.set_format(type="torch", columns=["prompt_ids", "chosen_ids", "rejected_ids"])

    print(f"After tokenization and filtering - Train size: {len(dataset['train'])}")
    print(f"After tokenization and filtering - Validation size: {len(dataset['validation'])}")
    print(f"After tokenization and filtering - Test size: {len(dataset['test'])}")

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

    # Create DataLoader with smaller batch size for MPS
    train_loader = DataLoader(dataset["train"], batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["validation"], batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset["test"], batch_size=1, shuffle=False, collate_fn=collate_fn)

    return dataset, train_loader, val_loader, test_loader

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

def compute_log_probs(logits, labels, mask):
    # Clip logits to prevent overflow
    logits = torch.clamp(logits, min=-1e9, max=1e9)
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # Add small epsilon to avoid nan in masked regions
    return (gathered * mask + 1e-10 * (1 - mask)).sum(dim=-1) / (mask.sum(dim=-1) + 1e-10)

def compute_dpo_loss_batch(batch, policy_model, reference_model, beta, tokenizer, device):
    # Move input tensors to device
    prompt_ids = batch["prompt_ids"].to(device)
    chosen_ids = batch["chosen_ids"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)

    # Compute attention masks
    chosen_mask = (chosen_ids != tokenizer.pad_token_id).to(device).float()
    rejected_mask = (rejected_ids != tokenizer.pad_token_id).to(device).float()

    # Debug: Check for invalid masks
    if chosen_mask.sum() == 0 or rejected_mask.sum() == 0:
        print("Warning: Invalid mask detected (all zeros)")
        print(f"Sample prompt: {tokenizer.decode(batch['prompt_ids'][0], skip_special_tokens=True)[:50]}")
        print(f"Sample chosen: {tokenizer.decode(batch['chosen_ids'][0], skip_special_tokens=True)[:50]}")
        print(f"Sample rejected: {tokenizer.decode(batch['rejected_ids'][0], skip_special_tokens=True)[:50]}")
        return torch.tensor(0.0, requires_grad=True).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

    # Compute logits for chosen and rejected responses
    with torch.no_grad():
        ref_chosen_outputs = reference_model(chosen_ids, attention_mask=chosen_mask)
        ref_rejected_outputs = reference_model(rejected_ids, attention_mask=rejected_mask)
    with autocast(device_type=device.type, dtype=torch.float16 if device.type == "mps" else torch.float32):
        policy_chosen_outputs = policy_model(chosen_ids, attention_mask=chosen_mask)
        policy_rejected_outputs = policy_model(rejected_ids, attention_mask=rejected_mask)

    # Debug: Check for nan in logits
    if torch.isnan(policy_chosen_outputs.logits).any() or torch.isnan(policy_rejected_outputs.logits).any():
        print("Warning: NaN detected in policy logits")
        print(f"Sample prompt: {tokenizer.decode(batch['prompt_ids'][0], skip_special_tokens=True)[:50]}")
        print(f"Sample chosen: {tokenizer.decode(batch['chosen_ids'][0], skip_special_tokens=True)[:50]}")
        print(f"Sample rejected: {tokenizer.decode(batch['rejected_ids'][0], skip_special_tokens=True)[:50]}")
        return torch.tensor(0.0, requires_grad=True).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

    # Compute log probabilities
    chosen_logprobs = compute_log_probs(policy_chosen_outputs.logits, chosen_ids, chosen_mask)
    rejected_logprobs = compute_log_probs(policy_rejected_outputs.logits, rejected_ids, rejected_mask)
    ref_chosen_logprobs = compute_log_probs(ref_chosen_outputs.logits, chosen_ids, chosen_mask)
    ref_rejected_logprobs = compute_log_probs(ref_rejected_outputs.logits, rejected_ids, rejected_mask)

    # Debug: Check for nan in log probabilities
    if torch.isnan(chosen_logprobs).any() or torch.isnan(rejected_logprobs).any():
        print("Warning: NaN detected in log probabilities")
        print(f"Sample prompt: {tokenizer.decode(batch['prompt_ids'][0], skip_special_tokens=True)[:50]}")
        print(f"Sample chosen: {tokenizer.decode(batch['chosen_ids'][0], skip_special_tokens=True)[:50]}")
        print(f"Sample rejected: {tokenizer.decode(batch['rejected_ids'][0], skip_special_tokens=True)[:50]}")
        return torch.tensor(0.0, requires_grad=True).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

    # Compute DPO loss
    log_ratio_chosen = chosen_logprobs - ref_chosen_logprobs
    log_ratio_rejected = rejected_logprobs - ref_rejected_logprobs
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -F.logsigmoid(logits).mean()

    # Debug: Check for nan in loss
    if torch.isnan(loss):
        print("Warning: NaN detected in loss")
        print(f"Sample prompt: {tokenizer.decode(batch['prompt_ids'][0], skip_special_tokens=True)[:50]}")
        print(f"Sample chosen: {tokenizer.decode(batch['chosen_ids'][0], skip_special_tokens=True)[:50]}")
        print(f"Sample rejected: {tokenizer.decode(batch['rejected_ids'][0], skip_special_tokens=True)[:50]}")
        return torch.tensor(0.0, requires_grad=True).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

    # Compute rewards for monitoring
    chosen_rewards = beta * log_ratio_chosen
    rejected_rewards = beta * log_ratio_rejected

    return loss, chosen_rewards.mean(), rejected_rewards.mean()

def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, tokenizer, device, num_batches=None):
    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    if len(data_loader) == 0:
        return float("nan"), float("nan"), float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

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
            # Print memory usage
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"Batch {i+1}/{num_batches}, Memory usage: {mem_info.rss / 1024**2:.2f} MB")

    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards

def train_and_evaluate(policy_model, reference_model, tokenizer, train_loader, val_loader, test_loader, device):
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-6)  # Reduced learning rate
    num_epochs = 3
    beta = 0.05  # Reduced beta for stability

    for epoch in range(num_epochs):
        policy_model.train()
        total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            with autocast(device_type=device.type, dtype=torch.float16 if device.type == "mps" else torch.float32):
                loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                    batch, policy_model, reference_model, beta, tokenizer, device
                )
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()
            # Print memory usage
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Memory usage: {mem_info.rss / 1024**2:.2f} MB")

        avg_loss = total_loss / len(train_loader)
        avg_chosen_rewards = total_chosen_rewards / len(train_loader)
        avg_rejected_rewards = total_rejected_rewards / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Chosen Rewards: {avg_chosen_rewards:.4f}, Rejected Rewards: {avg_rejected_rewards:.4f}")

        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            val_loader, policy_model, reference_model, beta, tokenizer, device
        )
        print(f"Validation Loss: {val_loss:.4f}, Chosen Rewards: {val_chosen_rewards:.4f}, Rejected Rewards: {val_rejected_rewards:.4f}")

    policy_model.save_pretrained("./gpt2_dpo_final")
    tokenizer.save_pretrained("./gpt2_dpo_final")

    test_loss, test_chosen_rewards, test_rejected_rewards = compute_dpo_loss_loader(
        test_loader, policy_model, reference_model, beta, tokenizer, device
    )
    print(f"Test Loss: {test_loss:.4f}, Chosen Rewards: {test_chosen_rewards:.4f}, Rejected Rewards: {test_rejected_rewards:.4f}")

    policy_model.eval()
    prompt = dataset["test"][0]["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = policy_model.generate(**inputs, max_length=512)
    print("\nExample generation:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    device, torch_dtype, policy_model, reference_model, tokenizer = setup_environment()
    dataset, train_loader, val_loader, test_loader = load_and_preprocess_dataset(tokenizer)
    train_and_evaluate(policy_model, reference_model, tokenizer, train_loader, val_loader, test_loader, device)
