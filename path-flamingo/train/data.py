"""
Preprocess and load datasets for training.
"""
import json
import math
import re
import torch
import random
import numpy as np
from open_flamingo.train.data_utils import *
from train_utils import get_cast_dtype

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class PathDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, feature_loader, cohort="tcga",
                 max_tokens=256, min_images=1, max_images=4):
        self.jsonl_file = jsonl_file
        self.tokenizer = tokenizer
        self.feature_loader = feature_loader
        self.cohort = cohort
        self.max_tokens = max_tokens
        self.min_images = min_images
        self.max_images = max_images
        self.data = self._load_entries()

    def _load_entries(self):
        entries = []
        with open(self.jsonl_file, "r") as file:
            for line in file:
                entry = json.loads(line)
                entries.append(entry)
        return entries

    def _process_text(self, text):
        text = re.sub(r"\n\s+", "\n\n", text)
        user_index = text.find("\n\nUser:")
        if user_index != -1:
            text = text[user_index:]
        text = text.replace("\n\nUser:", "<image>", 1)
        text = text.replace("\nUser:", "<image>", 1)
        text = text.replace("User:", "<image>", 1)
        text = re.sub(r"\n\nUser:", "<|endofchunk|>\n<image>", text)
        text = text.replace("\n\nAssistant:", "\nAnswer:")
        text = text.replace("\nAssistant:", "\nAnswer:")
        text = text.replace("Assistant:", "\nAnswer:")
        text = text.replace("\n\nPathology Assistant:", "\nAnswer:")
        text = text.replace("\nPathology Assistant:", "\nAnswer:")
        text = text.replace("Pathology Assistant:", "\nAnswer:")
        #return text
        lines = text.strip().split("\n")
        new_lines = []
        inside_answer = False

        for i, line in enumerate(lines):
            new_lines.append(line)

            if line.strip().startswith("Answer:"):
                inside_answer = True
            elif inside_answer and (
                i == len(lines) - 1 or lines[i + 1].strip().startswith("<image>")
            ):
                if not line.strip().endswith("<|endofchunk|>"):
                    new_lines[-1] = new_lines[-1].strip() + " <|endofchunk|>"
                inside_answer = False

        return "\n".join(new_lines)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        file_path = entry["file_path"]
        result = self._process_text(entry["result"])

        parts = result.split("<image>")
        num_image_tokens = len(parts) - 1

        truncated_result = "<image>".join(parts[:num_image_tokens + 1])
        lines = truncated_result.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Answer:") and not line.endswith("<|endofchunk|>"):
                lines[i] = line.strip() + " <|endofchunk|>"
        truncated_result = "\n".join(lines)

        text = self.tokenizer(
            truncated_result,
            max_length=self.max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        feature = self.feature_loader(file_path)
        if isinstance(feature, np.ndarray):
            feature = torch.tensor(feature, dtype=torch.float32)

        if feature.ndim == 1:
            # Case: shape [768] → make it [1, 768]
            feature = feature.unsqueeze(0)
        elif feature.ndim == 2:
            # Case: shape already [image_tokens_per_image, 768]
            pass
        else:
            raise ValueError(f"Unexpected feature shape: {feature.shape}")

        # Repeat along Tm (number of <image> tokens)
        repeated = feature.unsqueeze(0).repeat(num_image_tokens, 1, 1)  # [Tm, image_tokens_per_image, 768]

        return {
            "file_path": file_path,
            "raw_text": truncated_result,
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "features": repeated,  # shape: [Tm, image_tokens_per_image, 768]
            "token_count": num_image_tokens
        }

def greedy_collate_fn(batch, max_total_image_tokens=10, min_images=1, cast_dtype=None):
    batch_size = len(batch)

    # Step 1: Initial allocation (safe)
    token_allocations = [
        min(sample["token_count"], max(min_images, max_total_image_tokens // batch_size))
        for sample in batch
    ]
    remaining_budget = max_total_image_tokens - sum(token_allocations)

    # Step 2: Greedy top-up
    indices = list(range(batch_size))
    random.shuffle(indices)
    for idx in indices:
        sample = batch[idx]
        available = sample["token_count"] - token_allocations[idx]
        if available > 0 and remaining_budget > 0:
            give = min(available, remaining_budget)
            token_allocations[idx] += give
            remaining_budget -= give
        if remaining_budget <= 0:
            break

    # Step 3: Pad each sample to max_alloc (only along Tm)
    max_alloc = max(token_allocations)
    padded_features = []
    for alloc, sample in zip(token_allocations, batch):
        feats = sample["features"][:alloc]  # shape: [alloc, image_tokens_per_image, 768]
        pad_len = max_alloc - alloc
        if pad_len > 0:
            pad = torch.zeros(
                (pad_len, feats.shape[1], feats.shape[2]), dtype=feats.dtype
            )
            feats = torch.cat([feats, pad], dim=0)
        padded_features.append(feats)

    images = torch.stack(padded_features)  # [B, max_alloc, image_tokens_per_image, 768]
    if cast_dtype is not None:
        images = images.to(dtype=cast_dtype)

    return {
        "file_path": [s["file_path"] for s in batch],
        "raw_text": [s["raw_text"] for s in batch],
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "images": images,  # [B, Tm, image_tokens_per_image, 768]
        "token_count": token_allocations
    }

def get_tcga_dataset(args, tokenizer, feature_loader, epoch=0, floor=False):
    shared_epoch = SharedEpoch(epoch=epoch)

    dataset = PathDataset(
        jsonl_file=args.tcga_jsonl_file,
        tokenizer=tokenizer,
        feature_loader=feature_loader,
        cohort="tcga",
        max_tokens=args.max_tokens,
        min_images=args.tcga_min_num_images,
        max_images=args.tcga_max_num_images,
    )

    sampler = None
    if args.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=(shared_epoch.get_value() == 0)
        )

    global_batch_size = args.batch_size_tcga * args.world_size
    num_samples = args.train_num_samples_tcga
    round_fn = math.floor if floor else math.ceil
    num_batches = round_fn(num_samples / global_batch_size)
    num_samples = num_batches * global_batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_tcga,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=True,
        collate_fn=lambda x: greedy_collate_fn(
        x, max_total_image_tokens=args.tcga_max_num_images, min_images=args.tcga_min_num_images, cast_dtype=get_cast_dtype(args.precision)
        )
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, sampler=sampler, shared_epoch=shared_epoch)

def get_gtex_dataset(args, tokenizer, feature_loader, epoch=0, floor=False):
    shared_epoch = SharedEpoch(epoch=epoch)

    dataset = PathDataset(
        jsonl_file=args.gtex_jsonl_file,
        tokenizer=tokenizer,
        feature_loader=feature_loader,
        cohort="gtex",
        max_tokens=args.max_tokens,
        min_images=args.gtex_min_num_images,
        max_images=args.gtex_max_num_images,
    )

    sampler = None
    if args.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=(shared_epoch.get_value() == 0)
        )

    global_batch_size = args.batch_size_gtex * args.world_size
    num_samples = args.train_num_samples_gtex
    round_fn = math.floor if floor else math.ceil
    num_batches = round_fn(num_samples / global_batch_size)
    num_samples = num_batches * global_batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_gtex,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=True,
        collate_fn=lambda x: greedy_collate_fn(
            x, max_total_image_tokens=args.gtex_max_num_images, min_images=args.gtex_min_num_images, cast_dtype=get_cast_dtype(args.precision)
            )
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, sampler=sampler, shared_epoch=shared_epoch)

def get_dataset_fn(dataset_type):
    """
    Helper function to get the dataset function based on the dataset type
    """
    if dataset_type == "tcga":
        return get_tcga_dataset
    elif dataset_type == "gtex":
        return get_gtex_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, feature_loader, tokenizer, dataset_type, epoch=0):
    """
    Interface for getting the datasets
    """
    return get_dataset_fn(dataset_type)(
        args, feature_loader=feature_loader, epoch=epoch, tokenizer=tokenizer
    )
