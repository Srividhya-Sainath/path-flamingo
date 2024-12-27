"""
Preprocess and load datasets for training.
"""

import json
import math
import re
# from scipy.optimize import linear_sum_assignment ## इसकी ज़रूरत नहीं है क्योंकि Similarity Matrix वगैरह अभी नहीं चाहिए। हमारे पास तो पहले से ही matched captions हैं।

from data_utils import *

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class PathDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, feature_loader, max_tokens=256):
        """
        Custom Dataset for JSONL data with preprocessing.

        Args:
            jsonl_file (str): Path to the JSONL file.
            tokenizer: Tokenizer for text preprocessing.
            image_processor (function): Placeholder for image preprocessing function.
            max_tokens (int): Maximum token length for truncation.
        """
        self.jsonl_file = jsonl_file
        self.tokenizer = tokenizer
        self.feature_loader = feature_loader
        #self.image_processor = image_processor or (lambda x: x)  # Placeholder. Didi yahaan TITAN aega 
        self.max_tokens = max_tokens
        self.data = self._load_and_preprocess()

    def _load_and_preprocess(self):
        """
        Load and preprocess the JSONL file.
        """
        processed_data = []
        with open(self.jsonl_file, "r") as file:
            for line in file:
                
                entry = json.loads(line) # Parse each line as a JSON object

                
                file_path = entry["file_path"] # Get file_path (image ID) and preprocess text
                result = entry["result"]

                # if self.tokenizer.pad_token is None: # Iski Zaroorat nahi hai kyunki iska solution factory.py mein implement kiya hai
                #     self.tokenizer.pad_token = self.tokenizer.eos_token # अगर Tokenizer के पास कोई Padding token नहीं है तो EOS token का use कर लें

                result = entry["result"]
                result = re.sub(r"\n\s+", "\n\n", result)

                ## यहाँ से शुरू होती है text preprocessing का झमेला!
                # Step 1: Remove lines before the first "User:"
                user_index = result.find("\n\nUser:")
                if user_index != -1:
                    result = result[user_index:]
            
                # Step 2: Replace the first "User:" with "<image>" (without <|endofchunk|>)
                result = result.replace("\n\nUser:", "<image>", 1)
                result = result.replace("User:", "<image>", 1)

                # Step 3: Add <|endofchunk|> before every subsequent <image>
                result = re.sub(
                    r"\n\nUser:", "<|endofchunk|>\n<image>", result
                )
                result = result.replace("\n\nAssistant:", "\nAnswer:")
                result = result.replace("\n\nPathology Assistant:", "\nAnswer:")

                lines = result.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("Answer:"):
                    # Check if the line already ends with <|endofchunk|>
                        if not line.endswith("<|endofchunk|>"):
                            lines[i] = line.strip() + " <|endofchunk|>"
                result = "\n".join(lines)
                pairs = result.split("<image>")
                for pair in pairs:
                    if pair.strip():  # Ignore empty pairs
                        pair_text = "<image>" + pair.strip()
                        if not pair_text.endswith("<|endofchunk|>"):
                            pair_text += " <|endofchunk|>"

                        text = self.tokenizer(
                            pair_text,
                            max_length=self.max_tokens,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                        )

                        # Yahaan mein image ke features load kar rahi hun
                        features = self.feature_loader(file_path)

                        # Processed data contains: (file_path, input_ids, attention_mask)
                        ## file_path में ideally CONCHV1.5 के features आने चाहिए। - TODO
                        processed_data.append((file_path, features, text["input_ids"], text["attention_mask"], pair_text))
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, features, input_ids, attention_mask, raw_text = self.data[idx]
        # Preprocess the image using the placeholder processor
        #image = self.image_processor(file_path)  # TODO - ADD TITAN HERE Placeholder (Maybe Later. Pehle h5 files se chala lo)
        return {
            "file_path": file_path,
            "image": features,
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "attention_mask": attention_mask.squeeze(0),
            "raw_text": raw_text,
        }


def get_tcga_dataset(args, tokenizer, feature_loader, epoch=0, floor=False):
    """
    Initialize a DataLoader for the Llama dataset with epoch synchronization and meta-data.

    Args:
        args: Arguments with configurations like batch size.
        tokenizer: Tokenizer object for text preprocessing.
        feature_loader (function): Function to load preprocessed image features from h5 files.
        epoch (int): Current epoch index.
        floor (bool): Whether to round down the number of batches.

    Returns:
        DataInfo: Contains the DataLoader, sampler, and shared_epoch.
    """
    
    shared_epoch = SharedEpoch(epoch=epoch)


    path_dataset = PathDataset(
    jsonl_file=args.tcga_jsonl_file,
    tokenizer=tokenizer,
    feature_loader=feature_loader,
    max_tokens=getattr(args, "max_tokens", 256),  # Fallback to 256 if max_tokens is missing
    )

    num_samples = args.train_num_samples_tcga if hasattr(args, "train_num_samples_tcga") else len(path_dataset)
    global_batch_size = args.batch_size_tcga * args.world_size
    round_fn = math.floor if floor else math.ceil
    num_batches = round_fn(num_samples / global_batch_size)
    num_samples = num_batches * global_batch_size

    sampler = None
    if args.world_size > 1:
        sampler = DistributedSampler(
            path_dataset,
            num_replicas=args.world_size,
            rank=args.rank,  # Rank of the current process
            shuffle=(shared_epoch.get_value() == 0),  # Shuffle only for the first epoch
        )

    dataloader = DataLoader(
        path_dataset,
        batch_size=args.batch_size_tcga,
        sampler=sampler,  # DistributedSampler ensures each worker processes unique data
        shuffle=sampler is None,  # Shuffle if no sampler is used
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=True,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, sampler=sampler, shared_epoch=shared_epoch)

def get_gtex_dataset(args, tokenizer, feature_loader, epoch=0, floor=False):
    """
    Initialize a DataLoader for the Llama dataset with epoch synchronization and meta-data.

    Args:
        args: Arguments with configurations like batch size.
        tokenizer: Tokenizer object for text preprocessing.
        feature_loader (function): Function to load preprocessed image features from h5 files.
        epoch (int): Current epoch index.
        floor (bool): Whether to round down the number of batches.

    Returns:
        DataInfo: Contains the DataLoader, sampler, and shared_epoch.
    """
    
    shared_epoch = SharedEpoch(epoch=epoch)

    path_dataset = PathDataset(
    jsonl_file=args.gtex_jsonl_file,
    tokenizer=tokenizer,
    feature_loader=feature_loader,
    max_tokens=getattr(args, "max_tokens", 256),  # Fallback to 256 if max_tokens is missing
    )

    num_samples = args.train_num_samples_gtex if hasattr(args, "train_num_samples_gtex") else len(path_dataset)
    global_batch_size = args.batch_size_gtex * args.world_size
    round_fn = math.floor if floor else math.ceil
    num_batches = round_fn(num_samples / global_batch_size)
    num_samples = num_batches * global_batch_size

    sampler = None
    if args.world_size > 1:
        sampler = DistributedSampler(
            path_dataset,
            num_replicas=args.world_size,
            rank=args.rank,  # Rank of the current process
            shuffle=(shared_epoch.get_value() == 0),  # Shuffle only for the first epoch
        )

    # Create a DataLoader
    dataloader = DataLoader(
        path_dataset,
        batch_size=args.batch_size_gtex,
        sampler=sampler,  # DistributedSampler ensures each worker processes unique data
        shuffle=sampler is None,  # Shuffle if no sampler is used
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=True,
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
