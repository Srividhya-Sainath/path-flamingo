import time
import h5py
from contextlib import suppress
import torch
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
import os
import wandb
from einops import rearrange


def create_feature_loader(base_path):
    """
    Creates a feature loader function that searches for files in a given base path. 
    Args:
        base_path (str): The root directory where feature files are stored.
    Returns:
        function: A feature loader function that returns torch.FloatTensor of shape [Tm, image_tokens_per_image, 768]
    """
    def feature_loader(file_path):
        full_path = os.path.join(base_path, file_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Feature file not found: {full_path}")
        with h5py.File(full_path, "r") as h5_file:
            features = h5_file["feats"][:]
            if features.ndim == 1:
                features = features.reshape(1, -1)  # shape becomes [1, 768]
        return torch.tensor(features, dtype=torch.float32)  # shape [image_tokens_per_image, 768]
    return feature_loader

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_mp_policy_dtype(precision: str):
    if "bfloat16" in precision or "bf16" in precision:
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    else:
        return torch.float32


def get_autocast(precision, cache_enabled=True):
    #if precision == "amp":
    if precision in ["amp", "amp_fp16", "fp16"]:
        return lambda: torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=cache_enabled)
        #return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    #elif precision == "amp_bfloat16" or precision == "amp_bf16":
    elif precision in ["amp_bfloat16", "amp_bf16"]:
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress


def train_one_epoch(
    args,
    model,
    epoch,
    tcga_loader,
    gtex_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    # setup loaders
    num_batches_per_epoch_tcga = tcga_loader.num_batches
    num_batches_per_epoch_gtex = gtex_loader.num_batches
    print("Number of batches in TCGA dataset: ", num_batches_per_epoch_tcga)
    print("Number of batches in GTEX dataset: ", num_batches_per_epoch_gtex)
    assert (
        num_batches_per_epoch_tcga == num_batches_per_epoch_gtex
    ), "Number of batches in TCGA and GTEX datasets must be the same"
    num_batches_per_epoch = num_batches_per_epoch_gtex
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    # setup model
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through dataloader
    for num_steps, (batch_tcga, batch_gtex) in tqdm(
        enumerate(zip(tcga_loader, gtex_loader)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        #### TCGA FORWARD PASS ####
        #images = batch_tcga[0].to(device_id, dtype=cast_dtype, non_blocking=True) ## Nahi chahiye
        images = batch_tcga["images"].to(device_id, dtype=cast_dtype, non_blocking=True) ## (batch_size, feature_dim)
        #images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1) ## Nahi chahiye
        #input_ids = batch_tcga[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        # input_ids = batch_tcga["input_ids"].to(device_id, dtype=cast_dtype, non_blocking=True)
        input_ids = batch_tcga["input_ids"].to(device_id, non_blocking=True)
        # attention_mask = batch_tcga[1][1].to(
        #     device_id, dtype=cast_dtype, non_blocking=True
        # )
        # attention_mask = batch_tcga["attention_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_tcga["attention_mask"].to(device_id, non_blocking=True)

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            loss_tcga = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        divided_loss_tcga = loss_tcga / args.gradient_accumulation_steps
        (divided_loss_tcga * args.loss_multiplier_tcga).backward()

        #### GTEX FORWARD PASS ####
        #images = batch_gtex[0].to(device_id, dtype=cast_dtype, non_blocking=True) # Nahi chahiye
        images = batch_gtex["images"].to(device_id, dtype=cast_dtype, non_blocking=True) ## (batch_size, feature_dim)
        #images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1) #Nahi chahiye
        # input_ids = torch.stack([x[0] for x in batch_gtex[1]]).squeeze(1)
        # attention_mask = torch.stack([x[1] for x in batch_gtex[1]]).squeeze(1)
        #input_ids = batch_gtex["input_ids"].to(device_id, dtype=cast_dtype, non_blocking=True)
        input_ids = batch_gtex["input_ids"].to(device_id, non_blocking=True)
        #attention_mask = batch_gtex["attention_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_gtex["attention_mask"].to(device_id, non_blocking=True)

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            loss_gtex = model(
                vision_x=images,
                lang_x=input_ids.to(device_id),
                attention_mask=attention_mask.to(device_id),
                labels=labels,
            )[0]

            # if loss is nan, skip this batch
            # this hack of skipping the batch is not FSDP-compatible
            if torch.isnan(loss_gtex):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad(set_to_none=True)
                continue

        divided_loss_gtex = loss_gtex / args.gradient_accumulation_steps
        (divided_loss_gtex * args.loss_multiplier_gtex).backward()

        if (not args.freeze_lm_embeddings) and (
            not args.fsdp or args.fsdp_use_orig_params
        ):
            # Mask gradients for input embeddings s.t. we only update the added tokens <image> and <|endofchunk|>
            if args.fsdp:
                embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
            else:
                embed_grad = (
                    model.module.lang_encoder.get_input_embeddings().weight.grad
                )
            zero_mask = torch.zeros_like(embed_grad)
            zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
            zero_mask[endofchunk_token_id] = torch.ones_like(
                zero_mask[endofchunk_token_id]
            )
            if args.fsdp:
                model.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )
            else:
                model.module.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )

        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                tcga_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_tcga
                    * args.world_size
                    / step_time_m.val
                )
                tcga_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_tcga
                    / step_time_m.val
                )
                gtex_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_gtex
                    * args.world_size
                    / step_time_m.val
                )
                gtex_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_gtex
                    / step_time_m.val
                )
                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "tcga_samples_per_second": tcga_samples_per_second,
                        "tcga_samples_per_second_per_gpu": tcga_samples_per_second_per_gpu,
                        "gtex_samples_per_second": gtex_samples_per_second,
                        "gtex_samples_per_second_per_gpu": gtex_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_tcga": loss_tcga.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )
                wandb.log(
                    {"loss_gtex": loss_gtex.item(), "global_step": global_step},
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss TCGA: {loss_tcga.item():.3f} // Loss GTEX: {loss_gtex.item():.3f}"
            )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (
        name,
        p,
    ) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # also remove the keys in state_dict generated from
    # lang_encoder.old_decoder_blocks and lang_encoder.gated_cross_attn_layers
    # because these are already saved in lang_encoder.model...
    to_delete = [
        n
        for n in state_dict.keys()
        if ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
    ]
    for name in to_delete:
        del state_dict[name]
    return state_dict


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True),
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer, group=args.my_group)

    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        if not (args.fsdp and not args.fsdp_use_orig_params):
            model_state = filter_state_dict_to_trainable(model, model_state)

        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        print(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt")
        torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/checkpoint_{epoch}.pt")

        if args.delete_previous_checkpoint:
            if epoch > 0:
                os.remove(f"{args.run_name}/checkpoint_{epoch-1}.pt")
