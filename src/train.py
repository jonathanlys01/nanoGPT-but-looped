import math
import os
import pickle
import time
from contextlib import nullcontext
from dataclasses import dataclass
from os.path import join as pjoin

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed import all_reduce, destroy_process_group, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset

# Import the Loop-Residual model
from config import Config, GPTConfig
from model import LoopResidualGPT


#######################################################################
# DDP (Distributed Data Parallel) config


@dataclass
class DDPConfig:
    master_process: bool
    seed_offset: int
    world_size: int
    gradient_accumulation_steps: int
    local_rank: int


def _setup_ddp(config: Config) -> DDPConfig:
    if config.ddp:
        import idr_torch

        ddp_rank = idr_torch.rank
        ddp_local_rank = idr_torch.local_rank
        ddp_world_size = idr_torch.size

        init_process_group(
            backend=config.backend,
            init_method="env://",
            world_size=ddp_world_size,
            rank=ddp_rank,
        )

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert config.gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps = config.gradient_accumulation_steps // ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        gradient_accumulation_steps = config.gradient_accumulation_steps
        ddp_local_rank = None

    return DDPConfig(
        master_process=master_process,
        seed_offset=seed_offset,
        world_size=ddp_world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        local_rank=ddp_local_rank,
    )


def _reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


#######################################################################
# Data


def _move(x: torch.Tensor, y: torch.Tensor, device, pin_memory=False):
    if "cuda" in device:
        if pin_memory:
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:  # default
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_batch(config: Config, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(config.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(config.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - config.model.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + config.model.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + config.model.block_size]).astype(np.int64)) for i in ix])

    return x, y


class MemmapDataset(IterableDataset):
    def __init__(self, config: Config, split: str):
        super().__init__()
        self.config = config
        self.split = split

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            torch.manual_seed(torch.initial_seed() + worker_info.id)

        while True:
            x, y = get_batch(config=self.config, split=self.split)
            yield x, y


def get_dataloader(config: Config, split: str):
    dataset = MemmapDataset(config=config, split=split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # already batched
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
    )
    return dataloader


#######################################################################
# Utils
# Helps estimate an arbitrarily accurate loss over either split using many batches


@torch.no_grad()
def estimate_loss(model, config: Config, ctx):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters).to(config.device)
        for k in range(config.eval_iters):
            X, Y = get_batch(config=config, split=split)
            X, Y = _move(X, Y, config.device, pin_memory=True)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, config: Config):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


#######################################################################
# Debug


def _debug_ddp():
    import socket

    print(
        f"[Rank {os.environ.get('RANK')}] Host: {socket.gethostname()}, "
        f"Local Rank: {os.environ.get('LOCAL_RANK')}, "
        f"World Size: {os.environ.get('WORLD_SIZE')}, ",
    )

    print(f"Distributed initialized: {torch.distributed.is_initialized()}")


class Monitor:
    def __init__(self, config: Config, is_master: bool):
        self.active = config.debug and is_master
        if not self.active:
            return
        self.cpt = time.time()

    def step(self, step_name):
        if not self.active:
            return
        now = time.time()
        print(f"\033[92m> {step_name} | {(now - self.cpt) * 1000:.2f}ms\033[0m")
        self.cpt = now


#######################################################################
# Main


def run(config: Config):  # noqa: C901, PLR0912, PLR0915
    ddp = _setup_ddp(config)
    monitor = Monitor(config, ddp.master_process)
    tokens_per_iter = ddp.gradient_accumulation_steps * ddp.world_size * config.batch_size * config.model.block_size

    if ddp.master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        os.makedirs(config.out_dir, exist_ok=True)

    torch.manual_seed(config.seed + ddp.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in config.device else "cpu"

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    meta_path = pjoin(config.data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = config.model

    if config.debug:
        _debug_ddp()

    if config.init_from == "scratch":
        # init a new model from scratch
        if ddp.master_process:
            print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None and ddp.master_process:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = LoopResidualGPT(gptconf)  # Use LoopResidualGPT instead of GPT

    elif config.init_from == "resume":
        print(f"Resuming training from {config.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = pjoin(config.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in [
            "n_layer",
            "n_head",
            "n_embd",
            "block_size",
            "bias",
            "vocab_size",
            "n_loops",
            "loop_layers",
        ]:  # Add Loop-Residual parameters
            if k in checkpoint_model_args:  # Only set if the key exists in the checkpoint
                model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = LoopResidualGPT(gptconf)  # Use LoopResidualGPT instead of GPT
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    elif config.init_from.startswith("gpt2"):
        raise NotImplementedError("GPT-2 model loading not implemented yet")

    # crop down the model block size if desired, using model surgery
    if config.model.block_size < model.config.block_size:
        model.crop_block_size(config.model.block_size)
        model_args["block_size"] = config.model.block_size  # so that the checkpoint will have the right value
    model.to(config.device)

    monitor.step("model init")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(device="cuda", enabled=(config.dtype == "float16"))
    # optimizer
    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2))
    if config.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory
    if config.compile:
        if ddp.master_process:
            print("compiling the model... (takes a ~minute)")
        model = torch.compile(
            model,
            mode="default",
            fullgraph=True,
            dynamic=True,
        )  # requires PyTorch 2.0

    # wrap model into DDP container
    if config.ddp:
        model = FSDP(model, device_id=ddp.local_rank) if config.fsdp else DDP(model, device_ids=[ddp.local_rank])

    # logging
    if config.wandb_log and ddp.master_process:
        import wandb

        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config)

    monitor.step("optimizer/compile/(wandb) init")

    # training loop

    loader = get_dataloader(config=config, split="train")
    iterator = iter(loader)
    monitor.step("get dataloader")

    X, Y = next(iterator)
    X, Y = _move(X, Y, config.device)  # move to GPU
    monitor.step("get first batch")
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if config.ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    grad_norm = 0.0

    checkpoint_future = None

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        monitor.step(f"[{iter_num}] get learning rate")

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config, ctx)
            # reduce the losses across all processes
            if config.ddp:
                for k in losses:
                    losses[k] = _reduce_mean(losses[k], ddp.world_size)
            torch.cuda.synchronize()  # wait for all processes to finish
            if ddp.master_process:
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if config.wandb_log:
                    # Add Loop-Residual info to wandb logs

                    loop_weights = raw_model.transformer.get_loop_weights()
                    labels = [f"Loop {i + 1}" for i in range(len(loop_weights))]
                    data = [[label, weight] for (label, weight) in zip(labels, loop_weights)]
                    table = wandb.Table(data=data, columns=["label", "weight"])
                    plot = wandb.plot.bar(
                        table,
                        "label",
                        "weight",
                        title="Loop Weights",
                    )

                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                            # Add Loop-Residual specific info
                            "weights": plot,
                            "grad_norm": grad_norm,
                        },
                    )

            if losses["val"] < best_val_loss and config.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }

                    if ddp.master_process:
                        print(f"saving checkpoint to {config.out_dir}")

                    # torch.save(checkpoint, os.path.join(config.out_dir, "ckpt.pt"))
                    if checkpoint_future is not None:
                        checkpoint_future.result()

                    checkpoint_future = dcp.async_save(checkpoint, checkpoint_id=os.path.join(config.out_dir, "ckpt"))

        if iter_num == 0 and config.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(ddp.gradient_accumulation_steps):
            if config.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = micro_step == ddp.gradient_accumulation_steps - 1
            with ctx:
                _, loss = model(X, Y)
                loss = loss / ddp.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            monitor.step(f"[{iter_num}/{micro_step}] forward")
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = next(iterator)
            X, Y = _move(X, Y, config.device)  # move to GPU
            monitor.step(f"[{iter_num}/{micro_step}] get next batch")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            monitor.step(f"[{iter_num}/{micro_step}] backward")
        # clip the gradient
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
            monitor.step(f"[{iter_num}/{micro_step}] clip grad")
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        monitor.step(f"[{iter_num}] scaler step")
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        monitor.step(f"[{iter_num}] zero grad")

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config.log_interval == 0 and ddp.master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * ddp.gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(config.batch_size * ddp.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.95 * running_mfu + 0.05 * mfu

            log = (
                f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%, g-norm {grad_norm:.2f}"
            )
            print(log)
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config.max_iters:
            # save last model
            if ddp.master_process:
                print(f"saving final model to {config.out_dir}")
                torch.save(
                    raw_model.state_dict(),
                    os.path.join(config.out_dir, "final_model.pt"),
                )
            break

    if config.ddp:
        destroy_process_group()


if __name__ == "__main__":
    from config import get_config

    args = get_config()

    run(args)
