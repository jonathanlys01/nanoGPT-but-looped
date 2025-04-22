"""Profiler and FLOP counter for the model."""

import time

import torch
import torch.profiler
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.flop_counter import FlopCounterMode

from config import Config, GPTConfig, get_config
from model import LoopResidualGPT
from train import _move, _setup_ddp, get_batch, get_dataloader


def dist_profiled_run(config: Config):
    ddp = _setup_ddp(config)

    torch.manual_seed(config.seed + ddp.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    model_args = config.model
    gptconf = GPTConfig(**model_args)
    model = LoopResidualGPT(gptconf)

    model.to(config.device)
    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2))
    model = torch.compile(model)

    model = DDP(model, device_ids=[ddp.local_rank])

    loader = get_dataloader(config=config, split="train")
    iterator = iter(loader)

    X, Y = next(iterator)
    X, Y = _move(X, Y, device=config.device)

    acc = 16

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=acc - 4, warmup=1, active=4),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_dir"),
    ) as prof:
        for _ in range(acc):
            with torch.profiler.record_function("forward"):
                model.require_backward_grad_sync = False
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(X, Y)
                    loss = loss / acc

                X, Y = next(iterator)
                X, Y = _move(X, Y, device=config.device)
            with torch.profiler.record_function("backward"):
                loss.backward()
                prof.step()
        with torch.profiler.record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        prof.step()

    destroy_process_group()


def estimate_flop(config: Config):
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    model_args = config.model
    gptconf = GPTConfig(**model_args)
    model = LoopResidualGPT(gptconf)

    model.to(config.device)

    # Flop Counter

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        X, Y = get_batch(config=config, split="train")
        X, _ = _move(X, Y, device=config.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(X)[0].sum().backward()
    total_flops = flop_counter.get_total_flops()
    print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")

    N = 100

    t0 = time.time()
    for i in range(N):
        X, Y = get_batch(config=config, split="train")
        X, Y = _move(X, Y, device=config.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(X)[0].sum().backward()
    t1 = time.time()

    mfu = model.estimate_mfu(config.batch_size * N, dt=t1 - t0)
    print(f"MFU: {mfu:.2f}%")


if __name__ == "__main__":
    args = get_config()
    print("Estimating FLOPs...")
    estimate_flop(args)
    print("Profiling...")
    dist_profiled_run(args)
