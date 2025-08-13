import os
from dataclasses import dataclass, field
from datetime import datetime

from omegaconf import OmegaConf


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_geglu: bool = True  # Whether to use GEGLU activation (https://arxiv.org/abs/2002.05202), will use GELU if False
    # Loop-Residual parameters
    skip_first: bool = False  # Whether to skip the first residual connection
    n_loop: int = 1  # Number of times to loop over the blocks
    use_loop_pe: bool = False  # Whether to use loop positional encoding
    use_loop_weight: bool = False  # Whether to use loop weight
    # Latent transformer parameters
    n_encoder: int = 0
    n_decoder: int = 0
    # use flash attention
    use_flash_attention: bool = True


@dataclass
class Config:
    # I/O
    out_dir: str = "out"
    eval_interval: int = 100
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = True  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    seed: int = 1337  # random seed for initialization
    num_workers: int = 4  # number of data loading workers
    prefetch_factor: int = 3  # number of batches to prefetch
    # wandb logging
    wandb_log: bool = True  # disabled by default
    wandb_project: str = "looped-gpt2"
    wandb_run_name: str = f"run{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    # data
    dataset: str = "openwebtext"  # openwebtext, fineweb-edu
    gradient_accumulation_steps: int = 16 * 2  # used to simulate larger batch sizes
    batch_size: int = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
    # model
    model: GPTConfig = field(default_factory=GPTConfig)
    # adamw optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600_000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2_000  # how many steps to warm up for
    lr_decay_iters: int = 600_000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    debug: bool = False
    backend: str = "cpu:gloo,cuda:nccl"  # 'nccl', 'gloo', etc.
    # try "cpu:gloo,cuda:nccl" for async save to work
    ddp: bool = True
    fsdp: bool = False
    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

    def __post_init__(self):
        self.data_dir: str = os.path.join("data", self.dataset)


# Export some default configs for different models

default_config = Config()

eval_gpt2_large = Config(
    batch_size=8,
    eval_iters=500,  # use more iterations to get good estimate
    eval_only=True,
    wandb_log=False,
    init_from="gpt2-large",
)

train_gpt2 = Config(
    wandb_log=True,
    wandb_project="owt",
    wandb_run_name="gpt2-124M",
    # these make the total batch size be ~0.5M
    # 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
    batch_size=12,
    gradient_accumulation_steps=5 * 8,
    # this makes total number of tokens be 300B
    max_iters=600000,
    lr_decay_iters=600000,
    # eval stuff
    eval_interval=1000,
    eval_iters=200,
    log_interval=10,
    # weight decay
    weight_decay=1e-1,
)

CONFIG_FLAGS = ("--config", "-c", "config", "cfg")


def get_config() -> Config:
    """
    Read args from CLI, merge with defaults and return a read-only Config object.
    The config file can be specified with the --config/-c/config argument.
    """
    sys_args = OmegaConf.from_cli()

    if any(flag in sys_args for flag in CONFIG_FLAGS):
        flag = next(flag for flag in CONFIG_FLAGS if flag in sys_args)
        cfg_file = sys_args.pop(flag)  # remove the flag from sys_args
        add_args = OmegaConf.load(cfg_file)
    else:
        add_args = sys_args

    default_cfg = OmegaConf.structured(Config)
    args = OmegaConf.merge(default_cfg, add_args)
    return Config(**args)


if __name__ == "__main__":
    cfg = get_config()
    print(OmegaConf.to_yaml(cfg))
