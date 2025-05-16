import builtins
import os

import idr_torch
import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn.parallel import DistributedDataParallel as DDP


def print(*args, **kwargs):
    if idr_torch.local_rank == 0:
        builtins.print(*args, **kwargs)


class TrainerState(Stateful):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        super(TrainerState, self).__init__()
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def _zero_weights(self):
        for param in self.parameters():
            if param.requires_grad:
                param.data.zero_()

    def forward(self, x):
        return self.linear(x)


def setup_ddp():
    init_process_group(
        backend="cpu:gloo,cuda:nccl",
        world_size=idr_torch.size,
        rank=idr_torch.local_rank,
    )

    device = f"cuda:{idr_torch.local_rank}"
    torch.cuda.set_device(device)

    return device


def test():
    # Initialize the process group

    device = setup_ddp()
    raw_model = DummyModel().to(device)

    model = DDP(raw_model, device_ids=[idr_torch.local_rank], output_device=idr_torch.local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    if os.path.exists("test"):
        state_dict = {"app": TrainerState(model, optimizer)}
        dcp.load(state_dict, checkpoint_id="test")

        print("Checkpoint loaded successfully.")

        print("Model state dict:")
        for name, param in model.named_parameters():
            print(name, param.data)

    else:
        raw_model._zero_weights()

        state_dict = {"app": TrainerState(model, optimizer)}

        checkpoint_future = dcp.async_save(state_dict, checkpoint_id="test")

        checkpoint_future.result()

        print("Checkpoint saved successfully.")

    destroy_process_group()


if __name__ == "__main__":
    test()
