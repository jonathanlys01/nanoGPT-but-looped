# Loop-Residual Neural Networks Implementation

This repository implements the Loop-Residual Neural Network architecture from the paper "Loop-Residual Neural Networks for Iterative Refinement" (Ng & Wang, 2024). The Loop-Residual approach enables better performance without increasing the model size by utilizing iterative refinement through loops with residual connections.

## Key Concept

The core idea of Loop-Residual Neural Networks is simple but powerful:
- Instead of passing input through many sequential layers (standard approach)
- We loop over a smaller set of layers multiple times, refining the prediction with each pass
- The model learns to predict the residual between the current state and the desired state

This approach follows the formula: x^(n) = x^(n-1) + fθ(x^(n-1))


Where:
- x^(n) is the hidden state at iteration n
- x^(0) is the initial hidden state
- fθ is the function that predicts the residual

## install

```bash
uv sync && uv pip install --no-build-isolation flash-attn
```

Dependencies:

    pytorch <3
    numpy <3
    transformers for huggingface transformers <3 (to load GPT-2 checkpoints)
    datasets for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
    tiktoken for OpenAI's fast BPE code <3
    wandb for optional logging <3
    tqdm for progress bars <3


## Project Structure

- `model.py` - Contains the implementation of the Loop-Residual GPT model
- `train.py` - Training script adapted to support Loop-Residual architecture
- `run_small_experiments.sh` - Script to run small-scale experiments
- `run_experiments.sh` - Script to reproduce the paper's experiments

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/loop-residual-nn.git
cd loop-residual-nn
```

2. Install the required dependencies:
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## Reproducing Experiments

### Quick Start with Small Models

For quick experimentation and validation of the Loop-Residual concept, we provide a script to train small models:

```bash
chmod +x run_small_experiments.sh
./run_small_experiments.sh
```

This script runs four small-scale experiments:
1. **Tiny-4L**: Baseline 4-layer model without looping
2. **Tiny-2L-2Loop**: 2 layers looped twice (to compare with the 4-layer baseline)
3. **Tiny-2L-3Loop**: 2 layers looped three times
4. **Tiny-1L-4Loop**: 1 layer looped four times

These small models train quickly even on modest hardware, allowing you to validate the concept before scaling up.

### Full Paper Reproduction

To reproduce the experiments from the paper, use:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

This runs the following key experiments:
1. **GPT2-124M**: Standard GPT-2 model with 12 layers (baseline)
2. **Loop-Residual GPT2-81M**: 6 layers looped 6 times - this is the main experiment from the paper
3. **GPT2-45M-Lite**: Small model with 1 layer (baseline for small models)
4. **Loop-Residual GPT2-45M**: 1 layer looped twice

Note: These full-scale experiments require significant computational resources. For running on multiple GPUs or nodes, refer to the distributed training section below.

### Custom Experiments

To run a custom experiment, you can use the training script directly:

```bash
python train.py \
  --out_dir="experiments/my_experiment" \
  --use_loop_residual=True \
  --n_loops=4 \
  --loop_layers=3 \
  --n_layer=3 \
  --n_head=4 \
  --n_embd=256 \
  --batch_size=8 \
  --max_iters=10000
```

## Distributed Training

For training larger models on multiple GPUs:

```bash
# On a single node with 4 GPUs:
torchrun --standalone --nproc_per_node=4 train.py \
  --out_dir="experiments/multi_gpu" \
  --use_loop_residual=True \
  --n_loops=6 \
  --loop_layers=6

# Across multiple nodes:
# On the first (master) node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py \
  --out_dir="experiments/multi_node" \
  --use_loop_residual=True \
  --n_loops=6 \
  --loop_layers=6

# On the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py \
  --out_dir="experiments/multi_node" \
  --use_loop_residual=True \
  --n_loops=6 \
  --loop_layers=6
```

## Results

According to the paper, these are the expected results:

| Model | Parameters | Layers | Loops | Validation Loss |
|-------|------------|--------|-------|----------------|
| GPT2-124M | 124M | 12 | 1 | 3.12 |
| Loop-Residual GPT2-81M | 81M | 6 | 6 | 3.11 |
| Lite GPT2-45M | 45M | 1 | 1 | 3.98 |
| Loop-Residual GPT2-45M | 45M | 1 | 2 | 3.67 |

The key finding is that Loop-Residual GPT2-81M achieves comparable performance to the larger GPT2-124M despite having 35% fewer parameters, by using iterative refinement through looping.

## How Loop-Residual Works

In a standard Transformer, the input passes through all layers sequentially, with each layer processing the input once:

```
Input → Layer 1 → Layer 2 → ... → Layer N → Output
```

In the Loop-Residual architecture, the input passes through a subset of layers multiple times, with each pass refining the previous output:

```
Input → Loop 1: [Layer 1 → Layer 2] → Loop 2: [Layer 1 → Layer 2] → ... → Output
```

This is implemented in the `LoopBlock` class in our code:

```python
class LoopBlock(nn.Module):
    def forward(self, x):
        # Initial state x^(0)
        x_initial = x

        # Loop n times for iterative refinement
        for _ in range(self.n_loops):
            # Compute the residual through the blocks
            residual = x.clone()
            for block in self.blocks:
                residual = block(residual)

            # Update the state with the residual: x^(n) = x^(n-1) + fθ(x^(n-1))
            x = x + (residual - x)

        return x
```

## Customizing for Your Own Use

To adapt the Loop-Residual architecture for your own models:

1. Modify the `LoopBlock` class to fit your architecture
2. Adjust the looping parameters to balance computation vs. model size
3. Experiment with different numbers of layers and loops

## Citing

If you use this implementation in your research, please cite the original paper:

```
@article{ng2024loop,
  title={Loop-Residual Neural Networks for Iterative Refinement},
  author={Ng, Kei-Sing and Wang, Qingchen},
  journal={arXiv preprint arXiv:2409.14199},
  year={2024}
}
```

## Acknowledgements

This implementation is built on nanoGPT by Andrej Karpathy. The Loop-Residual architecture is based on the paper by Ng & Wang (2024).