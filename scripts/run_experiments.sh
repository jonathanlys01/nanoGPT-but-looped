#!/bin/bash
# Bash script to run all the Loop-Residual Neural Network experiments from the paper
# This reproduces the experiments with different configurations and saves results to separate directories

set -e  # Exit immediately if a command exits with a non-zero status

# Common configuration parameters
BATCH_SIZE=12
LEARNING_RATE=6e-4
DATASET="openwebtext"
MAX_ITERS=100000  # Reduced for faster experimentation, paper used more
COMPILE=False     # Set to False for compatibility, can enable for speed
WANDB_LOG=True    # Enable wandb logging

# Function to run an experiment with the given parameters
run_experiment() {
    NAME=$1
    OUTPUT_DIR=$2
    MODEL_SIZE=$3
    USE_LOOP=$4
    LOOPS=$5
    LOOP_LAYERS=$6
    N_LAYER=$7

    echo "============================================================"
    echo "Running experiment: $NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "Model size: $MODEL_SIZE"
    echo "Loop-Residual: $USE_LOOP"
    if [ "$USE_LOOP" = "True" ]; then
        echo "Loops: $LOOPS, Loop layers: $LOOP_LAYERS"
    else
        echo "Standard layers: $N_LAYER"
    fi
    echo "============================================================"

    python train.py \
        --out_dir="$OUTPUT_DIR" \
        --wandb_run_name="$NAME" \
        --init_from="$MODEL_SIZE" \
        --use_loop_residual="$USE_LOOP" \
        --n_loops="$LOOPS" \
        --loop_layers="$LOOP_LAYERS" \
        --n_layer="$N_LAYER" \
        --batch_size="$BATCH_SIZE" \
        --learning_rate="$LEARNING_RATE" \
        --dataset="$DATASET" \
        --max_iters="$MAX_ITERS" \
        --compile="$COMPILE" \
        --wandb_log="$WANDB_LOG"

    echo "Experiment $NAME completed."
    echo ""
}

# Make sure we have the required directories
mkdir -p experiments

# Experiment 1: GPT2-124M (baseline)
# Standard GPT-2 model with 12 layers
run_experiment \
    "GPT2-124M" \
    "experiments/gpt2-124m" \
    "gpt2" \
    "False" \
    "1" \
    "1" \
    "12"

# Experiment 2: Loop-Residual GPT2-81M (6 loops over 6 layers)
# This is the main experiment described in the paper
run_experiment \
    "Loop-Residual-GPT2-81M" \
    "experiments/loop-residual-gpt2-81m" \
    "gpt2" \
    "True" \
    "6" \
    "1" \
    "6"

# Experiment 3: GPT2-45M-Lite (baseline)
# Small model with just 1 layer
run_experiment \
    "GPT2-45M-Lite" \
    "experiments/gpt2-45m-lite" \
    "scratch" \
    "False" \
    "1" \
    "1" \
    "1"

# Experiment 4: Loop-Residual GPT2-45M (2 loops over 1 layer)
# Small model with loop-residual architecture
run_experiment \
    "Loop-Residual-GPT2-45M" \
    "experiments/loop-residual-gpt2-45m" \
    "scratch" \
    "True" \
    "2" \
    "1" \
    "1"

# Additional experiment: Medium scale - Loop-Residual GPT2-175M (4 loops over 12 layers)
# This is an additional experiment not explicitly in the paper
run_experiment \
    "Loop-Residual-GPT2-175M" \
    "experiments/loop-residual-gpt2-175m" \
    "gpt2-medium" \
    "True" \
    "4" \
    "12" \
    "12"

echo "All experiments completed successfully!"