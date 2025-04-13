#!/bin/bash
# Bash script to run smaller-scale Loop-Residual Neural Network experiments
# This script focuses on smaller models for faster experimentation

set -e  # Exit immediately if a command exits with a non-zero status

# Common configuration parameters - using smaller values for quicker experiments
BATCH_SIZE=8
LEARNING_RATE=6e-4
DATASET="openwebtext"
MAX_ITERS=10000  # Reduced significantly for faster experimentation
COMPILE=False    # Set to False for compatibility, can enable for speed
WANDB_LOG=True   # Enable wandb logging
DROPOUT=0.1      # Adding some dropout for these smaller models

# Function to run an experiment with the given parameters
run_experiment() {
    NAME=$1
    OUTPUT_DIR=$2
    INIT_FROM=$3
    USE_LOOP=$4
    LOOPS=$5
    LOOP_LAYERS=$6
    N_LAYER=$7
    N_HEAD=$8
    N_EMBD=$9

    echo "============================================================"
    echo "Running experiment: $NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "Init from: $INIT_FROM"
    echo "Loop-Residual: $USE_LOOP"
    if [ "$USE_LOOP" = "True" ]; then
        echo "Loops: $LOOPS, Loop layers: $LOOP_LAYERS"
    else
        echo "Standard layers: $N_LAYER"
    fi
    echo "Model dimensions: heads=$N_HEAD, embedding=$N_EMBD"
    echo "============================================================"

    python train.py \
        --out_dir="$OUTPUT_DIR" \
        --wandb_run_name="$NAME" \
        --init_from="$INIT_FROM" \
        --use_loop_residual="$USE_LOOP" \
        --n_loops="$LOOPS" \
        --loop_layers="$LOOP_LAYERS" \
        --n_layer="$N_LAYER" \
        --n_head="$N_HEAD" \
        --n_embd="$N_EMBD" \
        --batch_size="$BATCH_SIZE" \
        --learning_rate="$LEARNING_RATE" \
        --dataset="$DATASET" \
        --max_iters="$MAX_ITERS" \
        --compile="$COMPILE" \
        --wandb_log="$WANDB_LOG" \
        --dropout="$DROPOUT"

    echo "Experiment $NAME completed."
    echo ""
}

# Make sure we have the required directories
mkdir -p small_experiments

# Experiment 1: Tiny-4L (baseline)
# Small 4-layer model without loop-residual
run_experiment \
    "Tiny-4L" \
    "small_experiments/tiny-4l" \
    "scratch" \
    "False" \
    "1" \
    "1" \
    "4" \
    "4" \
    "256"

# Experiment 2: Tiny-2L-2Loop (loop-residual)
# Small model with 2 layers looped twice
run_experiment \
    "Tiny-2L-2Loop" \
    "small_experiments/tiny-2l-2loop" \
    "scratch" \
    "True" \
    "2" \
    "2" \
    "2" \
    "4" \
    "256"

# Experiment 3: Tiny-2L-3Loop (loop-residual with more loops)
# Small model with 2 layers looped three times (more computation)
run_experiment \
    "Tiny-2L-3Loop" \
    "small_experiments/tiny-2l-3loop" \
    "scratch" \
    "True" \
    "3" \
    "2" \
    "2" \
    "4" \
    "256"

# Experiment 4: Tiny-1L-4Loop (single-layer loop-residual)
# Small model with 1 layer looped four times
run_experiment \
    "Tiny-1L-4Loop" \
    "small_experiments/tiny-1l-4loop" \
    "scratch" \
    "True" \
    "4" \
    "1" \
    "1" \
    "4" \
    "256"

echo "All small-scale experiments completed successfully!"
echo "To view detailed results, check the output directories or Weights & Biases if enabled."