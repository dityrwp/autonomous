#!/bin/bash

# Training launch script
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Default settings
NUM_GPUS=1
CONFIG_DIR="configs"
MODEL_CONFIG="model.yaml"
TRAIN_CONFIG="train.yaml"
OUTPUT_DIR="outputs"
DATAROOT="/home/mevi/Documents/bev/nuscenes07"
BEV_LABELS_DIR="/home/mevi/Documents/bev/test"
DEBUG=0
SIMULATE_VLP16=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpus)
            NUM_GPUS="$2"
            shift
            shift
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift
            shift
            ;;
        --model-config)
            MODEL_CONFIG="$2"
            shift
            shift
            ;;
        --train-config)
            TRAIN_CONFIG="$2"
            shift
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --dataroot)
            DATAROOT="$2"
            shift
            shift
            ;;
        --bev-labels-dir)
            BEV_LABELS_DIR="$2"
            shift
            shift
            ;;
        --debug)
            DEBUG=1
            shift
            ;;
        --simulate-vlp16)
            SIMULATE_VLP16=1
            shift
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy configs for reproducibility
cp "$CONFIG_DIR/$MODEL_CONFIG" "$OUTPUT_DIR/"
cp "$CONFIG_DIR/$TRAIN_CONFIG" "$OUTPUT_DIR/"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

# Print settings
echo "Training with the following settings:"
echo "  Model config: $CONFIG_DIR/$MODEL_CONFIG"
echo "  Train config: $CONFIG_DIR/$TRAIN_CONFIG"
echo "  Output directory: $OUTPUT_DIR"
echo "  Data root: $DATAROOT"
echo "  BEV labels directory: $BEV_LABELS_DIR"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Debug mode: $DEBUG"
echo "  Simulate VLP-16: $SIMULATE_VLP16"

# Launch training
if [ $NUM_GPUS -gt 1 ]; then
    # Multi-GPU training
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        train.py \
        --model-config "$CONFIG_DIR/$MODEL_CONFIG" \
        --train-config "$CONFIG_DIR/$TRAIN_CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        --dataroot "$DATAROOT" \
        --bev-labels-dir "$BEV_LABELS_DIR" \
        ${DEBUG:+--debug} \
        ${SIMULATE_VLP16:+--simulate-vlp16}
else
    # Single-GPU training
    python train.py \
        --model-config "$CONFIG_DIR/$MODEL_CONFIG" \
        --train-config "$CONFIG_DIR/$TRAIN_CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        --dataroot "$DATAROOT" \
        --bev-labels-dir "$BEV_LABELS_DIR" \
        ${DEBUG:+--debug} \
        ${SIMULATE_VLP16:+--simulate-vlp16}
fi 