#!/bin/bash

# Testing launch script
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Default settings
CONFIG_DIR="configs"
MODEL_CONFIG="model.yaml"
CHECKPOINT="best_model.pth"
OUTPUT_DIR="predictions"
DATAROOT="/home/mevi/Documents/bev/nuscenes07"
BEV_LABELS_DIR="/home/mevi/Documents/bev/test"
SPLIT="val"
BATCH_SIZE=4
NUM_WORKERS=4
VISUALIZE=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
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
        --checkpoint)
            CHECKPOINT="$2"
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
        --split)
            SPLIT="$2"
            shift
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift
            shift
            ;;
        --no-vis)
            VISUALIZE=0
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

# Print settings
echo "Testing with the following settings:"
echo "  Model config: $CONFIG_DIR/$MODEL_CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Data root: $DATAROOT"
echo "  BEV labels directory: $BEV_LABELS_DIR"
echo "  Split: $SPLIT"
echo "  Batch size: $BATCH_SIZE"
echo "  Visualize: $VISUALIZE"

# Launch testing
python test.py \
    --model-config "$CONFIG_DIR/$MODEL_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --dataroot "$DATAROOT" \
    --bev-labels-dir "$BEV_LABELS_DIR" \
    --split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    ${VISUALIZE:+--visualize} 