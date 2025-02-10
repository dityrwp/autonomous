#!/bin/bash

# Testing launch script
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Default settings
CONFIG_DIR="configs"
MODEL_CONFIG="model.yaml"
CHECKPOINT="best_model.pth"
OUTPUT_DIR="predictions"
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

# Launch testing
python test.py \
    --model-config "$CONFIG_DIR/$MODEL_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    ${VISUALIZE:+--visualize} 