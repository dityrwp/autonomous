"""Single-sample BEV inference.

Runs the trained model on one camera image + one LiDAR point cloud and writes a
colorized BEV segmentation map (optionally next to the input image for a quick
side-by-side). Useful as a minimal demo / smoke test for a checkpoint without
needing the full nuScenes dataset wired up.

Example:
    python inference.py \
        --model-config configs/model.yaml \
        --checkpoint outputs/checkpoints/best_model.pth \
        --image sample.jpg \
        --lidar sample.bin \
        --output prediction.png
"""

import argparse

import cv2
import numpy as np
import torch
import yaml

from models.bev_model import BEVFusionModel
from utils.visualization import colorize_bev, CLASS_NAMES, CLASS_COLORS


def load_image(path: str, width: int, height: int) -> torch.Tensor:
    """Load an RGB image and return a normalized [1, 3, H, W] tensor."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)


def load_lidar(path: str, num_features: int = 5) -> torch.Tensor:
    """Load a nuScenes-style LiDAR .bin (float32, N x num_features)."""
    points = np.fromfile(path, dtype=np.float32)
    points = points.reshape(-1, num_features)
    return torch.from_numpy(points)


def parse_args():
    parser = argparse.ArgumentParser(description='Single-sample BEV inference')
    parser.add_argument('--model-config', type=str, required=True,
                        help='Path to model configuration YAML')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the input camera image')
    parser.add_argument('--lidar', type=str, required=True,
                        help='Path to the input LiDAR .bin point cloud')
    parser.add_argument('--output', type=str, default='prediction.png',
                        help='Path to save the colorized BEV prediction')
    parser.add_argument('--side-by-side', action='store_true',
                        help='Also save the input image beside the prediction')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    height, width = model_config.get('input_config', {}).get('image_size', [480, 640])
    num_features = model_config.get('lidar_backbone', {}).get('num_point_features', 5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = BEVFusionModel.from_checkpoint(
        args.checkpoint, model_config=model_config, device=device
    )

    image = load_image(args.image, width, height).to(device)
    points = load_lidar(args.lidar, num_features).to(device)

    pred = model.predict(image, [points])[0].cpu().numpy()  # [H, W] class indices
    pred_rgb = colorize_bev(pred)

    present = sorted(np.unique(pred))
    print('Predicted classes present:',
          ', '.join(CLASS_NAMES[c] for c in present if c < len(CLASS_NAMES)))

    out_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
    if args.side_by_side:
        cam = cv2.resize(cv2.imread(args.image),
                         (pred_rgb.shape[1], pred_rgb.shape[0]))
        out_bgr = np.hstack([cam, out_bgr])

    cv2.imwrite(args.output, out_bgr)
    print(f'Saved BEV prediction to {args.output}')


if __name__ == '__main__':
    main()
