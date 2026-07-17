"""Evaluate a trained BEV fusion model on a nuScenes split.

Runs the full camera + LiDAR + fusion + head pipeline (via ``BEVFusionModel``)
over a dataset, reports IoU / precision / recall, and optionally saves
prediction and ground-truth visualizations.
"""

import os
import time
import argparse
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import yaml
from tqdm import tqdm

from models.bev_model import BEVFusionModel
from datasets.precomputed_bev_dataset import PrecomputedBEVDataset
from utils.metrics import SegmentationMetrics
from utils.visualization import colorize_bev


def custom_collate_fn(batch):
    """Collate that keeps variable-sized LiDAR point clouds as a list."""
    elem = batch[0]
    result = {}
    for key in elem:
        if key == 'lidar' or key in ('calib', 'ego_pose'):
            result[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            result[key] = torch.stack([d[key] for d in batch])
        else:
            result[key] = [d[key] for d in batch]
    return result


def _to_index_map(label: torch.Tensor) -> torch.Tensor:
    """Convert a BEV label tensor to [H, W] class indices."""
    if label.dim() == 3 and label.size(0) > 1:  # one-hot [C, H, W]
        return label.argmax(dim=0)
    if label.dim() == 3 and label.size(0) == 1:  # [1, H, W]
        return label.squeeze(0)
    return label  # already [H, W]


class Tester:
    def __init__(self, model_config: Dict, checkpoint_path: str, device: str = 'cuda'):
        self.device = device
        self.model = BEVFusionModel.from_checkpoint(
            checkpoint_path, model_config=model_config, device=device
        )
        self.num_classes = model_config['segmentation_head']['num_classes']
        self.metrics = SegmentationMetrics(self.num_classes, device)

    @torch.no_grad()
    def test_dataset(
        self,
        test_loader: DataLoader,
        save_predictions: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        self.metrics.reset()
        inference_times = []

        vis_dir = None
        if save_predictions and output_dir:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

        has_labels = False
        for i, batch in enumerate(tqdm(test_loader, desc='Testing')):
            images = batch['image'].to(self.device)
            points_list = [p.to(self.device) for p in batch['lidar']]

            start = time.time()
            predictions = self.model(images, points_list)  # [B, C, H, W] probs
            inference_times.append(time.time() - start)

            pred_indices = predictions.argmax(dim=1)  # [B, H, W]

            if 'bev_label' in batch:
                has_labels = True
                targets = batch['bev_label'].to(self.device)
                target_indices = torch.stack(
                    [_to_index_map(t) for t in targets]
                ).long()
                self.metrics.update(predictions, target_indices)

            if vis_dir:
                tokens = batch.get(
                    'sample_token',
                    [f'sample_{i}_{j}' for j in range(len(pred_indices))],
                )
                for j, token in enumerate(tokens):
                    cv2.imwrite(
                        os.path.join(vis_dir, f'{token}_pred.png'),
                        cv2.cvtColor(
                            colorize_bev(pred_indices[j].cpu().numpy()),
                            cv2.COLOR_RGB2BGR,
                        ),
                    )
                    if 'bev_label' in batch:
                        gt = _to_index_map(batch['bev_label'][j]).cpu().numpy()
                        cv2.imwrite(
                            os.path.join(vis_dir, f'{token}_gt.png'),
                            cv2.cvtColor(colorize_bev(gt), cv2.COLOR_RGB2BGR),
                        )

        metrics = self.metrics.get_metrics() if has_labels else {}
        metrics['mean_inference_time'] = float(np.mean(inference_times))
        metrics['fps'] = 1.0 / float(np.mean(inference_times))
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Test BEV Fusion model')
    parser.add_argument('--model-config', type=str, required=True,
                        help='Path to model configuration YAML')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to NuScenes dataset')
    parser.add_argument('--bev-labels-dir', type=str, required=True,
                        help='Path to precomputed BEV labels')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to test on')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Directory to save outputs')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-vis', action='store_false', dest='visualize',
                        help='Disable visualization saving')
    parser.set_defaults(visualize=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    test_dataset = PrecomputedBEVDataset(
        dataroot=args.dataroot,
        bev_labels_dir=args.bev_labels_dir,
        split=args.split,
        return_tokens=True,
    )
    print(f'Created dataset with {len(test_dataset)} samples')

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
    )

    tester = Tester(model_config, args.checkpoint, device=str(device))

    print(f'Testing model on {args.split} split...')
    metrics = tester.test_dataset(
        test_loader=test_loader,
        save_predictions=args.visualize,
        output_dir=args.output_dir,
    )

    print('\nTest Results:')
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')

    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f'{k}: {v}\n')
    print(f'\nSaved metrics to {metrics_path}')
    print(f'Saved predictions to {args.output_dir}')


if __name__ == '__main__':
    main()
