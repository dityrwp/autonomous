import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

class NuScenesRCDataset(Dataset):
    def __init__(self, data_path, version="v1.0-trainval", bev_size=(256, 256), lidar_tilt_angle=10):
        """
        Custom DataLoader for RC-BEV segmentation and 3D object detection.
        
        Args:
            data_path (str): Path to the nuScenes dataset.
            version (str): Dataset version (e.g., "v1.0-trainval").
            bev_size (tuple): Size of the BEV grid (height, width).
            lidar_tilt_angle (float): Tilt angle of the LiDAR in degrees.
        """
        self.data_path = data_path
        self.bev_size = bev_size
        self.lidar_tilt_angle = np.deg2rad(lidar_tilt_angle)
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=False)
        self.samples = self.nusc.sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load front camera image
        cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        img_path = os.path.join(self.data_path, cam_front_data['filename'])
        img = Image.open(img_path)
        
        # Load LiDAR point cloud
        lidar_top_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = os.path.join(self.data_path, lidar_top_data['filename'])
        pcd = LidarPointCloud.from_file(lidar_path).points[:3, :]  # x, y, z
        
        # Rotate LiDAR points to compensate for tilt
        R = np.array([
            [1, 0, 0],
            [0, np.cos(self.lidar_tilt_angle), -np.sin(self.lidar_tilt_angle)],
            [0, np.sin(self.lidar_tilt_angle), np.cos(self.lidar_tilt_angle)]
        ])
        pcd = (R @ pcd).T  # Rotate and transpose to (N, 3)
        
        # Voxelize LiDAR to BEV grid
        bev_grid = self.voxelize(pcd)
        
        # Load 3D object detection labels (bounding boxes)
        boxes = self.get_3d_boxes(sample)
        
        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        bev_tensor = torch.from_numpy(bev_grid).float()  # [H_bev, W_bev]
        boxes_tensor = torch.from_numpy(boxes).float()  # [N_boxes, 7] (x, y, z, w, l, h, yaw)
        
        return img_tensor, bev_tensor, boxes_tensor

    def voxelize(self, points, voxel_size=0.1):
        """
        Convert LiDAR points to a BEV grid.
        
        Args:
            points (np.ndarray): LiDAR points (N, 3).
            voxel_size (float): Size of each voxel in meters.
        
        Returns:
            bev_grid (np.ndarray): BEV grid (H_bev, W_bev).
        """
        # Convert points to BEV grid indices
        indices = (points[:, :2] / voxel_size).astype(int)
        indices[:, 0] = np.clip(indices[:, 0], 0, self.bev_size[0] - 1)
        indices[:, 1] = np.clip(indices[:, 1], 0, self.bev_size[1] - 1)
        
        # Create BEV grid
        bev_grid = np.zeros(self.bev_size)
        bev_grid[indices[:, 0], indices[:, 1]] = 1  # Binary occupancy
        return bev_grid

    def get_3d_boxes(self, sample):
        """
        Extract 3D bounding boxes for a sample.
        
        Args:
            sample (dict): nuScenes sample dictionary.
        
        Returns:
            boxes (np.ndarray): 3D bounding boxes (N_boxes, 7).
        """
        boxes = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            box = self.nusc.get_box(ann_token)
            boxes.append([
                box.center[0], box.center[1], box.center[2],  # x, y, z
                box.wlh[0], box.wlh[1], box.wlh[2],           # width, length, height
                box.orientation.yaw_pitch_roll[0]             # yaw
            ])
        return np.array(boxes) if boxes else np.zeros((0, 7))  # Handle empty cases