import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from nuscenes.nuscenes import NuScenes
import cv2

class PrecomputedBEVDataset(Dataset):
    """Dataset that loads precomputed BEV labels and corresponding sensor inputs.
    
    This dataset is much more efficient than generating BEV labels on-the-fly
    as it loads precomputed labels from disk.
    """
    
    def __init__(self, 
                 dataroot,
                 bev_labels_dir,
                 split='train',
                 nuscenes_version='v1.0-trainval',
                 transform=None,
                 return_tokens=False,
                 camera_only=False,
                 lidar_only=False,
                 simulate_vlp16=False):
        """
        Args:
            dataroot: Path to NuScenes dataset (for camera/LiDAR data)
            bev_labels_dir: Path to precomputed BEV labels directory
            split: Dataset split ('train', 'val', etc.)
            nuscenes_version: NuScenes dataset version
            transform: Optional transform to apply to inputs
            return_tokens: Whether to return sample tokens
            camera_only: Only return camera data (no LiDAR)
            lidar_only: Only return LiDAR data (no camera)
            simulate_vlp16: Whether to simulate VLP-16 Puck Hi-Res LiDAR
        """
        self.dataroot = dataroot
        self.bev_labels_dir = bev_labels_dir
        self.split = split
        self.transform = transform
        self.return_tokens = return_tokens
        self.camera_only = camera_only
        self.lidar_only = lidar_only
        self.simulate_vlp16 = simulate_vlp16
        
        # Initialize NuScenes for loading camera/LiDAR data
        # Note: We still need NuScenes for calibration data and raw sensor inputs
        self.nusc = NuScenes(version=nuscenes_version, dataroot=dataroot, verbose=False)
        
        # Load mapping file
        mapping_file = os.path.join(bev_labels_dir, split, 'sample_mapping.json')
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}. "
                                    f"Please run precompute_bev_labels.py first.")
        
        with open(mapping_file, 'r') as f:
            self.mapping = json.load(f)
        
        self.samples = self.mapping['samples']
        logging.info(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Get class map from first sample metadata file
        first_sample = self.samples[0]
        first_label_json = os.path.join(bev_labels_dir, split, 
                                       os.path.dirname(first_sample['bev_label_path']),
                                       os.path.basename(first_sample['bev_label_path']).replace(
                                           os.path.splitext(first_sample['bev_label_path'])[1], '.json'))
        
        with open(first_label_json, 'r') as f:
            metadata = json.load(f)
            self.class_map = metadata['class_map']
        
        # Grid properties
        self.grid_size = self.mapping['grid_size']
        self.resolution = self.mapping['resolution']
    
    def _simulate_vlp16(self, lidar_pc):
        """
        Modify LiDAR point cloud to simulate VLP-16 Puck Hi-Res specifications.
        
        Args:
            lidar_pc: LiDAR point cloud tensor of shape (N, 5)
                      Format: [x, y, z, intensity, ring_index]
        
        Returns:
            Modified point cloud tensor
        """
        # Convert to numpy for easier manipulation
        points = lidar_pc.numpy()
        
        # Calculate vertical angles for each point
        # arcsin(z / sqrt(x^2 + y^2 + z^2))
        r_xy = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        vertical_angles = np.degrees(np.arctan2(points[:, 2], r_xy))
        
        # Filter points to match VLP-16 vertical FOV (-10° to 10°)
        vlp16_fov_mask = (vertical_angles >= -10.0) & (vertical_angles <= 10.0)
        filtered_points = points[vlp16_fov_mask]
        
        # If we have too few points after filtering, return the original
        if len(filtered_points) < 100:
            print(f"Warning: Too few points ({len(filtered_points)}) after FOV filtering. Using original point cloud.")
            return lidar_pc
        
        # Simulate 16 channels by quantizing the vertical angles
        # VLP-16 has evenly spaced channels from -10° to 10°
        vertical_angles_filtered = vertical_angles[vlp16_fov_mask]
        
        # Create 16 evenly spaced bins between -10° and 10°
        bins = np.linspace(-10, 10, 17)  # 17 edges for 16 bins
        
        # Digitize the angles into 16 bins
        channel_indices = np.digitize(vertical_angles_filtered, bins) - 1
        channel_indices = np.clip(channel_indices, 0, 15)  # Ensure valid indices
        
        # For each channel, keep a subset of points to simulate lower resolution
        unique_channels = np.unique(channel_indices)
        kept_indices = []
        
        for channel in unique_channels:
            channel_points = np.where(channel_indices == channel)[0]
            
            # If we have more points than we want for this channel, randomly sample
            if len(channel_points) > 0:
                # Keep approximately the same point density ratio
                # NuScenes has 32 channels, we want 16
                keep_ratio = 0.5  # 16/32
                num_to_keep = max(1, int(len(channel_points) * keep_ratio))
                
                # Randomly sample points to keep
                kept_points = np.random.choice(channel_points, size=num_to_keep, replace=False)
                kept_indices.extend(kept_points)
        
        # Get the final simulated point cloud
        simulated_points = filtered_points[kept_indices]
        
        # Add some noise to simulate real-world sensor characteristics
        # VLP-16 has slightly different noise characteristics than the 32-channel LiDAR
        distance = np.sqrt(np.sum(simulated_points[:, :3]**2, axis=1))
        
        # Distance-dependent noise (increases with distance)
        noise_factor = 0.003 + 0.0003 * distance  # Typical for VLP-16
        position_noise = np.random.normal(0, noise_factor[:, np.newaxis], size=simulated_points[:, :3].shape)
        
        # Apply noise only to XYZ coordinates
        simulated_points[:, :3] += position_noise
        
        # Convert back to tensor
        return torch.from_numpy(simulated_points).float()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load BEV label
        label_path = os.path.join(self.bev_labels_dir, self.split, sample['bev_label_path'])
        
        # Check file extension and load accordingly
        if label_path.endswith('.npy'):
            bev_label = np.load(label_path)
        elif label_path.endswith('.png'):
            # If using PNG, we need to convert back to class IDs
            bev_label = np.array(Image.open(label_path))
        else:
            raise ValueError(f"Unsupported label format: {label_path}")
        
        # Convert to tensor
        bev_label = torch.from_numpy(bev_label).long()
        
        # Create a batch dictionary
        batch = {
            'bev_label': bev_label
        }
        
        # Load camera image if requested
        if not self.lidar_only:
            image_path = os.path.join(self.dataroot, sample['image_path'])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transform if provided
            if self.transform:
                image = self.transform(image)
            else:
                # Basic normalization if no transform provided
                image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            
            batch['image'] = image
        
        # Load LiDAR data if requested
        if not self.camera_only:
            lidar_path = os.path.join(self.dataroot, sample['lidar_path'])
            lidar_pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
            
            # Convert to tensor
            lidar_pc = torch.from_numpy(lidar_pc).float()
            
            # Simulate VLP-16 if requested
            if self.simulate_vlp16:
                lidar_pc = self._simulate_vlp16(lidar_pc)
                
            batch['lidar'] = lidar_pc
        
        # Get calibration data
        sample_token = sample['sample_token']
        calib = self._load_calibration(sample_token)
        batch['calib'] = calib
        
        # Add sample tokens if requested
        if self.return_tokens:
            batch['sample_token'] = sample_token
            batch['scene_token'] = sample['scene_token']
        
        return batch
    
    def _load_calibration(self, sample_token):
        """Load calibration data for a sample."""
        sample_record = self.nusc.get('sample', sample_token)
        
        # Get camera and LiDAR sample data
        cam_token = sample_record['data']['CAM_FRONT']
        lidar_token = sample_record['data']['LIDAR_TOP']
        
        cam = self.nusc.get('sample_data', cam_token)
        lidar = self.nusc.get('sample_data', lidar_token)
        
        # Get ego pose for the sample
        ego_pose = self.nusc.get('ego_pose', cam['ego_pose_token'])
        
        # Get calibrated sensor data
        cam_calib = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        lidar_calib = self.nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
        
        # Extract intrinsic and extrinsic matrices
        intrinsic = np.array(cam_calib['camera_intrinsic'])
        
        # Camera to ego transform
        cam2ego_rot = torch.tensor(cam_calib['rotation']).float()
        cam2ego_trans = torch.tensor(cam_calib['translation']).float()
        
        # LiDAR to ego transform
        lidar2ego_rot = torch.tensor(lidar_calib['rotation']).float()
        lidar2ego_trans = torch.tensor(lidar_calib['translation']).float()
        
        def quaternion_to_matrix(quaternion):
            """Convert quaternion to rotation matrix."""
            import numpy as np
            from pyquaternion import Quaternion
            return Quaternion(quaternion).rotation_matrix
            
        def to_transform_matrix(rotation, translation):
            """Create 4x4 transformation matrix."""
            if isinstance(rotation, list):
                rotation = np.array(rotation)
            
            if len(rotation) == 4:  # Quaternion
                rotation_matrix = quaternion_to_matrix(rotation)
            else:  # Already a matrix
                rotation_matrix = rotation
                
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = translation
            return transform
            
        # Create transformation matrices
        cam2ego = to_transform_matrix(cam_calib['rotation'], cam_calib['translation'])
        lidar2ego = to_transform_matrix(lidar_calib['rotation'], lidar_calib['translation'])
        
        # Combine to get LiDAR to camera transform
        ego2cam = np.linalg.inv(cam2ego)
        lidar2cam = ego2cam @ lidar2ego
        
        return {
            'intrinsics': torch.tensor(intrinsic).float(),
            'extrinsics': torch.tensor(lidar2cam).float(),
            'cam2ego': torch.tensor(cam2ego).float(),
            'lidar2ego': torch.tensor(lidar2ego).float(),
            'ego_pose': {
                'translation': ego_pose['translation'],
                'rotation': ego_pose['rotation']
            }
        } 