import os
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

class NuScenesFilteredDataset(Dataset):
    """Filters nuScenes samples and provides raw sensor data"""
    def __init__(self, dataroot, version='v1.0-trainval', split='train', 
                 lidar_sensor='LIDAR_TOP', cam_sensor='CAM_FRONT'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot
        self.split = split
        self.lidar_sensor = lidar_sensor
        self.cam_sensor = cam_sensor
        
        # Filter samples with both sensors
        self.samples = self._filter_samples()
        self.class_map = self._create_class_map()

    def _filter_samples(self):
        scenes = create_splits_scenes()[self.split]
        samples = []
        
        for scene in self.nusc.scene:
            if scene['name'] not in scenes:
                continue
                
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                if self._has_valid_sensors(sample):
                    samples.append(sample_token)
                sample_token = sample['next']
                
        return samples

    def _has_valid_sensors(self, sample):
        lidar_data = self.nusc.get('sample_data', sample['data'][self.lidar_sensor])
        cam_data = self.nusc.get('sample_data', sample['data'][self.cam_sensor])
        return lidar_data['is_key_frame'] and cam_data['is_key_frame']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # Load camera data
        cam_data = self.nusc.get('sample_data', sample['data'][self.cam_sensor])
        cam_path = os.path.join(self.dataroot, cam_data['filename'])
        
        # Load lidar data
        lidar_data = self.nusc.get('sample_data', sample['data'][self.lidar_sensor])
        lidar_path = os.path.join(self.dataroot, lidar_data['filename'])
        
        # Load calibration data
        calib = self._load_calibration(sample)
        
        return {
            'image': cam_path,
            'lidar': lidar_path,
            'calib': calib,
            'timestamp': sample['timestamp'],
            'sample_token': sample_token
        }

    def _load_calibration(self, sample):
        cam_data = self.nusc.get('sample_data', sample['data'][self.cam_sensor])
        lidar_data = self.nusc.get('sample_data', sample['data'][self.lidar_sensor])
        
        # Get sensor transforms
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        lidar_calib = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        
        return {
            'cam_intrinsic': np.array(cam_calib['camera_intrinsic']),
            'cam2ego': np.array(cam_calib['rotation'] + cam_calib['translation']),
            'lidar2ego': np.array(lidar_calib['rotation'] + lidar_calib['translation']),
            'ego_pose': self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        }

class NuScenesBEVLabelDataset(Dataset):
    """Generates BEV segmentation labels from filtered samples"""
    def __init__(self, filtered_dataset, grid_size=128, resolution=0.2):
        self.filtered_dataset = filtered_dataset
        self.grid_size = grid_size
        self.resolution = resolution
        self.nusc = filtered_dataset.nusc
        self.class_map = filtered_dataset.class_map
        
    def __len__(self):
        return len(self.filtered_dataset)

    def __getitem__(self, idx):
        sample = self.filtered_dataset[idx]
        sample_token = sample['sample_token']
        
        # Get ego pose at sample time
        ego_pose = self.nusc.get('ego_pose', sample['calib']['ego_pose']['token'])
        
        # Generate BEV labels
        bev_label = self._generate_bev_label(sample_token, ego_pose)
        
        return {
            'bev_label': torch.from_numpy(bev_label).long(),
            'sample_token': sample_token
        }

    def _generate_bev_label(self, sample_token, ego_pose):
        label = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        sample = self.nusc.get('sample', sample_token)
        
        # Get HD map layers around ego vehicle
        ego_translation = ego_pose['translation']
        ego_rotation = Quaternion(ego_pose['rotation'])
        
        # Convert to BEV grid coordinates
        map_radius = self.grid_size * self.resolution / 2
        map_patch = self.nusc.get_map_mask(
            ego_translation[:2], 
            (map_radius, map_radius), 
            ['drivable_area', 'road_segment', 'lane', 'road_divider', 'lane_divider'],
            ego_rotation.yaw_pitch_roll[0]
        )
        
        # Render map elements to BEV grid
        for layer_name, geometries in map_patch.items():
            class_id = self.class_map[layer_name]
            for geom in geometries:
                self._render_geometry(geom, label, class_id)
                
        return label

    def _render_geometry(self, geom, label, class_id):
        # Convert geometry to grid coordinates
        coords = np.array(geom.exterior.coords)
        grid_coords = self._world_to_grid(coords)
        
        # Rasterize based on geometry type
        if geom.geom_type == 'Polygon':
            self._fill_polygon(grid_coords, label, class_id)
        elif geom.geom_type == 'LineString':
            self._draw_line(grid_coords, label, class_id)

    def _world_to_grid(self, coords):
        center = self.grid_size // 2
        return ((coords / self.resolution) + center).astype(int)

    def _fill_polygon(self, coords, label, class_id):
        from matplotlib.path import Path
        grid_x, grid_y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
        
        path = Path(coords)
        mask = path.contains_points(grid_points).reshape(self.grid_size, self.grid_size)
        label[mask] = class_id

    def _draw_line(self, coords, label, class_id, thickness=1):
        from skimage.draw import line_aa
        for i in range(len(coords)-1):
            x0, y0 = coords[i]
            x1, y1 = coords[i+1]
            rr, cc, _ = line_aa(y0, x0, y1, x1)
            valid = (rr >= 0) & (rr < self.grid_size) & (cc >= 0) & (cc < self.grid_size)
            label[rr[valid], cc[valid]] = class_id

    @staticmethod
    def _create_class_map():
        return {
            'drivable_area': 0,
            'road_segment': 1,
            'road_block': 2,
            'lane': 3,
            'road_divider': 4,
            'lane_divider': 5
        } 