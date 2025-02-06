import os
import numpy as np
import cv2
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.data_classes import LidarPointCloud
from mmdet3d.datasets import Det3DDataset
import pickle
from datetime import datetime

class NuScenesBEVDataset(Det3DDataset):
    # NameMapping = {
    #     "movable_object.barrier": "barrier",
    #     "vehicle.bicycle": "bicycle",
    #     "vehicle.bus.bendy": "bus",
    #     "vehicle.bus.rigid": "bus",
    #     "vehicle.car": "car",
    #     "vehicle.construction": "construction_vehicle",
    #     "vehicle.motorcycle": "motorcycle",
    #     "human.pedestrian.adult": "pedestrian",
    #     "human.pedestrian.child": "pedestrian",
    #     "human.pedestrian.construction_worker": "pedestrian",
    #     "human.pedestrian.police_officer": "pedestrian",
    #     "movable_object.trafficcone": "traffic_cone",
    #     "vehicle.trailer": "trailer",
    #     "vehicle.truck": "truck",
    # }
    METAINFO = {
        'classes': (
            "car", "truck", "trailer", "bus", "construction_vehicle",
            "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"
        )
    }
 # ✅ Converts nuScenes categories to BEV segmentation labels
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }

    # ✅ Default attributes for objects
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }

    # ✅ Attribute mapping (if using attributes in segmentation)
    AttrMapping = {
        "cycle.with_rider": 0,
        "cycle.without_rider": 1,
        "pedestrian.moving": 2,
        "pedestrian.standing": 3,
        "pedestrian.sitting_lying_down": 4,
        "vehicle.moving": 5,
        "vehicle.parked": 6,
        "vehicle.stopped": 7,
    }
    def __init__(self, dataroot, split='train',
                 front_cam='CAM_FRONT', lidar='LIDAR_TOP',
                 transform=None, lidar_fov=(-10, 10), sample_ratio=1.0,
                 bev_size=(128, 128), pipeline=None, modality=None):
        """
        Args:
            dataroot (str): Path to the nuScenes dataset.
            split (str): Data split to use ('train' or 'val').
            front_cam (str): Camera sensor key.
            lidar (str): LiDAR sensor key.
            transform (callable, optional): Image and LiDAR augmentation.
            lidar_fov (tuple): (lower_angle, upper_angle) in degrees to filter LiDAR points.
            sample_ratio (float): Fraction of dataset to use (e.g., 0.1 for 10%).
            bev_size (tuple): Output BEV segmentation resolution.
            pipeline (list): MMDetection3D pipeline for preprocessing.
            modality (dict): Defines enabled sensor modalities.
        """
        self.dataroot = dataroot
        self.nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
        self.split = split
        self.front_cam = front_cam
        self.lidar = lidar
        self.transform = transform
        self.lidar_fov = lidar_fov
        self.sample_ratio = sample_ratio
        self.bev_size = bev_size
        self.modality = modality if modality else {'use_lidar': True, 'use_camera': True}
        
        # Initialize samples before parent class
        self.samples = self._get_samples()
        
        # Create annotation file path
        ann_file = os.path.join(dataroot, 'annotations', f'nuscenes_infos_{split}.pkl')
        
        # Create annotations directory if it doesn't exist
        os.makedirs(os.path.dirname(ann_file), exist_ok=True)
        
        # Generate annotation file if it doesn't exist
        if not os.path.exists(ann_file):
            self.create_annotations(ann_file)
        
        # Initialize parent class with correct arguments
        super().__init__(
            data_root=dataroot,
            ann_file=ann_file,
            pipeline=pipeline,
            metainfo=self.METAINFO,
            modality=self.modality,
            box_type_3d='LiDAR'
        )

    def _get_samples(self):
        """ Collect samples that have both front camera and top LiDAR data and select a fraction. """
        all_samples = [sample for sample in self.nusc.sample
                       if self.front_cam in sample['data'] and self.lidar in sample['data']]
        total_samples = len(all_samples)
        num_selected = int(total_samples * self.sample_ratio)
        return all_samples[:num_selected]

    def get_data_info(self, index):
        """Get data info for BEV segmentation."""
        sample = self.samples[index]
        
        # Load LiDAR points
        lidar_token = sample['data'][self.lidar]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.dataroot, lidar_data['filename'])
        points = LidarPointCloud.from_file(lidar_path).points.T
        
        # Filter points by FOV
        filtered_points = self.filter_points_by_fov(points)
        
        # Generate BEV segmentation labels
        bev_label = self.load_bev_label(sample)
        
        return {
            'lidar_points': filtered_points,
            'bev_label': bev_label,
            'sample_idx': sample['token']
        }

    def load_bev_label(self, sample):
        """
        Generate BEV segmentation labels from nuScenes 3D bounding boxes.
        """
        # Initialize with 2 channels (class and attribute)
        bev_label = np.zeros((2, *self.bev_size), dtype=np.uint8)
        resolution = 0.2  # meters per pixel
        ego_offset = self.bev_size[0] // 2  # 64 for 128x128 BEV

        # Get LiDAR data for ego pose
        lidar_token = sample['data'][self.lidar]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'])

        for ann_token in sample['anns']:
            try:
                ann = self.nusc.get('sample_annotation', ann_token)
                box = self.nusc.get_box(ann_token)

                # Transform box to ego coordinates
                box_copy = Box(
                    center=box.center.copy(),
                    size=box.wlh.copy(),
                    orientation=box.orientation,  # Quaternion doesn't need copy
                    name=box.name,
                    token=box.token
                )
                box_copy.translate(-ego_translation)

                # Get 2D polygon coordinates
                corners = box_copy.bottom_corners()[:2].T  # Get bottom corners (x,y)
                grid_coords = ((corners / resolution) + ego_offset).astype(int)
                grid_coords = np.clip(grid_coords, 0, self.bev_size[0]-1)

                # Skip invalid polygons
                if len(grid_coords) < 3:
                    continue

                # Get class and attribute IDs
                class_name = self.NameMapping.get(ann['category_name'])
                if not class_name:
                    continue

                class_id = self.METAINFO['classes'].index(class_name) + 1
                attr_id = self.AttrMapping.get(
                    self.DefaultAttribute.get(class_name, ""), 0
                )

                # Draw on both channels
                cv2.fillPoly(bev_label[0], [grid_coords], color=class_id)
                cv2.fillPoly(bev_label[1], [grid_coords], color=attr_id)

            except Exception as e:
                print(f"Skipping invalid annotation {ann_token}: {str(e)}")
                continue

        return bev_label

    def get_class_id(self, category_name):
        """
        Convert nuScenes category name to numeric class ID.
        """
        class_name = self.NameMapping.get(category_name, None)
        if class_name is None:
            return 0  # Default to background
        try:
            return self.METAINFO['classes'].index(class_name) + 1  # +1 to reserve 0 for background
        except ValueError:
            return 0  # Default to background if class not found

    def __len__(self):
        return len(self.samples)

    def reduce_lidar_channels(self, points, num_channels=16):
        """
        Reduces LiDAR points to simulate 16-channel sensor.
        Args:
            points: (N, 4) array of points (x, y, z, intensity)
            num_channels: target number of channels
        Returns:
            Reduced point cloud
        """
        # Calculate vertical angles
        distances = np.linalg.norm(points[:, :2], axis=1)
        angles = np.degrees(np.arctan2(points[:, 2], distances))
        
        # Create angle bins matching your sensor's resolution
        angle_range = self.lidar_fov[1] - self.lidar_fov[0]  # 20 degrees
        bin_size = angle_range / num_channels
        angle_bins = np.arange(self.lidar_fov[0], self.lidar_fov[1] + bin_size, bin_size)
        
        # Assign points to bins and randomly sample within each bin
        kept_points = []
        for i in range(len(angle_bins) - 1):
            bin_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
            bin_points = points[bin_mask]
            if len(bin_points) > 0:
                # Randomly sample points in this channel
                sample_size = min(len(bin_points), 2000)  # Adjust based on your sensor's point density
                indices = np.random.choice(len(bin_points), sample_size, replace=False)
                kept_points.append(bin_points[indices])
        
        return np.vstack(kept_points) if kept_points else np.zeros((0, points.shape[1]))

    def filter_points_by_fov(self, points):
        """
        Filter LiDAR points by vertical field of view.
        
        Args:
            points (np.ndarray): (N, 4) array of points (x, y, z, intensity)
        
        Returns:
            np.ndarray: Filtered points within the specified FOV
        """
        # Calculate distances in x-y plane
        distances = np.linalg.norm(points[:, :2], axis=1)
        
        # Calculate vertical angles in degrees
        vertical_angles = np.degrees(np.arctan2(points[:, 2], distances))
        
        # Filter points within FOV range
        lower_fov, upper_fov = self.lidar_fov
        valid_indices = (vertical_angles >= lower_fov) & (vertical_angles <= upper_fov)
        
        return points[valid_indices]

    def get_camera_intrinsics(self, cam_data):
        """
        Get camera calibration parameters.
        Args:
            cam_data: nuScenes camera data dictionary
        Returns:
            dict: Camera intrinsics including focal length, principal point
        """
        # Get calibration data
        calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        # Extract intrinsic matrix
        intrinsic = np.array(calib['camera_intrinsic'])
        
        return {
            'focal_length': (intrinsic[0, 0], intrinsic[1, 1]),  # fx, fy
            'principal_point': (intrinsic[0, 2], intrinsic[1, 2]),  # cx, cy
            'intrinsic_matrix': intrinsic
        }

    def get_lidar_pose(self, sample):
        """
        Get LiDAR pose in ego vehicle frame.
        Args:
            sample: nuScenes sample dictionary
        Returns:
            dict: LiDAR pose information
        """
        # Get LiDAR data
        lidar_token = sample['data'][self.lidar]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        
        # Get calibration data (LiDAR to ego vehicle transform)
        calib = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        
        # Get ego pose from sample data instead of sample
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        return {
            'translation': calib['translation'],  # LiDAR position relative to ego vehicle
            'rotation': calib['rotation'],        # LiDAR orientation as quaternion
            'ego_pose': {                         # Ego vehicle pose in global frame
                'translation': ego_pose['translation'],
                'rotation': ego_pose['rotation']
            }
        }

    def create_annotations(self, ann_file):
        """Create annotation file in MMDetection3D compatible format."""
        print(f"Creating annotation file: {ann_file}")
        
        # Create dictionary structure
        annotations_dict = {
            'metainfo': {
                'dataset': 'nuscenes',
                'version': 'v1.0-mini',
                'split': self.split,
                'classes': self.METAINFO['classes']
            },
            'data_list': []  # List to store individual annotations
        }
        
        for sample in self.samples:
            try:
                # Get sample data paths
                lidar_token = sample['data'][self.lidar]
                cam_token = sample['data'][self.front_cam]
                
                lidar_data = self.nusc.get('sample_data', lidar_token)
                cam_data = self.nusc.get('sample_data', cam_token)
                
                # Basic sample info
                info = {
                    'sample_idx': sample['token'],
                    # LiDAR info
                    'lidar_points': {
                        'lidar_path': lidar_data['filename'],
                        'num_pts_feats': 4,  # x, y, z, intensity
                        'timestamp': lidar_data['timestamp']
                    },
                    # Camera info (required by MMDet3D)
                    'images': {
                        'CAM_FRONT': {
                            'img_path': cam_data['filename'],
                            'height': 900,  # NuScenes image height
                            'width': 1600,  # NuScenes image width
                            'cam2img': self.get_camera_intrinsics(cam_data)['intrinsic_matrix'].tolist(),
                            'timestamp': cam_data['timestamp']
                        }
                    },
                    # Transforms
                    'lidar2ego': self.get_lidar_pose(sample),
                    # BEV Segmentation label
                    'bev_seg_label': self.load_bev_label(sample).tolist(),
                    # Required by MMDet3D
                    'instances': [],  # Empty for BEV segmentation
                    'scene_token': sample['scene_token'],
                    'timestamp': sample['timestamp']
                }
                annotations_dict['data_list'].append(info)
                
            except Exception as e:
                print(f"Warning: Error processing sample {sample['token']}: {str(e)}")
                continue
        
        if not annotations_dict['data_list']:
            raise ValueError("No valid annotations could be created!")
        
        # Save as dictionary
        os.makedirs(os.path.dirname(ann_file), exist_ok=True)
        with open(ann_file, 'wb') as f:
            pickle.dump(annotations_dict, f)
        print(f"Saved {len(annotations_dict['data_list'])} annotations to {ann_file}")

    def _get_instances(self, sample):
        """
        Get instance-level annotations for a sample.
        Args:
            sample: nuScenes sample dictionary
        Returns:
            list: List of instance annotations
        """
        instances = []
        for ann_token in sample['anns']:
            try:
                ann = self.nusc.get('sample_annotation', ann_token)
                box = self.nusc.get_box(ann_token)
                
                # Get category name and map to dataset classes
                category_name = ann['category_name']
                mapped_name = self.NameMapping.get(category_name)
                
                if mapped_name is None:
                    continue  # Skip if category is not in our mapping
                    
                instance = {
                    'bbox_3d': [
                        *box.center,  # x, y, z
                        *box.wlh,     # width, length, height
                        box.orientation.yaw_pitch_roll[0]  # yaw angle
                    ],
                    'bbox_label_3d': self.METAINFO['classes'].index(mapped_name),
                    'bbox': self._get_2d_bbox(box),  # Optional 2D bbox
                    'attr_label': self._get_attribute_label(mapped_name),
                    'velocity': box.velocity[:2] if box.velocity is not None else [0, 0],
                    'num_lidar_pts': ann['num_lidar_pts'],
                    'num_radar_pts': ann['num_radar_pts'],
                    'instance_token': ann['instance_token']
                }
                instances.append(instance)
                
            except Exception as e:
                print(f"Warning: Error processing annotation {ann_token}: {str(e)}")
                continue
            
        return instances

    def _get_2d_bbox(self, box):
        """Get 2D bounding box (placeholder - implement if needed)"""
        return [0, 0, 0, 0]  # [x1, y1, x2, y2]

    def _get_attribute_label(self, class_name):
        """Get attribute label for a class"""
        default_attr = self.DefaultAttribute.get(class_name, "")
        return self.AttrMapping.get(default_attr, 0)
