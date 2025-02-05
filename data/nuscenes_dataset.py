import os
import numpy as np
import cv2
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.data_classes import LidarPointCloud
from mmdet3d.datasets import Det3DDataset


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
        self.pipeline = pipeline
        self.modality = modality if modality else {'use_lidar': True, 'use_camera': True}
        
        self.samples = self._get_samples()
        super().__init__(data_root=dataroot, pipeline=pipeline, ann_file=None, modality=self.modality, metainfo=self.METAINFO)


    def _get_samples(self):
        """ Collect samples that have both front camera and top LiDAR data and select a fraction. """
        all_samples = [sample for sample in self.nusc.sample
                       if self.front_cam in sample['data'] and self.lidar in sample['data']]
        total_samples = len(all_samples)
        num_selected = int(total_samples * self.sample_ratio)
        return all_samples[:num_selected]

    def get_data_info(self, index):
        """ Get image, LiDAR, and BEV segmentation labels for the given index. """
        sample = self.samples[index]
        
        # Load front camera image
        cam_token = sample['data'][self.front_cam]
        cam_data = self.nusc.get('sample_data', cam_token)
        cam_filepath = os.path.join(self.dataroot, cam_data['filename'])
        image = cv2.imread(cam_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
        
        # Load LiDAR point cloud
        lidar_token = sample['data'][self.lidar]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_filepath = os.path.join(self.dataroot, lidar_data['filename'])
        lidar_pc = LidarPointCloud.from_file(lidar_filepath)
        points = lidar_pc.points.T
        
        # Filter LiDAR points by vertical FOV (-10° to 10°)
        lower_fov, upper_fov = self.lidar_fov
        distances = np.linalg.norm(points[:, :2], axis=1)
        vertical_angles = np.degrees(np.arctan2(points[:, 2], distances))
        valid_indices = (vertical_angles >= lower_fov) & (vertical_angles <= upper_fov)
        filtered_points = points[valid_indices]
        
        # Load BEV segmentation labels from nuScenes annotations
        bev_label = self.load_bev_label(sample)
        
        # Apply Depth-Augmented Feature Warping
        #depth_warped_image = self.depth_augmented_warping(image, filtered_points)
        
        return {
            'image': image,
            'lidar': filtered_points,
            'bev_label': bev_label
        }


    def load_bev_label(self, sample):
        """
        Generate BEV segmentation labels from nuScenes 3D bounding boxes.
        Args:
            sample (dict): nuScenes sample dictionary.
        Returns:
            bev_label (np.ndarray): BEV segmentation map (128x128).
        """
        bev_label = np.zeros(self.bev_size, dtype=np.uint8)
        bev_size = self.bev_size[0]  # 128
        resolution = 0.2  # 0.2 meters per pixel
        ego_offset = bev_size // 2  # Center ego-vehicle in BEV

        # Get ego pose
        ego_pose = self.nusc.get('ego_pose', sample['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'])

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            box = self.nusc.get_box(ann_token)

            # Create a copy of the box before translating
            box_copy = Box(box.center, box.wlh, box.orientation, name=box.name, token=box.token)
            box_copy.translate(-ego_translation)  # Convert to ego frame

            # Get the 2D corners (x, y) in ego frame
            corners = box_copy.corners()[:2, :]
            grid_x = np.clip(((corners[0, :] / resolution) + ego_offset).astype(int), 0, bev_size - 1)
            grid_y = np.clip(((corners[1, :] / resolution) + ego_offset).astype(int), 0, bev_size - 1)

            # Ensure polygon validity
            if len(grid_x) < 3 or len(grid_y) < 3:
                continue  # Skip invalid polygons

            # Assign class ID based on nuScenes category
            class_id = self.get_class_id(ann['category_name'])

            # Separate class ID and attribute ID into separate channels
            class_mask = np.zeros_like(bev_label)
            attr_mask = np.zeros_like(bev_label)

            cv2.fillPoly(class_mask, [np.array(list(zip(grid_x, grid_y)))], color=class_id)
            
            # Assign attribute ID separately
            class_name = self.NameMapping.get(ann['category_name'], None)
            default_attr = self.DefaultAttribute.get(class_name, "")
            attr_id = self.AttrMapping.get(default_attr, 0)
            cv2.fillPoly(attr_mask, [np.array(list(zip(grid_x, grid_y)))], color=attr_id)

            # Merge into final BEV label (multi-channel)
            bev_label = np.stack([class_mask, attr_mask], axis=0)  # (2, 128, 128)

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

    # def depth_augmented_warping(self, image, lidar_points):
    #     """ Apply Depth-Augmented Feature Warping using LiDAR depth information. """
    #     # TODO: Implement depth-aware warping logic to align image features with BEV space
    #     return image

    def __len__(self):
        return len(self.samples)
