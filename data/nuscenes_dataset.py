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
from nuscenes.utils.data_classes import Quaternion  # Correct source module
from nuscenes.map_expansion.map_api import NuScenesMap
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
        
        # Debug print map structure
        print("\nDebug - Map Records:")
        for i, map_record in enumerate(self.nusc.map):
            print(f"\nMap Record {i}:")
            for key, value in map_record.items():
                print(f"{key}: {value}")
        
        # Initialize samples before parent class
        self.samples = self._get_samples()
        
        # Create annotation file path
        self.ann_file = os.path.join(dataroot, 'annotations', f'nuscenes_infos_{split}.pkl')
        
        # Create annotations directory if it doesn't exist
        os.makedirs(os.path.dirname(self.ann_file), exist_ok=True)
        
        # Generate annotation file if it doesn't exist
        if not os.path.exists(self.ann_file):
            self.create_annotations(self.ann_file)
        
        # Initialize parent class with correct arguments
        super().__init__(
            data_root=dataroot,
            ann_file=self.ann_file,
            pipeline=pipeline,
            metainfo=self.METAINFO,
            modality=self.modality,
            box_type_3d='LiDAR'
        )

    def _get_samples(self):
        """ Collect samples that have both front camera and top LiDAR data and select a fraction. """
        # Get all samples from the first scene only
        first_scene = self.nusc.scene[0]
        first_scene_samples = []
        
        # Get first sample from the scene
        sample_token = first_scene['first_sample_token']
        while sample_token:
            sample = self.nusc.get('sample', sample_token)
            if self.front_cam in sample['data'] and self.lidar in sample['data']:
                first_scene_samples.append(sample)
            sample_token = sample['next']
        
        print(f"\nProcessing only first scene with {len(first_scene_samples)} samples")
        return first_scene_samples

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

    def load_bev_labels(self, sample):
        """
        Generate BEV segmentation labels from nuScenes HD map data.
        
        This implementation uses the NuScenesMap API to retrieve map polygons
        (such as drivable area, road segments, road blocks, lanes, road dividers, and lane dividers)
        within a certain range of the ego vehicle. The polygons are then rasterized into a 128x128 grid.
        
        Args:
            sample (dict): nuScenes sample dictionary.
        
        Returns:
            bev_label (np.ndarray): BEV segmentation map with shape (H, W), where H, W = 128.
                                    Each pixel is labeled with a class ID:
                                    0: background, 
                                    1: drivable_area, 
                                    2: road_segment, 
                                    3: road_block, 
                                    4: lane, 
                                    5: road_divider, 
                                    6: lane_divider.
        """
        bev_h, bev_w = self.bev_size  # (128, 128)
        resolution = 0.2  # meters per pixel
        bev_label = np.zeros((bev_h, bev_w), dtype=np.uint8)
        
        # Determine the ego vehicle's position (in global coordinates) from the LiDAR ego_pose.
        lidar_token = sample['data'][self.lidar]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'])  # global position of ego vehicle

        # Determine the BEV grid's half-range (in meters)
        half_range = (bev_h * resolution) / 2.0  # e.g., 128*0.2/2 = 12.8 m
        # Define the BEV region in global coordinates (centered at ego_translation)
        # Here, we assume the ego vehicle is at the center of the BEV.
        # You may need to adjust this based on your coordinate conventions.
        x_min = ego_translation[0] - half_range
        x_max = ego_translation[0] + half_range
        y_min = ego_translation[1] - half_range
        y_max = ego_translation[1] + half_range

        # Get the scene's map information.
        # The map name is determined from the scene's location.
        scene = self.nusc.get('scene', sample['scene_token'])
        location = scene['location']  # e.g., "singapore-hollandvillage"
        nusc_map = NuScenesMap(self.dataroot, location)

        # For debugging purposes, we'll assume a simplified procedure:
        # Call a function from the map API to get a binary mask for drivable areas.
        # In a full implementation, you would extract polygons for all your classes:
        # 'drivable_area', 'road_segment', 'road_block', 'lane', 'road_divider', 'lane_divider'.
        # Here, we'll assume that for now, drivable area = 1 and all else = 0.
        try:
            # This function is hypothetical; in practice, you need to implement the extraction.
            # For example, nusc_map.get_map_mask(center, half_range, bev_size, resolution)
            # could return a BEV mask with integer labels.
            bev_mask = nusc_map.get_map_mask(ego_translation, half_range, self.bev_size, resolution=resolution)
            # Suppose the returned mask uses the following convention:
            # 0: background,  1: drivable_area, 2: road_segment, 3: road_block,
            # 4: lane, 5: road_divider, 6: lane_divider.
            bev_label = bev_mask.astype(np.uint8)
            print(f"BEV label shape: {bev_label.shape}")
        except Exception as e:
            print(f"Warning: Failed to load map mask, using placeholder label. Error: {e}")
            # Placeholder: For debugging, fill drivable area as 1.
            bev_label.fill(1)

        return bev_label

    def load_bev_label(self, sample):
        """Generate rear-cropped BEV segmentation labels with ego vehicle at bottom."""
        bev_size = self.bev_size  # Output resolution (128, 128)
        resolution = 0.2  # meters per pixel (50m/128)
        
        # Initialize label array (3 channels: semantic + instance + map_layers)
        bev_label = np.zeros((3, *bev_size), dtype=np.uint8)
        
        # Get ego pose
        lidar_token = sample['data'][self.lidar]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        try:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            
            print(f"\nDebug - Scene Info:")
            print(f"Scene token: {scene['token']}")
            print(f"Log token: {scene['log_token']}")
            print(f"Ego position: ({ego_translation[0]:.2f}, {ego_translation[1]:.2f})")
            
            # Find corresponding map for this scene
            map_record = None
            for record in self.nusc.map:
                if scene['log_token'] in record['log_tokens']:
                    map_record = record
                    break
            
            if map_record:
                print(f"\nDebug - Map Info:")
                print(f"Map token: {map_record['token']}")
                print(f"Map filename: {map_record['filename']}")
                
                # Load map image
                map_path = os.path.join(self.dataroot, map_record['filename'])
                print(f"Full map path: {map_path}")
                print(f"Map file exists: {os.path.exists(map_path)}")
                
                if os.path.exists(map_path):
                    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
                    if map_img is not None:
                        try:
                            # Open debug log file
                            with open('map_debug.log', 'w') as debug_log:
                                def log_debug(msg):
                                    print(msg, flush=True)
                                    debug_log.write(msg + '\n')
                                    debug_log.flush()
                                
                                log_debug(f"Map image loaded successfully. Shape: {map_img.shape}")
                                
                                # Get map resolution from mask object
                                map_resolution = 0.1  # meters/pixel (default for semantic prior maps)
                                log_debug(f"Map resolution: {map_resolution} meters/pixel")
                                
                                # Convert ego pose to map coordinates
                                # The map's origin is at the top-left corner, and y increases downward
                                map_x = int(ego_translation[0] / map_resolution)
                                map_y = int(ego_translation[1] / map_resolution)
                                log_debug(f"Raw map coordinates: ({map_x}, {map_y})")
                                
                                # Transform to image coordinates (origin at top-left)
                                # For x: add offset to center
                                img_x = map_x + map_img.shape[1] // 2
                                
                                # For y: invert y-axis and add offset
                                # Since map coordinates increase upward but image coordinates increase downward
                                img_y = map_img.shape[0] // 2 + map_y
                                
                                log_debug(f"Image dimensions: {map_img.shape}")
                                log_debug(f"Center offset: ({map_img.shape[1] // 2}, {map_img.shape[0] // 2})")
                                log_debug(f"Final image coordinates: ({img_x}, {img_y})")
                                
                                # Calculate map patch size in pixels
                                patch_size = int(50 / map_resolution)  # 50m patch
                                half_patch = patch_size // 2
                                log_debug(f"Patch size in pixels: {patch_size}")
                                
                                # Calculate patch boundaries with boundary checking
                                x_min = max(0, min(img_x - half_patch, map_img.shape[1]-1))
                                x_max = max(0, min(img_x + half_patch, map_img.shape[1]-1))
                                y_min = max(0, min(img_y - half_patch, map_img.shape[0]-1))
                                y_max = max(0, min(img_y + half_patch, map_img.shape[0]-1))
                                
                                log_debug(f"Clipped patch bounds: x[{x_min}, {x_max}], y[{y_min}, {y_max}]")
                                
                                # Ensure we have a valid patch size (at least 100x100 pixels)
                                min_patch_size = 100
                                if x_max - x_min >= min_patch_size and y_max - y_min >= min_patch_size:
                                    log_debug("Patch bounds are valid, extracting patch...")
                                    # Extract map patch
                                    map_patch = map_img[y_min:y_max, x_min:x_max]
                                    log_debug(f"Extracted patch shape: {map_patch.shape}")
                                    
                                    # Save debug visualization of full map
                                    debug_full = cv2.cvtColor(map_img.copy(), cv2.COLOR_GRAY2BGR)
                                    cv2.circle(debug_full, (img_x, img_y), 20, (0, 0, 255), -1)  # Red dot for ego position
                                    cv2.rectangle(debug_full, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box for patch
                                    cv2.imwrite('map_debug_full.png', debug_full)
                                    log_debug("Saved full map debug visualization")
                                    
                                    # Continue with the rest of the processing...
                                    # Rotate patch to align with ego vehicle orientation
                                    rot_mat = ego_rotation.rotation_matrix[:2, :2]
                                    yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
                                    center = (map_patch.shape[1] // 2, map_patch.shape[0] // 2)
                                    rot_mat = cv2.getRotationMatrix2D(center, np.degrees(yaw), 1.0)
                                    map_patch = cv2.warpAffine(map_patch, rot_mat, map_patch.shape[::-1])
                                    
                                    # Save debug visualization of extracted patch
                                    cv2.imwrite('map_debug_patch.png', map_patch)
                                    log_debug("Saved patch debug visualization")
                                    
                                    # Resize to match BEV resolution
                                    map_patch = cv2.resize(map_patch, bev_size[::-1])
                                    
                                    # Process map layers
                                    thresholds = [50, 100, 150, 200, 250]
                                    layer_priorities = [1, 2, 4, 5, 6]
                                    combined_mask = np.zeros(bev_size, dtype=np.uint8)
                                    
                                    for threshold, priority in zip(thresholds, layer_priorities):
                                        mask = map_patch > threshold
                                        combined_mask[mask] = priority
                                    
                                    # Add to map layer channel
                                    bev_label[2] = combined_mask
                                    
                                    # Add drivable area to semantic channel
                                    bev_label[0][combined_mask == 1] = 11  # Class 11 for drivable area
                                    
                                    # Save final visualization
                                    debug_img = np.zeros((*bev_size, 3), dtype=np.uint8)
                                    colors = {
                                        1: (176, 224, 230),  # Drivable area - Light blue
                                        2: (152, 251, 152),  # Road segment - Pale green
                                        4: (255, 215, 0),    # Lane - Gold
                                        5: (255, 0, 0),      # Road divider - Red
                                        6: (255, 255, 0)     # Lane divider - Yellow
                                    }
                                    for priority, color in colors.items():
                                        debug_img[combined_mask == priority] = color
                                    cv2.imwrite('map_debug.png', debug_img)
                                    log_debug("Saved final visualization")
                                else:
                                    log_debug("Invalid patch bounds after clipping!")
                        except Exception as e:
                            print(f"Error during map processing: {str(e)}", flush=True)
                            import traceback
                            traceback.print_exc()
                    else:
                        print("Failed to load map image", flush=True)
                else:
                    print("Map file not found")
            else:
                print("No matching map record found for this scene")
        
        except Exception as e:
            print(f"\nMap error: {str(e)}")
            import traceback
            traceback.print_exc()

        # Draw ego vehicle at bottom center (in both semantic and instance channels)
        ego_width = int(2.0 / resolution)  # Typical car width ~2m
        ego_length = int(4.5 / resolution)  # Typical car length ~4.5m
        ego_y = bev_size[0] - ego_length - 5  # 5 pixels from bottom
        ego_x = bev_size[1] // 2 - ego_width // 2
        
        # Draw ego vehicle in semantic channel
        bev_label[0, ego_y:ego_y+ego_length, ego_x:ego_x+ego_width] = 1  # Class 1 for car
        
        # Draw ego vehicle in instance channel with unique ID
        bev_label[1, ego_y:ego_y+ego_length, ego_x:ego_x+ego_width] = 1  # Instance ID 1 for ego vehicle

        # Process nearby objects (only in rear view)
        instance_id = 2  # Start from 2 since 1 is ego vehicle
        for ann_token in sample['anns']:
            try:
                ann = self.nusc.get('sample_annotation', ann_token)
                box = self.nusc.get_box(ann_token)
                
                # Skip distant objects (>25m) or objects outside rear view angle
                rel_pos = box.center[:2] - ego_translation[:2]
                distance = np.linalg.norm(rel_pos)
                angle = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
                if distance > 25 or abs(angle) > 90:  # Only include ±90° rear view
                    continue

                # Transform to ego frame
                box.translate(-ego_translation)
                box.rotate(ego_rotation.inverse)

                # Project to BEV
                corners = box.bottom_corners()[:2].T
                grid_coords = ((corners / resolution) + np.array([bev_size[1]//2, bev_size[0]-1])).astype(int)
                grid_coords = np.clip(grid_coords, 0, np.array(bev_size)-1)

                if len(grid_coords) >= 3:
                    class_name = self.NameMapping.get(ann['category_name'])
                    if class_name:
                        # Add to semantic channel
                        class_id = self.METAINFO['classes'].index(class_name) + 1
                        cv2.fillPoly(bev_label[0], [grid_coords], color=class_id)
                        
                        # Add to instance channel
                        cv2.fillPoly(bev_label[1], [grid_coords], color=instance_id)
                        instance_id += 1

            except Exception as e:
                continue

        return bev_label

    def get_local_map(self, nmap, ego_translation, stretch, layer_names, line_names):
        """Get local map centered around ego vehicle."""
        # Define patch coordinates
        box_coords = (
            ego_translation[0] - stretch,  # min x
            ego_translation[1] - stretch,  # min y
            ego_translation[0] + stretch,  # max x
            ego_translation[1] + stretch,  # max y
        )

        polys = {}

        # Get polygons
        records_in_patch = nmap.get_records_in_patch(box_coords,
                                                    layer_names=layer_names,
                                                    mode='intersect')
        for layer_name in layer_names:
            polys[layer_name] = []
            for token in records_in_patch[layer_name]:
                poly_record = nmap.get(layer_name, token)
                if layer_name == 'drivable_area':
                    polygon_tokens = poly_record['polygon_tokens']
                else:
                    polygon_tokens = [poly_record['polygon_token']]

                for polygon_token in polygon_tokens:
                    polygon = nmap.extract_polygon(polygon_token)
                    if not polygon.is_empty:  # Skip empty polygons
                        polys[layer_name].append(np.array(polygon.exterior.xy).T)

        # Get lines
        for layer_name in line_names:
            polys[layer_name] = []
            for record in getattr(nmap, layer_name):
                token = record['token']
                line = nmap.extract_line(record['line_token'])
                if not line.is_empty:  # Skip empty lines
                    xs, ys = line.xy
                    polys[layer_name].append(np.array([xs, ys]).T)

        return polys

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
        Filter LiDAR points to match Velodyne Puck Hi-Res specifications:
        - 16 channels
        - -10° to 10° vertical FOV
        - ~300k points/sec (vs 1.4M in nuScenes)
        
        Args:
            points (np.ndarray): (N, 4) array of points (x, y, z, intensity)
        
        Returns:
            np.ndarray: Filtered points matching our sensor specs
        """
        # Calculate distances in x-y plane
        distances = np.linalg.norm(points[:, :2], axis=1)
        
        # Calculate vertical angles in degrees
        vertical_angles = np.degrees(np.arctan2(points[:, 2], distances))
        
        # Filter points within FOV range (-10° to 10°)
        valid_indices = (vertical_angles >= -10) & (vertical_angles <= 10)
        filtered_points = points[valid_indices]
        
        # Reduce to 16 channels (from 32)
        filtered_points = self.reduce_lidar_channels(filtered_points, num_channels=16)
        
        # Reduce point density to match Velodyne Puck (~300k points/sec)
        target_points = 300000 // 10  # Assuming 10Hz scan rate
        if len(filtered_points) > target_points:
            indices = np.random.choice(len(filtered_points), target_points, replace=False)
            filtered_points = filtered_points[indices]
        
        return filtered_points

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
        
        # Initialize instance counters
        instance_counts = {class_name: 0 for class_name in self.METAINFO['classes']}
        
        # Process only samples from the first scene
        for sample in self.samples:
            try:
                # Get sample data paths
                lidar_token = sample['data'][self.lidar]
                cam_token = sample['data'][self.front_cam]
                
                lidar_data = self.nusc.get('sample_data', lidar_token)
                cam_data = self.nusc.get('sample_data', cam_token)
                
                # Count instances in this sample
                for ann_token in sample['anns']:
                    ann = self.nusc.get('sample_annotation', ann_token)
                    category_name = ann['category_name']
                    mapped_name = self.NameMapping.get(category_name)
                    if mapped_name in self.METAINFO['classes']:
                        instance_counts[mapped_name] += 1
                
                # Basic sample info
                info = {
                    'sample_idx': sample['token'],
                    'lidar_points': {
                        'lidar_path': lidar_data['filename'],
                        'num_pts_feats': 4,
                        'timestamp': lidar_data['timestamp']
                    },
                    'images': {
                        'CAM_FRONT': {
                            'img_path': cam_data['filename'],
                            'height': 900,
                            'width': 1600,
                            'cam2img': self.get_camera_intrinsics(cam_data)['intrinsic_matrix'].tolist(),
                            'timestamp': cam_data['timestamp']
                        }
                    },
                    'lidar2ego': self.get_lidar_pose(sample),
                    'bev_seg_label': self.load_bev_label(sample).tolist(),
                    'instances': self._get_instances(sample),
                    'scene_token': sample['scene_token'],
                    'timestamp': sample['timestamp']
                }
                annotations_dict['data_list'].append(info)
                
            except Exception as e:
                print(f"Warning: Error processing sample {sample['token']}: {str(e)}")
                continue
        
        if not annotations_dict['data_list']:
            raise ValueError("No valid annotations could be created!")
        
        # Print instance counts
        print("\nInstance counts per category:")
        for class_name, count in instance_counts.items():
            print(f"{class_name}: {count}")
        
        # Save as dictionary
        os.makedirs(os.path.dirname(ann_file), exist_ok=True)
        with open(ann_file, 'wb') as f:
            pickle.dump(annotations_dict, f)
        print(f"\nSaved {len(annotations_dict['data_list'])} annotations to {ann_file}")

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
