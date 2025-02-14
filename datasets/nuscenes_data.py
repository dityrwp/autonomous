import os
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
import logging
from shapely.geometry import Polygon, LineString
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def patched_load_table(self, table_name):
    filepath = os.path.join(self.dataroot, self.version, table_name + '.json')
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

NuScenes.__load_table__ = patched_load_table

class NuScenesFilteredDataset(Dataset):
    """Filters nuScenes samples to only include frames with both front camera and LiDAR."""
    
    def __init__(self, dataroot, version='v1.0-trainval', split='train', 
                 lidar_sensor='LIDAR_TOP', cam_sensor='CAM_FRONT'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot
        self.split = split
        self.lidar_sensor = lidar_sensor
        self.cam_sensor = cam_sensor

        # Filter samples with both sensors
        logging.info(f"Filtering {version} samples for {split} split...")
        self.samples = self._filter_samples()
        logging.info(f"Found {len(self.samples)} valid samples")
        
        # Shared class map
        self.class_map = {
            'drivable_area': 0,
            'road_segment': 1,
            'road_block': 2,
            'lane': 3,
            'road_divider': 4,
            'lane_divider': 5
        }

    def _filter_samples(self):
        """Finds samples that contain both front camera and LiDAR frames."""
        scenes = create_splits_scenes()[self.split]
        samples = []
        scene_count = 0
        missing_files = []
        
        # First, check which scenes are actually available in the dataset
        available_scenes = set()
        for scene in self.nusc.scene:
            if scene['name'] not in scenes:
                continue
                
            first_sample = self.nusc.get('sample', scene['first_sample_token'])
            try:
                # Try to access the camera file to verify scene availability
                cam_data = self.nusc.get('sample_data', first_sample['data'][self.cam_sensor])
                cam_path = os.path.join(self.dataroot, cam_data['filename'].replace('/', os.sep))
                if os.path.exists(cam_path):
                    available_scenes.add(scene['name'])
                    logging.debug(f"Scene {scene['name']} is available")
                else:
                    logging.debug(f"Scene {scene['name']} camera file not found: {cam_path}")
            except Exception as e:
                logging.debug(f"Scene {scene['name']} not available: {str(e)}")
                continue
        
        logging.info(f"Found {len(available_scenes)} available scenes out of {len(scenes)} total scenes")
        
        # Then filter samples only from available scenes
        for scene in self.nusc.scene:
            if scene['name'] not in scenes or scene['name'] not in available_scenes:
                continue
                
            scene_count += 1
            sample_token = scene['first_sample_token']
            scene_samples = 0
            
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                
                # Check if files exist before adding to samples
                try:
                    cam_data = self.nusc.get('sample_data', sample['data'][self.cam_sensor])
                    lidar_data = self.nusc.get('sample_data', sample['data'][self.lidar_sensor])
                    
                    cam_path = os.path.join(self.dataroot, cam_data['filename'].replace('/', os.sep))
                    lidar_path = os.path.join(self.dataroot, lidar_data['filename'].replace('/', os.sep))
                    
                    if not os.path.exists(cam_path):
                        missing_files.append(cam_path)
                        sample_token = sample['next']
                        continue
                        
                    if not os.path.exists(lidar_path):
                        missing_files.append(lidar_path)
                        sample_token = sample['next']
                        continue
                    
                    if self._has_valid_sensors(sample):
                        samples.append(sample_token)
                        scene_samples += 1
                except Exception as e:
                    logging.warning(f"Error processing sample {sample_token}: {str(e)}")
                    
                sample_token = sample['next']
            
            logging.debug(f"Scene {scene['name']}: found {scene_samples} valid samples")
        
        if missing_files:
            logging.warning(f"Found {len(missing_files)} missing files. First 5:")
            for f in missing_files[:5]:
                logging.warning(f"  {f}")
        
        logging.info(f"Found {len(samples)} samples from {scene_count} scenes")
        if len(samples) == 0:
            raise RuntimeError(
                "No valid samples found! Please check:\n"
                "1. Dataset structure and scene availability\n"
                "2. File naming conventions\n"
                "3. Dataset extraction completeness"
            )
        return samples

    def _has_valid_sensors(self, sample):
        """Ensures that the sample contains key frames for both sensors."""
        try:
            lidar_data = self.nusc.get('sample_data', sample['data'][self.lidar_sensor])
            cam_data = self.nusc.get('sample_data', sample['data'][self.cam_sensor])
            return lidar_data['is_key_frame'] and cam_data['is_key_frame']
        except KeyError as e:
            logging.warning(f"Missing sensor data: {e}")
            return False
        except Exception as e:
            logging.error(f"Error checking sensors: {e}")
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Loads camera and LiDAR paths along with calibration data."""
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        
        try:
            # Load camera data
            cam_data = self.nusc.get('sample_data', sample['data'][self.cam_sensor])
            cam_filename = cam_data['filename'].replace('/', os.sep)
            cam_path = os.path.join(self.dataroot, cam_filename)
            
            # Load LiDAR data
            lidar_data = self.nusc.get('sample_data', sample['data'][self.lidar_sensor])
            lidar_filename = lidar_data['filename'].replace('/', os.sep)
            lidar_path = os.path.join(self.dataroot, lidar_filename)
            
            # Debug logging for paths
            logging.debug(f"\nProcessing sample {sample_token}:")
            logging.debug(f"Camera filename from dataset: {cam_data['filename']}")
            logging.debug(f"Constructed camera path: {cam_path}")
            logging.debug(f"LiDAR filename from dataset: {lidar_data['filename']}")
            logging.debug(f"Constructed LiDAR path: {lidar_path}")
            
            # Verify camera directory exists
            cam_dir = os.path.dirname(cam_path)
            if not os.path.exists(cam_dir):
                raise FileNotFoundError(
                    f"Camera directory not found: {cam_dir}\n"
                    f"Please check dataset structure."
                )
            
            # List available files in camera directory for debugging
            if not os.path.exists(cam_path):
                cam_files = os.listdir(cam_dir)
                logging.debug(f"\nAvailable files in {cam_dir}:")
                for f in sorted(cam_files)[:5]:  # Show first 5 files
                    logging.debug(f"  {f}")
                raise FileNotFoundError(
                    f"Image not found: {cam_path}\n"
                    f"Original filename: {cam_data['filename']}\n"
                    f"Dataroot: {self.dataroot}\n"
                    f"Camera directory exists but file not found.\n"
                    f"Please check file naming and dataset extraction."
                )
            
            # Verify LiDAR directory exists
            lidar_dir = os.path.dirname(lidar_path)
            if not os.path.exists(lidar_dir):
                raise FileNotFoundError(
                    f"LiDAR directory not found: {lidar_dir}\n"
                    f"Please check dataset structure."
                )
            
            # List available files in LiDAR directory for debugging
            if not os.path.exists(lidar_path):
                lidar_files = os.listdir(lidar_dir)
                logging.debug(f"\nAvailable files in {lidar_dir}:")
                for f in sorted(lidar_files)[:5]:  # Show first 5 files
                    logging.debug(f"  {f}")
                raise FileNotFoundError(
                    f"LiDAR data not found: {lidar_path}\n"
                    f"Original filename: {lidar_data['filename']}\n"
                    f"Dataroot: {self.dataroot}\n"
                    f"LiDAR directory exists but file not found.\n"
                    f"Please check file naming and dataset extraction."
                )
            
            # Load calibration data
            calib = self._load_calibration(sample)
            
            # Get ego pose
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

            return {
                'image': cam_path,
                'lidar': lidar_path,
                'calib': calib,
                'timestamp': sample['timestamp'],
                'sample_token': sample_token,
                'ego_pose': ego_pose
            }
            
        except Exception as e:
            logging.error(f"Error loading sample {sample_token}: {e}")
            raise

    def _load_calibration(self, sample):
        """Retrieves intrinsics and extrinsics as 4x4 transformation matrices."""
        cam_data = self.nusc.get('sample_data', sample['data'][self.cam_sensor])
        lidar_data = self.nusc.get('sample_data', sample['data'][self.lidar_sensor])
        
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        lidar_calib = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        def to_matrix(rotation, translation):
            """Converts quaternion rotation + translation to 4x4 transformation matrix."""
            rot = Quaternion(rotation).rotation_matrix
            trans = np.array(translation).reshape(3, 1)
            return np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))

        return {
            'cam_intrinsic': np.array(cam_calib['camera_intrinsic']),
            'cam2ego': to_matrix(cam_calib['rotation'], cam_calib['translation']),
            'lidar2ego': to_matrix(lidar_calib['rotation'], lidar_calib['translation'])
        }

class NuScenesBEVLabelDataset(Dataset):
    """Generates BEV segmentation labels from filtered samples."""
    
    def __init__(self, filtered_dataset, grid_size=128, resolution=0.2):
        self.filtered_dataset = filtered_dataset
        self.grid_size = grid_size
        self.resolution = resolution
        self.nusc = filtered_dataset.nusc
        self.class_map = filtered_dataset.class_map
        
        # Verify dataset version
        if 'trainval' not in self.nusc.version:
            raise ValueError(
                f"This implementation requires the full trainval dataset. "
                f"Found version: {self.nusc.version}. Please use v1.0-trainval."
            )
        
        # Initialize maps and validate data
        self.map_cache = {}
        self.valid_samples = []
        self._initialize_maps()
        self._validate_samples()
        
        logging.info(f"Initialized BEV dataset with {len(self.valid_samples)} valid samples")

    def _initialize_maps(self):
        """Initialize and cache maps for all locations."""
        map_locations = ['singapore-onenorth', 'singapore-hollandvillage', 
                        'singapore-queenstown', 'boston-seaport']
        
        for location in map_locations:
            try:
                # Check map files exist
                json_path = os.path.join(self.nusc.dataroot, 'maps', 'expansion', f'{location}.json')
                
                if not os.path.exists(json_path):
                    logging.error(f"Map files missing for {location}")
                    continue
                
                # Load map
                nusc_map = NuScenesMap(dataroot=self.nusc.dataroot, map_name=location)
                self.map_cache[location] = nusc_map
                logging.info(f"Loaded map for {location}")
                
            except Exception as e:
                logging.error(f"Failed to load map for {location}: {str(e)}")
                continue
        
        if not self.map_cache:
            raise RuntimeError(
                "No maps loaded! Please check:\n"
                "1. Map files exist and are readable\n"
                f"2. Dataset root path is correct: {self.nusc.dataroot}"
            )

    def _validate_samples(self):
        """Validate all samples and filter those with proper map data."""
        valid_samples_by_location = {}
        location_counts = {}  # Track all locations
        
        for idx in range(len(self.filtered_dataset)):
            try:
                sample = self.filtered_dataset[idx]
                sample_token = sample['sample_token']
                
                # Get scene and log for location
                scene_token = self.nusc.get('sample', sample_token)['scene_token']
                scene = self.nusc.get('scene', scene_token)
                log = self.nusc.get('log', scene['log_token'])
                location = log['location']
                
                # Track all locations
                location_counts[location] = location_counts.get(location, 0) + 1
                
                if location in self.map_cache:
                    nusc_map = self.map_cache[location]
                    ego_pose = sample['ego_pose']
                    ego_x, ego_y = ego_pose['translation'][:2]
                    
                    # Define patch box around ego vehicle (x_min, y_min, x_max, y_max)
                    patch_size = self.grid_size * self.resolution  # Convert grid cells to meters
                    patch_box = (
                        ego_x - patch_size/2,  # x_min
                        ego_y - patch_size/2,  # y_min
                        ego_x + patch_size/2,  # x_max
                        ego_y + patch_size/2   # y_max
                    )
                    
                    # Check for any map data in the patch
                    found_data = False
                    for layer in ['drivable_area', 'road_segment', 'lane']:
                        try:
                            records = nusc_map.get_records_in_patch(patch_box, layer_names=[layer])
                            if records and len(records[layer]) > 0:
                                # Verify we can extract at least one polygon
                                record = nusc_map.get(layer, records[layer][0])
                                if layer == 'drivable_area':
                                    poly_token = record['polygon_tokens'][0]
                                else:
                                    poly_token = record['polygon_token']
                                
                                poly = nusc_map.extract_polygon(poly_token)
                                if not poly.is_empty:
                                    found_data = True
                                    break
                        except Exception as e:
                            logging.debug(f"Error checking {layer} at {location}: {str(e)}")
                            continue
                    
                    if found_data:
                        self.valid_samples.append(idx)
                        valid_samples_by_location[location] = valid_samples_by_location.get(location, 0) + 1
                        
                else:
                    logging.debug(f"No map cache for location: {location}")
                    
            except Exception as e:
                logging.warning(f"Error validating sample {idx}: {str(e)}")
                continue
        
        # Log statistics
        total_samples = len(self.filtered_dataset)
        valid_samples = len(self.valid_samples)
        logging.info(f"\nDataset Statistics:")
        logging.info(f"Total samples: {total_samples}")
        logging.info(f"Valid samples: {valid_samples} ({valid_samples/total_samples*100:.1f}%)")
        logging.info("\nSamples by location:")
        for loc in location_counts:
            valid = valid_samples_by_location.get(loc, 0)
            total = location_counts[loc]
            logging.info(f"{loc}: {valid}/{total} valid ({valid/total*100:.1f}%)")
        
        if not self.valid_samples:
            # Provide detailed error message
            logging.error("\nMap loading summary:")
            logging.error(f"Total locations in dataset: {len(location_counts)}")
            logging.error(f"Successfully loaded maps: {len(self.map_cache)}")
            logging.error("\nAvailable locations:")
            for location in location_counts:
                logging.error(f"  {location}")
            logging.error("\nLoaded maps:")
            for location in self.map_cache:
                logging.error(f"  {location}")
            
            raise RuntimeError(
                "No valid samples found with map data! Please check:\n"
                "1. Map files exist in the correct location\n"
                "2. Map files contain valid data\n"
                "3. Sample locations match available maps\n"
                f"4. Dataset root path is correct: {self.nusc.dataroot}"
            )
        
        logging.info(f"\nFound {len(self.valid_samples)} valid samples out of "
                    f"{len(self.filtered_dataset)} total samples")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """Retrieves BEV segmentation label for a sample."""
        try:
            # Get the valid sample index
            valid_idx = self.valid_samples[idx]
            sample = self.filtered_dataset[valid_idx]
            sample_token = sample['sample_token']
            ego_pose = sample['ego_pose']
            
            bev_label = self._generate_bev_label(sample_token, ego_pose)
            
            return {
                'bev_label': torch.from_numpy(bev_label).long(),
                'sample_token': sample_token
            }
            
        except Exception as e:
            logging.error(f"Error generating BEV label for sample {sample_token}: {e}")
            raise

    def _generate_bev_label(self, sample_token, ego_pose):
        """Generate BEV segmentation label."""
        label = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Get scene location
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        location = log['location']
        
        logging.info(f"\nGenerating BEV label for sample {sample_token}")
        logging.info(f"Location: {location}")
        
        nusc_map = self.map_cache.get(location)
        if nusc_map is None:
            raise RuntimeError(f"Map not found for location: {location}")
        
        # Calculate patch box
        patch_size = self.grid_size * self.resolution
        ego_x, ego_y = ego_pose['translation'][:2]
        patch_box = (
            ego_x - patch_size/2,
            ego_y - patch_size/2,
            ego_x + patch_size/2,
            ego_y + patch_size/2
        )
        
        logging.info(f"Ego position: ({ego_x:.2f}, {ego_y:.2f})")
        logging.info(f"Patch box: {patch_box}")
        
        # Process each map layer
        for layer_name, class_id in self.class_map.items():
            try:
                # Get records in patch
                records = nusc_map.get_records_in_patch(patch_box, layer_names=[layer_name])
                layer_records = records.get(layer_name, [])
                
                logging.info(f"\nProcessing layer {layer_name}: found {len(layer_records)} records")
                
                for record_token in layer_records:
                    try:
                        record = nusc_map.get(layer_name, record_token)
                        
                        if layer_name == 'drivable_area':
                            # Handle multiple polygons for drivable area
                            for poly_token in record['polygon_tokens']:
                                polygon = nusc_map.extract_polygon(poly_token)
                                if not polygon.is_empty:
                                    coords = np.array(polygon.exterior.coords)
                                    if len(coords) > 2:
                                        coords_ego = self._world_to_grid(coords, ego_pose)
                                        self._fill_polygon(coords_ego, label, class_id)
                        
                        elif layer_name in ['road_divider', 'lane_divider']:
                            # Handle line segments
                            line = nusc_map.extract_line(record['line_token'])
                            if not line.is_empty:
                                coords = np.array(line.coords)
                                if len(coords) > 1:
                                    coords_ego = self._world_to_grid(coords, ego_pose)
                                    self._draw_line(coords_ego, label, class_id)
                        
                        else:
                            # Handle single polygon layers
                            polygon = nusc_map.extract_polygon(record['polygon_token'])
                            if not polygon.is_empty:
                                coords = np.array(polygon.exterior.coords)
                                if len(coords) > 2:
                                    coords_ego = self._world_to_grid(coords, ego_pose)
                                    self._fill_polygon(coords_ego, label, class_id)
                        
                    except Exception as e:
                        logging.warning(f"Error processing record in {layer_name}: {str(e)}")
                        continue
                
                # Log statistics for this layer
                layer_pixels = np.sum(label == class_id)
                logging.info(f"Layer {layer_name}: {layer_pixels} pixels filled")
                
            except Exception as e:
                logging.warning(f"Error processing layer {layer_name}: {str(e)}")
                continue
        
        # Log final label statistics
        unique_labels, counts = np.unique(label, return_counts=True)
        logging.info("\nFinal label statistics:")
        for label_id, count in zip(unique_labels, counts):
            if label_id > 0:  # Skip background
                layer_name = [k for k, v in self.class_map.items() if v == label_id][0]
                logging.info(f"{layer_name}: {count} pixels")
        
        return label

    def _render_geometry(self, polygon, label, class_id, ego_pose):
        """Converts world coordinates to BEV grid and rasterizes."""
        try:
            # Convert to shapely polygon
            if isinstance(polygon['polygon'], (Polygon, LineString)):
                geom = polygon['polygon']
            else:
                geom = Polygon(polygon['polygon'])
            
            # Get coordinates
            if isinstance(geom, Polygon):
                coords = np.array(geom.exterior.coords)
            else:  # LineString
                coords = np.array(geom.coords)
            
            # Transform to grid
            grid_coords = self._world_to_grid(coords, ego_pose)
            
            # Render based on geometry type
            if isinstance(geom, Polygon) and len(grid_coords) >= 3:
                self._fill_polygon(grid_coords, label, class_id)
            elif isinstance(geom, LineString) and len(grid_coords) >= 2:
                self._draw_line(grid_coords, label, class_id)
                
        except Exception as e:
            logging.warning(f"Error rendering geometry: {e}")

    def _world_to_grid(self, coords, ego_pose):
        """Transforms world coordinates to BEV grid indices."""
        center = self.grid_size // 2
        rotation = Quaternion(ego_pose['rotation']).rotation_matrix
        translation = np.array(ego_pose['translation'][:2])

        # Transform coordinates using the inverse rotation (transpose)
        coords_transformed = np.dot(rotation[:2, :2].T, (coords[:, :2] - translation).T).T

        grid_coords = ((coords_transformed / self.resolution) + center).astype(int)
        
        # Clip to grid bounds
        grid_coords = np.clip(grid_coords, 0, self.grid_size - 1)
        
        return grid_coords

    def _fill_polygon(self, coords, label, class_id):
        """Fills polygons in the BEV grid."""
        try:
            from matplotlib.path import Path
            grid_x, grid_y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
            points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
            
            mask = Path(coords).contains_points(points)
            mask = mask.reshape(self.grid_size, self.grid_size)
            label[mask] = class_id
        except Exception as e:
            logging.warning(f"Error filling polygon: {e}")

    def _draw_line(self, coords, label, class_id, thickness=1):
        """Draws lines in the BEV grid."""
        try:
            from skimage.draw import line_aa
            for i in range(len(coords) - 1):
                rr, cc, _ = line_aa(coords[i, 1], coords[i, 0], 
                                  coords[i+1, 1], coords[i+1, 0])
                valid = (rr >= 0) & (rr < self.grid_size) & \
                       (cc >= 0) & (cc < self.grid_size)
                label[rr[valid], cc[valid]] = class_id
        except Exception as e:
            logging.warning(f"Error drawing line: {e}")

    def _verify_map_data(self, location, nusc_map):
        """Verify that map data exists and is accessible."""
        # Test different areas of the map
        test_boxes = [
            (-100, -100, 100, 100),  # Center
            (0, 0, 200, 200),        # Positive quadrant
            (-200, -200, 0, 0)       # Negative quadrant
        ]
        
        for test_box in test_boxes:
            for layer in ['drivable_area', 'road_segment', 'lane']:
                records = nusc_map.get_records_in_patch(test_box, layer_names=[layer])
                if len(records[layer]) > 0:
                    logging.info(f"Found {len(records[layer])} {layer} records in {location}")
                    return True
        
        return False

    def run_all_tests(self):
        """Run all validation tests"""
        try:
            print("\nRunning validation tests for trainval dataset...")
            # self.test_calibration()
            # self.test_data_loading()
            self.test_bev_visualization()
            self.test_coordinate_mapping()
            self.test_map_statistics()
            # self.test_dataloader()
            print("\nAll validation tests completed successfully!")
            
        except Exception as e:
            logging.error(f"Validation failed: {e}")
            raise 