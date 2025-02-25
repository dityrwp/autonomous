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
from shapely.geometry import Polygon, LineString, MultiPolygon, Point,box
import json
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
from matplotlib.colors import ListedColormap
from shapely.prepared import prep
from shapely.geometry import MultiPoint


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def patched_load_table(self, table_name):
    filepath = os.path.join(self.dataroot, self.version, table_name + '.json')
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

NuScenes._load_table_ = patched_load_table

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
            'background': 0,
            'drivable_area': 1,
            'lane_divider': 2,
            'road_divider': 3,
            'ped_crossing': 4,
            'walkway': 5
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
        
        # Define consistent color map for visualization
        self.color_map = {
            'background': '#fcfcfc',     # Black for background
            'drivable_area': '#a6cee3',  # Light blue
            'road_divider': '#cab2d6',   # Gray
            'lane_divider': '#6a3d9a',   # Orange
            'walkway': '#e04a4c',        # Red
            'ped_crossing': '#fb9a99'    # Pink
        }
        
        # Convert hex colors to RGB for visualization
        self.rgb_colors = {}
        for layer, hex_color in self.color_map.items():
            hex_color = hex_color.lstrip('#')
            self.rgb_colors[layer] = tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
            
        logging.info(f"Initialized BEV dataset with {len(self.valid_samples)} valid samples")

    def _initialize_maps(self):
        """Initialize and cache maps for all locations."""
        map_locations = [
            'singapore-onenorth',
            'singapore-hollandvillage',
            'singapore-queenstown',
            'boston-seaport'
        ]
        
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
                    
                    # Define the target view space with expanded query area
                    forward_range = 25.6  # meters ahead of ego
                    side_range = 12.8    # meters to each side of ego

                    # Expand patch box for querying (larger than view area)
                    query_expansion = 10.0  # meters
                    patch_box = (
                        ego_pose['translation'][0] - (side_range + query_expansion),
                        ego_pose['translation'][1] - query_expansion,  # query behind ego too
                        ego_pose['translation'][0] + (side_range + query_expansion),
                        ego_pose['translation'][1] + (forward_range + query_expansion)
                    )

                    # Keep the target view boundaries the same for clipping
                    x_min = -side_range
                    x_max = side_range
                    y_min = 0.0  # start at ego
                    y_max = forward_range
                    
                    # Check for any map data in the patch
                    found_data = False
                    for layer in ['drivable_area', 'lane']:  # Removed road_segment
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

    def box_to_polygon(self, box):
        """Convert box coordinates to Shapely Polygon."""
        xmin, ymin, xmax, ymax = box
        return Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    def _generate_bev_label(self, sample_token, ego_pose):
        """Generate BEV segmentation label by direct rasterization from map."""
        # Initialize label with background class
        label = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)  # Will be class 0 (background)
        
        # Get scene location and sample data
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        location = log['location']
        
        # Get front camera image for validation
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        cam_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        
        # Get map and validate
        nusc_map = self.map_cache.get(location)
        if nusc_map is None:
            raise RuntimeError(f"Map not found for location: {location}")
        
        # Define crop ratio (e.g., 0.7 means 70% of the grid is in front of the ego)
        crop_ratio = 0.7
        
        # Calculate patch box based on crop ratio and shift for bottom-aligned ego
        patch_size = self.grid_size * self.resolution
        patch_box = (
            ego_pose['translation'][0] - patch_size/2,  # x_min (centered horizontally)
            ego_pose['translation'][1],                 # y_min (starting from ego)
            ego_pose['translation'][0] + patch_size/2,  # x_max
            ego_pose['translation'][1] + patch_size     # y_max (full range in front)
        )
        
        # Get rotation matrix and heading
        rotation_matrix = Quaternion(ego_pose['rotation']).rotation_matrix
        # Calculate heading angle (front is positive x-axis in vehicle frame)
        heading_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) 
        
        # Create figure with 2 subplots
        fig = plt.figure(figsize=(8, 8))
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1.2])
        
        # Front camera view
        ax_cam = fig.add_subplot(gs[0])
        ax_cam.set_title("Front Camera View")
        if os.path.exists(cam_path):
            img = plt.imread(cam_path)
            ax_cam.imshow(img)
        ax_cam.axis('off')
        
        # BEV grid view
        ax_bev = fig.add_subplot(gs[1])
        ax_bev.set_title(f"BEV Grid View ({self.grid_size}x{self.grid_size} @ {self.resolution}m/pixel)")
        
        # Define rendering order (from bottom to top)
        layer_order = [
            'drivable_area',  # Render drivable area first
            'walkway',        # Render walkways on top of drivable area
            'ped_crossing',   # Render pedestrian crossings next
            'road_divider',   # Render road and lane dividers last
            'lane_divider'    
        ]
        
        # Initialize BEV grid with background color
        bev_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        bev_grid[:] = self.class_map['background']
        
        # Process layers in order
        for layer_name in layer_order:
            if layer_name not in self.class_map:
                continue
            
            class_id = self.class_map[layer_name]
            
            try:
                records = nusc_map.get_records_in_patch(patch_box, layer_names=[layer_name])
                layer_records = records.get(layer_name, [])
                
                logging.info(f"\nProcessing layer {layer_name}: found {len(layer_records)} records")
                
                # Collect all polygons for this layer
                polygons = []
                
                for record_token in layer_records:
                    try:
                        record = nusc_map.get(layer_name, record_token)
                        
                        if layer_name == 'drivable_area':
                            for poly_token in record['polygon_tokens']:
                                polygon = nusc_map.extract_polygon(poly_token)
                                if not polygon.is_empty:
                                    polygons.append(polygon)
                        
                        elif layer_name in ['road_divider', 'lane_divider']:
                            line = nusc_map.extract_line(record['line_token'])
                            if not line.is_empty:
                                # Increase buffer size for dividers to make them more visible
                                buffer_size = 0.25 if layer_name == 'road_divider' else 0.15
                                polygon = line.buffer(buffer_size)
                                if not polygon.is_empty:
                                    polygons.append(polygon)
                        
                        else:
                            polygon = nusc_map.extract_polygon(record['polygon_token'])
                            if not polygon.is_empty:
                                polygons.append(polygon)
                    
                    except Exception as e:
                        logging.warning(f"Error processing record in {layer_name}: {str(e)}")
                        continue
                
                if polygons:
                    # Define the target view space with expanded query area
                    forward_range = 25.6  # meters ahead of ego
                    side_range = 12.8     # meters to each side of ego

                    # Expand patch box for querying (larger than view area)
                    query_expansion = 10.0  # meters
                    patch_box = (
                        ego_pose['translation'][0] - (side_range + query_expansion),
                        ego_pose['translation'][1] - query_expansion,  # query behind ego too
                        ego_pose['translation'][0] + (side_range + query_expansion),
                        ego_pose['translation'][1] + (forward_range + query_expansion)
                    )

                    # Keep the target view boundaries the same for clipping
                    x_min = -side_range
                    x_max = side_range
                    y_min = 0.0  # start at ego
                    y_max = forward_range
                    
                    transformed_polys = []
                    for polygon in polygons:
                        # Get polygon coordinates
                        poly_coords = np.array(polygon.exterior.coords)
                        
                        # 1. Transform to ego vehicle coordinate system
                        # Translate such that ego is at (0, 0)
                        poly_coords[:, 0] -= ego_pose['translation'][0]
                        poly_coords[:, 1] -= ego_pose['translation'][1]
                        
                        # Rotate to align with ego heading
                        theta = heading_angle -np.pi/2
                        c, s = np.cos(theta), np.sin(theta)
                        R = np.array(((c, -s), (s, c)))
                        poly_coords = np.dot(poly_coords, R)
                        
                        # 2. Clip polygons to target view boundaries
                        poly = Polygon(poly_coords)
                        target_box = box(x_min, y_min, x_max, y_max)
                        clipped_poly = poly.intersection(target_box)
                        
                        if not clipped_poly.is_empty:
                            # Handle both Polygon and MultiPolygon types
                            if clipped_poly.geom_type == 'Polygon':
                                # Process single polygon
                                clipped_coords = np.array(clipped_poly.exterior.coords)
                                
                                # Convert to grid coordinates
                                grid_coords = np.zeros_like(clipped_coords)
                                grid_coords[:, 0] = (clipped_coords[:, 0] - x_min) / (x_max - x_min) * self.grid_size
                                grid_coords[:, 1] = (y_max - clipped_coords[:, 1]) / (y_max - y_min) * self.grid_size
                                
                                transformed_polys.append(Polygon(grid_coords))
                            
                            elif clipped_poly.geom_type == 'MultiPolygon':
                                # Process each polygon in the multipolygon
                                for poly_part in clipped_poly.geoms:
                                    clipped_coords = np.array(poly_part.exterior.coords)
                                    
                                    # Convert to grid coordinates
                                    grid_coords = np.zeros_like(clipped_coords)
                                    grid_coords[:, 0] = (clipped_coords[:, 0] - x_min) / (x_max - x_min) * self.grid_size
                                    grid_coords[:, 1] = (y_max - clipped_coords[:, 1]) / (y_max - y_min) * self.grid_size
                                    
                                    transformed_polys.append(Polygon(grid_coords))
                    
                    # Create points grid for contains test
                    x = np.linspace(0, self.grid_size-1, self.grid_size)
                    y = np.linspace(0, self.grid_size-1, self.grid_size)
                    grid_x, grid_y = np.meshgrid(x, y)
                    points = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
                    
                    # Test points against transformed polygons
                    mask = np.zeros(len(points), dtype=bool)
                    points = [Point(p) for p in points]
                    
                    if transformed_polys:
                        merged_poly = transformed_polys[0]
                        for poly in transformed_polys[1:]:
                            merged_poly = merged_poly.union(poly)
                        
                        prepared_poly = prep(merged_poly)
                        mask = np.array([prepared_poly.contains(point) for point in points])
                    
                    # Reshape mask back to grid
                    mask = mask.reshape(self.grid_size, self.grid_size)
                    bev_grid[mask] = class_id
            
            except Exception as e:
                logging.error(f"Error processing layer {layer_name}: {str(e)}")
                continue
        
        # Convert class IDs to RGB colors for visualization
        bev_vis = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for class_name, class_id in self.class_map.items():
            mask = (bev_grid == class_id)
            # Convert hex color to RGB values
            hex_color = self.color_map[class_name].lstrip('#')
            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            bev_vis[mask] = rgb_color
        
        # Show composite BEV
        ax_bev.imshow(bev_vis)
        
        # Draw ego vehicle at the bottom center of the grid
        ego_bev_x = self.grid_size // 2  # Center X
        ego_bev_y = self.grid_size - 10   # Near bottom Y (10 pixels from bottom)
        
        # Draw ego vehicle as a triangle pointing upward
        triangle_height = 15
        triangle_width = 10
        triangle = plt.Polygon([
            [ego_bev_x, ego_bev_y - triangle_height],  # Top point
            [ego_bev_x - triangle_width/2, ego_bev_y],  # Bottom left
            [ego_bev_x + triangle_width/2, ego_bev_y]   # Bottom right
        ], color='red', alpha=1.0, label='Ego Vehicle')
        ax_bev.add_patch(triangle)
        
        # Add distance markers in front of ego (semicircles)
        for dist in [10, 20, 30, 40]:  # distances in meters
            pixels = dist / self.resolution
            # Draw only the front half of the circle
            circle = plt.matplotlib.patches.Arc(
                (ego_bev_x, ego_bev_y), 
                pixels * 2, pixels * 2,  # width, height
                theta1=180, theta2=360,  # Draw only top half
                color='gray', linestyle='--', alpha=0.3)
            ax_bev.add_patch(circle)
            # Add distance label
            ax_bev.text(
                ego_bev_x, ego_bev_y - pixels, 
                f'{dist}m', 
                horizontalalignment='center',
                color='gray', alpha=0.5
            )
        
        # Add coordinate axes labels
        ax_bev.set_xlabel("Lateral Distance (meters)")
        ax_bev.set_ylabel("Forward Distance (meters)")
        
        # Adjust tick labels to show distances relative to ego vehicle
        meter_ticks_x = np.arange(-25, 26, 5)
        meter_ticks_y = np.arange(0, 51, 5)  # Only positive distances for forward view
        pixel_ticks_x = meter_ticks_x / self.resolution + self.grid_size // 2
        pixel_ticks_y = self.grid_size - (meter_ticks_y / self.resolution)
        
        ax_bev.set_xticks(pixel_ticks_x[::2])
        ax_bev.set_yticks(pixel_ticks_y[::2])
        ax_bev.set_xticklabels([f'{x}m' for x in meter_ticks_x[::2]])
        ax_bev.set_yticklabels([f'{y}m' for y in meter_ticks_y[::2]])
        
        # Add legend
        ax_bev.legend()
        
        plt.tight_layout()
        plt.savefig(f"bev_debug_{sample_token}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return bev_grid  # Return class IDs for training

    def _world_to_grid(self, coords, ego_pose):
        """Transforms world coordinates to BEV grid indices."""
        # Get rotation matrix from quaternion (only 2x2 part for 2D rotation)
        rotation = Quaternion(ego_pose['rotation']).rotation_matrix[:2, :2]
        translation = np.array(ego_pose['translation'][:2])

        # Debug prints for transformation components - only print first point as example
        logging.info(f"Sample coordinate transformation:")
        logging.info(f"- Original first point: {coords[0]}")
        
        # 1. Center coordinates at ego vehicle position
        coords_centered = coords[:, :2] - translation
        logging.info(f"- After centering: {coords_centered[0]}")
        
        # 2. Rotate coordinates to align with ego vehicle orientation
        coords_rotated = np.dot(rotation, coords_centered.T).T
        logging.info(f"- After rotation: {coords_rotated[0]}")
        
        # 3. Scale to grid resolution and shift to center of grid
        grid_coords = (coords_rotated / self.resolution + self.grid_size/2).astype(int)
        logging.info(f"- Final grid coords: {grid_coords[0]}")
        
        # Clip coordinates to grid bounds
        grid_coords = np.clip(grid_coords, 0, self.grid_size - 1)
        
        # Print summary statistics instead of all coordinates
        logging.info(f"Coordinate ranges:")
        logging.info(f"- X range: [{grid_coords[:, 0].min()}, {grid_coords[:, 0].max()}]")
        logging.info(f"- Y range: [{grid_coords[:, 1].min()}, {grid_coords[:, 1].max()}]")
        
        return grid_coords

    def _fill_polygon(self, coords, label, class_id):
        """Fills polygons in the BEV grid."""
        try:
            from matplotlib.path import Path
            grid_x, grid_y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
            points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
            
            path = Path(coords)
            mask = path.contains_points(points)
            mask = mask.reshape(self.grid_size, self.grid_size)
            label[mask] = class_id
        except Exception as e:
            logging.warning(f"Error filling polygon: {e}")

    def _draw_line(self, coords, label, class_id):
        """Draws lines in the BEV grid."""
        try:
            from skimage.draw import line
            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i + 1]
                
                # Skip if either point is outside the grid
                if not (0 <= start[0] < self.grid_size and 0 <= start[1] < self.grid_size and
                       0 <= end[0] < self.grid_size and 0 <= end[1] < self.grid_size):
                    continue
                
                # Draw the line
                rr, cc = line(int(start[1]), int(start[0]), int(end[1]), int(end[0]))
                
                # Filter points within grid bounds
                valid = (rr >= 0) & (rr < self.grid_size) & (cc >= 0) & (cc < self.grid_size)
                label[rr[valid], cc[valid]] = class_id
                
                # # Draw thicker line by adding adjacent pixels
                # for offset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                #     rr_offset = rr[valid] + offset[0]
                #     cc_offset = cc[valid] + offset[1]
                #     valid_offset = (rr_offset >= 0) & (rr_offset < self.grid_size) & \
                #                  (cc_offset >= 0) & (cc_offset < self.grid_size)
                #     label[rr_offset[valid_offset], cc_offset[valid_offset]] = class_id
                
        except Exception as e:
            logging.warning(f"Error drawing line: {e}")
            return

    def _verify_map_data(self, location, nusc_map):
        """Verify that map data exists and is accessible."""
        # Test different areas of the map
        test_boxes = [
            (-100, -100, 100, 100),  # Center
            (0, 0, 200, 200),        # Positive quadrant
            (-200, -200, 0, 0)       # Negative quadrant
        ]
        
        for test_box in test_boxes:
            for layer in ['drivable_area', 'lane']:  # Removed road_segment
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