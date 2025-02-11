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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
            cam_path = os.path.join(self.dataroot, cam_data['filename'])
            
            # Load LiDAR data
            lidar_data = self.nusc.get('sample_data', sample['data'][self.lidar_sensor])
            lidar_path = os.path.join(self.dataroot, lidar_data['filename'])
            
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
        try:
            # For trainval dataset, use location-based maps
            map_locations = ['singapore-onenorth', 'singapore-hollandvillage', 
                           'singapore-queenstown', 'boston-seaport']
            
            for location in map_locations:
                map_path = os.path.join(self.nusc.dataroot, 'maps', 
                                      'expansion', f'{location}.json')
                if os.path.exists(map_path):
                    self.map_cache[location] = NuScenesMap(self.nusc.dataroot, location)
                    logging.info(f"Loaded map for location: {location}")
                else:
                    raise FileNotFoundError(f"Map file not found: {map_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error initializing maps: {e}")

    def _validate_samples(self):
        """Validate all samples and filter those with proper map data."""
        for idx in range(len(self.filtered_dataset)):
            sample = self.filtered_dataset[idx]
            sample_token = sample['sample_token']
            
            # Get scene and location
            scene = self.nusc.get('sample', sample_token)['scene_token']
            scene = self.nusc.get('scene', scene)
            location = scene['location']
            
            # Check if we have valid map data
            if location in self.map_cache:
                self.valid_samples.append(idx)
            
        if not self.valid_samples:
            raise RuntimeError("No valid samples found with map data!")
        
        logging.info(f"Found {len(self.valid_samples)} valid samples out of "
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
        """Generates a BEV grid with HD map elements."""
        label = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Get scene and location
        scene = self.nusc.get('sample', sample_token)['scene_token']
        scene = self.nusc.get('scene', scene)
        location = scene['location']
        
        # Get map (should be cached)
        nusc_map = self.map_cache.get(location)
        if nusc_map is None:
            raise RuntimeError(f"Map not found for location: {location}")
        
        try:
            # Get patch coordinates centered around ego pose
            patch_box = (ego_pose['translation'][0], ego_pose['translation'][1],
                        self.grid_size * self.resolution, self.grid_size * self.resolution)
            patch_angle = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]
            
            # Process each map layer
            for layer_name, class_id in self.class_map.items():
                try:
                    # Get records in the patch
                    records = nusc_map.get_records_in_patch(patch_box, layer_names=[layer_name], mode='intersect')
                    
                    # Get geometries for the records
                    geometries = []
                    for token in records[layer_name]:
                        if layer_name in ['road_divider', 'lane_divider']:
                            line = nusc_map.get('line', token)
                            geom = {'polygon': nusc_map.extract_line(line['line_token'])}
                        else:
                            poly_record = nusc_map.get(layer_name, token)
                            if layer_name == 'drivable_area':
                                for poly_token in poly_record['polygon_tokens']:
                                    geom = {'polygon': nusc_map.extract_polygon(poly_token)}
                                    geometries.append(geom)
                                continue
                            else:
                                geom = {'polygon': nusc_map.extract_polygon(poly_record['polygon_token'])}
                        geometries.append(geom)
                    
                    # Render each geometry
                    for geom in geometries:
                        self._render_geometry(geom, label, class_id, ego_pose)
                    
                except Exception as e:
                    logging.warning(f"Error processing layer {layer_name}: {e}")
                    continue
                    
        except Exception as e:
            raise RuntimeError(f"Error generating BEV label: {e}")
        
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

        # Transform coordinates
        coords_2d = coords[:, :2]  # Only use x,y coordinates
        coords_transformed = np.dot(rotation[:2, :2], (coords_2d - translation).T).T
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