import os
import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigDict(dict):
    """
    Extended dictionary for easier access to nested configurations.
    Allows both dict['key'] and dict.key access.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = ConfigDict(v)
            elif isinstance(v, list):
                self[k] = [ConfigDict(x) if isinstance(x, dict) else x for x in v]


def load_config(config_path: str) -> ConfigDict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return ConfigDict(config)


def merge_configs(*configs: Dict) -> ConfigDict:
    """Merge multiple configurations, later configs override earlier ones"""
    merged = {}
    for config in configs:
        _deep_update(merged, config)
    return ConfigDict(merged)


def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """Recursively update a dictionary"""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def validate_config(config: ConfigDict) -> None:
    """Validate configuration values"""
    # Validate model config
    assert config.model.camera_backbone.name == "EfficientNetV2Backbone"
    assert config.model.lidar_backbone.name == "SECONDBackbone"
    assert config.model.fusion.name == "BEVFusion"
    assert config.model.segmentation_head.name == "BEVSegmentationHead"
    
    # Validate input dimensions
    assert len(config.model.input_config.image_size) == 2
    assert config.model.input_config.num_cameras > 0
    assert config.model.input_config.num_lidar_channels > 0
    
    # Validate training config
    assert config.train.batch_size > 0
    assert config.train.epochs > 0
    assert config.train.num_workers >= 0
    
    # Validate optimizer config
    assert all(v > 0 for v in [
        config.optimizer.lr_camera,
        config.optimizer.lr_lidar,
        config.optimizer.lr_fusion,
        config.optimizer.lr_head
    ])
    
    # Validate scheduler config
    assert config.scheduler.warmup_epochs >= 0
    assert 0 <= config.scheduler.warmup_ratio <= 1
    
    # Validate loss config
    assert len(config.loss.class_weights) == config.model.segmentation_head.num_classes
    assert all(w > 0 for w in config.loss.class_weights)


def save_config(config: ConfigDict, save_path: str) -> None:
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(dict(config), f, default_flow_style=False)


def load_all_configs(model_config_path: str, train_config_path: str) -> ConfigDict:
    """Load and merge model and training configurations"""
    model_config = load_config(model_config_path)
    train_config = load_config(train_config_path)
    
    # Create a unified config
    config = merge_configs(
        {'model': model_config},
        {'train': train_config}
    )
    
    # Validate the merged config
    validate_config(config)
    
    return config 