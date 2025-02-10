import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, List, Callable


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    dist_url: str = "tcp://localhost:12345"
) -> None:
    """
    Initialize distributed training environment
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: PyTorch distributed backend
        dist_url: URL for distributed coordination
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = dist_url.split(":")[-1]
    
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    # Enable cuDNN benchmark mode for better performance
    torch.backends.cudnn.benchmark = True


def cleanup_distributed():
    """Clean up distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_distributed(
    fn: Callable,
    world_size: int,
    backend: str = "nccl",
    dist_url: str = "tcp://localhost:12345",
    args: tuple = (),
    kwargs: dict = {}
):
    """
    Launch distributed training
    Args:
        fn: Function to run in distributed mode
        world_size: Number of processes to launch
        backend: Distributed backend
        dist_url: URL for coordination
        args: Arguments for the function
        kwargs: Keyword arguments for the function
    """
    mp.spawn(
        _distributed_worker,
        args=(fn, world_size, backend, dist_url, args, kwargs),
        nprocs=world_size,
        join=True
    )


def _distributed_worker(
    rank: int,
    fn: Callable,
    world_size: int,
    backend: str,
    dist_url: str,
    args: tuple,
    kwargs: dict
):
    """Worker function for distributed training"""
    setup_distributed(rank, world_size, backend, dist_url)
    try:
        fn(rank, *args, **kwargs)
    finally:
        cleanup_distributed()


def get_world_size() -> int:
    """Get total number of distributed processes"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get rank of current process"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if current process is the main process"""
    return get_rank() == 0


def synchronize():
    """Synchronize all processes"""
    if not dist.is_initialized():
        return
    dist.barrier()


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce dictionary values across processes
    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average or sum the values
    Returns:
        Reduced dictionary
    """
    if not dist.is_initialized():
        return input_dict
    
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k, v in sorted(input_dict.items()):
            names.append(k)
            values.append(v)
            
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
            
        reduced_dict = {k: v for k, v in zip(names, values)}
        
    return reduced_dict


class DistributedWrapper:
    """
    Wrapper for distributed training setup
    Handles device placement and distributed data parallel
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        find_unused_parameters: bool = False
    ):
        self.model = model
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Wrap with DistributedDataParallel if in distributed mode
        if dist.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=find_unused_parameters
            )
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        self.model.eval()
        return self
    
    def __call__(self, *args, **kwargs):
        """Forward pass"""
        return self.model(*args, **kwargs)
    
    def state_dict(self):
        """Get state dictionary"""
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dictionary"""
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict) 