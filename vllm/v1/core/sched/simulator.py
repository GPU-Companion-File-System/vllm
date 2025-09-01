import torch
import math
from typing import Tuple, Optional
from vllm.v1.core.sched.scheduler import init_logger
from vllm.config import ModelConfig, VllmConfig

from typing import List, Dict

import flashinfer.green_ctx as green_ctx


logger = init_logger(__name__)

def partition_sms(total_sm, min_sm_partition, step):
    partitions = []
    remaining = total_sm
    current = min_sm_partition
    while remaining > 0:
        part = min(current, remaining)   # 最后一个不要超过 total_sm
        if part > 0 and part % min_sm_partition != 0:
            part = part - part % min_sm_partition
        if part == 0:
            break
        partitions.append(part)
        remaining -= part
        current = min(current + step, total_sm)
    return partitions


class StreamController:
    """
    Manages a pool of CUDA streams, each partitioned to a specific number of SMs.

    This controller pre-creates multiple streams with different SM resource allocations
    using green contexts. It provides an interface to get the most
    appropriate stream for a given number of concurrent tasks (e.g., I/O operations),
    effectively "padding" the resource request to the next available stream size.

    Args:
        device (torch.device): The CUDA device to create streams on.
        min_sm_partition (int): The number of SMs for the smallest stream partition.
        sm_partition_step (int): The increment in SMs for each subsequent stream partition.
        sms_per_io_task (int): An estimated heuristic for how many SMs are needed
                               per concurrent I/O task. This is used to calculate
                               the total required SMs.
    """
    def __init__(self,
                 devices: List[torch.device],
                 min_sm_partition: int = 8,
                 sm_partition_step: int = 8,
                 sms_per_io_task: int = 1):
        if not torch.cuda.is_available() or devices[0].type != 'cuda':
            raise ValueError("StreamController requires a CUDA device.")

        self.devices = devices
        self.per_device_total_sm: Dict[torch.device, int] = {}
        self.per_device_streams: Dict[torch.device, Dict[int, torch.cuda.Stream]] = {}
        self.per_device_sorted_sm_counts: Dict[torch.device, List[int]] = {}
        
        self.min_sm_partition = min_sm_partition
        self.sm_partition_step = sm_partition_step
        # 根据经验测试值设定, 每个io task(chunk粒度)需要多少sm
        # 对于通信而言，最大的sm数是32
        self.sms_per_io_task = sms_per_io_task
        # A dictionary mapping the SM count of a partition to its stream
        self.sm_count_to_stream: Dict[torch.device, Dict[int, torch.cuda.Stream]] = {}
        
        for device in self.devices:
            if not torch.cuda.is_available() or device.type != 'cuda':
                raise ValueError(f"Device {device} is not a valid CUDA device.")
            
            # 初始化数据结构
            self.per_device_streams[device] = {}
            self.per_device_sorted_sm_counts[device] = []
            self.per_device_total_sm[device] = torch.cuda.get_device_properties(device).multi_processor_count
            
            # 为该设备创建流
            self._init_stream_pool_for_device(device)

    def _init_stream_pool_for_device(self, device: torch.device):
        """
        Creates the pool of SM-partitioned streams for a specific device.
        """
        total_sm = self.per_device_total_sm[device]
        logger.info(f"Initializing stream pool on device {device} with {total_sm} total SMs.")
        
        sm_partitions = partition_sms(total_sm, self.min_sm_partition, self.sm_partition_step) #TODO 改进划分方法。
        
        if not sm_partitions:
            logger.warning(f"Could not create any SM partitions on device {device} with the given configuration.")
            return

        try:
            streams, _ = green_ctx.split_device_green_ctx_by_sm_count(device, sm_partitions)
            
            for sm_count, stream in zip(sm_partitions, streams):
                self.per_device_streams[device][sm_count] = stream
                
            self.per_device_sorted_sm_counts[device] = sorted(self.per_device_streams[device].keys())
            logger.info(f"Successfully created {len(self.per_device_sorted_sm_counts[device])} streams on {device} with SM counts: {self.per_device_sorted_sm_counts[device]}")

        except Exception as e:
            logger.error(f"Failed to initialize flashinfer green contexts on {device}: {e}")
            raise

    def get_matched_stream(self, device: torch.device, io_count: int) -> Optional[torch.cuda.Stream]:
        """
        Gets the most suitable stream for a given number of I/O tasks on a specific device.

        Args:
            device (torch.device): The device for which to retrieve a stream.
            io_count (int): The number of concurrent I/O requests to be processed.

        Returns:
            Optional[torch.cuda.Stream]: The best-matched CUDA stream, or None.
        """
        if io_count <= 0:
            return None
        
        # 根据 device 参数查找对应的流池
        sorted_sm_counts = self.per_device_sorted_sm_counts.get(device)
        sm_count_to_stream = self.per_device_streams.get(device)
        
        if not sorted_sm_counts or not sm_count_to_stream:
            logger.error(f"No streams available in the pool for device {device}. Cannot match a stream.")
            return None

        required_sms = io_count * self.sms_per_io_task

        for sm_count in sorted_sm_counts:
            if sm_count >= required_sms:
                logger.debug(f"Device {device}: Matched {io_count} tasks (needs {required_sms} SMs) to stream with {sm_count} SMs.")
                return sm_count_to_stream[sm_count]
        
        largest_sm_count = sorted_sm_counts[-1]
        logger.warning(
            f"Device {device}: Required SMs ({required_sms}) exceeds the largest partition ({largest_sm_count}). "
            f"Returning the largest available stream."
        )
        return sm_count_to_stream[largest_sm_count]
    
class Simulator:
    def __init__(self, model : ModelConfig, comm_io_bandwidth, chunk_layer_size, streamController : StreamController, tp_size, dp_size,
                 chunk_size_bytes, storage_io_bandwidth_Bps, num_layers, num_heads, hidden_size, gpu_tflops):
        self.model = model
        self.comm_io_bandwidth = comm_io_bandwidth 
        self.chunk_layer_size = chunk_layer_size # 一个chunk一层的数据大小
        self.streamController = streamController 
        self.tp_size = tp_size
        self.dp_size = dp_size
        self._chunk_size_bytes = chunk_size_bytes
        self.storage_io_bandwidth_Bps = storage_io_bandwidth_Bps
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.gpu_tflops = gpu_tflops

        self.compute_efficiency = 0.5

    def _estimate_prefill_comm_time_ms(self, num_tokens: int) -> float:
        bytes_per_input_id = 8  # torch.int64

        bytes_per_position = 8 # torch.int64

        # 总字节数 = (input_ids 大小 + positions 大小)
        # 我们假设还有一些其他元数据，给一个 1.2 的系数作为缓冲
        total_bytes_to_transfer = (num_tokens * (bytes_per_input_id + bytes_per_position)) * 1.2

        if self.comm_io_bandwidth == 0:
            return 0.0

        time_seconds = total_bytes_to_transfer / self.comm_io_bandwidth

        time_ms = time_seconds * 1000

        return time_ms

    def _estimate_decode_step_time_ms(self, num_tokens: int) -> float:
        attn_flops = 8 * self.hidden_size * self.hidden_size # 暂时忽略了 softmax耗时

        ffn_flops = 2 * self.hidden_size * (4 * self.hidden_size) + 2 * self.hidden_size * 4 * self.hidden_size

        layer_flops = attn_flops + ffn_flops

        total_flops = num_tokens * self.num_layers * layer_flops

        gpu_flops_per_s = self.gpu_tflops * 1e12 * self.compute_efficiency

        est_time = total_flops / gpu_flops_per_s * 1000
        return est_time

    def execute(self, num_tokens: int, device: torch.device, stage: str = "prefill") -> Tuple[int, Optional[torch.cuda.Stream]]:
        if stage == "prefill": 
            overlap_window_ms = self._estimate_prefill_comm_time_ms(num_tokens) 
        elif stage == "decode": 
            overlap_window_ms = self._estimate_decode_step_time_ms(num_tokens) 
        else: 
            raise ValueError(f"Invalid stage: {stage}")
        if overlap_window_ms <= 0:             
            return 0, None

        overlap_window_s = overlap_window_ms / 1000.0         
        total_bytes_to_write = overlap_window_s * self.storage_io_bandwidth_Bps
        io_count = math.floor(total_bytes_to_write / self._chunk_size_bytes)
        stream = self.streamController.get_matched_stream(device, io_count)  
        return [io_count, stream]
