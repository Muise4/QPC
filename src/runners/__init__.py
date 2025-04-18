REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .SSD_parallel_runner import SSD_ParallelRunner
REGISTRY["SSD_parallel"] = SSD_ParallelRunner

from .MAT_parallel_runner import MAT_ParallelRunner
REGISTRY["MAT_parallel"] = MAT_ParallelRunner