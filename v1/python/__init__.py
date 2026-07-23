"""V1 helpers with backend-specific imports resolved lazily.

Keeping package import side-effect free lets the portable CPU/MPS backend run on
machines where the optional ``v0_core`` extension is not installed.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "GpuStateBatch": (".mcts_gpu", "GpuStateBatch"),
    "V1RootMCTS": (".mcts_gpu", "V1RootMCTS"),
    "V1RootMCTSConfig": (".mcts_gpu", "V1RootMCTSConfig"),
    "self_play_v1_gpu": (".self_play_gpu_runner", "self_play_v1_gpu"),
    "train_network_from_tensors": (".train_bridge", "train_network_from_tensors"),
    "TensorSelfPlayBatch": (".trajectory_buffer", "TensorSelfPlayBatch"),
    "TensorTrajectoryBuffer": (".trajectory_buffer", "TensorTrajectoryBuffer"),
    "PortableMCTS": (".portable_mcts", "PortableMCTS"),
    "PortableMCTSConfig": (".portable_mcts", "PortableMCTSConfig"),
    "PortableTree": (".portable_mcts", "PortableTree"),
    "self_play_v1_portable": (".portable_self_play", "self_play_v1_portable"),
    "PortableCppMCTS": (".portable_cpp_mcts", "PortableCppMCTS"),
    "self_play_v1_portable_cpp": (
        ".portable_cpp_self_play",
        "self_play_v1_portable_cpp",
    ),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "GpuStateBatch",
    "TensorSelfPlayBatch",
    "TensorTrajectoryBuffer",
    "V1RootMCTS",
    "V1RootMCTSConfig",
    "self_play_v1_gpu",
    "train_network_from_tensors",
    "PortableMCTS",
    "PortableMCTSConfig",
    "PortableTree",
    "self_play_v1_portable",
    "PortableCppMCTS",
    "self_play_v1_portable_cpp",
]
