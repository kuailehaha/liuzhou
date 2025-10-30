"""
Neural-network interfaces for the tensorized pipeline.

Reuses the existing ChessNet architecture but augments the input/output glue to
handle batched tensorized states and flat action encodings.
"""

__all__ = ["encoding", "policy_decoder"]

