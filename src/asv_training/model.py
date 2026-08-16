"""Compatibility import for the packaged policy architecture.

The ROS runtime uses :mod:`asv_vla.policy_model` directly so an installed
Jetson package does not depend on a checkout-level ``training`` directory.
Keep this module for existing PC training scripts and tests.

# Compatibility marker retained for the v1 configuration contract:
# entity_attention_mode: str = "legacy"
# language_conditioned_entity_attention: bool = False
"""

from asv_vla.policy_model import (
    PolicyOutput,
    SmallActionPolicy,
    SmallPolicyConfig,
)

__all__ = [
    "PolicyOutput",
    "SmallActionPolicy",
    "SmallPolicyConfig",
]
