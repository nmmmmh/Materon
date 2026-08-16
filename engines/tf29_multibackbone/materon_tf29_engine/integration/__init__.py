"""Package initialization for integration in the TF2.9 multi-backbone engine."""

from .official_builder_patch import build_official_with_materon, inspect_builder_contract
from .weight_pairing import copy_backbone_weights, assert_backbone_parity
