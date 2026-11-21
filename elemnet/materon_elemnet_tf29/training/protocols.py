from __future__ import absolute_import, division, print_function

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class TrainingProtocol:
    name: str
    input_scaler: str = "standard"
    target_scaling: str = "none"
    loss_name: str = "mae"
    huber_delta: float = 1.0
    learning_rate: float = 1.0e-3
    batch_size: int = 32
    epochs: int = 800
    patience: int = 80
    lr_patience: int = 25
    min_learning_rate: float = 1.0e-6
    clipnorm: float = 0.0
    # V14 formal evaluation uses the exact EarlyStopping-restored discovery
    # checkpoint. It does not restart from random initialization for a refit.
    formal_strategy: str = "discovery_checkpoint"
    refit_full_outer_train: bool = False
    refit_multiplier: float = 1.0
    min_refit_epochs: int = 20
    max_refit_epochs: int = 800

    def to_dict(self):
        return asdict(self)


def protocol_registry() -> Dict[str, TrainingProtocol]:
    return {
        "V14_P1_stdX_rawY_mae_b32_checkpoint": TrainingProtocol(
            name="V14_P1_stdX_rawY_mae_b32_checkpoint",
            input_scaler="standard",
            target_scaling="none",
            loss_name="mae",
            batch_size=32,
            formal_strategy="discovery_checkpoint",
            refit_full_outer_train=False,
        ),
        "P1_stdX_rawY_mae_b32": TrainingProtocol(
            name="P1_stdX_rawY_mae_b32"
        ),
        "P2_stdX_stdY_mae_b32": TrainingProtocol(
            name="P2_stdX_stdY_mae_b32",
            input_scaler="standard",
            target_scaling="standard",
            loss_name="mae",
            batch_size=32,
        ),
    }


def get_protocol(name: str) -> TrainingProtocol:
    values = protocol_registry()
    if name not in values:
        raise KeyError("Unknown protocol=%r; available=%r" % (name, sorted(values)))
    return values[name]
