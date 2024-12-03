"""Defines the default training configuration."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Get the default training configuration."""
    config = ml_collections.ConfigDict()

    config.device="cuda"
    config.batch_size = 64

    config.input_irreps='1x0e'
    config.output_irreps='0e'
    config.diameter=5.0
    config.num_radial_basis=5
    config.steps=(1, 1, 1)
    config.n=2
    config.n_downsample=3
    config.scalar_upsampling = False

    return config
