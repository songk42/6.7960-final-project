"""Defines the default training configuration."""

from e3nn.o3 import Irreps
import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Get the default training configuration."""
    config = ml_collections.ConfigDict()

    config.device="cuda"
    config.batch_size = 64

    config.unet = ml_collections.ConfigDict()
    config.unet.input_irreps='1x0e'
    config.unet.output_irreps='0e'
    config.unet.diameter=5.0
    config.unet.num_radial_basis=5
    config.unet.steps=(1, 1, 1)
    config.unet.n=2
    config.unet.n_downsample=3
    config.unet.batch_norm='instance'
    config.unet.equivariance = 'SO3'
    config.unet.lmax = 2
    config.unet.down_op = 'maxpool3d'
    config.unet.scale = 2
    config.unet.is_bias = True
    config.unet.dropout_prob=0

    return config
