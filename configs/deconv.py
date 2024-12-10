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
    config.unet.hidden_irreps=[  # only even parity for SO3 equivariance
        Irreps(f"1x0e + 1x1e + 1x2e"),
        Irreps(f"2x0e + 2x1e + 2x2e"),
        Irreps(f"4x0e + 4x1e + 4x2e"),
        Irreps(f"8x0e + 8x1e + 8x2e"),
    ]
    config.unet.output_irreps='0e'
    config.unet.diameter=5.0
    config.unet.num_radial_basis=5
    config.unet.steps=(1, 1, 1)
    config.unet.n=2
    config.unet.n_downsample=len(config.unet.hidden_irreps)-2  # TOdO this is so sus
    config.unet.batch_norm='instance'
    config.unet.equivariance = 'SO3'
    config.unet.lmax = 2
    config.unet.down_op = 'maxpool3d'
    config.unet.scale = 2
    config.unet.is_bias = True
    config.unet.dropout_prob=0
    config.unet.min_rad=0.0
    config.unet.max_rad=5.0
    config.unet.n_radii=64
    config.unet.res_beta=90
    config.unet.res_alpha=89

    return config
