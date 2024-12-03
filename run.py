import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from config import get_config
from unet import UNet


config = get_config()
unet_model = UNet(
    input_irreps=config.input_irreps,
    output_irreps=config.output_irreps,
    diameter=config.diameter,
    num_radial_basis=config.num_radial_basis,
    steps=config.steps,
    n_downsample=config.n_downsample,
    scalar_upsampling=config.scalar_upsampling,
).to(config.device)

def loss(output, target):
    return torch.nn.functional.mse_loss(output, target)

def train_loop(dataloader, model, loss_fn, optimizer, num_epochs=10):
    total_loss = np.inf
    pbar = tqdm.tqdm(range(num_epochs), desc=f'Loss = {total_loss:.6f} | Epochs')
    for _ in pbar:
        total_loss = 0
        for data in dataloader:
            pred = model(data)
            loss = loss_fn(pred, data)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_description(f'Loss = {total_loss:.6f} | Epochs')


f = 'Head_VolumeTemplate_MMRR-21.nii.gz'
img1 = nib.load(f)
img1_data = img1.get_fdata().astype(np.float32)
img1_data = img1_data[:20, :20, :20]  # for memory reasons :/
img1_data = np.expand_dims(img1_data, axis=0)
img1_data = np.expand_dims(img1_data, axis=0)
img1_data = torch.from_numpy(img1_data).to(config.device)
dataloader = DataLoader(img1_data, batch_size=1, shuffle=True)

# (batch, irreps_in.dim, x, y, z)
# print(unet_model.down.down_blocks)
# print(unet_model.up.up_blocks)

train_loop(dataloader, unet_model, loss, torch.optim.Adam(unet_model.parameters()), num_epochs=20)