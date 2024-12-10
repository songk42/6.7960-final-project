from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from configs.deconv import get_config
from unet import UNet
from unet_deconv import UNetDeconv


def loss(output, target):
    return torch.nn.functional.mse_loss(output, target)

def train_loop(dataloader, model, loss_fn, optimizer, num_epochs=10):
    pbar = tqdm.tqdm(range(num_epochs), desc='Loss = N/A | Epochs')
    for _ in pbar:
        total_loss = 0
        for data in dataloader:
            pred = model(data)
            print(pred.shape, data.shape)
            loss = loss_fn(pred, data)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_description(f'Loss = {total_loss:.6f} | Epochs')


config = get_config()
unet_model = UNetDeconv(**config.unet).to(config.device)
# print(unet_model.down.down_blocks)
print(unet_model.up.up_blocks)

stripes_x = np.load("/home/songk/6.7960-final-project/data/stripes_x_32x32.npy")

input_img = stripes_x.astype(np.float32)
input_img = np.expand_dims(input_img, axis=0)
input_img = np.expand_dims(input_img, axis=0)
input_img = torch.from_numpy(input_img).to(config.device)
dataloader = DataLoader(input_img, batch_size=1, shuffle=True)

train_loop(dataloader, unet_model, loss, torch.optim.Adam(unet_model.parameters()), num_epochs=1)
