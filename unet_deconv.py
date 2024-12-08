import math
from functools import partial
import torch
from torch import nn
from e3nn.nn import BatchNorm, Gate, Dropout
from e3nn.o3 import Irreps, Linear, ToS2Grid, FullyConnectedTensorProduct
import numpy as np
from nnunet.network_architecture.neural_network import SegmentationNetwork
from e3nn.nn.models.v2104.voxel_convolution import Convolution
from unet_pooling import DynamicPool3d


class ConvolutionBlock(nn.Module):
    def __init__(self, input, irreps_hidden, activation, irreps_sh, normalization,diameter,num_radial_basis,steps,dropout_prob):
        super().__init__()

        if normalization == 'None':
            BN = Identity
        elif normalization == 'batch':
            BN = BatchNorm
        elif normalization == 'instance':
            BN = partial(BatchNorm,instance=True)


        irreps_scalars = Irreps( [ (mul, ir) for mul, ir in irreps_hidden if ir.l == 0 ] )
        irreps_gated   = Irreps( [ (mul, ir) for mul, ir in irreps_hidden if ir.l > 0  ] )
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")

        #fe = sum(mul for mul,ir in irreps_gated if ir.p == 1)
        #fo = sum(mul for mul,ir in irreps_gated if ir.p == -1)
        #irreps_gates = Irreps(f"{fe}x0e+{fo}x0o").simplify()
        if irreps_gates.dim == 0:
            irreps_gates = irreps_gates.simplify()
            activation_gate = []
        else:
            activation_gate = [torch.sigmoid]
            #activation_gate = [torch.sigmoid, torch.tanh][:len(activation)]

        self.gate1 = Gate(irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated)
        self.conv1 = Convolution(input, self.gate1.irreps_in, irreps_sh, diameter,num_radial_basis,steps)
        self.batchnorm1 = BN(self.gate1.irreps_in)
        self.dropout1 = Dropout(self.gate1.irreps_out, dropout_prob)

        self.gate2 = Gate(irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated)
        self.conv2 = Convolution(self.gate1.irreps_out, self.gate2.irreps_in, irreps_sh, diameter,num_radial_basis,steps)
        self.batchnorm2 = BN(self.gate2.irreps_in)
        self.dropout2 = Dropout(self.gate2.irreps_out, dropout_prob)

        self.irreps_out = self.gate2.irreps_out

    def forward(self, x):
 
        x = self.conv1(x)
        x = self.batchnorm1(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate1(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout1(x.transpose(1, 4)).transpose(1, 4)

        x = self.conv2(x)
        x = self.batchnorm2(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate2(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout2(x.transpose(1, 4)).transpose(1, 4)
        return x

class Down(nn.Module):

    def __init__(self, n_downsample,activation,irreps_sh,ne,no,BN,input,diameters,num_radial_basis,steps,down_op,scale,dropout_prob):
        super().__init__()

        blocks = []
        self.down_irreps_out = []

        for n in range(n_downsample+1):
            irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e +  {2*no}x1o + {ne}x2e + {no}x2o").simplify()
            block = ConvolutionBlock(input,irreps_hidden,activation,irreps_sh,BN, diameters[n],num_radial_basis,steps[n],dropout_prob)
            blocks.append(block)
            self.down_irreps_out.append(block.irreps_out)
            input = block.irreps_out
            ne *= 2
            no *= 2

        self.down_blocks = nn.ModuleList(blocks)

        pooling = []
        for n in range(n_downsample):
            pooling.append(DynamicPool3d(scale[n],steps[n],down_op,self.down_irreps_out[n]))

        self.down_pool = nn.ModuleList(pooling)

    def forward(self, x):
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i < len(self.down_blocks)-1:
                x = self.down_pool[i](x)
        return x

class Up(nn.Module):
    def __init__(self, n_blocks_up,irreps_sh,ne,no,downblock_irreps,steps,scale,min_rad, max_rad, n_radii=64):
        '''we replace the upsample+conv combo with a deconvolution thingy'''
        super().__init__()

        self.n_blocks_up = n_blocks_up
        self.n_radii = n_radii
        self.radii = torch.linspace(min_rad, max_rad, n_radii)
        self.to_s2grid = ToS2Grid(lmax = irreps_sh.lmax)

        downblock_irreps_reversed = downblock_irreps[::-1]
        blocks = []
        blocks_rebase = []
        self.scale_factors = []

        for n in range(n_blocks_up):
            irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e + {ne}x2e + {2*no}x1o + {no}x2o").simplify()
            irreps_in = downblock_irreps_reversed[n]
            # FullyConnectedTensorProduct(irreps_in, irreps_sh, irreps_hidden)
            lin1 = Linear(irreps_in, irreps_hidden)
            lin2 = Linear(irreps_hidden, irreps_sh * self.n_radii * irreps_hidden.dim)
            blocks.append(lin1)
            blocks_rebase.append(lin2)
            upsample_scale_factor = tuple([math.floor(scale[n]/step) if step < scale[n] else 1 for step in steps[n]]) #same as pooling kernel
            self.scale_factors.append(upsample_scale_factor)
            
            ne //= 2
            no //= 2

        self.up_blocks = nn.ModuleList(blocks)
        self.rebase_blocks = nn.ModuleList(blocks_rebase)

    
    def _coarse_to_fine(self, x, scale_factor):
        '''upsample x by scale_factor, return indices'''
        shape_3d = x.shape[-2:]
        ndx_3d = torch.tensor(
            [[[[i, j, k] for i in range(shape_3d[0])] for j in range(shape_3d[1])] for k in range(shape_3d[2])]
        ).to(x.device)
        ndx_new = nn.functional.interpolate(
            ndx_3d + torch.ones(3, device=x.device) * shape_3d / 2,
            scale_factor,
            mode='nearest'
        )
        return ndx_new
    
    def _get_radii_and_angles(self, x, scale_factor):
        ndx_centers = self._coarse_to_fine(x, scale_factor)
        shape_3d = ndx_centers.shape[-2:]
        ndx = torch.tensor(
            [[[[i, j, k] for i in range(shape_3d[0])] for j in range(shape_3d[1])] for k in range(shape_3d[2])]
        ).to(x.device)
        return torch.norm(ndx - ndx_centers, dim=-1)
    
    def _get_closest_radius(self, r):
        '''get the closest radius to r in self.radii'''
        return (self.radii - r).abs().argmin()

    def forward(self, x):

        for i in range(self.n_blocks_up):
            x = self.up_blocks[i](x)
            x_sh = self.rebase_blocks[i](x)
            print(x_sh.shape)
            x_grid = self.to_s2grid(x_sh)
            print(x_grid.shape)
            radii = self._get_radii(x, self.scale_factors[i])
            closest_radii = torch.vmap(
                lambda r: self._get_closest_radius(r)
            )(radii.flatten()).reshape(radii.shape)

        return x

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        return x

class UNetDeconv(SegmentationNetwork):
    def __init__(
            self, 
            input_irreps, 
            output_irreps, 
            diameter, 
            num_radial_basis, 
            steps, 
            batch_norm, 
            n, 
            n_downsample, 
            equivariance,
            lmax, 
            down_op, 
            scale,
            is_bias,
            dropout_prob,
            min_rad,
            max_rad,
            n_radii
            ):
        """Equivariant UNet with physical units

        Parameters
        ----------
        input_irreps : str
            input representations
            example: "1x0e" when one channel of scalar values
        output_irreps : str
            output representations
            example: "4x0e" when four channels of scalar values
        n_classes_vector : int
            number of vector classes
        diameter : float
            diameter of input convolution kernel in physical units
        num_radial_basis : int
            number of radial basis functions
        steps : float
            physical dimension of a pixel in physical units
        batch_norm : str, optional
            normalization: can be 'batch', 'instance' or 'None'.
            by default 'instance'
        n : int, optional
            multiplication factor of number of irreps
            between successive convolution blocks, by default 2
        n_downsample : int, optional
            number of downsampling operations, by default 2
        equivariance : str, optional
            type of equivariance, can be 'O3' or 'SO3'
            by default 'SO3'
        lmax : int, optional
            maximum spherical harmonics l
            by default 2
        down_op : str, optional
            type of downsampling operation
            can be 'maxpool3d', 'average' or 'lowpass'
            by default 'maxpool3d'
        scale : int, optional
            size of pooling diameter
            in physical units, by default 2
        is_bias : bool, optional
            defines whether or not to add a bias, by default True
        scalar_upsampling : bool, optional
            flag to use scalar_upsampling, by default False
        dropout_prob : float, optional
            dropout probability between 0 and 1.0, by default 0

        """
        super().__init__()

        self.n_classes_scalar = Irreps(output_irreps).count('0e')
        self.num_classes = self.n_classes_scalar
        
        self.n_downsample = n_downsample
        self.conv_op = nn.Conv3d #Needed in order to use nnUnet predict_3D

        assert batch_norm in ['None','batch','instance'], "batch_norm needs to be 'batch', 'instance', or 'None'"
        assert down_op in ['maxpool3d','average','lowpass'], "down_op needs to be 'maxpool3d', 'average', or 'lowpass'"

        if down_op == 'lowpass':
            up_op = 'lowpass' 
            self.odd_resize = True
       
        else:
            up_op = 'upsample' 
            self.odd_resize = False

        if equivariance == 'SO3':
            activation = [torch.relu]
            irreps_sh = Irreps.spherical_harmonics(lmax, 1)
            ne = n
            no = 0
        elif equivariance == 'O3':
            activation = [torch.relu,torch.tanh]
            irreps_sh = Irreps.spherical_harmonics(lmax, -1)
            ne = n
            no = n
        scales = [scale*2**i for i in range(n_downsample)] #TODO change 2 to variable factor
        diameters = [diameter*2**i for i in range(n_downsample+1)] #TODO change 2 to variable factor

        steps_array = [steps]
        for i in range(n_downsample):
            
            output_steps = []
            for step in steps:
                if step < scales[i]:
                    kernel_dim = math.floor(scales[i]/step)
                    output_steps.append(kernel_dim*step)
                else:
                    output_steps.append(step)

            steps_array.append(tuple(output_steps))


        self.down = Down(
            n_downsample,
            activation,
            irreps_sh,
            ne,
            no,
            batch_norm,
            input_irreps,
            diameters,
            num_radial_basis,
            steps_array,
            down_op,
            scales,
            dropout_prob,
        )
        ne *= 2**(n_downsample-1)
        no *= 2**(n_downsample-1)
        self.up = Up(
            n_downsample,
            irreps_sh,
            ne,
            no,
            self.down.down_irreps_out,
            steps_array[::-1][1:],
            scales[::-1],
            min_rad,
            max_rad,
            n_radii
        )
        self.out = Linear(self.up.up_blocks[-1].irreps_out, output_irreps)

        if is_bias:
            #self.bias = nn.parameter.Parameter(torch.Tensor(n_classes_scalar))
            self.bias = nn.parameter.Parameter(torch.zeros(self.n_classes_scalar))
        else:
            self.register_parameter('bias', None)


    def forward(self, x):

        def pad_size(image_shape,odd):

            pooling_factor = np.ones(3,dtype='int')
            for pool in self.down.down_pool:
                pooling_factor *= np.array(pool.kernel_size)

            pad = [] 
            for f,s in zip(pooling_factor,image_shape):
                p = 0  #padding for current dimension
                if odd:
                    t = (s - 1) % f
                else:
                    t = s % f

                if t != 0:
                    p = f - t
                pad.append(p)
            
            return pad

        pad = pad_size(x.shape[-3:],self.odd_resize)
        x = torch.nn.functional.pad(x, (pad[-1], 0, pad[-2], 0, pad[-3], 0))

        down_output = self.down(x)
        x = self.up(down_output)
        x = self.out(x.transpose(1, 4)).transpose(1, 4)

        if self.bias is not None:
            bias = self.bias.reshape(-1, 1, 1, 1)
            x = torch.cat([x[:, :self.n_classes_scalar,...] + bias, x[:, self.n_classes_scalar:,...]], dim=1)
        
        x = x[..., pad[0]:, pad[1]:, pad[2]:]

        return x
