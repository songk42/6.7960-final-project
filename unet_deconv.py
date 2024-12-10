import math
from functools import partial
import torch
from torch import nn
import e3nn
from e3nn.nn import BatchNorm, Gate, Dropout
from e3nn.o3 import Irreps, Linear, ToS2Grid, FullyConnectedTensorProduct, xyz_to_angles
import numpy as np
from nnunet.network_architecture.neural_network import SegmentationNetwork
from e3nn.nn.models.v2104.voxel_convolution import Convolution
from unet_pooling import DynamicPool3d
from s2point import ToS2Point


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

    def __init__(self, n_downsample,activation,irreps_hidden,irreps_sh,ne,no,BN,input,diameters,num_radial_basis,steps,down_op,scale,dropout_prob):
        super().__init__()

        blocks = []
        self.down_irreps_out = []

        for n in range(n_downsample+1):
            irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e +  {2*no}x1o + {ne}x2e + {no}x2o").simplify()
            block = ConvolutionBlock(input,irreps_hidden,activation,irreps_sh,BN, diameters[n],num_radial_basis,steps[n],dropout_prob)
            # block = ConvolutionBlock(input,irreps_hidden[n],activation,irreps_sh,BN, diameters[n],num_radial_basis,steps[n],dropout_prob)
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
    def __init__(self, n_downsample,activation,irreps_hidden,irreps_sh,ne,no,BN,downblock_irreps,diameters,num_radial_basis,steps,scale,min_rad, max_rad, n_radii, res_beta, res_alpha,dropout_prob):
        '''we replace the upsample+conv combo with a deconvolution thingy'''
        super().__init__()

        self.n_blocks_up = n_downsample
        self.n_radii = n_radii
        self.radii = torch.linspace(min_rad, max_rad, n_radii)
        self.to_s2grid = ToS2Grid(lmax = irreps_sh.lmax, res=(res_beta, res_alpha))
        self.to_s2point = ToS2Point(
            lmax = irreps_sh.lmax, 
            res=(res_beta, res_alpha), 
            dtype=torch.float32, 
            normalization="integral",
            device="cuda"
        )
        self.lmax = irreps_sh.lmax

        self.irreps_hidden = irreps_hidden
        blocks = []
        blocks_pos = []
        blocks_rebase = []
        self.scale_factors = []

        for n in range(self.n_blocks_up):
            irreps_in = downblock_irreps[n]
            # irreps_hidden = self.irreps_hidden[n]
            irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e + {ne}x2e + {2*no}x1o + {no}x2o").simplify()
            conv = ConvolutionBlock(irreps_in,irreps_hidden,activation,irreps_sh,BN,diameters[n],num_radial_basis,steps[n],dropout_prob)
            # lin = Linear(irreps_in, irreps_hidden)  # TODO try out convolution?
            pos_linear = Linear(Irreps("1x1o"), Irreps("1x0e + 1x1o + 1x2e"))
            upsample_scale_factor = tuple([math.floor(scale[n]/step) if step < scale[n] else 1 for step in steps[n]]) #same as pooling kernel
            # TODO rn all terms are tensor producting with each other, but we want to keep voxels separate
            tp = FullyConnectedTensorProduct(irreps_hidden, Irreps("1x0e + 1x1o + 1x2e"), irreps_hidden)
            # tp = FullyConnectedTensorProduct(irreps_hidden, Irreps("1x0e"), irreps_sh * irreps_hidden.dim)
            blocks.append(conv)
            blocks_pos.append(pos_linear)
            blocks_rebase.append(tp)
            self.scale_factors.append(upsample_scale_factor)
            
            ne //= 2
            no //= 2

        self.up_blocks = nn.ModuleList(blocks)
        self.position_blocks = nn.ModuleList(blocks_pos)
        self.rebase_blocks = nn.ModuleList(blocks_rebase)

    
    def _coarse_to_fine(self, x, scale_factor):
        '''upsample x by scale_factor, return indices'''
        shape_3d = torch.tensor(x.shape[-3:], device=x.device)
        ndx_3d = torch.tensor(
            [[[[i, j, k] for i in range(shape_3d[0])] for j in range(shape_3d[1])] for k in range(shape_3d[2])]
        ).to(x.device)
        upsample = nn.Upsample(scale_factor=scale_factor)
        ndx_shifted = (ndx_3d + (shape_3d // 2).reshape(1, 1, 1, 3)).reshape(1, 1, *shape_3d, 3)
        ndx_new = torch.concat([
            (upsample(ndx_shifted[:, :, :, :, :, i].float())).unsqueeze(-1) for i in range(3)
        ], axis=-1)
        return ndx_new[0,0,]
    
    def _get_radii_and_angles(self, x, scale_factor):
        ndx_centers = self._coarse_to_fine(x, scale_factor)
        shape_3d = torch.tensor(ndx_centers.shape[:-1], device=x.device)
        ndx = torch.tensor(
            [[[[i, j, k] for i in range(shape_3d[0])] for j in range(shape_3d[1])] for k in range(shape_3d[2])]
        ).to(x.device)
        vecs = ndx - ndx_centers
        alphas, betas = xyz_to_angles(vecs)
        return vecs, torch.norm(vecs, dim=-1), alphas, betas
    
    def _get_closest(self, x, arr):
        '''get the index of the closest value in arr to x'''
        return (arr - x).abs().argmin()


    def forward(self, x):
        for n in range(self.n_blocks_up):
            x = self.up_blocks[n](x)
            # x = self.up_blocks[n](x.transpose(1, 4)).transpose(1, 4)
            vecs, radii, alphas, betas = self._get_radii_and_angles(x, self.scale_factors[n])
            # radii = radii.unsqueeze(0)

            upsample = nn.Upsample(scale_factor=self.scale_factors[n])
            x_upsampled = upsample(x)
            vecs_embed = self.position_blocks[n](vecs)
            print(x[0, 0])
            x = self.rebase_blocks[n](x_upsampled.transpose(1, 4), vecs_embed.unsqueeze(0)).transpose(1, 4)
            # # x_sh = self.rebase_blocks[n](x_upsampled.transpose(1, 4), radii.unsqueeze(-1))
            # print("sh", x_sh.shape)

            # # alpha_ndx = torch.vmap(
            # #     lambda a: self._get_closest(a, self.to_s2grid.alphas)
            # # )(alphas.flatten()).reshape(alphas.shape)
            # # beta_ndx = torch.vmap(
            # #     lambda a: self._get_closest(a, self.to_s2grid.betas)
            # # )(betas.flatten()).reshape(betas.shape)

            # def f(voxel, vec):
            #     x_grid = self.to_s2point(
            #         voxel.reshape((*voxel.shape[:-1], -1, (self.lmax + 1)**2)),
            #         vec,
            #     )
            #     # x_grid = self.to_s2grid(voxel.reshape((*voxel.shape[:-1], -1, (self.lmax + 1)**2)))  # [..., res_beta, res_alpha]
            #     return x_grid.squeeze()
            # # def g(grid):
            # #     out = torch.zeros(grid.shape[:3] + (self.irreps_hidden[n].dim,), device=grid.device)
            # #     for i in range(grid.shape[0]):
            # #         for j in range(grid.shape[1]):
            # #             for k in range(grid.shape[2]):
            # #                 out[i,j,k] = f(grid[i,j,k:k+1], vecs[i,j,k])
            # #     return out

            # print("voxel shape:", (*x_sh[0,0,0,0].shape[:-1], -1, (self.lmax + 1)**2))
            # x = torch.vmap(
            #     lambda square, vec_square: torch.vmap(
            #         lambda row, vec_row: torch.vmap(
            #             lambda voxel, vec: f(voxel, vec))(row, vec_row)
            #     )(square, vec_square)
            # )(x_sh[0], vecs).unsqueeze(0)
            # # print(x_sh.shape)
            # # x = g(x_sh[0]).unsqueeze(0)
            # print(x.shape)

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
            hidden_irreps,
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
            n_radii,
            res_beta,
            res_alpha
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
            hidden_irreps[1:],
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
            activation,
            hidden_irreps[::-1][1:],
            irreps_sh,
            ne,
            no,
            batch_norm,
            self.down.down_irreps_out[::-1],
            diameters[::-1][1:],
            num_radial_basis,
            steps_array[::-1][1:],
            scales[::-1],
            min_rad,
            max_rad,
            n_radii,
            res_beta,
            res_alpha,
            dropout_prob
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
