import torch
import torch.nn.functional as F
from .utils.utils import bilinear_sampler, coords_grid

try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = corr_sampler.forward(volume, coords, radius)
        return corr
    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None

class CorrBlockFast1D:
    def __init__(self, fmap2, fmap3, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        # all pairs correlation
        corr = CorrBlockFast1D.corr(fmap2, fmap3)
        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, 1, w2)
        for i in range(self.num_levels):
            self.corr_pyramid.append(corr.view(batch, h1, w1, -1, w2//2**i))
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])

    def __call__(self, coords):
        out_pyramid = []
        bz, _, ht, wd = coords.shape
        coords = coords[:, [0]]
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i].squeeze(3), coords/2**i, self.radius)
            out_pyramid.append(corr.view(bz, -1, ht, wd))
        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap2, fmap3):
        B, D, H, W1 = fmap2.shape
        _, _, _, W2 = fmap3.shape
        fmap2 = fmap2.view(B, D, H, W1)
        fmap3 = fmap3.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap2, fmap3)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class PytorchAlternateCorrBlock1D:
    def __init__(self, fmap2, fmap3, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.fmap2 = fmap2
        self.fmap3 = fmap3

    def corr(self, fmap2, fmap3, coords):
        B, D, H, W = fmap3.shape
        # map grid coordinates to [-1,1]
        xgrid, ygrid = coords.split([1,1], dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        output_corr = []
        for grid_slice in grid.unbind(3):
            fmapw_mini = F.grid_sample(fmap3, grid_slice, align_corners=True)
            corr = torch.sum(fmapw_mini * fmap2, dim=1)
            output_corr.append(corr)
        corr = torch.stack(output_corr, dim=1).permute(0,2,3,1)

        return corr / torch.sqrt(torch.tensor(D).float())

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        fmap2 = self.fmap2
        fmap3 = self.fmap3
        out_pyramid = []
        for i in range(self.num_levels):
            dx = torch.zeros(1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)
            centroid_lvl = coords.reshape(batch, h1, w1, 1, 2).clone()
            centroid_lvl[...,0] = centroid_lvl[...,0] / 2**i
            coords_lvl = centroid_lvl + delta.view(-1, 2)
            corr = self.corr(fmap2, fmap3, coords_lvl)
            fmap3 = F.avg_pool2d(fmap3, [1, 2], stride=[1, 2])
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


class CorrBlock1D:
    def __init__(self, fmap2, fmap3, hints=None, validhints=None, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap2, fmap3, hints, validhints)
        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap2, fmap3, hints=None, validhints=None):
        B, D, H, W2 = fmap2.shape
        _, _, _, W3 = fmap3.shape
        fmap2 = fmap2.view(B, D, H, W2)
        fmap3 = fmap3.view(B, D, H, W3)
        corr = torch.einsum('aijk,aijh->ajkh', fmap2, fmap3)
        corr = corr.reshape(B, H, W2, 1, W3).contiguous()
        corr = corr / torch.sqrt(torch.tensor(D).float())

        if hints is not None:
            ### MODULATION ###
            GAUSSIAN_HEIGHT = 10.
            GAUSSIAN_WIDTH = 1.

            # image features are one fourth the original size: subsample the hints and divide them by four
            SUBSAMPLE = 4
            hints = F.upsample(hints, [hints.size()[2]//SUBSAMPLE, hints.size()[3]//SUBSAMPLE], mode='nearest').squeeze(1)
            validhints = F.upsample(validhints, [validhints.size()[2]//SUBSAMPLE, validhints.size()[3]//SUBSAMPLE], mode='nearest').squeeze(1)
            hints = hints*validhints / float(SUBSAMPLE)
            hints = coords_grid(hints.shape[0], hints.shape[1], hints.shape[2]).to(hints.device)[:,0]*validhints - hints
            #GAUSSIAN_WIDTH /= float(SUBSAMPLE)
            
            # add feature and disparity dimensions to hints and validhints
            # and repeat their values along those dimensions, to obtain the same size as cost
            hints = hints.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, W3)#(-1, features, num_disp, -1, -1)
            validhints = validhints.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, W3)#(-1, features, num_disp, -1, -1)

            # create a tensor of the same size as cost, with disparities
            # between 0 and num_disp-1 along the disparity dimension
            disparities = torch.linspace(start=0, end=W3 - 1, steps=W3).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, H, W2, 1, -1)#(batch_size, features, -1, height, width)
            corr = corr * ((1 - validhints) + validhints * GAUSSIAN_HEIGHT * torch.exp(-(disparities - hints) ** 2 / (2 * GAUSSIAN_WIDTH ** 2)))




        return corr 


class AlternateCorrBlock:
    def __init__(self, fmap2, fmap3, num_levels=4, radius=4):
        raise NotImplementedError
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap2, fmap3)]
        for i in range(self.num_levels):
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            fmap3 = F.avg_pool2d(fmap3, 2, stride=2)
            self.pyramid.append((fmap2, fmap3))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap2_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap3_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap2_i, fmap3_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
