import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import BasicMultiUpdateBlock
from .extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from .corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from .utils.utils import coords_grid, upflow8


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.corr_implementation = "reg"
        self.slow_fast_gru = False
        self.mixed_precision = False
        self.shared_backbone = False
        self.n_downsample = 2
        self.corr_radius = 4
        self.corr_levels = 4
        self.n_gru_layers = 3
        
        context_dims = [128]*3

        self.cnet = MultiBasicEncoder(output_dim=[context_dims, context_dims], norm_fn="batch", downsample=self.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=context_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], context_dims[i]*3, 3, padding=3//2) for i in range(self.n_gru_layers)])

        if self.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=self.n_downsample)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_for_finetuning(self):
        for layer in [self.cnet, self.update_block, self.context_zqr_convs]:
            for param in layer.parameters():
                param.requires_grad = False
            

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask, n_downsample=2):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)


    def forward(self, image0, image2, image3, hints=None, validhints=None, iters=12, flow_init=None, test_mode=False, normalize=False):
        """ Estimate optical flow between pair of frames """

        if normalize:
            image0 = (2 * (image0) - 1.0).contiguous()
            image2 = (2 * (image2) - 1.0).contiguous()
            image3 = (2 * (image3) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.mixed_precision):
            if self.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image2, image3), dim=0), dual_inp=True, num_layers=self.n_gru_layers)
                fmap2, fmap3 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(image0, num_layers=self.n_gru_layers)
                fmap2, fmap3 = self.fnet([image2, image3])
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap2, fmap3 = fmap2.float(), fmap3.float()
        elif self.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap2, fmap3 = fmap2.float(), fmap3.float()
        elif self.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.corr_implementation == "alt_cuda": # Faster version of alt
            corr_block = AlternateCorrBlock
            
        if hints is not None and validhints is not None and self.corr_implementation == "reg":
            corr_fn = corr_block(fmap2, fmap3, hints=hints, validhints=validhints, radius=self.corr_radius, num_levels=self.corr_levels)
        else:
            corr_fn = corr_block(fmap2, fmap3, radius=self.corr_radius, num_levels=self.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                if self.n_gru_layers == 3 and self.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.n_gru_layers >= 2 and self.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.n_gru_layers==3, iter16=self.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
