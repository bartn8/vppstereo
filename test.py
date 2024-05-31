from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.psmnet import *
from models.raft_stereo import *
from models.rsgm import *

import cmapy

from vpp_standalone import vpp
from filter import occlusion_heuristic

from losses import *
import cv2

from dataloaders.datasets import fetch_dataloader
import tqdm

from utils import color_error_image_kitti, guided_visualize

parser = argparse.ArgumentParser(description='Virtual Pattern Projection (VPP) - Test script')

parser.add_argument('--maxdisp', type=int ,default=192, help='maxium disparity')
parser.add_argument('--stereomodel', default='raft-stereo', help='select model')
parser.add_argument('--datapath', default='dataset/oak_dataset/', help='datapath')
parser.add_argument('--dataset', default='middlebury', help='dataset type')
parser.add_argument('--outdir', default=None)           
parser.add_argument('--loadstereomodel', required=True, help='load stereo model')             
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--iscale', type=int, default=1, help='Downsampling factor for input images')
parser.add_argument('--oscale', type=int, default=1, help='Downsampling factor for groundtruth')
parser.add_argument('--vpp', action='store_true')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--wsize', type=int, default=3, help='Patch size')
parser.add_argument('--wsizeAgg_x', type=int, default=64, help='Patch width for histogram pattern search')
parser.add_argument('--wsizeAgg_y', type=int, default=3, help='Patch height for histogram pattern search')
parser.add_argument('--guideperc', type=float, default=0.05, help='Sampling percentage for datasets without raw hints')
parser.add_argument('--blending', type=float, default=0.4, help='Pattern alpha blending')
parser.add_argument('--valsize', type=int, default=0, help='validation max size (0=unlimited)')
parser.add_argument('--normalize', action='store_true', help="Normalize RAFT-Stereo input between [-1,1] instead of [0,1]")
parser.add_argument('--maskocc', action='store_true', help='Use occlusion mask during virtual projection')
parser.add_argument('--cblending', type=float, default=0.0, help='Pattern alpha blending on occluded points')
parser.add_argument('--discard_occ', action='store_true', help='Discard occluded points')
parser.add_argument('--r2l', action='store_true', help="Right to left virtual projection")
parser.add_argument('--colormethod', default='rnd', choices=['rnd', 'maxDistance'], help='Virtual pattering method (random (vi) or histogram based (vii))')
parser.add_argument('--uniform_color', action='store_true', help='Enable patches with uniform color')
parser.add_argument('--guided', action='store_true', help='Use Guided Stereo Matching strategy')
parser.add_argument('--tries', type=int, default=1, help='Repeat the experiment n times and returns the mean results')

#TPAMI new arguments
parser.add_argument('--bilateralpatch', action='store_true', help='Use adaptive patch based on bilateral filter')
parser.add_argument('--bilateral_spatial_variance', type=float, default=1, help='Spatial variance of the adaptive patch')
parser.add_argument('--bilateral_color_variance', type=float, default=2, help='Color variance of the adaptive patch')
parser.add_argument('--bilateral_threshold', type=float, default=.001, help='Adaptive patch classification threshold')
parser.add_argument('--distancepatch', action='store_true', help='Use distance-based patch size')
parser.add_argument('--distance_gamma', type=float, default=.3, help='Distance-based patch size curve hyper-parameter')

parser.add_argument('--csv_path', default=None, help='Save experiment\'s results in a CSV file')
parser.add_argument('--search_header', action='store_true', help='Store search arguments inside CSV file')
parser.add_argument('--verbose', action='store_true', help='Show intermediate results during experiment')
parser.add_argument('--errormetric', default='bad 3.0', choices=['bad 1.0', 'bad 2.0', 'bad 3.0', 'bad 4.0', 'avgerr', 'rms'], help='Metric used for errormap\'s text')
parser.add_argument('--dilation', type=int, default=1, help='Use dilation for saved results (hints, errormaps, ...)')

parser.add_argument('--rsgm_subpixel', action='store_true', help='Enable RSGM with subpixel disparities to reproduce TPAMI experiments.')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

stereonet = None

if args.stereomodel == 'psmnet':
    stereonet = PSMNet(args.maxdisp)
elif args.stereomodel == 'raft-stereo':
    stereonet = RAFTStereo(args)
elif args.stereomodel == 'rsgm':
    stereonet = None
else:
    print('no model')
    exit()

if stereonet is not None:
    if args.cuda:
        stereonet = nn.DataParallel(stereonet)
        stereonet.cuda()
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print('Load pretrained stereo model')

    if args.stereomodel in ['psmnet']:
        state_dict = torch.load(args.loadstereomodel, map_location=device)
        stereonet.load_state_dict(state_dict['state_dict'], strict=False)
    elif args.stereomodel == 'raft-stereo':
        state_dict = torch.load(args.loadstereomodel, map_location=device)
        state_dict  = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
        stereonet.load_state_dict(state_dict)
    else:
        print('no model')
        exit()
else:
    device = torch.device('cpu')

@torch.no_grad()
def run(data):
    if stereonet is not None:
        stereonet.eval()

    # GENERATE PATTERN   
    if 'hints' not in data:   
        if 'raw' not in data:
            data['hints'], data['validhints'] = sample_hints(data['gt'], data['validgt']>0, probability=args.guideperc)  
        else:
            data['hints'], data['validhints'] = sample_hints(data['raw'], data['validraw']>0, probability=args.guideperc) 

    if args.stereomodel != 'rsgm' and args.iscale != 1:
        data['im2'] = F.interpolate(data['im2'], scale_factor=1./args.iscale)
        data['im3'] = F.interpolate(data['im3'], scale_factor=1./args.iscale)
        data['hints'] = F.interpolate(data['hints'], scale_factor=1./args.iscale, mode='nearest') / args.iscale
        data['validhints'] = F.interpolate(data['validhints'], scale_factor=1./args.iscale, mode='nearest')
    elif args.stereomodel == 'rsgm' and args.iscale != 1:
        data_im2 = []
        data_im3 = []

        for b in range(args.batch_size):
            h,w = data['im2'].shape[-2:]
            data_im2.append(torch.from_numpy(cv2.resize((255.0*data['im2'][b]).permute(1,2,0).cpu().numpy().astype(np.uint8), (w//args.iscale,h//args.iscale))).permute(2,0,1).unsqueeze(0)/255.)
            data_im3.append(torch.from_numpy(cv2.resize((255.0*data['im3'][b]).permute(1,2,0).cpu().numpy().astype(np.uint8), (w//args.iscale,h//args.iscale))).permute(2,0,1).unsqueeze(0)/255.)
            
        data['im2'] = torch.cat(data_im2, 0).to(device)
        data['im3'] = torch.cat(data_im3, 0).to(device)        
        data['hints'] = F.interpolate(data['hints'], scale_factor=1./args.iscale, mode='nearest') / args.iscale
        data['validhints'] = F.interpolate(data['validhints'], scale_factor=1./args.iscale, mode='nearest')

    if args.oscale != 1:
        data['gt'] = F.interpolate(data['gt'], scale_factor=1./args.oscale, mode='nearest') / args.oscale
        data['validgt'] = F.interpolate(data['validgt'], scale_factor=1./args.oscale, mode='nearest')

    mask_occ = occlusion_heuristic(data['hints'][0,0].numpy())[1] if args.maskocc else None
    
    wsize = args.wsize
    
    data['im2_blended'], data['im3_blended'] = vpp( (255*data['im2'][0].permute(1,2,0).numpy()).astype(np.uint8),
                                                    (255*data['im3'][0].permute(1,2,0).numpy()).astype(np.uint8),
                                                    data['hints'][0,0].numpy(),
                                                    blending=args.blending,
                                                    use_distance_patch=args.distancepatch,
                                                    distance_gamma=args.distance_gamma,
                                                    use_bilateral_patch=args.bilateralpatch,
                                                    bilateral_o_xy=args.bilateral_spatial_variance,
                                                    bilateral_o_i=args.bilateral_color_variance,
                                                    bilateral_th=args.bilateral_threshold,
                                                    wsize=wsize,
                                                    wsizeAgg_x=args.wsizeAgg_x,
                                                    wsizeAgg_y=args.wsizeAgg_y,
                                                    c_occ=args.cblending,
                                                    g_occ=mask_occ,
                                                    left2right=(not args.r2l),
                                                    method=args.colormethod,
                                                    uniform_color=args.uniform_color,
                                                    discard_occ=args.discard_occ, )


    data['im2_blended'] = torch.from_numpy(data['im2_blended']/255.).permute(2,0,1).unsqueeze(0).float()
    data['im3_blended'] = torch.from_numpy(data['im3_blended']/255.).permute(2,0,1).unsqueeze(0).float()

    if args.cuda:
        data['im2'], data['im3'] = data['im2'].cuda(), data['im3'].cuda()
        data['im2_blended'], data['im3_blended'] = data['im2_blended'].cuda(), data['im3_blended'].cuda()
        data['hints'], data['validhints'] = data['hints'].cuda(), data['validhints'].cuda()
    
    ht, wt = data['im2'].shape[-2], data['im2'].shape[-1]

    if args.stereomodel in ['psmnet','raft-stereo']:
        pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
        pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
    
    if args.stereomodel != 'rsgm':
        _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        data['im2'] = F.pad(data['im2'], _pad, mode='replicate')
        data['im3'] = F.pad(data['im3'], _pad, mode='replicate')
        data['im2_blended'] = F.pad(data['im2_blended'], _pad, mode='replicate')
        data['im3_blended'] = F.pad(data['im3_blended'], _pad, mode='replicate')
        data['hints'] = F.pad(data['hints'], _pad, mode='replicate')
        data['validhints'] = F.pad(data['validhints'], _pad, mode='replicate')

    if args.vpp:
        data['im2_vpp'], data['im3_vpp'] = data['im2_blended'], data['im3_blended']
    else:
        data['im2_vpp'], data['im3_vpp'] = data['im2'], data['im3']

    inject_hints = data['hints'] if args.guided else None
    inject_validhints = data['validhints'] if args.guided else None

    if args.stereomodel == 'rsgm':
        pred_disps_list = []
        for b in range(args.batch_size):
            left = (data['im2'][b].permute(1,2,0).cpu().numpy()*255.).astype(np.uint8)
            left_vpp = (data['im2_vpp'][b].permute(1,2,0).cpu().numpy()*255.).astype(np.uint8)
            right_vpp = (data['im3_vpp'][b].permute(1,2,0).cpu().numpy()*255.).astype(np.uint8)
            
            if inject_hints is not None:
                tmphints = inject_hints[b,0].cpu().numpy()
                tmpvalidhints = inject_validhints[b,0].cpu().numpy()
            else:
                tmphints,tmpvalidhints = 2*[None]

            dmap = compute_rsgm(left, left_vpp, right_vpp, hints=tmphints, validhints=tmpvalidhints, dmax=args.maxdisp, subpixel=args.rsgm_subpixel)
            pred_disps_list.append(torch.from_numpy(dmap).unsqueeze(0).unsqueeze(0))
        pred_disps = torch.cat(pred_disps_list, 0)
    elif args.stereomodel == 'raft-stereo':
        _,pred_disps = stereonet(data['im2'], data['im2_vpp'], data['im3_vpp'], 
        test_mode=True, iters=32, normalize=args.normalize,
        hints=inject_hints, validhints=inject_validhints)
    elif args.stereomodel == 'psmnet':
        pred_disps = stereonet(im2=data['im2_vpp'], im3=data['im3_vpp'], hints=inject_hints, validhints=inject_validhints)
    else:
        pred_disps = stereonet(data['im2_vpp'], data['im3_vpp'])
    
    if args.stereomodel in ['psmnet']:  
        pred_disp = pred_disps[0]
    elif args.stereomodel == 'raft-stereo':
        pred_disp = -pred_disps.squeeze(1)
    elif args.stereomodel == 'rsgm':
        pred_disp = pred_disps.squeeze(1)

    if args.stereomodel != 'rsgm':
        ht, wd = pred_disp.shape[-2:]
        c = [_pad[2], ht-_pad[3], _pad[0], wd-_pad[1]]
        pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]

    if args.iscale != 1 and args.iscale/args.oscale != 1:
        pred_disp = F.interpolate(pred_disp.unsqueeze(0), scale_factor=args.iscale/args.oscale, mode='nearest').squeeze(0) * args.iscale / args.oscale

    result = {}
    if 'gt' in data:
        result = guided_metrics(pred_disp.cpu().numpy(), data['gt'].numpy(), data['validgt'].numpy())

    result['disp'] = pred_disp
    result['im2_vpp'] = 255*data['im2_vpp'].squeeze().permute(1,2,0)
    result['im3_vpp'] = 255*data['im3_vpp'].squeeze().permute(1,2,0)

    return result

def write_csv_header(file, args, metrics):
    if args.search_header:
        header = "VPP,GUIDED,WSIZE,BLENDING,COLORMETHOD,UNIFORM_COLOR,BILATERALPATCH,DISTANCEPATCH,BILATERAL_XY,BILATERAL_I,BILATERAL_TH,DISTANCE_GAMMA,MASKOCC,DISCARD_OCC,"
    else:
        header = "VPP,GUIDED,GUIDEPERC,DATASET,DATAPATH,STEREOMODEL,STEREOMODEL_PATH,TRIES,ISCALE,MAXDISP,BLENDING,NORMALIZE,"
    keys = list(metrics.keys())
    for k in keys[:-1]:
        header += f"{k.upper()},"
    header += f"{keys[-1].upper()}\n"

    file.write(header)

def write_csv_row(file, args, metrics):
    if args.search_header:
        parameters = f'{args.vpp},{args.guided},{args.wsize},{args.blending},{args.colormethod},{args.uniform_color},{args.bilateralpatch},{args.distancepatch},{args.bilateral_spatial_variance},{args.bilateral_color_variance},{args.bilateral_threshold},{args.distance_gamma},{args.maskocc},{args.discard_occ},'
    else:
        parameters = f"{args.vpp},{args.guided},{args.guideperc},{args.dataset},{args.datapath},{args.stereomodel},{args.loadstereomodel},{args.tries},{args.iscale},{args.maxdisp},{args.blending},{args.normalize},"
    keys = list(metrics.keys())
    for k in keys[:-1]: 
        if 'bad' not in k:
            parameters += f"{metrics[k]:.2f},"
        else:
            parameters += f"{metrics[k]*100:.2f},"
    
    if 'bad' not in keys[-1]:
        parameters += f"{metrics[keys[-1]]:.2f}\n"
    else:
        parameters += f"{metrics[keys[-1]]*100:.2f}\n"

    file.write(parameters)

def main():
    args.test = True
    args.batch_size = 1
    demo_loader = fetch_dataloader(args)
    
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    ## demo ##
    acc_list = []
    
    for asd in range(args.tries):

        acc = {}
        pbar = tqdm.tqdm(total=len(demo_loader))
        val_len = min(len(demo_loader), args.valsize) if args.valsize > 0 else len(demo_loader)
        for batch_idx, datablob in enumerate(demo_loader):
            if batch_idx >= val_len:
                break

            result = run(datablob)

            if args.outdir is not None and asd == 0:
                for dirname in ['dmap', 'left', 'right', 'maemap', 'hints', 'metricmap']:
                    if not os.path.exists(os.path.join(args.outdir, dirname)):
                        os.mkdir(os.path.join(args.outdir, dirname))

                max_val = torch.where(torch.isinf(datablob['gt'][0]), -float('inf'), datablob['gt'][0]).max()

                myleft = cv2.cvtColor(result['im2_vpp'].squeeze().detach().cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.outdir, "left", '%s.png'%(batch_idx)), myleft)
                myright = cv2.cvtColor(result['im3_vpp'].squeeze().detach().cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.outdir, "right", '%s.png'%(batch_idx)), myright)

                mygt = cv2.applyColorMap(((torch.clamp(datablob['gt'][0,0],0,max_val)/max_val*255).detach().cpu().numpy()).astype(np.uint8), cmapy.cmap('magma'))
                myhints = cv2.applyColorMap(((torch.clamp(datablob['hints'][0,0],0,max_val)/max_val*255).detach().cpu().numpy()).astype(np.uint8), cmapy.cmap('magma'))
                maemap = color_error_image_kitti(torch.abs(datablob['gt'].cpu()-result['disp'].cpu()).squeeze().numpy(), scale=1, mask=datablob['gt']>0, dilation=args.dilation)
                metricmap = guided_visualize(result['disp'].squeeze().cpu().numpy(), datablob['gt'].squeeze().cpu().numpy(), datablob['gt'].squeeze().cpu().numpy()>0, dilation=args.dilation)[args.errormetric]

                if args.dilation>0:
                    kernel = np.ones((args.dilation, args.dilation))
                    kernelhints = np.ones((5, 5))
                    mygt = cv2.dilate(mygt, kernel)
                    maemap = cv2.dilate(maemap, kernel)
                    metricmap = cv2.dilate(metricmap, kernel)
                    myhints = cv2.dilate(myhints, kernelhints)

                
                cv2.imwrite(os.path.join(args.outdir, "hints", '%s.png'%(batch_idx)), myhints)
                cv2.imwrite(os.path.join(args.outdir, "maemap", '%s.png'%(batch_idx)), maemap)
                cv2.imwrite(os.path.join(args.outdir, "metricmap", '%s.png'%(batch_idx)), metricmap)

                mydmap = cv2.applyColorMap(((torch.clamp(result['disp'][0],0,max_val)/max_val*255).detach().cpu().numpy()).astype(np.uint8), cmapy.cmap('magma'))
                cv2.imwrite(os.path.join(args.outdir, "dmap", '%s.png'%(batch_idx)), mydmap)                

            for k in result:
                if k != 'disp' and k!= 'errormap' and k != 'im2_vpp' and k != 'im3_vpp':
                    if k not in acc:
                        acc[k] = []
                    acc[k].append(result[k])
                    
                    if args.verbose:
                        print(f"{batch_idx}) {k}: {result[k]}")

            pbar.update(1)
        pbar.close()

        acc_list.append(acc)

    #print("Demo complete! Bye!")
    
    #print(["%s: %.4f "%(k,np.array(acc[k]).mean()) for k in acc])
    #print(acc)

    acc_mean = {}
    acc_std = {}

    for acc in acc_list:
        for k in acc:
            if k not in acc_mean:
                acc_mean[k] = []
            if k not in acc_std:
                acc_std[k] = []
            
            acc_mean[k].append(np.array(acc[k]).mean())
            acc_std[k].append(np.array(acc[k]).mean())
    
    for k in acc_mean:
        acc_mean[k] = np.mean(acc_mean[k])
        acc_std[k] = np.std(acc_std[k])

    print("MEAN Metrics:")

    metrs = ''
    for k in acc_mean:
        metrs += f" {k.upper()} &"
    print(metrs)

    metrs = ''
    for k in acc_mean:
            if 'bad' not in k:
                metrs += f" {acc_mean[k]:.2f} &"
            else:
                metrs += f" {acc_mean[k]*100:.2f} &"

    print(metrs)

    print("STD Metrics:")

    metrs = ''
    for k in acc_std:
            if 'bad' not in k:
                metrs += f" {acc_std[k]:.2f} &"
            else:
                metrs += f" {acc_std[k]*100:.2f} &"

    print(metrs)

    if args.csv_path is not None:
        if os.path.exists(args.csv_path):
            csv_file = open(args.csv_path, "a")
        else:
            csv_file = open(args.csv_path, "w")
            write_csv_header(csv_file, args, acc_mean)
        
        write_csv_row(csv_file, args, acc_mean)

        csv_file.close()


if __name__ == '__main__':
   main()
    
