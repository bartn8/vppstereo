# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data

import random
from glob import glob
import os.path as osp


from .frame_utils import read_gen, readPFM, readDispKITTI
from .noise import add_kinectv1_noise

class MiddleburyDataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False):
        self.augmentor = None
        self.is_test = test
        self.init_seed = False
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, '*/im0.png')))
        for i in range(len(image_list)):
            if (not self.is_test and ('ArtL' not in image_list[i] and 'Teddy' not in image_list[i])) or self.is_test:
                self.image_list += [ [image_list[i].replace('im0.png','disp0GT.pfm'), image_list[i], image_list[i].replace('im0', 'im1')] ]
                self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        if not self.init_seed and not self.is_test:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        data = {}

        data['im2'] = read_gen(self.image_list[index][1])
        data['im3'] = read_gen(self.image_list[index][2])
        
        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        if self.is_test:
            data['im2'] = data['im2'] / 255.0
            data['im3'] = data['im3'] / 255.0

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        data['gt'], data['validgt'] = [None]*2

        data['gt'] = np.expand_dims( readPFM(self.image_list[index][0]), -1)
        data['validgt'] = data['gt'] < 5000

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)
               
        
        if self.is_test:
            data['im2_aug'] = data['im2']
            data['im3_aug'] = data['im3']
        else:
            if self.augmentor is not None:
                augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
                data['im2'] = augm_data['im2']
                data['im3'] = augm_data['im3']
                data['im2_aug'] = augm_data['im2_aug']
                data['im3_aug'] = augm_data['im3_aug']
                data['gt'] = augm_data['gt']
                data['validgt'] = augm_data['validgt']
            

        data['im2f'] = np.ascontiguousarray(data['im2'][:,::-1])
        data['im3f'] = np.ascontiguousarray(data['im3'][:,::-1])

        for k in data:
            if data[k] is not None:
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float() 
        
        data['extra_info'] = self.extra_info[index]

        return data

    def __len__(self):
        return len(self.image_list)

class MiddleburyAdditionalDataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False):
        self.augmentor = None
        self.is_test = test
        self.init_seed = False
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, '*/im0.png')))
        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('im0', 'im1')] ]
            #self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('im0', 'im1E')] ]
            #self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('im0', 'im1L')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
            #self.extra_info += [ image_list[i].split('/')[-1] ]
            #self.extra_info += [ image_list[i].split('/')[-1] ]
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = read_gen(self.image_list[index][1])
            data['im3'] = read_gen(self.image_list[index][2])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]
            data['gt'] = np.expand_dims( readPFM(self.image_list[index][0].replace('im0.png','disp0.pfm')), -1)
            data['validgt'] = data['gt'] < 5000

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = read_gen(self.image_list[index][1])
        data['im3'] = read_gen(self.image_list[index][2])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'] = np.expand_dims( readPFM(self.image_list[index][0].replace('im0.png','disp0.pfm')), -1)
        data['validgt'] = data['gt'] < 5000

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)

class Middlebury2021Dataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False):
        self.augmentor = None
        self.is_test = test
        self.init_seed = False
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, '*/im0.png')))
        for i in range(len(image_list)):
            if (not self.is_test and ('ArtL' not in image_list[i] and 'Teddy' not in image_list[i])) or self.is_test:
                self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('im0', 'im1')] ]
                self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = read_gen(self.image_list[index][1])
            data['im3'] = read_gen(self.image_list[index][2])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]
            data['gt'] = np.expand_dims( readPFM(self.image_list[index][0].replace('im0.png','disp0.pfm')), -1)
            data['validgt'] = data['gt'] < 5000

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = read_gen(self.image_list[index][1])
        data['im3'] = read_gen(self.image_list[index][2])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'] = np.expand_dims( readPFM(self.image_list[index][0].replace('im0.png','disp0.pfm')), -1)
        data['validgt'] = data['gt'] < 5000

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)

class KITTI142Dataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False):
        self.augmentor = None
        self.is_test = test
        self.init_seed = False
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, 'image_2/*_10.png')))
        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('image_2', 'image_3')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = read_gen(self.image_list[index][1])
            data['im3'] = read_gen(self.image_list[index][2])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]
            data['gt'], data['validgt'] = readDispKITTI(self.image_list[index][0].replace('image_2','disp_occ'))
            data['hints'], data['validhints'] = readDispKITTI(self.image_list[index][0].replace('image_2','lidar_disp_2'))

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['hints'] = np.array(data['hints']).astype(np.float32)
            data['validhints'] = np.array(data['validhints']).astype(np.float32)
            
            data['hints'] = torch.from_numpy(data['hints']).permute(2, 0, 1).float()
            data['validhints'] = torch.from_numpy(data['validhints']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

    def __len__(self):
        return len(self.image_list)

class SimStereoDataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False):
        self.augmentor = None
        
        self.is_test = test
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, 'rgbColormanaged/*_left.jpg')))
        disp_list = sorted(glob(osp.join(datapath, 'pfmDisp/*_left.pfm')))

        for i in range(len(image_list)):
                self.image_list += [ [disp_list[i], image_list[i], image_list[i].replace('_left.jpg','_right.jpg')] ]
                self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):
        data = {}

        data['im2'] = read_gen(self.image_list[index][1])
        data['im3'] = read_gen(self.image_list[index][2])
        
        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        if self.is_test:
            data['im2'] = data['im2'] / 255.0
            data['im3'] = data['im3'] / 255.0

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        data['gt'], data['validgt'] = [None]*2

        gt = readPFM(self.image_list[index][0])

        data['gt'] = np.expand_dims(gt, -1)
        data['validgt'] = np.logical_and(data['gt'] < 5000, data['gt'] > 0)

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        baseline = 0.16
        focal = 888.89

        gt_depth = gt.copy()
        gt_depth[gt_depth>0] = (baseline*focal) / gt_depth[gt_depth>0]
        raw_depth = add_kinectv1_noise(gt_depth)
        raw_depth[raw_depth>0] = (baseline*focal) / raw_depth[raw_depth>0]

        data['raw'] = np.expand_dims(raw_depth, -1)
        data['validraw'] = np.logical_and(data['raw'] < 5000, data['raw'] > 0).astype(np.float32)
        
        if self.is_test:
            data['im2_aug'] = data['im2']
            data['im3_aug'] = data['im3']           

        data['im2f'] = np.ascontiguousarray(data['im2'][:,::-1])
        data['im3f'] = np.ascontiguousarray(data['im3'][:,::-1])

        for k in data:
            if data[k] is not None:
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float() 
        
        data['extra_info'] = self.extra_info[index]

        return data

    def __len__(self):
        return len(self.image_list)

class SimStereoIRDataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False):
        self.augmentor = None

        self.is_test = test
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, 'nirColormanaged/*_left.jpg')))
        disp_list = sorted(glob(osp.join(datapath, 'pfmDisp/*_left.pfm')))

        for i in range(len(image_list)):
                self.image_list += [ [disp_list[i], image_list[i], image_list[i].replace('_left.jpg','_right.jpg')] ]
                self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):
        data = {}

        data['im2'] = read_gen(self.image_list[index][1])
        data['im3'] = read_gen(self.image_list[index][2])
        
        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        if self.is_test:
            data['im2'] = data['im2'] / 255.0
            data['im3'] = data['im3'] / 255.0

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        data['gt'], data['validgt'] = [None]*2

        data['gt'] = np.expand_dims( readPFM(self.image_list[index][0]), -1)
        data['validgt'] = data['gt'] < 5000

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)
               
        
        if self.is_test:
            data['im2_aug'] = data['im2']
            data['im3_aug'] = data['im3']           

        data['im2f'] = np.ascontiguousarray(data['im2'][:,::-1])
        data['im3f'] = np.ascontiguousarray(data['im3'][:,::-1])

        for k in data:
            if data[k] is not None:
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float() 
        
        data['extra_info'] = self.extra_info[index]

        return data

    def __len__(self):
        return len(self.image_list)

class MyCustomDataset(data.Dataset):
    def __init__(self, datapath, test=False, overfit=False):
        self.augmentor = None
        
        self.is_test = test
        
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, 'left/*.png')))

        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i].replace('left', 'right'), image_list[i].replace('left', 'gt'), image_list[i].replace('left', 'hints')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}

        if self.is_test:
            data['im2'] = read_gen(self.image_list[index][0])
            data['im3'] = read_gen(self.image_list[index][1])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = readDispKITTI(self.image_list[index][2])
            data['hints'], data['validhints'] = readDispKITTI(self.image_list[index][3])

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)

            data['hints'] = np.array(data['hints']).astype(np.float32)
            data['validhints'] = np.array(data['validhints']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['hints'] = torch.from_numpy(data['hints']).permute(2, 0, 1).float()
            data['validhints'] = torch.from_numpy(data['validhints']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

    def __len__(self):
        return len(self.image_list)

#----------------------------------------------------------------------------------------------

def worker_init_fn(worker_id):                                                          
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    if args.dataset == 'kitti_stereo142':
        if args.test:
            dataset = KITTI142Dataset(args.datapath,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=1, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))     
        else:
            raise Exception("Not implemented yet")               

    elif args.dataset == 'middlebury_add':
        if args.test:
            dataset = MiddleburyAdditionalDataset(args.datapath,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=1, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise Exception("Not implemented yet")             

    elif args.dataset == 'middlebury2021':
        if args.test:
            dataset = Middlebury2021Dataset(args.datapath,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=1, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise Exception("Not implemented yet")             
    
    elif args.dataset in ['middlebury', 'eth3d']:
        if args.test:
            dataset = MiddleburyDataset(args.datapath,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=1, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise Exception("Not implemented yet")             

    elif args.dataset == 'simstereo':
        if args.test:
            dataset = SimStereoDataset(args.datapath,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=1, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise Exception("Not implemented yet")             

    elif args.dataset == 'simstereoir':
        if args.test:
            dataset = SimStereoIRDataset(args.datapath,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=1, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise Exception("Not implemented yet")             

    elif args.dataset in ['dsec', 'm3ed', 'm3ed_active']:
        if args.test:
            dataset = MyCustomDataset(args.datapath, test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=1, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise Exception("Not implemented yet")  

    return loader
