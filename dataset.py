import torch
from torch.utils.data import Dataset
import numpy as np
import os
from os.path import isfile, join
import SimpleITK as sitk
from skimage import data, filters
import csv
import nibabel as nib
import pydicom as pd


class UnetDataset(Dataset):
    def __init__(self, root_dir, patch_size=128):# /home/sci/hdai/Projects/LymphNodes
#         construct case_info dict
        self.case_info = []
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.field_list = ['Series UID', 'Collection', '3rd Party Analysis', 
                      'Data Description URI', 'Subject ID', 'Study UID', 
                      'Study Description', 'Study Date', 'Series Description', 
                      'Manufacturer', 'Modality', 'SOP Class Name', 
                      'SOP Class UID', 'Number of Images', 'File Size', 
                      'File Location', 'Download Timestamp']
        with open(f'{root_dir}/metadata.csv', mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                self.case_info.append({self.field_list[i]:row[i] for i in range(len(row))})
#         self.case_info.pop(0)
        self.case_info = self.case_info[87:]
        
    def __len__(self):
        # return len(self.case_info)
        return 30
        
    def __getitem__(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         construct 3d CT from dicom folder
        # '/CT Lymph Nodes/ABD_LYMPH_003/09-14-2014-ABDLYMPH003-abdominallymphnodes-39052/abdominallymphnodes-65663'
        relative_ct_folder_path = self.case_info[idx]['File Location'][1:].replace('\\','/')
        # '/home/sci/hdai/Projects/LymphNodes/CT Lymph Nodes/ABD_LYMPH_003/09-14-2014-ABDLYMPH003-abdominallymphnodes-39052/abdominallymphnodes-65663'
        ct_folder_path = f'{self.root_dir}{relative_ct_folder_path}'
        slice_name_list = [f for f in os.listdir(ct_folder_path)]
        slice_name_list.sort()
        slice_list = []
        for slice_name in slice_name_list:
            ds = pd.dcmread(f'{ct_folder_path}/{slice_name}')
            slice_list.append(torch.from_numpy(ds.pixel_array.transpose()))
        img = torch.stack(slice_list,-1).to(device)
        
#         load 3d mask
        case_name = self.case_info[idx]['File Location'][17:30].replace('\\','/')
        mask_path = f'{self.root_dir}/MED_ABD_LYMPH_MASKS/{case_name}/{case_name}_mask.nii.gz'
        mask = torch.from_numpy(nib.load(mask_path).get_fdata()).to(device)
        mask[mask>1] = 1
        
        half_patch_size = int(self.patch_size/2)
        idx_x, idx_y, idx_z = torch.where(mask!=0)
        centroid_x, centroid_y, centroid_z = 256, 256, 300
        if int(torch.mean(idx_x.float())) < mask.shape[0]-half_patch_size and int(torch.mean(idx_x.float())) > half_patch_size:
            centroid_x = int(torch.mean(idx_x.float()))
        if int(torch.mean(idx_y.float())) < mask.shape[1]-half_patch_size and int(torch.mean(idx_y.float())) > half_patch_size:
            centroid_y = int(torch.mean(idx_y.float()))
        if int(torch.mean(idx_z.float())) < mask.shape[2]-half_patch_size and int(torch.mean(idx_z.float())) > half_patch_size:
            centroid_z = int(torch.mean(idx_z.float()))
        
        img = img[centroid_x-half_patch_size:centroid_x+half_patch_size, centroid_y-half_patch_size:centroid_y+half_patch_size, centroid_z-half_patch_size:centroid_z+half_patch_size]
        mask = mask[centroid_x-half_patch_size:centroid_x+half_patch_size, centroid_y-half_patch_size:centroid_y+half_patch_size, centroid_z-half_patch_size:centroid_z+half_patch_size]


        sample = {  'name' : case_name,
                    'img'  : img.unsqueeze(0),
                    'mask' : torch.stack((mask.long(),1-mask.long()),0)}#.unsqueeze(0)
        
        return sample


class FnetDataset(Dataset):
    def __init__(self, root_dir, patch_size=128):# /home/sci/hdai/Projects/LymphNodes
#         construct case_info dict
        self.case_info = []
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.field_list = ['Series UID', 'Collection', '3rd Party Analysis', 
                      'Data Description URI', 'Subject ID', 'Study UID', 
                      'Study Description', 'Study Date', 'Series Description', 
                      'Manufacturer', 'Modality', 'SOP Class Name', 
                      'SOP Class UID', 'Number of Images', 'File Size', 
                      'File Location', 'Download Timestamp']
        with open(f'{root_dir}/metadata.csv', mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                self.case_info.append({self.field_list[i]:row[i] for i in range(len(row))})
#                 only use mediastinal lymph node
        self.case_info = self.case_info[87:]
        
    def __len__(self):
        return len(self.case_info)
        
    def __getitem__(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         construct 3d CT from dicom folder
        # '/CT Lymph Nodes/ABD_LYMPH_003/09-14-2014-ABDLYMPH003-abdominallymphnodes-39052/abdominallymphnodes-65663'
        relative_ct_folder_path = self.case_info[idx]['File Location'][1:].replace('\\','/')
        # '/home/sci/hdai/Projects/LymphNodes/CT Lymph Nodes/ABD_LYMPH_003/09-14-2014-ABDLYMPH003-abdominallymphnodes-39052/abdominallymphnodes-65663'
        ct_folder_path = f'{self.root_dir}{relative_ct_folder_path}'
        slice_name_list = [f for f in os.listdir(ct_folder_path)]
        slice_name_list.sort()
        slice_list = []
        for slice_name in slice_name_list:
            ds = pd.dcmread(f'{ct_folder_path}/{slice_name}')
            slice_list.append(torch.from_numpy(ds.pixel_array.transpose()))
        img = torch.stack(slice_list,-1).to(device)
        
#         load 3d mask
        case_name = self.case_info[idx]['File Location'][17:30].replace('\\','/')
        mask_path = f'{self.root_dir}/MED_ABD_LYMPH_MASKS/{case_name}/{case_name}_mask.nii.gz'
        mask = torch.from_numpy(nib.load(mask_path).get_fdata()).to(device)
        mask[mask>1] = 1
        
        half_patch_size = int(self.patch_size/2)
        idx_x, idx_y, idx_z = torch.where(mask!=0)
        centroid_x, centroid_y, centroid_z = 256, 256, 300
        if int(torch.mean(idx_x.float())) < mask.shape[0]-half_patch_size and int(torch.mean(idx_x.float())) > half_patch_size:
            centroid_x = int(torch.mean(idx_x.float()))
        if int(torch.mean(idx_y.float())) < mask.shape[1]-half_patch_size and int(torch.mean(idx_y.float())) > half_patch_size:
            centroid_y = int(torch.mean(idx_y.float()))
        if int(torch.mean(idx_z.float())) < mask.shape[2]-half_patch_size and int(torch.mean(idx_z.float())) > half_patch_size:
            centroid_z = int(torch.mean(idx_z.float()))
        
        image_list, mask_list = [], []
        
        mask = mask[centroid_x-half_patch_size:centroid_x+half_patch_size, \
                    centroid_y-half_patch_size:centroid_y+half_patch_size, \
                    centroid_z-half_patch_size:centroid_z+half_patch_size]
        
        for i in range(4):
            image_list.append(img[centroid_x-int(half_patch_size/2**i):centroid_x+int(half_patch_size/2**i), \
                                  centroid_y-int(half_patch_size/2**i):centroid_y+int(half_patch_size/2**i), \
                                  centroid_z-int(half_patch_size/2**i):centroid_z+int(half_patch_size/2**i)])
#             mask_list.append(mask[centroid_x-int(half_patch_size/2**i):centroid_x+int(half_patch_size/2**i), \
#                                   centroid_y-int(half_patch_size/2**i):centroid_y+int(half_patch_size/2**i), \
#                                   centroid_z-int(half_patch_size/2**i):centroid_z+int(half_patch_size/2**i)])

        sample = {  'name' : case_name,
                    'img0' : image_list[0].unsqueeze(0),
                    'img1' : image_list[1].unsqueeze(0),
                    'img2' : image_list[2].unsqueeze(0),
                    'img3' : image_list[3].unsqueeze(0),
                    'mask' : torch.stack((mask.long(),1-mask.long()),0)}
        
        return sample