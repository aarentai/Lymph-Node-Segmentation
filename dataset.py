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


def intensity_mean(original):
    original = original.flatten()
    new_ori = np.delete(original, np.where(original == 0))
    return np.mean(new_ori)


def intensity_std(original):
    original = original.flatten()
    new_ori = np.delete(original, np.where(original == 0))
    return np.std(new_ori)


def intensity_norm(original, mean, std):
    normalized = (original - mean)/std
    normalized[normalized > 5] = 5
    # normalized[normalized < -5] = -5
    # normalized = (normalized + 5)/(2*5)
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    return normalized


class BratsDataset(Dataset):
    def __init__(self, root_dir, img_folder, label_folder, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, img_folder)
        self.label_dir = os.path.join(root_dir, label_folder)
        self.folder_name = [f for f in os.listdir(self.img_dir)]
        self.folder_name.sort()
        self.transform = transform

    def __len__(self):
        return len(self.folder_name)

    def __getitem__(self, index):
        folder = self.folder_name[index]
        t1_path = self.img_dir + folder + '/' + folder + '_t1.nii.gz'
        # occ_path = self.img_dir + folder + '/'  + folder + '_occ.nii.gz'
        t1ce_path = self.img_dir + folder + '/' + folder + '_t1ce.nii.gz'
        t2_path = self.img_dir + folder + '/' + folder + '_t2.nii.gz'
        flair_path = self.img_dir + folder + '/' + folder + '_flair.nii.gz'
        label_path = self.label_dir + folder + '_seg.nii.gz'

        # load 4-channel input
        t1 = nib.load(t1_path).get_data()
        # occ = nib.load(occ_path).get_data()
        t1ce = nib.load(t1ce_path).get_data()
        t2 = nib.load(t2_path).get_data()
        flair = nib.load(flair_path).get_data()

        # calculate mean and std
        mean = [intensity_mean(t1), intensity_mean(t1ce), intensity_mean(t2), intensity_mean(flair)]
        # mean = [intensity_mean(occ), intensity_mean(t1ce), intensity_mean(t2), intensity_mean(flair)]
        std = [intensity_std(t1), intensity_std(t1ce), intensity_std(t2), intensity_std(flair)]
        # std = [intensity_std(occ), intensity_std(t1ce), intensity_std(t2), intensity_std(flair)]

        # normalization
        t1 = intensity_norm(t1, mean[0], std[0])
        # occ = intensity_norm(occ, mean[0], std[0])
        t1ce = intensity_norm(t1ce, mean[1], std[1])
        t2 = intensity_norm(t2, mean[2], std[2])
        flair = intensity_norm(flair, mean[3], std[3])

        # crop
        t1 = t1[56:56 + 128, 56:56 + 128, 14:14 + 128]
        # occ = occ[56:56 + 128, 56:56 + 128, 14:14 + 128]
        t1ce = t1ce[56:56 + 128, 56:56 + 128, 14:14 + 128]
        t2 = t2[56:56 + 128, 56:56 + 128, 14:14 + 128]
        flair = flair[56:56 + 128, 56:56 + 128, 14:14 + 128]

        image = np.stack([t1, t1ce, t2, flair])
        # image = np.stack([occ, t1ce, t2, flair])

        label = nib.load(label_path).get_data()
        label = label[56:56 + 128, 56:56 + 128, 14:14 + 128]
        label[label > 3] = 3

        sample = {'name': folder,
                  'image': torch.from_numpy(image).type('torch.DoubleTensor'),
                  'label': torch.from_numpy(label).type('torch.DoubleTensor')}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

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
        self.case_info = self.case_info[87:-18]
        
    def __len__(self):
        return len(self.case_info)
        # return 30
        
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

        img[img<70-750]=70-750
        img[img>70+750]=70+750
        img = img - torch.min(img)
        img = img/(torch.max(img)-torch.min(img))

        sample = {  'name' : case_name,
                    'img'  : img.unsqueeze(0),
                    'mask' : mask.unsqueeze(0)}
        
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
        self.case_info = self.case_info[87:-18]
        
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
        
        img[img<70-750]=70-750
        img[img>70+750]=70+750
        img = img - torch.min(img)
        img = img/(torch.max(img)-torch.min(img))
        
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
                    'mask' : mask.long().unsqueeze(0)}
#                     'mask' : torch.stack((mask.long(),1-mask.long()),0)}
        
        return sample