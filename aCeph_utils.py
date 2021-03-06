# -*- coding: utf-8 -*-
"""
@author: runits
"""
#%%
import os
import csv
import dicom
from glob import glob
import SimpleITK as sitk
import numpy as np
import scipy.ndimage

#%%
def make_block_label(im, lbl, x, y, z, blsz=0):
    xb = np.arange(x-blsz, x+blsz+1)
    yb = np.arange(y-blsz, y+blsz+1)
    zb = np.arange(z-blsz, z+blsz+1)
    
    for z in zb:
        for y in yb:
            for x in xb:
                # print (x, y, z)
                im.SetPixel(int(x), int(y), int(z), lbl)
    
def get_csv(fname):
    try:
        with open(fname, 'r') as f:
        # with open(fname, 'r', encoding='mac_roman') as f:
            reader = csv.reader(f)
            csvlst = list(reader)

        print('---->', fname, 'load ok')
        return csvlst
    except EnvironmentError:
        print('error open error')
        raise SystemExit

def get_fov(path):    
    err = ''
    try:
        # we just want info. so, we get 2 slice only
        slices = \
            [dicom.read_file(s) for s in glob(os.path.join(path, '*.dcm'))[:2]]
            # [dicom.read_file(path + '/' + s) for s in os.listdir(path)[:2] if s.endswith('.dcm')]
    except:
        err = 'error !! invalid dicom path or other reason'
        return err
    
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - \
                                 slices[1].ImagePositionPatient[2])            
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - \
                                 slices[1].SliceLocation)
        
    fov = slices[0].ReconstructionDiameter
    dZ = slice_thickness
    
    return fov


def get_dcm_image(fname):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames( fname )
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img16 = sitk.Cast(image, sitk.sitkInt16)
    return img16

def find_coord(name, csvlst):
    find = False
    for l in csvlst:        
        if 'Point' in l:
            preXoffset = l.index('X-pre(mm)')
            preYoffset = l.index('Y-pre(mm)')
            preZoffset = l.index('Z-pre(mm)')            
            break

    for l in csvlst:
        if name in l:
            x = l[preXoffset]
            y = l[preYoffset]
            z = l[preZoffset]
            find = True
            break
    
    if (not find):
        print ('!!!!!!! CAUTION!!!!!!! no match ' + name + ' in csv file')
        x = str(0)
        y = str(0)
        z = str(0)
    
    return x, y, z, find

# we use 0-index
# if you need verify this index in dicom "voxel" coordinate (add 1 each coordinate)
def calc_dicom_voxel_idx(sx, sy, sz, ori, spc, fov):
    dvx = round(sx/spc[0])
    dvy = round((sy/spc[1]) + (fov/spc[1]))
    dvz = round((sz/spc[2]) - (ori[2]/spc[2]))    
    return dvx, dvy, dvz
    
# simplant to dicom voxel coordinate (-1ed)
# we use 0-index
# if you need verify this index in dicom "voxel" coordinate (add 1 each coordinate)
def calc_s_to_dv(sx, sy, sz, ori, spc, fov):
    dvx = round(sx/spc[0])
    dvy = round((sy/spc[1]) + (fov/spc[1]))
    dvz = round((sz/spc[2]) - (ori[2]/spc[2]))    
    return dvx, dvy, dvz

# conversion dicom voxel coord to dicom world(ITK) coordinate
def calc_dv_to_d(dvx, dvy, dvz, ori, spc):

    dx = ori[0] + (dvx * spc[0])
    dy = ori[1] + (dvy * spc[1])
    dz = ori[2] + (dvz * spc[2])
    return dx, dy, dz

    
# new spacing = z y x
def resample(image, spc, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing    
    spacing = np.array(spc)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing



def normalize(image, min_bound=-1000.0, max_bound=400.0):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
    
    