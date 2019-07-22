# -*- coding: utf-8 -*-
"""
@author: runits
"""
#%%
import os
import csv
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


#%%
down_sample = True
if down_sample:
    down_mm = 2

blsize = 0 # blsize 2 is --> up 2 down 2 left 2 right 2

#%%
paths = glob('../CT data/A*')
dcms = glob('../CT data/**/CT')
csvs = glob('../CT data/**/simplant_coord_v2.csv')

x_label_name = 'v1_label_pp.csv'
x_label = get_csv(x_label_name)



#%%
from sklearn.model_selection import train_test_split
nsamples = len(paths)
trn_idxs, tst_idxs = train_test_split(np.arange(nsamples), random_state=79, test_size=0.3)

trn_tst_phases = ['train/', 'test/']

#%%

for phase in trn_tst_phases:
    # for i, dcm in enumerate(dcms[0:1]):    
    #for i, dcm in enumerate(dcms[-1:]):    
    if phase == trn_tst_phases[0]:
        p_idx = trn_idxs
    else:
        p_idx = tst_idxs
        
    for ti in p_idx:    
        print (dcms[ti])
        dcm = dcms[ti]
        pid = dcm.split(os.sep)[1]
    
        fov = get_fov(dcm)
        print (fov)		
        img = get_dcm_image(dcm)    
        size = img.GetSize()
        origin = img.GetOrigin()
        spc = img.GetSpacing()
        
        sim_csv = get_csv(csvs[ti])
    
        if down_sample:
            savepath = ('mhd.'+str(down_mm)+'mm/'+phase+ pid)
            spc = [down_mm, down_mm, down_mm]
        else:
            savepath = ('mhd/'+phase+pid )        
        
        print (savepath)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
    
        # 1. make down sample data
        if down_sample:
            print ('--->Do down sampling to ', spc)
            old_spc = img.GetSpacing()
            npa = sitk.GetArrayViewFromImage(img)
            np_post, spc = resample(npa, list(reversed(list(old_spc))), spc)
            
            dw_img = sitk.GetImageFromArray(np_post)
            dw_img.SetSpacing(spc)
            dw_img.SetOrigin(origin)
            sitk.WriteImage(dw_img, os.path.join(savepath, pid+'.mhd'))
        else:
            print ('--->Do save mhd no downsample ', spc)
            sitk.WriteImage(img, os.path.join(savepath, pid+'.mhd'))
            
    
    
        # 2. make dicom voxel index csv file
        # 3. make labeling referecne volume data
        if down_sample:
            size = dw_img.GetSize()
        print ('x, y, z:', size)
        label_img = sitk.Image(size, sitk.sitkUInt8)
        label_img.SetOrigin(origin)    
        label_img.SetSpacing(spc)
        
        save_ref_csv_path = \
            os.path.join(savepath, os.path.basename(x_label_name)[:-4] + '.csv')
        with open(save_ref_csv_path, 'w', newline='') as csv_file:
            cw = csv.writer(csv_file, delimiter=',')
            
            csv_header = ['name', 'x', 'y', 'z', 'dvx', 'dvy', 'dvz']
            cw.writerow(csv_header) 
            
            for xl in x_label[1:]:
                # print (xl)
    
                row=[]
                row.append(xl[1])
                label = int(xl[2])
                
                x, y, z, find = find_coord(xl[0], sim_csv)
                if not find:
                    continue
                # print (x, y, z)
                row.append(x)
                row.append(y)
                row.append(z)
                
                dvx, dvy, dvz = calc_dicom_voxel_idx(float(x), float(y), float(z), origin, spc, fov)
                print (dvx, dvy, dvz)
                row.append(dvx)
                row.append(dvy)
                row.append(dvz)
                
                cw.writerow(row) 
                
                make_block_label(label_img, label, (int(dvx)+1), (int(dvy)+1), (int(dvz)+1), blsz=blsize)
                #label_img.SetPixel((int(dvx)+1), (int(dvy)+1), (int(dvz)+1), label)
        
        print ('save...', savepath)
        sitk.WriteImage(label_img, os.path.join(savepath, 'reference.mhd'))

