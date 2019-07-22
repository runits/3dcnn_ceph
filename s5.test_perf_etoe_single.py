#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: runits
"""

import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import csv
import numpy as np
from glob import glob
import SimpleITK as sitk


import aCeph_utils as aceph_ut


from keras.models import model_from_json
from conv3models import get_model_p2_1_conv4l_mxout_landmarks

#%%
dcms = glob('test_etoe/**/CT')
csvs = glob('test_etoe/**/simplant_coord_v2.csv')

x_label_name = 'v1_label_pp.csv'
x_label = aceph_ut.get_csv(x_label_name)



tar_spc = [2, 2, 2]
pad_size = np.array((152, 128, 128)) # z, y, x

#%%
pp_models = []

single_models = [ 'cfm', 'bregma', 'na', 'me' ]
single_w = [ 
            # 'f_w_1805xx/2.cfm.hdf5', \
            'weights/8.180727.02.cfm.aug15.nf.20.out1_conv4l_mxout.3352.hdf5', \
            'f_w_1805xx/4.bregma.hdf5', \
            'f_w_1805xx/5.na.hdf5', \
            # 'f_w_1805xx/94.me.hdf5'
			'weights/8.1808061110.94.me.aug15.nf.20.out1_conv4l_mxout.0238.hdf5'
            ]


for i in range(4):    
    model = get_model_p2_1_conv4l_mxout_landmarks(nf=20)
    model.load_weights(single_w[i])
    
    print ('load ok ...', single_w[i])
    pp_models.append(model)


#%%
simplant_name = \
    [
     'CFM', 'Bregma', 'Na', 'Me (anat)', \
     'R Or', 'L Or', 'R Po', 'L Po', \
     'R COR', 'L COR', 'R F', 'L F'
     ]
    
landmark_single = \
    [ '02.cfm.z', '02.cfm.y', '02.cfm.x', \
      '04.bregma.z', '04.bregma.y', '04.bregma.x', \
      '05.na.z', '05.na.y', '05.na.x', \
      '94.me.z', '94.me.y', '94.me.x'
    ]

landmark_bilateral = \
    [  '10.R Or.z', '10.R Or.y', '10.R Or.x', \
       '12.L Or.z', '12.L Or.y', '12.L Or.x', \
       '16.R Po.z', '16.R Po.y', '16.R Po.x', \
       '17.L Po.z', '17.L Po.y', '17.L Po.x', \
       '38.R COR.z', '38.R COR.y', '38.R COR.x', \
       '40.L COR.z', '40.L COR.y', '40.L COR.x', \
       '80.R F.z', '80.R F.y', '80.R F.x', \
       '83.L F.z', '83.L F.y', '83.L F.x'
    ]
    
landmark_names = landmark_single + landmark_bilateral


for i, dcm in enumerate(dcms):

    pid = dcm.split(os.sep)[1]
    fov = aceph_ut.get_fov(dcm)    
    print (i, pid, fov, dcm)

    img = aceph_ut.get_dcm_image(dcm)    
    ori_size = img.GetSize()
    origin = img.GetOrigin()
    ori_spc = img.GetSpacing()
    
    
    sim_csv = aceph_ut.get_csv(csvs[i])
    
    save_ref_csv_path = \
            os.path.join(os.path.dirname(csvs[i]), pid + '_ref_pred_mm.csv')
    print (save_ref_csv_path)
    
    #1. make down to 2mm 
    npa = sitk.GetArrayViewFromImage(img)
    np_rsz, spc = aceph_ut.resample(npa, list(reversed(list(ori_spc))), tar_spc)
    
    #2. padding
    pd = pad_size - np.array(np_rsz.shape)
    print ('---> before size:', np_rsz.shape)
    print ('---> before needed padding size:', pd)
    
    np_padded = np.pad(np_rsz, ((0, pd[0]), (0,pd[1]), (0,pd[2])), mode='edge')
    np_padded = np_padded[None, ..., None]

    #3. normalize
    tst = aceph_ut.normalize(np_padded)
    
    
    #4. predict
    preds=[]
    for i in range(8):
        pred = pp_models[i].predict(tst, batch_size=1, verbose=1)
        preds.append(pred)
    
    #5. summary
    preds_mm =[]
    for i in range(4):
        print (i)
        pred = preds[i]
        
        max_idx = np.argmax(pred[0]) # z
        pred_mm = ((max_idx + 1)*2) + origin[2] # index * spc z + origin z            
        preds_mm.append(pred_mm)
                
        max_idx = np.argmax(pred[1]) # y
        pred_mm = ((max_idx + 1)*2) - fov # index * spc y - fov(mm)        
        preds_mm.append(pred_mm)
        
        max_idx = np.argmax(pred[2]) # x
        pred_mm = ((max_idx + 1)*2) # index * spc x
        preds_mm.append(pred_mm)
         
    for i in range(4, 8):
        print (i)
        pred = preds[i]
        
        max_idx = np.argmax(pred[0]) # z
        pred_mm = ((max_idx + 1)*2) + origin[2] # index * spc z + origin z            
        preds_mm.append(pred_mm)
                
        max_idx = np.argmax(pred[1]) # y
        pred_mm = ((max_idx + 1)*2) - fov # index * spc y - fov(mm)        
        preds_mm.append(pred_mm)
        
        max_idx = np.argmax(pred[2]) # x
        pred_mm = ((max_idx + 1)*2) # index * spc x
        preds_mm.append(pred_mm)
        
        max_idx = np.argmax(pred[3]) # z
        pred_mm = ((max_idx + 1)*2) + origin[2] # index * spc z + origin z            
        preds_mm.append(pred_mm)
                
        max_idx = np.argmax(pred[4]) # y
        pred_mm = ((max_idx + 1)*2) - fov # index * spc y - fov(mm)        
        preds_mm.append(pred_mm)
        
        max_idx = np.argmax(pred[5]) # x
        pred_mm = ((max_idx + 1)*2) # index * spc x
        preds_mm.append(pred_mm)

    # write csv file
    with open(save_ref_csv_path, 'w', newline='') as csv_file:
        cw = csv.writer(csv_file, delimiter=',')

        pid_w = ['PID', str(pid)]
        cw.writerow(pid_w) 

        fov_w = ['FOV', str(fov)]
        cw.writerow(fov_w) 
        
        slice_thickness_w = ['slice_thickness', str(ori_spc[2])]
        cw.writerow(slice_thickness_w)
        
        csv_header = ['Point', 'X-pre(mm)', 'Y-pre(mm)', 'Z-pre(mm)']
        cw.writerow(csv_header) 
        
        
        for sn in simplant_name:
            # print (xl)    
            row=[]
            row.append(sn)
            
            x, y, z, find = aceph_ut.find_coord(sn, sim_csv)
            if not find:
                continue
            # print (x, y, z)
            row.append(x)
            row.append(y)
            row.append(z)
            cw.writerow(row) 
        
        for isn, sn in enumerate(simplant_name):
            # print (xl)    
            row=[]
            row.append(sn+'p')
            
            z = preds_mm[isn*3 + 0]
            y = preds_mm[isn*3 + 1]
            x = preds_mm[isn*3 + 2]
            
            # print (x, y, z)
            row.append(x)
            row.append(y)
            row.append(z)
            cw.writerow(row) 
        

            

