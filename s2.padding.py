# -*- coding: utf-8 -*-
"""
@author: runits
"""
#%%
from glob import glob
import SimpleITK as sitk
import numpy as np

#%%
trn_tst_phases = ['train/', 'test/']
targets = ['mhd', 'mhd.1mm', 'mhd.2mm', 'mhd.4mm']

target = targets[2]
print ('target is :', target)

#%%
for phase in trn_tst_phases:

    paths   = glob(target)
    x_paths = glob(target+'/'+phase+'/A*/A*.mhd')
    y_paths = glob(target+'/'+phase+'/A*/reference.mhd')

    if '2mm' in target:
        print ('2mm')
        t_mm = '2mm'
        fix_size = np.array((152, 128, 128)) # z, y, x
    elif '4mm' in target:
        print ('4mm')
        t_mm = '4mm'
        fix_size = np.array((76, 62, 62)) # z, y, x
    elif '1mm' in target:
        print ('1mm')
        t_mm = '1mm'
        fix_size = np.array((302, 256, 256)) # z, y, x
    else:
        t_mm = 'ori'
        fix_size = np.array((512, 512, 512)) # z, y, x


    x = []
    y = []
    for i, x_path in enumerate(x_paths):
        print (x_path)
        x_img =sitk.ReadImage(x_path)
        x_np_img = sitk.GetArrayFromImage(x_img)
        y_img =sitk.ReadImage(y_paths[i])
        y_np_img = sitk.GetArrayFromImage(y_img)


        np_img_size = np.array(x_np_img.shape)
        pd = fix_size - np_img_size
        print ('---> before size:', x_np_img.shape)
        print ('---> before needed padding size:', pd)

        x_padded = np.pad(x_np_img, ((0, pd[0]), (0,pd[1]), (0,pd[2])), mode='edge')
        y_padded = np.pad(y_np_img, ((0, pd[0]), (0,pd[1]), (0,pd[2])), mode='edge')

        x.append(x_padded[None, ...])
        y.append(y_padded[None, ...])

        print ('<--- after size:', x_padded.shape)
        print ('')

    xstck = np.vstack(x)
    ystck = np.vstack(y)

    print ('xstack size :', xstck.shape)
    print ('ystack size :', ystck.shape)

    np.save('npdata/'+t_mm+'/'+phase+'/x.npy', xstck)
    np.save('npdata/'+t_mm+'/'+phase+'/y.npy', ystck)

    print (phase, 'done ... :-)')







