#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: runits
"""

import os
import numpy as np
import matplotlib.pyplot as plt


from keras.models import model_from_json
from conv3models import get_model_p2_1_conv4l_mxout_landmarks

#%%
def show_all(xs,  outpath):
    for i, x in enumerate(xs):
        plt.figure(figsize=(10,10))
        plt.imshow(x)
    
        if True:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
                print (outpath, 'make directory')
        plt.savefig(outpath + '/' + str(i)+'.png', \
                        bbox_inches='tight', dpi=200)
        plt.close()
            
            
#%%		
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

#%%

trn_planes = ['z', 'y', 'x']
t_mms = ['ori', '1mm', '2mm', '4mm']
t_mm = t_mms[2]



Tname = '8.180516.05.na.aug15.nf.20.out1_conv4l_mxout'
epochs = [2855] # na

#02. CFM
#04. bregma
#05. Na                                    
#94. Me

land_idx = 5

for epoch in epochs:
    m_arch = 'model/' + Tname + '.json'
    m_weight = 'weights/' + Tname + '.'+str(epoch)+'.hdf5'

    if not os.path.exists(m_arch):
        raise SystemExit
    if not os.path.exists(m_weight):
        raise SystemExit

    print ('load ok ...', m_arch)
    print ('load ok ...', m_weight)

    #model = model_from_json(open(m_arch).read())
    model = get_model_p2_1_conv4l_mxout_landmarks(nf=20)

    model.load_weights(m_weight)
    x_tst = np.load('npdata/'+t_mm+ '/test/x.npy')[..., None].astype(np.float32)
    y_tst = np.load('npdata/'+t_mm+ '/test/y.npy')[..., None]

    x_tst = normalize(x_tst)
    ypreds = model.predict(x_tst, batch_size=1, verbose=1)

    y_1_idx = np.argwhere(y_tst == land_idx)
    y_1_z = y_1_idx[:, 1]
    y_1_y = y_1_idx[:, 2]
    y_1_x = y_1_idx[:, 3]

    ytrs= [y_1_z, y_1_y, y_1_x]
    landmark_names = ['05.na.z', '05.na.y', '05.na.x']

    dists = []
    for ipa, _ in enumerate(x_tst):
        print ()
        print (ipa)

        dist_ = []
        for ilm, lm in enumerate(landmark_names):
            #print (lm)
            ytr  =   ytrs[ilm]
            yprd = ypreds[ilm]
            ytr_idx  = ytr[ipa]
            yprd_idx = np.argmax(yprd[ipa])
            print (lm, ytr_idx, yprd_idx, ytr_idx - yprd_idx)
            dist_.append(ytr_idx - yprd_idx)
        dist = np.linalg.norm(dist_)
        dists.append(dist)
    print ()
    #print ('dist=', dists)
    print (epoch, 'dists=', np.average(dists))

