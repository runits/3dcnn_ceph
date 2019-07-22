# -*- coding: utf-8 -*-
"""
@author: runits
"""
#%%
import os
import numpy as np
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.image_3d_runits_2_1_6 import ImageDataGenerator
from conv3models import get_model_p2_1_conv4l_mxout_landmarks


#%%
DATA_AUG_FIT_GEN = True
RELAY = True
K.set_image_data_format('channels_last')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

t_mms = ['ori', '1mm', '2mm', '4mm']
t_mm = t_mms[2]


nf = 20
subject = '8.1808061110.94.me.aug15.nf.'+str(nf)+'.out1_conv4l_mxout'
print (subject, t_mm, nf)

#02. cfm
#04. bregma
#05. na
#94. me
land_idx = 94

#%%
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def make_onehot_label(indexes, sz):
    labels = []
    for i in range(len(indexes)):
        l = np.zeros((1, sz), dtype=np.bool)
        l[:, indexes[i]] = True
        labels.append(l)

    return np.vstack(labels)

def make_log_label(indexes, sz, step=3):
    idx = indexes
    lps = np.logspace(0.99, 0.0, step) / 10.0
    l = np.zeros((1, sz), dtype=np.float32)
    l[:, idx] = 1.0
    if (idx - step) > 0:
        for ii, jj in enumerate(range(idx-1, idx-1-step, -1)):
            l[:, jj] = lps[ii]
    if (idx + step) < sz:
        for ii, jj in enumerate(range(idx+1, idx+1+step, +1)):
            l[:, jj] = lps[ii]

    return l


def make_log_labels(indexes, sz, step=3):
    labels = []
    lps = np.logspace(0.99, 0.0, step) / 10.0
    for i in range(len(indexes)):
        idx = indexes[i]
        l = np.zeros((1, sz), dtype=np.float32)
        l[:, idx] = 1.0
        if (idx - step) > 0:
            for ii, jj in enumerate(range(idx-1, idx-1-step, -1)):
                l[:, jj] = lps[ii]
        if (idx + step) < sz:
            for ii, jj in enumerate(range(idx+1, idx+1+step, +1)):
                l[:, jj] = lps[ii]

        labels.append(l)

    return np.vstack(labels)


def aceph_flow(x, y, batch_sz):
    x_dg = ImageDataGenerator(width_shift_range=0.15,
                         height_shift_range=0.15,
                         rotation_range = 0.15,
                         horizontal_flip=False,
                         labels=[land_idx])
    missing_landmark = False
    seed = 1
    bs = batch_sz
    x_dg.fit(x, augment=True, seed=seed)
    x_g = x_dg.flow(x, y=y, batch_size=bs, seed=seed)
    while 1:
        # Get!!!
        x, y = x_g.next()

         # check all landmark
        for ii in y:
            if np.sum(ii) == 0:
                #            print (ii, np.sum(ii))
                missing_landmark = True

        if (missing_landmark):
            missing_landmark = False
            print ('missing landmark cause augmentation')
            continue

        sp = x.shape

        y_1_z = make_log_label(y[0, 1], sp[1]).astype(np.float32)
        y_1_y = make_log_label(y[0, 2], sp[2]).astype(np.float32)
        y_1_x = make_log_label(y[0, 3], sp[3]).astype(np.float32)

        yield x, [y_1_z, y_1_y, y_1_x]

#%%
print ('train data load start!')

x_trn = np.load('npdata/'+t_mm+'/train/x.npy')[..., None].astype(np.float32)
y_trn = np.load('npdata/'+t_mm+'/train/y.npy')[..., None]
x_tst = np.load('npdata/'+t_mm+ '/test/x.npy')[..., None].astype(np.float32)
y_tst = np.load('npdata/'+t_mm+ '/test/y.npy')[..., None]

x_trn = normalize(x_trn)
x_tst = normalize(x_tst)


sp = y_tst.shape
y_1_idx = np.argwhere(y_tst==land_idx)
#%%
if len(y_tst) != ( len(y_1_idx) ):
    print ('test data dirty !!!, so check!!!')

y_1_tst_z = make_log_labels(y_1_idx[:, 1], sp[1]).astype(np.float32)
y_1_tst_y = make_log_labels(y_1_idx[:, 2], sp[2]).astype(np.float32)
y_1_tst_x = make_log_labels(y_1_idx[:, 3], sp[3]).astype(np.float32)


#%%
print ('train data & test data load done!')

# get model.
model = get_model_p2_1_conv4l_mxout_landmarks(nf=nf)
metrics = ['accuracy']

model.compile(optimizer=Adadelta(), loss='binary_crossentropy', metrics=metrics)
model.summary()

json_string = model.to_json()
model_save_path = 'model'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

graph_p = 'graph/' + subject
if not os.path.exists(graph_p):
    os.makedirs(graph_p)
    print (graph_p, 'make directory')

weights_p = 'weights/'
if not os.path.exists(weights_p):
    os.makedirs(weights_p)

open(os.path.join(model_save_path, subject + '.json'), 'w').write(json_string)

i_epoch = 0
if RELAY:
    # m_weight = 'weights/8.180726.02.cfm.aug15.nf.20.out1_conv4l_mxout.1919.hdf5'
    m_weight = 'f_w_1805xx/94.me.hdf5'
    model.load_weights(m_weight)
    i_epoch = 1

model_checkpoint = ModelCheckpoint('weights/' + subject + '.{epoch:04d}.hdf5', \
                                   # monitor='val_loss', save_best_only=True)
                                   monitor='loss', save_best_only=True)


tbCallBack = TensorBoard(log_dir=graph_p, \
                         histogram_freq=0, write_graph=True, write_images=True)

batch_sz = 1
e = 50000
if DATA_AUG_FIT_GEN:
    print ('Augment using Fit generator!')

    model.fit_generator( aceph_flow(x_trn, y_trn, batch_sz),
                         steps_per_epoch = 36,
                         validation_data = (x_tst, [y_1_tst_z, y_1_tst_y, y_1_tst_x]),
                         epochs=e,
                         initial_epoch=i_epoch,
                         callbacks=[model_checkpoint, tbCallBack]
                        )
