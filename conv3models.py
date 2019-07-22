# -*- coding: utf-8 -*-
"""
@author: runits
"""
#%%
from keras.layers import Input, Dense, Dropout, Flatten, Lambda
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import LeakyReLU
from keras.models import Model
import keras.backend as K

#%%
def Maxout(x, num_unit=None):
    """
    Maxout as in the paper `Maxout Networks <http://arxiv.org/abs/1302.4389>`_.

    Args:
        x (tf.Tensor): a NHWC or NC tensor. Channel has to be known.
        num_unit (int): a int. Must be divisible by C.

    Returns:
        tf.Tensor: of shape NHW(C/num_unit) named ``output``.
    """
    input_shape = x.get_shape().as_list()
    # print (input_shape)
    ndim = len(input_shape)
    assert ndim == 5 or ndim == 4 or ndim == 2 

    data_format = K.image_data_format()

    if data_format == 'channels_first':
        ch = input_shape[1]
    else:
        ch = input_shape[-1]

    if num_unit == None:
        num_unit = int(ch / 2)
    assert ch is not None and ch % num_unit == 0

    if ndim == 4:
        if data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 2, 3, 1))
        x = K.reshape(x, (-1, input_shape[1], input_shape[2], ch // num_unit, num_unit))
        
        # print ('-->res ', x.get_shape())

        x = K.max(x, axis=3)

        # print ('-->max ', x.get_shape())

        if data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 3, 1, 2))
    elif (ndim == 5):
        if data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 2, 3, 4, 1))
        x = K.reshape(x, (-1, input_shape[1], input_shape[2], input_shape[3], ch // num_unit, num_unit))
        
        # print ('-->res ', x.get_shape())

        x = K.max(x, axis=4)

        # print ('-->max ', x.get_shape())

        if data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 3, 1, 2))
    else:
        x = K.reshape(x, (-1, ch // num_unit, num_unit))
        x = K.max(x, axis=1)

    return x

    
def get_model_p2_1_conv4l_mxout_landmarks(nf=8):
    print ('------>get_model_p2_3_conv4l_mxout_landmarks', nf)
    
    input_shape = (152, 128, 128, 1)
    
    input_img = Input(shape=input_shape, name='input_img')
    conv1 = Conv3D(nf, (3, 3, 3), padding='same')(input_img)
    conv1 = Lambda(Maxout)(conv1)
    conv1 = Conv3D(nf, (1, 3, 3), padding='same')(conv1)
    conv1 = Lambda(Maxout)(conv1)
    conv1 = Conv3D(nf, (3, 1, 3), padding='same')(conv1)
    conv1 = Lambda(Maxout)(conv1)    
    conv1 = Conv3D(nf, (3, 3, 1), padding='same')(conv1)
    conv1 = Lambda(Maxout)(conv1)
    conv1 = Conv3D(nf, (3, 3, 3), padding='same')(conv1)
    conv1 = Lambda(Maxout)(conv1)
    pool1 = MaxPooling3D((2, 2, 2), name='pool1')(conv1)   # 152, 128, 128 --> 76, 64, 64
    
    conv2 = Dropout(0.25)(pool1)
    conv2 = Conv3D(nf, (3, 3, 3), padding='same')(conv2)
    conv2 = Lambda(Maxout)(conv2)
    conv2 = Conv3D(nf, (1, 3, 3), padding='same')(conv2)
    conv2 = Lambda(Maxout)(conv2)
    conv2 = Conv3D(nf, (3, 1, 3), padding='same')(conv2)
    conv2 = Lambda(Maxout)(conv2)
    conv2 = Conv3D(nf, (3, 3, 1), padding='same')(conv2)
    conv2 = Lambda(Maxout)(conv2)
    conv2 = Conv3D(nf, (3, 3, 3), padding='same')(conv2)
    conv2 = Lambda(Maxout)(conv2)
    pool2 = MaxPooling3D((2, 2, 2), name='pool2')(conv2)   # 76, 64, 64 --> 38, 32, 32
    
    conv3 = Dropout(0.25)(pool2)
    conv3 = Conv3D(nf, (3, 3, 3), padding='same')(conv3)
    conv3 = Lambda(Maxout)(conv3)
    conv3 = Conv3D(nf, (1, 3, 3), padding='same')(conv3)
    conv3 = Lambda(Maxout)(conv3)
    conv3 = Conv3D(nf, (3, 1, 3), padding='same')(conv3)
    conv3 = Lambda(Maxout)(conv3)
    conv3 = Conv3D(nf, (3, 3, 1), padding='same')(conv3)
    conv3 = Lambda(Maxout)(conv3)
    conv3 = Conv3D(nf, (3, 3, 3), padding='same')(conv3)
    conv3 = Lambda(Maxout)(conv3)
    pool3 = MaxPooling3D((2, 2, 2), name='pool3')(conv3)   # 38, 32, 32 --> 19, 16, 16
    
    conv4 = Dropout(0.25)(pool3)
    conv4 = Conv3D(nf, (3, 3, 3), padding='same')(conv4)
    conv4 = Lambda(Maxout)(conv4)
    conv4 = Conv3D(nf, (1, 3, 3), padding='same')(conv4)
    conv4 = Lambda(Maxout)(conv4)
    conv4 = Conv3D(nf, (3, 1, 3), padding='same')(conv4)
    conv4 = Lambda(Maxout)(conv4)
    conv4 = Conv3D(nf, (3, 3, 1), padding='same')(conv4)
    conv4 = Lambda(Maxout)(conv4)
    conv4 = Conv3D(nf, (3, 3, 3), padding='same')(conv4)
    conv4 = Lambda(Maxout)(conv4)
    pool4 = MaxPooling3D((2, 2, 2), name='pool4')(conv4)   # 19, 16, 16 --> 9,8,8

    flatt = Flatten()(pool4)
    
    out_ = Dropout(0.75)(flatt)
    out_ = Dense(512)(out_)
    out_ = LeakyReLU()(out_)
    out_ = Dropout(0.75)(out_)    
    out_ = Dense(256)(out_)
    out_ = Dropout(0.75)(out_)
    out_ = LeakyReLU()(out_)

    out_1_z_ = Dense(152, activation='relu')(out_)
    out_1_z = Dense(152, activation='softmax', name='out_1_z')(out_1_z_)
    out_1_y_ = Dense(128, activation='relu')(out_)
    out_1_y = Dense(128, activation='softmax', name='out_1_y')(out_1_y_)
    out_1_x_ = Dense(128, activation='relu')(out_)
    out_1_x = Dense(128, activation='softmax', name='out_1_x')(out_1_x_)
    
    
    model = Model( \
        inputs=input_img, \
        outputs= [ \
                   out_1_z, out_1_y, out_1_x
                 ]
    )

    return model

    
 
