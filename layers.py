# -*- coding: utf-8 -*-
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import backend as K
# from tensorflow.keras.activations import sigmoid
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import *

from keras import layers
from keras import backend as K
from keras.activations import sigmoid
from keras.models import *
from keras.layers import *
from keras.optimizers import *

#Encoder
#定义一个可以接收任意数量关键词参数的**kwargs
def Conv(filters,kernel_size,stride=(1, 1),d=(1, 1),pd = "same",initializer="he_normal",**kwargs):

    def layer(x):
        x1 = Conv2D(filters, kernel_size, strides=stride,dilation_rate = d, padding = pd, kernel_initializer=initializer)(x)
        x1 = BatchNormalization()(x1,training=False)
        x1 = ReLU()(x1)

        out = x1
        return out

    return layer

def side_branch(factor):

    def layers(x):
        x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

        kernel_size = (2*factor, 2*factor)
        x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)

        return x
    return layers

def BD(index):

    def layers(inputs):
        x1 = Conv2D(1, 1, strides=1, kernel_initializer="he_normal")(inputs) #c=1,size=1
#         x1 = BatchNormalization()(x1,training=False)
#         x1 = ReLU()(x1)
#         pad1 = ZeroPadding2D(padding=(1, 1))(x1) #226
        p1 = MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(x1) #same dim? 224 valid
        print(p1.shape)
        subtract1 = subtract([x1, p1])
        add1 = add([subtract1,x1])
        # subtract1 = x1 - p1
        # add1 = subtract1 + x1
        return subtract1, add1

    return layers

def Ms_conv(filters):
    
    def layer(x):
        # x1 = Conv(filters, kernel_size = 1)(x)
        # x3 = Conv(filters, kernel_size = 3)(x)
        # x5 = Conv(filters, kernel_size = 5)(x)
        # ###concat 1*1 3*3 5*5
        # concat1_3_5 = concatenate([x1,x3,x5],axis=3) #filters*3
        x3_d1 = Conv(filters, kernel_size = 3,d=(1, 1))(x)
        x3_d2 = Conv(filters, kernel_size = 3,d=(2, 2))(x)
        x3_d4 = Conv(filters, kernel_size = 3,d=(4, 4))(x)
        #concat 3*3 d=1,d=2,d=4
        concatd1_d2_d4 = concatenate([x3_d1,x3_d2,x3_d4],axis=3) #filters*3

        #1*1
        y = Conv(filters, kernel_size = 1)(concatd1_d2_d4)

        out = y
        return out

    return layer

def Asy_conv(filters, kernel_size):
    
    def layer(x):
        
        cx_11 = Conv(filters, 1)(x)

        c33 = Conv(filters, kernel_size)(cx_11)
        c33_11 = Conv(filters, 1)(c33)
        
        c31 = Conv(filters, (kernel_size,1))(cx_11)
        c13 = Conv(filters, (1,kernel_size))(c31)
        c13_11 = Conv(filters, 1)(c13)

        c13_2 = Conv(filters, (1,kernel_size))(cx_11)
        c31_2 = Conv(filters, (kernel_size,1))(c13_2)
        c31_2_11 = Conv(filters, 1)(c31_2)
        
        concat1 = concatenate([c33_11,c13_11,c31_2_11],axis=3) #filters*3
        concat1_11 = Conv(filters, 1)(concat1)
        
        # short_cut = add([x,concat1_11])

        out = concat1_11
        return out
    
    return layer

def Asy_seconv(filters, kernel_size, ratio=16):
    
    def layer(x):
        #inception
        cx_11 = Conv(filters, 1)(x)

        c33 = Conv(filters, kernel_size)(cx_11)
        c33_11 = Conv(filters, 1)(c33)
        
        c31 = Conv(filters, (kernel_size,1))(cx_11)
        c13 = Conv(filters, (1,kernel_size))(c31)
        c13_11 = Conv(filters, 1)(c13)

        c13_2 = Conv(filters, (1,kernel_size))(cx_11)
        c31_2 = Conv(filters, (kernel_size,1))(c13_2)
        c31_2_11 = Conv(filters, 1)(c31_2)
        
        #se
        gap = GlobalAveragePooling2D(name='gap')(cx_11)
        fc_1 = Dense(filters//ratio,activation='relu',kernel_initializer='he_normal', use_bias=False)(gap)
        fc_2 = Dense(filters,activation='relu',kernel_initializer='he_normal', use_bias=False)(fc_1)
        se_weights = Conv2D(filters, 1, activation = 'sigmoid',kernel_initializer='he_normal')(fc_2)
        se_out = multiply([x, se_weights])

        concat1 = concatenate([c33_11,c13_11,c31_2_11,se_out],axis=3) #filters*3
        concat1_11 = Conv(filters, 1)(concat1)
        
        # short_cut = add([x,concat1_11])

        out = concat1_11
        return out
    
    return layer

def Double_conv(filters, kernel_size):
    
    def layer(x):
        x1 = Conv(filters, kernel_size)(x)
        x2 = Conv(filters, kernel_size)(x1)
        
        out = x2
        return out
    
    return layer

#SE
def squeeze_excite_block(input,ratio=8):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    print(init.shape[-1])
    # channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = int(init.shape[-1])
    mid_filters = (filters // ratio)
    se_shape = (1, 1, int(init.shape[-1]))

    se = GlobalAveragePooling2D()(init) #(None,input_filters)
    se = Reshape(se_shape)(se)#(1,1,filters)
    se = Dense(mid_filters, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    # se = Conv(filters//ratio,3)(se)
    # se = Conv(filters,3)(se)
    # se = Conv2D(filters,kernel_size = 1,activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

#SK
def SKConv(M=2, r=16, L=32, G=32, name='skconv'):#

    def wrapper(inputs):
        inputs_shape = tf.shape(inputs)
        b, h, w = inputs_shape[0], inputs_shape[1], inputs_shape[2]
        filters = inputs.get_shape().as_list()[-1]
        d = max(filters//r, L)

        x = inputs

        xs = []
        for m in range(1, M+1):
            if G == 1:
                _x = layers.Conv2D(filters, 3, dilation_rate=m, padding='same',
                              use_bias=False, name=name+'_conv%d'%m)(x)
            else:
                c = filters // G
                _x = layers.DepthwiseConv2D(3, dilation_rate=m, depth_multiplier=c, padding='same',
                                           use_bias=False, name=name+'_conv%d'%m)(x)

                _x = layers.Reshape([h, w, G, c, c], name=name+'_conv%d_reshape1'%m)(_x)
                _x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1),
                                  output_shape=[b, h, w, G, c],
                                  name=name+'_conv%d_sum'%m)(_x)
                _x = layers.Reshape([h, w, filters],
                                   name=name+'_conv%d_reshape2'%m)(_x)


                _x = layers.BatchNormalization(name=name+'_conv%d_bn'%m)(_x)
                _x = layers.Activation('relu', name=name+'_conv%d_relu'%m)(_x)

            xs.append(_x)

        U = layers.Add(name=name+'_add')(xs)
        s = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1,2], keepdims=True),
                          output_shape=[b, 1, 1, filters],
                          name=name+'_gap')(U)

        z = layers.Conv2D(d, 1, name=name+'_fc_z')(s)
        z = layers.BatchNormalization(name=name+'_fc_z_bn')(z)
        z = layers.Activation('relu', name=name+'_fc_z_relu')(z)

        x = layers.Conv2D(filters*M, 1, name=name+'_fc_x')(z)
        x = layers.Reshape([1, 1, filters, M],name=name+'_reshape')(x)
        scale = layers.Softmax(name=name+'_softmax')(x)

        x = layers.Lambda(lambda x: tf.stack(x, axis=-1),
                          output_shape=[b, h, w, filters, M],
                          name=name+'_stack')(xs) # b, h, w, c, M
        x = Axpby(name=name+'_axpby')([scale, x])

        return x
    return wrapper

#CBAM
def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat) 
    assert cbam_feature._keras_shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])

#DANet
#DANet

#mid
def mid_v1(filters,kernel_size,dilation_list=[1,3,5]):
    #input c=512
    #filters c=1024
    def layer(input):
        short_cut = Conv(filters,1)(input)
        #asy_FP
        d1 = Conv(filters,kernel_size,d=dilation_list[0])(input)
        d2 = Conv(filters,kernel_size,d=dilation_list[1])(input)
        d3 = Conv(filters,kernel_size,d=dilation_list[2])(input)
        concatd1_d2_d3 = concatenate([d1,d2,d3],axis=3)
        weight_d1_d2_d3 = Conv2D(filters,kernel_size=1,activation='sigmoid',kernel_initializer='he_normal',use_bias=False)(concatd1_d2_d3)
        weight_d1 = multiply([weight_d1_d2_d3, d1])
        weight_d2 = multiply([weight_d1_d2_d3, d2])
        weight_d3 = multiply([weight_d1_d2_d3, d3])
        add_1 = add([weight_d1,weight_d2,weight_d3]) #y 01.30 10:08 加入残差input test 3_fold——down/1_fold--down 

        #dconv_FN
        c33 = Double_conv(filters, kernel_size)(input)
        c31 = Conv(filters, (kernel_size,1))(input)
        c13 = Conv(filters, (1,kernel_size))(input)
#         c13_2 = Conv(filters, (1,kernel_size))(input)
#         c31_2 = Conv(filters, (kernel_size,1))(c13_2)
        add_2 = add([c33,c13,c31]) #z

        concat_2 = concatenate([add_1,add_2],axis=3)
        concat_2 = Conv(filters,1)(concat_2)
        output = add_1

        return output

    return layer

######-----------------------------------------------------------
def encoder(pre):

    def layer(inputs):
        
#         name = inputs
        index = inputs
        #pre_trained_block
#         pre_trained = pre.get_layer(name = name).output
        pre_trained = pre.get_layer(index=index).output
        print(type(pre_trained))
        print(pre_trained)
        return pre_trained
    
    return layer

#decoder
def unet_decoder(filters,mode = "upsampling"):
    
    def layer(inputs):
        inp, skip3 = inputs
        if mode == "transpose":
            x = Conv2DTranspose(filters, kernel_size=3,strides=(2, 2), padding='same')(inp)
            x = BatchNormalization(x)
            x = ReLU()(x)
        elif mode == 'upsampling':
            x = UpSampling2D(size=2)(inp) #w #28
        else:
            raise ValueError()
        
        # skip3 = UpSampling2D(size=2)(skip3) #28
        concat3 = concatenate([x, skip3], axis=3) #w 28 28
        
#         x3 = Conv2D(filters, kernel_size=3,padding = "same",kernel_initializer="he_normal",
#         kernel_regularizer=regularizers.l2(weight_decay))(concat3) #w 原Conv
        x3 = Double_conv(filters, kernel_size=3)(concat3)
        out = x3
        
        return out

    return layer


#res_decoder
def res_decoder(filters,mode = "upsampling"):
    
    def layer(inputs):
        inp, skip3 = inputs
        if mode == "transpose":
            x = Conv2DTranspose(filters, kernel_size=3,strides=(2, 2), padding='same')(inp)
            x = BatchNormalization(x)
            x = ReLU()(x)
        elif mode == 'upsampling':
            x = UpSampling2D(size=2)(inp) #w #28
        else:
            raise ValueError()
        
        concat3 = concatenate([x, skip3], axis=3) #w 28 28
        x3 = Double_conv(filters, kernel_size=3)(concat3)
        short_cut = Conv(filters,1)(concat3)
        add_1 = add([x3,short_cut])
        add_1 = ReLU()(add_1)
        out = add_1
        
        return out

    return layer
