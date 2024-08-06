from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.vgg16 import VGG16
# import keras

#self_define
from losses import *
from sota_layers import *

filters_list = [64,128,256,512,1024]
IMAGE_SIZE,c = 224,3
input_size=(IMAGE_SIZE,IMAGE_SIZE,c)
kinit = 'glorot_normal'
#FCN_8s
def FCN_8s(num_class=1,pretrained_weights=None):

    project_name = "fcn_segment"
    channels = 3
    std_shape = (224,224, channels) 
    img_input = Input(shape = std_shape, name = "input")

    block1_conv1 = Conv2D(64, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block1_conv1")(img_input)
    block1_conv2 = Conv2D(64, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block1_conv2")(block1_conv1)            
    block1_pool  = MaxPooling2D(pool_size = (2, 2), strides = (2, 2),name = "block1_pool")(block1_conv2)

    block2_conv1 = Conv2D(128, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block2_conv1")(block1_pool)
    block2_conv2 = Conv2D(128, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block2_conv2")(block2_conv1)            
    block2_pool  = MaxPooling2D(pool_size = (2, 2), strides = (2, 2),name = "block2_pool")(block2_conv2)

    block3_conv1 = Conv2D(256, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block3_conv1")(block2_pool)
    block3_conv2 = Conv2D(256, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block3_conv2")(block3_conv1)    
    block3_conv3 = Conv2D(256, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block3_conv3")(block3_conv2)           
    block3_pool  = MaxPooling2D(pool_size = (2, 2), strides = (2, 2),name = "block3_pool")(block3_conv3)

    block4_conv1 = Conv2D(512, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block4_conv1")(block3_pool)
    block4_conv2 = Conv2D(512, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block4_conv2")(block4_conv1)    
    block4_conv3 = Conv2D(512, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block4_conv3")(block4_conv2)           
    block4_pool  = MaxPooling2D(pool_size = (2, 2), strides = (2, 2),name = "block4_pool")(block4_conv3)

    block5_conv1 = Conv2D(512, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block5_conv1")(block4_pool)
    block5_conv2 = Conv2D(512, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block5_conv2")(block5_conv1)    
    block5_conv3 = Conv2D(512, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block5_conv3")(block5_conv2)           
    block5_pool  = MaxPooling2D(pool_size = (2, 2), strides = (2, 2),name = "block5_pool")(block5_conv3)

    block6_conv1 = Conv2D(4096, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block6_conv1")(block5_pool)
    block6_conv2 = Conv2D(4096, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block6_conv2")(block6_conv1)    
#     block6_conv3 = Conv2D(1000, kernel_size = (3, 3), activation = "relu",padding = "same", name = "block6_conv3")(block6_conv2)  


   # _32s = keras.layers.Conv2DTranspose(256, kernel_size = (3, 3),
   #                                     strides = (32, 32),
   #                                     padding = "same",
   #                                     kernel_initializer = "he_normal",
   #                                     name = "upsamping_6")(block6_conv3)                                 

    up6 = Conv2DTranspose(512, kernel_size = (3, 3),
                                       strides = (2, 2),
                                       padding = "same",
                                       kernel_initializer = "he_normal",
                                       name = "upsamping_6")(block6_conv2)
                    
    _16s = add([block4_pool, up6])

    # _16s 转置卷积上采样 2 倍和 max_pool_3 一样大
    up_16s = Conv2DTranspose(256, kernel_size = (3, 3),
                                          strides = (2, 2),
                                          padding = "same",
                                          kernel_initializer = "he_normal",
                                          name = "Conv2DTranspose_16s")(_16s)
                                      
    _8s = add([block3_pool, up_16s])

    # _16s 转置卷积上采样 2 倍和 max_pool_3 一样大
    up7 = Conv2DTranspose(256, kernel_size = (3, 3),
                                          strides = (8, 8),
                                          padding = "same",
                                          kernel_initializer = "he_normal",
                                          name = "up7")(_8s)

    # 这里 kernel 也是 3 * 3, 也可以同 FCN-32s 那样修改的
    conv_7 = Conv2D(num_class, kernel_size = (3, 3), activation = "sigmoid",
                                 padding = "same",name='outputs')(up7)

    model = Model(img_input, conv_7, name = project_name)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#UNet
def unet(pretrained_weights = None,input_size = (224,224,3),num_class = 1):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#SegNet
def SegNet(pretrained_weights = None,num_class = 1,input_size = (224,224,3), kernel=3, pool_size=(2, 2), output_mode="softmax"):
    # encoder
    inputs = Input(shape=input_size)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(num_class, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape(
        (input_shape[0] * input_shape[1], num_class),
        input_shape=(input_shape[0], input_shape[1], num_class),
    )(conv_26)

    outputs = Activation(output_mode,name='outputs')(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#PsPNet


#Att_UNet
def Att_UNet(pretrained_weights = None,input_size = (224,224,3),num_class = 1):   
    inputs = Input(shape=input_size)
    conv1 = UnetConv2D(inputs, 64, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UnetConv2D(pool1, 128, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 256, is_batchnorm=True, name='conv3')
    #conv3 = Dropout(0.2,name='drop_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 512, is_batchnorm=True, name='conv4')
    #conv4 = Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    center = UnetConv2D(pool4, 1024, is_batchnorm=True, name='center')
    
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 512, '_1')
    up1 = concatenate([Conv2DTranspose(512, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')
    
    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 256, '_2')
    up2 = concatenate([Conv2DTranspose(256, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 64, '_3')
    up3 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')

    up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')
    out = Conv2D(num_class, (1, 1), activation='sigmoid',  kernel_initializer=kinit, name='outputs')(up4)
    
    model = Model(inputs=[inputs], outputs=[out])
    # model.compile(optimizer=opt, loss=lossfxn, metrics=[losses.dsc,losses.tp,losses.tn])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

