#coding=utf-8

from keras.applications import vgg16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation
from tensorflow.keras.layers import *

def FCN8_helper(nClasses=1, input_height=224, input_width=224):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input)
    # assert isinstance(model, Model)
    """
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
    """
    o = Conv2D(
        filters=4096,
        kernel_size=(7,7),
        padding="same",
        activation="relu",
        name="fc6")(model.output)
    # o = Dropout(rate=0.5)(o)

    o = Conv2D(
        filters=4096,
        kernel_size=(1,1),
        padding="same",
        activation="relu",
        name="fc7")(o)
    # o = Dropout(rate=0.5)(o)

    o = Conv2D(filters=nClasses, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal",
               name="score_fr")(o)

    o = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score2")(o)

    fcn8 = Model(inputs=img_input, outputs=o)

    return fcn8



def FCN8(nClasses=1, input_height=224, input_width=224,pretrained_weights=None):


    fcn8 = FCN8_helper(nClasses, input_height=224, input_width=224)



    # Conv to be applied on Pool4
    skip_con1 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool4")(fcn8.get_layer("block4_pool").output)
    Summed = add(inputs=[skip_con1, fcn8.output])

    x = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score4")(Summed)

    ###
    skip_con2 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool3")(fcn8.get_layer("block3_pool").output)
    Summed2 = add(inputs=[skip_con2, x])

    #####
    Up = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8),
                         padding="valid", activation=None, name="upsample")(Summed2)

    

    o_shape = Model(inputs=fcn8.input, outputs=Up).output_shape



    outputHeight = o_shape[1]

    outputWidth = o_shape[2]

    # Up = Reshape((-1, nClasses))(Up)
    Up = Activation("sigmoid",name='outputs')(Up)

    model = Model(inputs=fcn8.input, outputs=Up)

    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model



	
