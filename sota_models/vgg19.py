from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19


#self_define
from losses import *
from layers import *

vgg16 = ['block1_conv2','block2_conv2','block3_conv3','block4_conv3','block5_conv3']
vgg19 = ['block1_conv2','block2_conv2','block3_conv4','block4_conv4','block5_conv4']

filters_list = [64,128,256,512,1024]
IMAGE_SIZE,c = 224,3
input_size=(IMAGE_SIZE,IMAGE_SIZE,c)
VGG16 = VGG16(input_shape = input_size,weights='imagenet',include_top=False)
VGG19 = VGG19(input_shape = input_size,weights='imagenet',include_top=False)

#baseline
def baseline(num_class=1,Pre =VGG16,Pre_list=vgg16,pretrained_weights=None):

    inp = Input(input_size)
    #Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #224 64
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #112 128
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #56 256
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #28 512
    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #14 512
    
    #Decoder
    d_x6 = res_decoder(filters_list[3])((pre_trained_b5,pre_trained_b4)) #in14 14 out28 512
    d_x7 = res_decoder(filters_list[2])((d_x6,pre_trained_b3)) #in28 28 out56 256
    d_x8 = res_decoder(filters_list[1])((d_x7,pre_trained_b2)) #in56 out56 128
    d_x9 = res_decoder(filters_list[0])((d_x8,pre_trained_b1)) #in112 out224 64
    # x = Conv(filters=2, kernel_size=2)(d_x9) 
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x9)
    model = Model(inputs = [Pre.input], outputs = outputs)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#baseline + ASAF
def baseline_ASAF(num_class=1,Pre =VGG16,Pre_list=vgg16,pretrained_weights=None):

    inp = Input(input_size)
    #Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #224 64
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #112 128
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #56 256   
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #28 512
    #Mid
    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #14 512
    mid = mid_v1(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #14 1024
    #Decoder
    d_x6 = res_decoder(filters_list[3])((mid,pre_trained_b4)) #in14 14 out28 512
    d_x7 = res_decoder(filters_list[2])((d_x6,pre_trained_b3)) #in28 28 out56 256
    d_x8 = res_decoder(filters_list[1])((d_x7,pre_trained_b2)) #in56 out56 128
    d_x9 = res_decoder(filters_list[0])((d_x8,pre_trained_b1)) #in112 out224 64
    # x = Conv(filters=2, kernel_size=2)(d_x9) 
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x9)
    model = Model(inputs = [Pre.input], outputs = outputs)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

'''
bd5 = side_branch(16)(pre_trained_b5) #add at 2021.5.31 10:36
bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd5) #224 1
'''
#baseline + BRM_BAM
def baseline_BRM_BAM(num_class=1,Pre =VGG16,Pre_list=vgg16,pretrained_weights=None):

    inp = Input(input_size)
    #Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #224 64
    bd1 = side_branch(1)(pre_trained_b1)
    bd1 = Conv2D(1, 1, activation = 'sigmoid',name='bd1')(bd1) #224 1

    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #112 128
    bd2 = side_branch(2)(pre_trained_b2)
    bd2 = Conv2D(1, 1, activation = 'sigmoid',name='bd2')(bd2) #224 1

    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #56 256 
    bd3 = side_branch(4)(pre_trained_b3)
    bd3 = Conv2D(1, 1, activation = 'sigmoid',name='bd3')(bd3)#224 1

    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #28 512
    bd4 = side_branch(8)(pre_trained_b4)
    bd4 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd4) #224 1

    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #14 512 
    bd5 = side_branch(16)(pre_trained_b5) #add at 2021.5.31 10:36
    bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd5')(bd5) #224 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse45 = Concatenate(axis=-1)([bd4,bd5]) #224 2
    bd_fuse45 = Conv(1, 1)(bd_fuse45) #224 1
    bd_fuse45 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse45')(bd_fuse45) # 224 1
    bd_fuse45_maxpooling = MaxPool2D((8,8))(bd_fuse45) #56 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse345 = Concatenate(axis=-1)([bd3, bd4,bd5]) #224 2
    bd_fuse345 = Conv(1, 1)(bd_fuse345) #224 1
    bd_fuse345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse345')(bd_fuse345) # 224 1
    bd_fuse345_maxpooling = MaxPool2D((4,4))(bd_fuse345) #56 1

    #bd_fuse234 bd2, bd3, bd4
    bd_fuse2345 = Concatenate(axis=-1)([bd2, bd3, bd4,bd5]) #224 3
    bd_fuse2345= Conv(1, 1)(bd_fuse2345) #224 1
    bd_fuse2345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse2345')(bd_fuse2345)
    bd_fuse2345_maxpooling = MaxPool2D((2,2))(bd_fuse2345) #112 1

    #bd_fuse1234 bd1, bd2, bd3, bd4
    bd_fuse12345 = Concatenate(axis=-1)([bd1, bd2, bd3, bd4,bd5]) #224 3
    bd_fuse12345 = Conv(1, 1)(bd_fuse12345) #224 1
    bd_fuse12345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse12345')(bd_fuse12345) #224 1

    #Decoder
    d_x6 = res_decoder(filters_list[3])((pre_trained_b5,pre_trained_b4)) #in14 14 out28 512
#     strong_d_x6 
    d_x6_multiply = multiply([d_x6,bd_fuse45_maxpooling])
#     d_x6_add = add([d_x6_multiply,d_x6])
#     d_x6_add = ReLU()(d_x6_add)

    d_x7 = res_decoder(filters_list[2])((d_x6_multiply,pre_trained_b3)) #in28 28 out56 256 
    #strong_d_x7
    d_x7_multiply = multiply([d_x7,bd_fuse345_maxpooling])
#     d_x7_add = add([d_x7_multiply,d_x7])
#     d_x7_add = ReLU()(d_x7_add)

    d_x8 = res_decoder(filters_list[1])((d_x7_multiply,pre_trained_b2)) #in56 out56 128
    #strong_d_x8
    d_x8_multiply = multiply([d_x8,bd_fuse2345_maxpooling])
#     d_x8_add = add([d_x8_multiply,d_x8])
#     d_x8_add = ReLU()(d_x8_add)

    d_x9 = res_decoder(filters_list[0])((d_x8_multiply,pre_trained_b1)) #in112 out224 64
    #strong_d_x9
    d_x9_multiply = multiply([d_x9,bd_fuse12345])
#     d_x9_add = add([d_x9_multiply,d_x9])
#     d_x9_add = ReLU()(d_x9_add)

    # x = Conv(filters=2, kernel_size=2)(d_x9) 
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x9_multiply)
    model = Model(inputs = [Pre.input], outputs = [bd1,bd2,bd3,bd4,bd5,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,outputs])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#baseline + RRM
def baseline_RRM(num_class=1,Pre =VGG16,Pre_list=vgg16,pretrained_weights=None):

    inp = Input(input_size)
    #Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #224 64
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #112 128
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #56 256 
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #28 512

    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #14 512
    mid_mask1 = side_branch(16)(pre_trained_b5)
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1)

    #Decoder
    d_x6 = res_decoder(filters_list[3])((pre_trained_b5,pre_trained_b4)) #in14 14 out28 512
    d_x6_mask2 = side_branch(8)(d_x6)
    d_x6_mask2 = Conv2D(1, 1, activation = 'sigmoid',name='d_x6_mask2')(d_x6_mask2)

    d_x7 = res_decoder(filters_list[2])((d_x6,pre_trained_b3)) #in28 28 out56 256
    d_x7_mask3 = side_branch(4)(d_x7)
    d_x7_mask3 = Conv2D(1, 1, activation = 'sigmoid',name='d_x7_mask3')(d_x7_mask3)

    d_x8 = res_decoder(filters_list[1])((d_x7,pre_trained_b2)) #in56 out56 128
    d_x8_mask4 = side_branch(2)(d_x8)
    d_x8_mask4 = Conv2D(1, 1, activation = 'sigmoid',name='d_x8_mask4')(d_x8_mask4)

    d_x9 = res_decoder(filters_list[0])((d_x8,pre_trained_b1)) #in112 out224 64
    d_x9_mask5 = side_branch(1)(d_x9)
    d_x9_mask5 = Conv2D(1, 1, activation = 'sigmoid',name='d_x9_mask5')(d_x9_mask5)

    #mask_fuse
    mask_fuse = Concatenate(axis=-1)([mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5])
    mask_fuse = Conv(1, 1)(mask_fuse) # 224 224 1
    mask_fuse = Conv2D(1, 1, activation = 'sigmoid',name='mask_fuse')(mask_fuse)

    # x = Conv(filters=2, kernel_size=2)(d_x9) 
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x9)
    model = Model(inputs = [Pre.input], outputs = [mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,mask_fuse,outputs])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#baseline + ASAF + BRM+BAM 
def baseline_ASAF_BRM_BAM(num_class=1,Pre =VGG16,Pre_list=vgg16,pretrained_weights=None):

    inp = Input(input_size)
    #Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #224 64
    bd1 = side_branch(1)(pre_trained_b1)
    bd1 = Conv2D(1, 1, activation = 'sigmoid',name='bd1')(bd1) #224 1

    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #112 128
    bd2 = side_branch(2)(pre_trained_b2)
    bd2 = Conv2D(1, 1, activation = 'sigmoid',name='bd2')(bd2) #224 1

    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #56 256 
    bd3 = side_branch(4)(pre_trained_b3)
    bd3 = Conv2D(1, 1, activation = 'sigmoid',name='bd3')(bd3)#224 1

    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #28 512
    bd4 = side_branch(8)(pre_trained_b4)
    bd4 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd4) #224 1

    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #14 512 
    mid = mid_v1(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #14 1024
    bd5 = side_branch(16)(pre_trained_b5) #add at 2021.5.31 10:36
    bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd5')(bd5) #224 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse45 = Concatenate(axis=-1)([bd4,bd5]) #224 2
    bd_fuse45 = Conv(1, 1)(bd_fuse45) #224 1
    bd_fuse45 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse45')(bd_fuse45) # 224 1
    bd_fuse45_maxpooling = MaxPool2D((8,8))(bd_fuse45) #56 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse345 = Concatenate(axis=-1)([bd3, bd4,bd5]) #224 2
    bd_fuse345 = Conv(1, 1)(bd_fuse345) #224 1
    bd_fuse345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse345')(bd_fuse345) # 224 1
    bd_fuse345_maxpooling = MaxPool2D((4,4))(bd_fuse345) #56 1

    #bd_fuse234 bd2, bd3, bd4
    bd_fuse2345 = Concatenate(axis=-1)([bd2, bd3, bd4,bd5]) #224 3
    bd_fuse2345= Conv(1, 1)(bd_fuse2345) #224 1
    bd_fuse2345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse2345')(bd_fuse2345)
    bd_fuse2345_maxpooling = MaxPool2D((2,2))(bd_fuse2345) #112 1

    #bd_fuse1234 bd1, bd2, bd3, bd4
    bd_fuse12345 = Concatenate(axis=-1)([bd1, bd2, bd3, bd4,bd5]) #224 3
    bd_fuse12345 = Conv(1, 1)(bd_fuse12345) #224 1
    bd_fuse12345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse12345')(bd_fuse12345) #224 1
    #Decoder
    d_x6 = res_decoder(filters_list[3])((mid,pre_trained_b4)) #in14 14 out28 512
    #strong_d_x6
    d_x6_multiply = multiply([d_x6,bd_fuse45_maxpooling])
#     d_x6_add = add([d_x6_multiply,d_x6])
#     d_x6_add = ReLU()(d_x6_add)

    d_x7 = res_decoder(filters_list[2])((d_x6_multiply,pre_trained_b3)) #in28 28 out56 256
    #strong_d_x7
    d_x7_multiply = multiply([d_x7,bd_fuse345_maxpooling])
#     d_x7_add = add([d_x7_multiply,d_x7])
#     d_x7_add = ReLU()(d_x7_add)

    d_x8 = res_decoder(filters_list[1])((d_x7_multiply,pre_trained_b2)) #in56 out56 128
    #strong_d_x8
    d_x8_multiply = multiply([d_x8,bd_fuse2345_maxpooling])
#     d_x8_add = add([d_x8_multiply,d_x8])
#     d_x8_add = ReLU()(d_x8_add)

    d_x9 = res_decoder(filters_list[0])((d_x8_multiply,pre_trained_b1)) #in112 out224 64
    #strong_d_x9
    d_x9_multiply = multiply([d_x9,bd_fuse12345])
#     d_x9_add = add([d_x9_multiply,d_x9])
#     d_x9_add = ReLU()(d_x9_add)

    # x = Conv(filters=2, kernel_size=2)(d_x9) 
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x9_multiply)
    model = Model(inputs = [Pre.input], outputs = [bd1,bd2,bd3,bd4,bd5,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,outputs])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#baseline + ASAF + RRM
def baseline_ASAF_RRM(num_class=1,Pre =VGG16,Pre_list=vgg16,pretrained_weights=None):

    inp = Input(input_size)
    #Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #224 64
    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #112 128
    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #56 256 
    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #28 512

    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #14 512
    mid = mid_v1(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #14 1024
    mid_mask1 = side_branch(16)(mid)
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1)

    #Decoder
    d_x6 = res_decoder(filters_list[3])((mid,pre_trained_b4)) #in14 14 out28 512
    d_x6_mask2 = side_branch(8)(d_x6)
    d_x6_mask2 = Conv2D(1, 1, activation = 'sigmoid',name='d_x6_mask2')(d_x6_mask2)

    d_x7 = res_decoder(filters_list[2])((d_x6,pre_trained_b3)) #in28 28 out56 256
    d_x7_mask3 = side_branch(4)(d_x7)
    d_x7_mask3 = Conv2D(1, 1, activation = 'sigmoid',name='d_x7_mask3')(d_x7_mask3)

    d_x8 = res_decoder(filters_list[1])((d_x7,pre_trained_b2)) #in56 out56 128
    d_x8_mask4 = side_branch(2)(d_x8)
    d_x8_mask4 = Conv2D(1, 1, activation = 'sigmoid',name='d_x8_mask4')(d_x8_mask4)

    d_x9 = res_decoder(filters_list[0])((d_x8,pre_trained_b1)) #in112 out224 64
    d_x9_mask5 = side_branch(1)(d_x9)
    d_x9_mask5 = Conv2D(1, 1, activation = 'sigmoid',name='d_x9_mask5')(d_x9_mask5)

    #mask_fuse
    mask_fuse = Concatenate(axis=-1)([mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5])
    mask_fuse = Conv(1, 1)(mask_fuse) # 224 224 1
    mask_fuse = Conv2D(1, 1, activation = 'sigmoid',name='mask_fuse')(mask_fuse)

    # x = Conv(filters=2, kernel_size=2)(d_x9) 
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x9)
    model = Model(inputs = [Pre.input], outputs = [mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,mask_fuse,outputs])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#baseline + BRM+BAM + RRM
def baseline_BRM_BAM_RRM(num_class=1,Pre =VGG16,Pre_list=vgg16,pretrained_weights=None):

    inp = Input(input_size)
    #Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #224 64
    bd1 = side_branch(1)(pre_trained_b1)
    bd1 = Conv2D(1, 1, activation = 'sigmoid',name='bd1')(bd1) #224 1

    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #112 128
    bd2 = side_branch(2)(pre_trained_b2)
    bd2 = Conv2D(1, 1, activation = 'sigmoid',name='bd2')(bd2) #224 1

    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #56 256 
    bd3 = side_branch(4)(pre_trained_b3)
    bd3 = Conv2D(1, 1, activation = 'sigmoid',name='bd3')(bd3)#224 1

    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #28 512
    bd4 = side_branch(8)(pre_trained_b4)
    bd4 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd4) #224 1

    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #14 512
    bd5 = side_branch(16)(pre_trained_b5)
    bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd5')(bd5) #224 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse45 = Concatenate(axis=-1)([bd4,bd5]) #224 2
    bd_fuse45 = Conv(1, 1)(bd_fuse45) #224 1
    bd_fuse45 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse45')(bd_fuse45) # 224 1
    bd_fuse45_maxpooling = MaxPool2D((8,8))(bd_fuse45) #56 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse345 = Concatenate(axis=-1)([bd3, bd4,bd5]) #224 2
    bd_fuse345 = Conv(1, 1)(bd_fuse345) #224 1
    bd_fuse345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse345')(bd_fuse345) # 224 1
    bd_fuse345_maxpooling = MaxPool2D((4,4))(bd_fuse345) #56 1

    #bd_fuse234 bd2, bd3, bd4
    bd_fuse2345 = Concatenate(axis=-1)([bd2, bd3, bd4,bd5]) #224 3
    bd_fuse2345= Conv(1, 1)(bd_fuse2345) #224 1
    bd_fuse2345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse2345')(bd_fuse2345)
    bd_fuse2345_maxpooling = MaxPool2D((2,2))(bd_fuse2345) #112 1

    #bd_fuse1234 bd1, bd2, bd3, bd4
    bd_fuse12345 = Concatenate(axis=-1)([bd1, bd2, bd3, bd4,bd5]) #224 3
    bd_fuse12345 = Conv(1, 1)(bd_fuse12345) #224 1
    bd_fuse12345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse12345')(bd_fuse12345) #224 1

#     mid = mid_v1(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #14 1024
    mid_mask1 = side_branch(16)(pre_trained_b5)
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1)

    #Decoder
    d_x6 = res_decoder(filters_list[3])((pre_trained_b5,pre_trained_b4)) #in14 14 out28 512
    #strong_d_x6
    d_x6_multiply = multiply([d_x6,bd_fuse45_maxpooling])
#     d_x6_add = add([d_x6_multiply,d_x6])
#     d_x6_add = ReLU()(d_x6_add)
    #mask_side1
    d_x6_mask2 = side_branch(8)(d_x6)
    d_x6_mask2 = Conv2D(1, 1, activation = 'sigmoid',name='d_x6_mask2')(d_x6_mask2)

    d_x7 = res_decoder(filters_list[2])((d_x6_multiply,pre_trained_b3)) #in28 28 out56 256
    #strong_d_x7
    d_x7_multiply = multiply([d_x7,bd_fuse345_maxpooling])
#     d_x7_add = add([d_x7_multiply,d_x7])
#     d_x7_add = ReLU()(d_x7_add)
    #mask_side2
    d_x7_mask3 = side_branch(4)(d_x7)
    d_x7_mask3 = Conv2D(1, 1, activation = 'sigmoid',name='d_x7_mask3')(d_x7_mask3)

    d_x8 = res_decoder(filters_list[1])((d_x7_multiply,pre_trained_b2)) #in56 out56 128
    #strong_d_x8
    d_x8_multiply = multiply([d_x8,bd_fuse2345_maxpooling])
#     d_x8_add = add([d_x8_multiply,d_x8])
#     d_x8_add = ReLU()(d_x8_add)
    #mask_side3
    d_x8_mask4 = side_branch(2)(d_x8)
    d_x8_mask4 = Conv2D(1, 1, activation = 'sigmoid',name='d_x8_mask4')(d_x8_mask4)

    d_x9 = res_decoder(filters_list[0])((d_x8_multiply,pre_trained_b1)) #in112 out224 64
    #strong_d_x9
    d_x9_multiply = multiply([d_x9,bd_fuse12345])
#     d_x9_add = add([d_x9_multiply,d_x9])
#     d_x9_add = ReLU()(d_x9_add)
    #mask_side4
    d_x9_mask5 = side_branch(1)(d_x9)
    d_x9_mask5 = Conv2D(1, 1, activation = 'sigmoid',name='d_x9_mask5')(d_x9_mask5)

    #mask_fuse
    mask_fuse = Concatenate(axis=-1)([mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5])
    mask_fuse = Conv(1, 1)(mask_fuse) # 224 224 1
    mask_fuse = Conv2D(1, 1, activation = 'sigmoid',name='mask_fuse')(mask_fuse)

    # x = Conv(filters=2, kernel_size=2)(d_x9) 
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x9_multiply)
    model = Model(inputs = [Pre.input], outputs = [bd1,bd2,bd3,bd4,bd5,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,mask_fuse,outputs])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#baseline + ASAF + BRM_BAM + RRM
def baseline_ASAF_BRM_BAM_RRM(num_class=1,Pre =VGG16,Pre_list=vgg16,pretrained_weights=None):

    inp = Input(input_size)
    #Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #224 64
    bd1 = side_branch(1)(pre_trained_b1)
    bd1 = Conv2D(1, 1, activation = 'sigmoid',name='bd1')(bd1) #224 1

    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #112 128
    bd2 = side_branch(2)(pre_trained_b2)
    bd2 = Conv2D(1, 1, activation = 'sigmoid',name='bd2')(bd2) #224 1

    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #56 256 
    bd3 = side_branch(4)(pre_trained_b3)
    bd3 = Conv2D(1, 1, activation = 'sigmoid',name='bd3')(bd3)#224 1

    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #28 512
    bd4 = side_branch(8)(pre_trained_b4)
    bd4 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd4) #224 1

    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #14 512
    bd5 = side_branch(16)(pre_trained_b5)
    bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd5')(bd5) #224 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse45 = Concatenate(axis=-1)([bd4,bd5]) #224 2
    bd_fuse45 = Conv(1, 1)(bd_fuse45) #224 1
    bd_fuse45 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse45')(bd_fuse45) # 224 1
    bd_fuse45_maxpooling = MaxPool2D((8,8))(bd_fuse45) #56 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse345 = Concatenate(axis=-1)([bd3, bd4,bd5]) #224 2
    bd_fuse345 = Conv(1, 1)(bd_fuse345) #224 1
    bd_fuse345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse345')(bd_fuse345) # 224 1
    bd_fuse345_maxpooling = MaxPool2D((4,4))(bd_fuse345) #56 1

    #bd_fuse234 bd2, bd3, bd4
    bd_fuse2345 = Concatenate(axis=-1)([bd2, bd3, bd4,bd5]) #224 3
    bd_fuse2345= Conv(1, 1)(bd_fuse2345) #224 1
    bd_fuse2345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse2345')(bd_fuse2345)
    bd_fuse2345_maxpooling = MaxPool2D((2,2))(bd_fuse2345) #112 1

    #bd_fuse1234 bd1, bd2, bd3, bd4
    bd_fuse12345 = Concatenate(axis=-1)([bd1, bd2, bd3, bd4,bd5]) #224 3
    bd_fuse12345 = Conv(1, 1)(bd_fuse12345) #224 1
    bd_fuse12345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse12345')(bd_fuse12345) #224 1

    mid = mid_v1(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #14 1024
    mid_mask1 = side_branch(16)(mid)
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1)

    #Decoder
    d_x6 = res_decoder(filters_list[3])((mid,pre_trained_b4)) #in14 14 out28 512
    #strong_d_x6
    d_x6_multiply = multiply([d_x6,bd_fuse45_maxpooling])
#     d_x6_add = add([d_x6_multiply,d_x6])
#     d_x6_add = ReLU()(d_x6_add)
    #mask_side1
    d_x6_mask2 = side_branch(8)(d_x6)
    d_x6_mask2 = Conv2D(1, 1, activation = 'sigmoid',name='d_x6_mask2')(d_x6_mask2)

    d_x7 = res_decoder(filters_list[2])((d_x6_multiply,pre_trained_b3)) #in28 28 out56 256
    #strong_d_x7
    d_x7_multiply = multiply([d_x7,bd_fuse345_maxpooling])
#     d_x7_add = add([d_x7_multiply,d_x7])
#     d_x7_add = ReLU()(d_x7_add)
    #mask_side2
    d_x7_mask3 = side_branch(4)(d_x7)
    d_x7_mask3 = Conv2D(1, 1, activation = 'sigmoid',name='d_x7_mask3')(d_x7_mask3)

    d_x8 = res_decoder(filters_list[1])((d_x7_multiply,pre_trained_b2)) #in56 out56 128
    #strong_d_x8
    d_x8_multiply = multiply([d_x8,bd_fuse2345_maxpooling])
#     d_x8_add = add([d_x8_multiply,d_x8])
#     d_x8_add = ReLU()(d_x8_add)
    #mask_side3
    d_x8_mask4 = side_branch(2)(d_x8)
    d_x8_mask4 = Conv2D(1, 1, activation = 'sigmoid',name='d_x8_mask4')(d_x8_mask4)

    d_x9 = res_decoder(filters_list[0])((d_x8_multiply,pre_trained_b1)) #in112 out224 64
    #strong_d_x9
    d_x9_multiply = multiply([d_x9,bd_fuse12345])
#     d_x9_add = add([d_x9_multiply,d_x9])
#     d_x9_add = ReLU()(d_x9_add)
    #mask_side4
    d_x9_mask5 = side_branch(1)(d_x9)
    d_x9_mask5 = Conv2D(1, 1, activation = 'sigmoid',name='d_x9_mask5')(d_x9_mask5)

    #mask_fuse
    mask_fuse = Concatenate(axis=-1)([mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5])
    mask_fuse = Conv(1, 1)(mask_fuse) # 224 224 1
    mask_fuse = Conv2D(1, 1, activation = 'sigmoid',name='mask_fuse')(mask_fuse)

    # x = Conv(filters=2, kernel_size=2)(d_x9) 
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x9_multiply)
    model = Model(inputs = [Pre.input], outputs = [bd1,bd2,bd3,bd4,bd5,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,mask_fuse,outputs])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#vgg19_res_decoder
def vgg19_res_decoder(num_class=1,Pre =VGG19,Pre_list=vgg19,pretrained_weights=None):

    inp = Input(input_size)
    #Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #224 64
    bd1 = side_branch(1)(pre_trained_b1)
    bd1 = Conv2D(1, 1, activation = 'sigmoid',name='bd1')(bd1) #224 1

    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #112 128
    bd2 = side_branch(2)(pre_trained_b2)
    bd2 = Conv2D(1, 1, activation = 'sigmoid',name='bd2')(bd2) #224 1

    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #56 256 
    bd3 = side_branch(4)(pre_trained_b3)
    bd3 = Conv2D(1, 1, activation = 'sigmoid',name='bd3')(bd3)#224 1

    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #28 512
    bd4 = side_branch(8)(pre_trained_b4)
    bd4 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd4) #224 1

    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #14 512
    bd5 = side_branch(16)(pre_trained_b5)
    bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd5')(bd5) #224 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse45 = Concatenate(axis=-1)([bd4,bd5]) #224 2
    bd_fuse45 = Conv(1, 1)(bd_fuse45) #224 1
    bd_fuse45 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse45')(bd_fuse45) # 224 1
    bd_fuse45_maxpooling = MaxPool2D((8,8))(bd_fuse45) #56 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse345 = Concatenate(axis=-1)([bd3, bd4,bd5]) #224 2
    bd_fuse345 = Conv(1, 1)(bd_fuse345) #224 1
    bd_fuse345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse345')(bd_fuse345) # 224 1
    bd_fuse345_maxpooling = MaxPool2D((4,4))(bd_fuse345) #56 1

    #bd_fuse234 bd2, bd3, bd4
    bd_fuse2345 = Concatenate(axis=-1)([bd2, bd3, bd4,bd5]) #224 3
    bd_fuse2345= Conv(1, 1)(bd_fuse2345) #224 1
    bd_fuse2345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse2345')(bd_fuse2345)
    bd_fuse2345_maxpooling = MaxPool2D((2,2))(bd_fuse2345) #112 1

    #bd_fuse1234 bd1, bd2, bd3, bd4
    bd_fuse12345 = Concatenate(axis=-1)([bd1, bd2, bd3, bd4,bd5]) #224 3
    bd_fuse12345 = Conv(1, 1)(bd_fuse12345) #224 1
    bd_fuse12345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse12345')(bd_fuse12345) #224 1

    mid = mid_v1(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #14 1024
    mid_mask1 = side_branch(16)(mid)
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1)

    #Decoder
    d_x6 = res_decoder(filters_list[3])((mid,pre_trained_b4)) #in14 14 out28 512
    #strong_d_x6
    d_x6_multiply = multiply([d_x6,bd_fuse45_maxpooling])
#     d_x6_add = add([d_x6_multiply,d_x6])
#     d_x6_add = ReLU()(d_x6_add)
    #mask_side1
    d_x6_mask2 = side_branch(8)(d_x6)
    d_x6_mask2 = Conv2D(1, 1, activation = 'sigmoid',name='d_x6_mask2')(d_x6_mask2)

    d_x7 = res_decoder(filters_list[2])((d_x6_multiply,pre_trained_b3)) #in28 28 out56 256
    #strong_d_x7
    d_x7_multiply = multiply([d_x7,bd_fuse345_maxpooling])
#     d_x7_add = add([d_x7_multiply,d_x7])
#     d_x7_add = ReLU()(d_x7_add)
    #mask_side2
    d_x7_mask3 = side_branch(4)(d_x7)
    d_x7_mask3 = Conv2D(1, 1, activation = 'sigmoid',name='d_x7_mask3')(d_x7_mask3)

    d_x8 = res_decoder(filters_list[1])((d_x7_multiply,pre_trained_b2)) #in56 out56 128
    #strong_d_x8
    d_x8_multiply = multiply([d_x8,bd_fuse2345_maxpooling])
#     d_x8_add = add([d_x8_multiply,d_x8])
#     d_x8_add = ReLU()(d_x8_add)
    #mask_side3
    d_x8_mask4 = side_branch(2)(d_x8)
    d_x8_mask4 = Conv2D(1, 1, activation = 'sigmoid',name='d_x8_mask4')(d_x8_mask4)

    d_x9 = res_decoder(filters_list[0])((d_x8_multiply,pre_trained_b1)) #in112 out224 64
    #strong_d_x9
    d_x9_multiply = multiply([d_x9,bd_fuse12345])
#     d_x9_add = add([d_x9_multiply,d_x9])
#     d_x9_add = ReLU()(d_x9_add)
    #mask_side4
    d_x9_mask5 = side_branch(1)(d_x9)
    d_x9_mask5 = Conv2D(1, 1, activation = 'sigmoid',name='d_x9_mask5')(d_x9_mask5)

    #mask_fuse
    mask_fuse = Concatenate(axis=-1)([mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5])
    mask_fuse = Conv(1, 1)(mask_fuse) # 224 224 1
    mask_fuse = Conv2D(1, 1, activation = 'sigmoid',name='mask_fuse')(mask_fuse)

    # x = Conv(filters=2, kernel_size=2)(d_x9) 
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x9_multiply)
    model = Model(inputs = [Pre.input], outputs = [bd1,bd2,bd3,bd4,bd5,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,mask_fuse,outputs])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#vgg16_decoder
def vgg16_decoder(num_class=1,Pre =VGG16,Pre_list=vgg16,pretrained_weights=None):

    inp = Input(input_size)
    #Encoder
    pre_trained_b1 = encoder(Pre)((Pre_list[0])) #224 64
    bd1 = side_branch(1)(pre_trained_b1)
    bd1 = Conv2D(1, 1, activation = 'sigmoid',name='bd1')(bd1) #224 1

    pre_trained_b2 = encoder(Pre)((Pre_list[1])) #112 128
    bd2 = side_branch(2)(pre_trained_b2)
    bd2 = Conv2D(1, 1, activation = 'sigmoid',name='bd2')(bd2) #224 1

    pre_trained_b3 = encoder(Pre)((Pre_list[2])) #56 256 
    bd3 = side_branch(4)(pre_trained_b3)
    bd3 = Conv2D(1, 1, activation = 'sigmoid',name='bd3')(bd3)#224 1

    pre_trained_b4 = encoder(Pre)((Pre_list[3])) #28 512
    bd4 = side_branch(8)(pre_trained_b4)
    bd4 = Conv2D(1, 1, activation = 'sigmoid',name='bd4')(bd4) #224 1

    pre_trained_b5 = encoder(Pre)((Pre_list[4])) #14 512
    bd5 = side_branch(16)(pre_trained_b5)
    bd5 = Conv2D(1, 1, activation = 'sigmoid',name='bd5')(bd5) #224 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse45 = Concatenate(axis=-1)([bd4,bd5]) #224 2
    bd_fuse45 = Conv(1, 1)(bd_fuse45) #224 1
    bd_fuse45 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse45')(bd_fuse45) # 224 1
    bd_fuse45_maxpooling = MaxPool2D((8,8))(bd_fuse45) #56 1
    
    #bd_fuse34 bd3, bd4
    bd_fuse345 = Concatenate(axis=-1)([bd3, bd4,bd5]) #224 2
    bd_fuse345 = Conv(1, 1)(bd_fuse345) #224 1
    bd_fuse345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse345')(bd_fuse345) # 224 1
    bd_fuse345_maxpooling = MaxPool2D((4,4))(bd_fuse345) #56 1

    #bd_fuse234 bd2, bd3, bd4
    bd_fuse2345 = Concatenate(axis=-1)([bd2, bd3, bd4,bd5]) #224 3
    bd_fuse2345= Conv(1, 1)(bd_fuse2345) #224 1
    bd_fuse2345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse2345')(bd_fuse2345)
    bd_fuse2345_maxpooling = MaxPool2D((2,2))(bd_fuse2345) #112 1

    #bd_fuse1234 bd1, bd2, bd3, bd4
    bd_fuse12345 = Concatenate(axis=-1)([bd1, bd2, bd3, bd4,bd5]) #224 3
    bd_fuse12345 = Conv(1, 1)(bd_fuse12345) #224 1
    bd_fuse12345 = Conv2D(1, 1, activation = 'sigmoid',name='bd_fuse12345')(bd_fuse12345) #224 1

    mid = mid_v1(filters_list[4],3,dilation_list=[1,3,5])(pre_trained_b5) #14 1024
    mid_mask1 = side_branch(16)(mid)
    mid_mask1 = Conv2D(1, 1, activation = 'sigmoid',name='mid_mask1')(mid_mask1)

    #Decoder
    d_x6 = unet_decoder(filters_list[3])((mid,pre_trained_b4)) #in14 14 out28 512
    #strong_d_x6
    d_x6_multiply = multiply([d_x6,bd_fuse45_maxpooling])
#     d_x6_add = add([d_x6_multiply,d_x6])
#     d_x6_add = ReLU()(d_x6_add)
    #mask_side1
    d_x6_mask2 = side_branch(8)(d_x6)
    d_x6_mask2 = Conv2D(1, 1, activation = 'sigmoid',name='d_x6_mask2')(d_x6_mask2)

    d_x7 = unet_decoder(filters_list[2])((d_x6_multiply,pre_trained_b3)) #in28 28 out56 256
    #strong_d_x7
    d_x7_multiply = multiply([d_x7,bd_fuse345_maxpooling])
#     d_x7_add = add([d_x7_multiply,d_x7])
#     d_x7_add = ReLU()(d_x7_add)
    #mask_side2
    d_x7_mask3 = side_branch(4)(d_x7)
    d_x7_mask3 = Conv2D(1, 1, activation = 'sigmoid',name='d_x7_mask3')(d_x7_mask3)

    d_x8 = unet_decoder(filters_list[1])((d_x7_multiply,pre_trained_b2)) #in56 out56 128
    #strong_d_x8
    d_x8_multiply = multiply([d_x8,bd_fuse2345_maxpooling])
#     d_x8_add = add([d_x8_multiply,d_x8])
#     d_x8_add = ReLU()(d_x8_add)
    #mask_side3
    d_x8_mask4 = side_branch(2)(d_x8)
    d_x8_mask4 = Conv2D(1, 1, activation = 'sigmoid',name='d_x8_mask4')(d_x8_mask4)

    d_x9 = unet_decoder(filters_list[0])((d_x8_multiply,pre_trained_b1)) #in112 out224 64
    #strong_d_x9
    d_x9_multiply = multiply([d_x9,bd_fuse12345])
#     d_x9_add = add([d_x9_multiply,d_x9])
#     d_x9_add = ReLU()(d_x9_add)
    #mask_side4
    d_x9_mask5 = side_branch(1)(d_x9)
    d_x9_mask5 = Conv2D(1, 1, activation = 'sigmoid',name='d_x9_mask5')(d_x9_mask5)

    #mask_fuse
    mask_fuse = Concatenate(axis=-1)([mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5])
    mask_fuse = Conv(1, 1)(mask_fuse) # 224 224 1
    mask_fuse = Conv2D(1, 1, activation = 'sigmoid',name='mask_fuse')(mask_fuse)

    # x = Conv(filters=2, kernel_size=2)(d_x9) 
    outputs = Conv2D(num_class, 1, activation = 'sigmoid',name='outputs')(d_x9_multiply)
    model = Model(inputs = [Pre.input], outputs = [bd1,bd2,bd3,bd4,bd5,bd_fuse45,bd_fuse345,bd_fuse2345,bd_fuse12345,mid_mask1,d_x6_mask2,d_x7_mask3,d_x8_mask4,d_x9_mask5,mask_fuse,outputs])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model



