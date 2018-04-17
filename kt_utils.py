
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.initializers import glorot_uniform


def identity_block(X,f,filters,stage,block):
    #参数含义：
    # X:输入的张量(m,h,w,c)
    #f:整数型，为主路径中间处卷积的窗口的大小
    #filters：python的整数列表，定义主路径CONV层中的过滤器数量
    #stage：整数型，用于命名层，依赖于在网络中的位置
    #block：字符串/字符型，用于命名层，依赖于在网络中的位置

    #Returns:
    #X:标志块的输出，形状的张量(h,w,c)

    #定义名称的基础
    conv_name_base='res'+str(stage)+block+'branch'
    bn_name_base='bn'+str(stage)+block+'branch'
    #检索过滤器
    F1,F2,F3=filters
    #保存输入变量
    X_shortcut=X
    #主路径的第一个分量
    X=Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2a')(X)
    X=Activation('relu')(X)

    #主路径的第二个分量
    X=Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding='same',name=conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X=Activation('relu')(X)

    #主路径的第三个分量
    X=Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding="valid",name=conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2c')(X)

    #最后一步,将快捷键添加到主路径
    X=layers.add([X,X_shortcut])
    X=Activation('relu')(X)

    return X

#当输入和输出维度不匹配时，您可以使用这种类型的块。与标识块的不同之处在于，在快捷路径中有一个“卷积”层：
def convolutional_block(X,f,filters,stage,block,s=2):
   #X,f,filters,stage,block等跟上一个方法中的含义相同，s：整数类型，指定步幅
   # 定义名称的基础
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
   # 检索过滤器
    F1, F2, F3 = filters  # Save the input value
    X_shortcut = X
   # 第一步的卷积
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '2a', padding='valid',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

   #第二部的卷积
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    #第三步的卷积
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    #最后一步，对输捷径进行卷积
    X_shortcut=Conv2D(filters=F3,strides=(s,s),name=conv_name_base+'1',padding='valid',kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X=layers.add(X,X_shortcut)
    return X

#定义一个50层的残差网络
def ResNet50(input_shape=(64,64,3),classes=6):
    #实现50层的残差网络的结构
    #conv2D->batchnorm->relu->maxpool->convblock->idblock*2->convblock->idblock*3->
    # convblock->idblock*5->convblock->idblock*2->avgpool->toplayer
    #参数含义：
    #input_shape：数据集图形的形状
    #classes：整数的类型，分类的数量

    #Returns:
    #model--a Model() instance in Keras

    #将输入定义成有形状的张量
    X_input = Input(input_shape)
    #进行0填充
    X=ZeroPadding2D((3,3))(X_input)
    #第一步
    #使用64个7*7的过滤器，步长为(2,2),名字为‘conv1’，进行卷积
    X=Conv2D(filters=64,strides=(2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name='nb_conv1')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)#池化层
    #第二步
    X=convolutional_block(X,3,filters=[64,64,256],stage=2,block='a',s=1)#三层
    X=identity_block(X,3,filters=[64,64,256],stage=2,block='b')#三层
    X=identity_block(X,filters=[64,64,256],stage=2,block='c')#三层

    #第三步
    X=convolutional_block(X,3,filters=[128,128,512],stage=3,block='a',s=2)
    X=identity_block(X,3,filters=[128,128,512],stage=3,block='b')
    X=identity_block(X,3,filters=[128,128,512],stage=3,block='c')
    X=identity_block(X,3,filters=[128,128,512],stage=3,block='d')

    #第四步
    X=convolutional_block(X,3,filters=[256,256,1024],stage=4,block='a',s=2)
    X=identity_block(X,3,[256,256,1024],stage=4,block='b')
    X=identity_block(X,3,[256,256,1024],stage=4,block='c')
    X=identity_block(X,3,[256,256,1024],stage=4,block='d')
    X=identity_block(X,3,[256,256,1024],stage=4,block='e')
    X=identity_block(X,3,[256,256,1024],stage=4,block='f')

    #第五步
    X=convolutional_block(X,3,filters=[512,512,2048],stage=5,block='a',s=2)
    X=identity_block(X,3,[512,512,2048],stage=5,block='b')
    X=identity_block(X,3,[512,512,2048],stage=5,block='c')

    #avg pool，#flatten，









