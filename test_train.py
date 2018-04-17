from keras.models import load_model
from keras.preprocessing import image #从Keras中导入image模块 进行图片处理
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import os
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  StratifiedShuffleSplit
my_model=load_model('model3_many.h5')
def read_and_process_image(data_dir, width=64, height=64, channels=3, preprocess=False):
    train_images = [data_dir + i for i in os.listdir(data_dir) ]#if i != '.DS_Store'#将所有的要训练的问图片的名称读取到train_images的列表中
    #print(train_images)
    random.shuffle(train_images)#将train_image中的内容随机排序
    #定义读取图片
    def read_image(file_path, preprocess):
        img = image.load_img(file_path, target_size=(height, width))
        #print("img",img)
        x = image.img_to_array(img)#使用keras中的img_to_array()函数，可以将一张图片转换成一个矩阵
        #print('img_to_array返回的结果：',x)
        x = np.expand_dims(x, axis=0)#将x增加维数
        if preprocess:
            x = preprocess_input(x)#进行图像预处理
        return x
    #将所有的图片转换成一个2进制的形式进行保存
    def prep_data(images, preprocess):
        count = len(images)
        data = np.ndarray((count, width, height, channels), dtype=np.float32)
        for i, image_file in enumerate(images):
            #print("i的值：",i)
            image = read_image(image_file, preprocess)
            #print("image的值：",image)
            data[i] = image
        print("data",data.shape)

        return data

    def read_labels(file_path):
        # Using 1 to represent dog and 0 for cat
        labels = []
        label_encoder = LabelEncoder()
        for i in file_path:
            label = i.split('/')[1].split('.')[0].split('_')[0]
            labels.append(label)
        labels = label_encoder.fit_transform(labels)

        return labels, label_encoder

    X = prep_data(train_images, preprocess)#调用前面写的函数 将所有的图片转换成向量的形式进行保存并且返回
    labels, label_encoder = read_labels(train_images)#将训练好的模型跟所有的名字进行调用并且保存，labels包括所有的结果集，label_encoder是训练好的模型

    assert X.shape[0] == len(labels)

    print("Train shape: {}".format(X.shape))

    return X,label_encoder
WIDTH = 48
HEIGHT = 48
CHANNELS = 3
#函数开始运行
X,label_encoder = read_and_process_image('test_image/', width = WIDTH, height = HEIGHT, channels = CHANNELS)
label_encoder.classes_
my_predict=my_model.predict(X)
X_value=my_predict[0]
X_value=X_value.tolist()
result=X_value.index(max(X_value))
print(result)
if(result==0):
    print("computer")
elif(result==1):
    print("tong")
else:
    print("wt310")

print('my_predict',my_predict)
#print('y_label',y_label)
# if not (my_predict-[[0.,1.]]).any():
#    print("wt310")
# elif not (my_predict-[[1.,0.]]).any():
#     print("computer")
# else:
#     print("burenshi")
