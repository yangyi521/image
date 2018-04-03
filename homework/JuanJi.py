import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#加载测试数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#定义权重的函数
def weight_variable(shape):#参数为一个w的形状，例如[-1,28,28,1]
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布，此函数原型为尺寸、均值、标准差
    return tf.Variable(initial)

#定义偏置量的函数，使用0.1进行初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#定义卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长

#定义最大池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 参数同上，ksize是池化块的大小


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 图像转化为一个四维张量，第一个参数代表样本数量，-1表示不定，第二三参数代表图像尺寸，最后一个参数代表图像通道数  
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积加池化  
w_conv1 = weight_variable([5, 5, 1, 32])  # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征
#使用5*5的过滤器进行卷积运算，第一个数跟第二个数代表过滤器的大小，第三个数代表通道的数量，最后一个代表输出的特征图的数量
b_conv1 = bias_variable([32])#创建的大小即生成图的数量的大小
#进行卷积运算
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
#进行池化运算
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积加池化   
w_conv2 = weight_variable([5, 5, 32, 64])  # 多通道卷积，卷积出64个特征
#仍然使用5*5的过滤器，上一个输出32个特征图，即本次输入的32个特征图，下次输出64个特征图
b_conv2 = bias_variable([64])#偏置的大小即本次输出特征图的数量，即64
#进行第二次的卷积运算，上一次池化完成，就是本次卷积的输入
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
#进行第二次池化的操作
h_pool2 = max_pool_2x2(h_conv2)
#######此处以上已经完成了卷积操跟池化操作###########
#卷积池化操作完成后，需要进入到全连接层进行操作。
# 原图像尺寸28*28，第一轮图像缩小为14*14，共有32张，第二轮后图像缩小为7*7，共有64张
w_fc1 = weight_variable([7 * 7 * 64, 2048])#全连接层的操作是在两次的卷积跟池化完成后进行的，其中经过第一次池化是输出是14*14，特征图的数量为32，经过第二次池化后为7*7，特征图的数量为64
b_fc1 = bias_variable([2048])#定义偏置量的个数为1024

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 展开，第一个参数为样本数量，-1未知，将卷积池化后的数据进行重新定义形状，为7*7*64列，行自动计算大小的矩阵
f_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)#进行神经网络的运算

# dropout操作，减少过拟合  
keep_prob = tf.placeholder(tf.float32)
#使用dropout随机减少神经元的数量，防止过拟合的现象产生
h_fc1_drop = tf.nn.dropout(f_fc1, keep_prob) #其中keep_prob是指的每个神经元保留的概率，即参与的神经元的概率
#跟深度网络的定义类似了现在，1024指的前一个的输入单元的输入，10指的是输出类别的个数
w_fc2 = weight_variable([2048, 10])
b_fc2 = bias_variable([10])
#卷积层池化层全连接层全部完成后，输出的结果就是预测的结果
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
#定义损失函数，y_为实际的标签的值，y_conv为预测的值，使用交叉熵函数定义损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 定义交叉熵为loss函数
#对损失函数进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 调用优化器优化
#预测函数跟实际的函数的值进行比较，算出准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(2000):
    batch = mnist.train.next_batch(50)#每次加载的batch_size为50
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_lost = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step: %d,lost: %d,training accuracy: %g" % (i, train_lost,train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(
    feed_dict={x: mnist.test.images[0:500], y_: mnist.test.labels[0:500], keep_prob: 1.0}) )
