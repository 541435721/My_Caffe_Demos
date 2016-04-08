# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/08_15:49 '

from numpy import *
import caffe
from caffe import layers as L
from caffe import params as P
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.datasets import load_iris

if __name__ == '__main__':
    dims = 4
    label_dims = 4
    # 创建训练集 10000个样本。4个类别
    x, y = sklearn.datasets.make_classification(n_samples=10000, n_features=dims, n_redundant=0, n_informative=dims,
                                                n_clusters_per_class=label_dims, hypercube=False, random_state=0)
    # 离散化分类标签
    y_ = zeros((len(y), label_dims))
    for i, t in enumerate(y):
        y_[i][t] = 1
    y = y_
    # 分离样本x,y为7000个训练样本，xt,yt为2550个验证样本，xtt,ytt为750个测试样本
    x, xt, y, yt = sklearn.cross_validation.train_test_split(x, y, test_size=0.3)
    xt, xtt, yt, ytt = sklearn.cross_validation.train_test_split(xt, yt, test_size=0.25)

    # 创建数据
    train_data = {}
    train_data['input'] = reshape(x, (len(x), 1, 1, dims))
    train_data['output'] = y

    test_data = {}
    test_data['input'] = reshape(xt, (len(xt), 1, 1, dims))
    test_data['output'] = yt

    pre_data = {}
    pre_data['input'] = reshape(xtt, (len(xtt), 1, 1, dims))
    pre_data['output'] = ytt

    # 写入h5文件
    with h5py.File('data/train_data.h5', 'w') as f:
        f['data'] = train_data['input'].astype(float32)
        f['label'] = train_data['output'].astype(float32)
    with open('data/train_data.txt', 'w') as f:
        f.write('data/train_data.h5' + '\n')

    with h5py.File('data/test_data.h5', 'w') as f:
        f['data'] = test_data['input'].astype(float32)
        f['label'] = test_data['output'].astype(float32)
    with open('data/test_data.txt', 'w') as f:
        f.write('data/test_data.h5' + '\n')

    with h5py.File('data/pre_data.h5', 'w') as f:
        f['data'] = pre_data['input'].astype(float32)
        f['label'] = pre_data['output'].astype(float32)
    with open('data/pre_data.txt', 'w') as f:
        f.write('data/pre_data.h5' + '\n')

    # 设置CPU模式
    caffe.set_mode_cpu()
    # 训练神经网络
    solver = caffe.get_solver('protofile/solver.prototxt')
    solver.solve()

    Net = caffe.Net('protofile/deploy.prototxt', 'model/iris__iter_1000000.caffemodel', caffe.TEST)

    out = Net.forward(data=pre_data['input'])  # 调用测试数据
    result = []
    for i in xrange(750):
        r = out['ip3'][i].argmax() == ytt[i].tolist().index(1)
        print "output:" + str(r)
        result.append(r)
    print result.count(True) * 1.0 / 750
