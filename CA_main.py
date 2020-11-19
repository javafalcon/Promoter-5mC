# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:08:27 2020

@author: lwzjc
"""
from dataset import SeqCAImageGenerator
from Capsule_Keras import Capsule

import numpy as np
import os
from Bio import SeqIO

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Reshape, Lambda
from tensorflow.keras import backend as K

from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix
from sklearn.utils import class_weight, shuffle, resample
from sklearn.model_selection import train_test_split



def CapsNet(num_classes=2, dim_capsule=16, num_routing=3):
    input_image = Input(shape=(None,None,1))
    cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
    cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = AveragePooling2D((2,2))(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Reshape((-1, 128))(cnn)
    capsule = Capsule(num_classes, dim_capsule, num_routing, True)(cnn)
    output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(2,))(capsule)
    
    model = Model(inputs=input_image, outputs=output)
    model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.summary()
    return model
    
def X_generator(generator):
    while 1:
        x_batch, _ = generator.next()
        yield x_batch
        
if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lam_recon', default=0.465, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=9, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_iteration', default=82, type=int)
    parser.add_argument('--num_rule', default=170, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # load seqs data
    print("load train and test data as sequences")
    with open('train_data/train_negative_data.fasta', 'r') as fn:
        train_negs = list(SeqIO.parse(fn, 'fasta'))
    with open('train_data/train_positive_data.fasta', 'r') as fp:
        train_poss = list(SeqIO.parse(fp, 'fasta'))
    with open('test_data/test_negative_data.fasta', 'r') as ftn:
        test_negs = list(SeqIO.parse(ftn, 'fasta'))
    with open('test_data/test_positive_data.fasta', 'r') as ftp:
        test_poss = list(SeqIO.parse(ftp, 'fasta'))
        
    # build test data 
    print("build test data")
    seq_test = test_negs + test_poss
    labels_test = [0] * len(test_negs) + [1] * len(test_poss)
    test_generator = SeqCAImageGenerator(seq_test, labels_test, batch_size=100,
                                         num_iteration=args.num_iteration, num_rule=args.num_rule)
    
    
    num_train_negs, num_train_poss = len(train_negs), len(train_poss)
    num_test_negs, num_test_poss = len(test_negs), len(test_poss)
    
    class_ratio = int(np.ceil( num_train_negs / num_train_poss))
    num_undersampling = num_train_negs // class_ratio
       
    y_pred = np.zeros(shape=(num_test_negs+num_test_poss, ))
    
    train_negs = shuffle(train_negs)
    
    for k in range(class_ratio):
        # 把训练集划分为class_ratio个子集
        start = k * num_undersampling
        end = min( (k+1)*num_undersampling, num_train_negs)
        
        neg = train_negs[start:end]
        train_data = neg + train_poss
        labels = [0] * len(neg) + [1] * len(train_poss)
        
        # 80%训练，20%验证
        seq_train, seq_val, labels_train, labels_val = train_test_split(
            train_data, labels, test_size=0.20, shuffle=True)
        
        train_generator = SeqCAImageGenerator(seq_train, labels_train, batch_size=100, 
                                num_iteration=args.num_iteration, num_rule=args.num_rule)
        val_generator = SeqCAImageGenerator(seq_val, labels_val, batch_size=100,
                                num_iteration=args.num_iteration, num_rule=args.num_rule)
        
        # define model
        model = CapsNet()
        
        model.fit(train_generator,
          epochs=10,
          verbose=1,
          validation_data=val_generator)
        
        score = model.predict(X_generator(test_generator)) #用模型进行预测
        
        y_pred = y_pred + np.argmax(score, axis=-1)
        K.clear_session()
        
    
    y_pred = y_pred / class_ratio
    y_p = (y_pred>0.5).astype(float)
    y_t = np.array(labels_test)
    print('Test Accuracy:', accuracy_score(y_t, y_p))
    print('Test mattews-corrcoef', matthews_corrcoef(y_t, y_p))
    print('Test confusion-matrix', confusion_matrix(y_t, y_p))
    