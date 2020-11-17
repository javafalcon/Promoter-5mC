# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 20:44:43 2020

@author: lwzjc
"""
from Bio import SeqIO
import numpy as np
import math
from CA import CAImage
from tensorflow.keras.utils import Sequence
import matplotlib

num_negative, num_positive = 823576, 69750
def generate_CAImage(num_iteration=82, num_rule=74):
    with open('all_negative.fasta', 'r') as fn:
        for i, record in enumerate(SeqIO.parse(fn, 'fasta')):
            seq = str(record.seq)
            ca = CAImage(seq, num_iteration=num_iteration, num_rule=num_rule)
            image_file = 'neg_CAImages/' + str(i) + ".jpg"
            matplotlib.image.imsave(image_file, ca)
            print("\rneg-{}".format(i), end='')
    with open('all_positive.fasta', 'r') as fp:
       for i, record in enumerate(SeqIO.parse(fp, 'fasta')):
           seq = str(record.seq)
           ca = CAImage(seq, num_iteration=num_iteration, num_rule=num_rule)
           image_file = 'pos_CAImages/' + str(i) + ".jpg"
           matplotlib.image.imsave(image_file, ca)
           print("\rpos-{}".format(i), end='') 
class SeqCAImageGenerator(Sequence):
    def __init__(self, datas, labels, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.labels = labels
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle
    
    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))
    
    def __getitem__(self, index):
        # 生成每个batch数据
        # 生成batch_size个索引
        batch_index = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_index]
        batch_labels = [self.labels[k] for k in batch_index]
        # 生成CA Image数据
        x = self.data_generation(batch_datas)
        
        return np.array(x), np.array(batch_labels)
    
    def on_epoch_end(self):
        # 在每一次epoch结束时，是否要进行一次随机，重新随机排列index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def data_generation(self, batch_datas):
        images = []
        for i, seq in enumerate(batch_datas):
            image = CAImage(seq, num_iteration, num_rule)
            images.append(image)
        return images
    
generate_CAImage()
        

