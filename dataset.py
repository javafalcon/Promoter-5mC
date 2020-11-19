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

def generate_CAImage(fastafile, save_dir, num_iteration=82, num_rule=74):
    i = 0
    with open(fastafile, 'r') as fn:
        for record in SeqIO.parse(fn, 'fasta'):
            seq = str(record.seq)
            if 'N' in seq:
                continue
            
            ca = CAImage(seq, num_iteration=num_iteration, num_rule=num_rule)
            image_file = save_dir + str(i) + ".jpg"
            matplotlib.image.imsave(image_file, ca)
            print("\r{}-{}".format(fastafile, i), end='')
            i += 1
    
           
class SeqCAImageGenerator(Sequence):
    def __init__(self, datas, labels, batch_size, 
                 num_iteration=82, num_rule=82, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.labels = labels
        self.indexes = np.arange(len(self.datas))
        self.num_iteration = num_iteration
        self.num_rule = num_rule
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
        x = np.array(x)
        y = np.eye(2)[batch_labels]
        return x.reshape((x.shape[0], x.shape[1], x.shape[2], 1)), y
    
    def on_epoch_end(self):
        # 在每一次epoch结束时，是否要进行一次随机，重新随机排列index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def data_generation(self, batch_datas):
        images = []
        for i, seq in enumerate(batch_datas):
            image = CAImage(seq, self.num_iteration, self.num_rule)
            images.append(image)
        return images
    

# 统计数据集A、C、G、T分布
# 结论如下：
# 正类A、C、G、T出现频率：0.1479，0.3489，0.3576，0.1457
# 负类A、C、G、T出现频率：0.2073, 0.3036, 0.2767, 0.2124
# 全部A、C、G、T出现频率：0.2026, 0.3072, 0.2830, 0.2072, 所以按哈夫曼编码如下：
# 在正类样本集中没有其它字符出现，在负类样本集中N出现34次
def statFreq():
    count = np.zeros((6,))
    with open('all_negative.fasta','r') as fn:
        for r in SeqIO.parse(fn,'fasta'):
            for c in str(r.seq):
                if c == 'A':
                    count[0] += 1
                elif c == 'C':
                    count[1] += 1
                elif c == 'G':
                    count[2] += 1
                elif c == 'T':
                    count[3] += 1
                elif c == 'N':
                    count[4] += 1
                else:
                    count[5] += 1
                    
    with open('all_positive.fasta','r') as fn:
        for r in SeqIO.parse(fn,'fasta'):
            for c in str(r.seq):
                if c == 'A':
                    count[0] += 1
                elif c == 'C':
                    count[1] += 1
                elif c == 'G':
                    count[2] += 1
                elif c == 'T':
                    count[3] += 1
                else:
                    count[4] += 1
                
 

