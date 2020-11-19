# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:50:20 2020

@author: lwzjc
"""
import numpy as np

def cellIterate(x,y,z,rule):
    k = int(7 - (x*4 + y*2 + z))
    return rule[k]
    
def CAImage(DNAseq, num_iteration, num_rule):
    ndic={'A':[0,0], 'C':[0,1], 'G':[1,0], 'T':[1,1], 'N':[0,0]}
    bs = []
    for d in DNAseq:
        bs += ndic[d]
    rule = [int(i) for i in "{:08b}".format(num_rule)]
    
    # 元胞自动机演化N次
    # [1 1 1;1 1 0;1 0 1;1 0 0;0 1 1;0 1 0;0 0 1;0 0 0]
    m = len(bs)
    c = np.zeros((num_iteration, m))
    c[0,:] = np.array(bs)
    for i in range(1, num_iteration):
        c[i,0] = cellIterate(c[i-1,m-1], c[i-1,0], c[i-1,1], rule)
        c[i,m-1] = cellIterate(c[i-1,m-2], c[i-1,m-1], c[i-1,0], rule)
        for j in range(1, m-1):
            c[i,j] = cellIterate(c[i-1,j-1], c[i-1,j], c[i-1,j+1], rule)
    
    return c

        
    
    
    

