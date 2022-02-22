'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 01:02:16
@LastEditTime: 2020-07-29 15:20:16
@Description:
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

def CPU(var):
    return var.cpu().detach()

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class DataBuffer:
    # numpy-based ring buffer to store data

    def __init__(self, input_dim, output_dim, max_len=5000000):
        self.input_buf = np.zeros(combined_shape(max_len, input_dim), dtype=np.float32)
        self.output_buf = np.zeros(combined_shape(max_len, output_dim), dtype=np.float32)
        self.ptr, self.max_size = 0, max_len
        self.full = 0 # indicate if the buffer is full and begin a new circle
        self.ptr_old = 0
        self.ptr_reset = 0 # indicate if we have reset prt to 0 since last data retrival

    def store(self, input_data, output_data):
        """
        Append one data to the buffer.
        @param input_data [ndarray, input_dim]
        @param input_data [ndarray, output_dim]
        """
        if self.ptr == self.max_size:
            self.full = 1 # finish a ring
            self.ptr_reset += 1
            self.ptr = 0
        self.input_buf[self.ptr] = input_data
        self.output_buf[self.ptr] = output_data
        self.ptr += 1

    def get_all(self):
        '''
        Return all the valid data in the buffer
        @return input_buf [ndarray, (size, input_dim)], output_buf [ndarray, (size, input_dim)]
        '''

        self.ptr_reset = 0
        if self.full:
            print("data buffer is full, return all data: ", self.max_size, self.ptr)
            return self.input_buf, self.output_buf
        # Buffer is not full
        print("return data util ", self.ptr)
        return self.input_buf[:self.ptr], self.output_buf[:self.ptr]

    def get_new(self):
        '''
        Return new input data in the buffer since last retrival by calling this method
        @return input_buf [ndarray, (size, input_dim)], output_buf [ndarray, (size, input_dim)]
        '''
        if self.ptr_reset > 1: # two round
            x, y = self.input_buf, self.output_buf
        elif self.ptr_reset == 1:
            if self.ptr < self.ptr_old:
                x = np.concatenate((input_buf[:self.ptr], input_buf[self.ptr_old]), axis=1)
                y = np.concatenate((output_buf[:self.ptr], output_buf[self.ptr_old]), axis=1)
            else:
                x, y = self.input_buf, self.output_buf
        else:
            x, y = self.input_buf[self.ptr_old:self.ptr], self.output_buf[self.ptr_old:self.ptr]
        self.ptr_old = self.ptr
        self.ptr_reset = 0
        return x, y

    def save(self, path=None):
        assert path is not None, "The saving path is not specified!"
        x, y = self.get_all()
        data = {"x":x,"y":y}
        joblib.dump(data, path)
        print("Successfully save data buffer to "+path)

    def load(self, path=None):
        assert path is not None, "The loading path is not specified!"
        data = joblib.load(path)
        x, y = data["x"], data["y"]
        data_num = x.shape[0]
        if data_num<self.max_size:
            self.input_buf[:data_num], self.output_buf[:data_num] = x, y
            self.ptr = data_num
        else:
            self.input_buf, self.output_buf = x[:self.max_size], y[:self.max_size]
            self.full = 1

def mlp(sizes, activation, output_activation=nn.Identity, dropout=0):
    layers = []
    #padding = int((kernel_size-1)/2)
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        dropout_layer = [ nn.Dropout(p=dropout) ] if (j < len(sizes)-2 and dropout>0) else []
        new_layer = [nn.Linear(sizes[j], sizes[j+1]), act()] + dropout_layer #+ maxpool_layer
        layers += new_layer
    return nn.Sequential(*layers)

class MLPRegression(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_sizes=(64,64), activation=nn.Tanh):
        """
            @param int - input_dim
            @param int - output_dim 
            @param list - hidden_sizes : such as [32,32,32]
        """ 
        super().__init__()
        self.net = mlp([input_dim] + list(hidden_sizes) + [output_dim], activation)

    def forward(self, x):
        """
            @param tensor - x: shape [batch, input dim]
            
            @return tensor - out : shape [batch, output dim]
        """ 
        out = self.net(x)
        return out

class MLPCategorical(nn.Module):

    def __init__(self, input_dim, output_dim=2, hidden_sizes=(256,256,256,256), activation=nn.ELU, dropout=0):
        """
            @param int - input_dim
            @param int - output_dim, default 2
            @param list - hidden_sizes : such as [32,32,32]
        """ 
        super().__init__()
        self.logits_net = mlp([input_dim] + list(hidden_sizes) + [output_dim], activation, dropout=dropout)

    def forward(self, x):
        """
            @param tensor - x: shape [batch, input dim]

            @return tensor - out : shape [batch, 2]
        """ 
        logits = self.logits_net(x)
        #out = Categorical(logits=logits)
        output = F.log_softmax(logits, dim=1)
        return output

class GRURegression(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_sizes=(256, 256), output_sizes=(100, 100), activation=nn.ELU):
        super().__init__()

        self.embedding_dim = embedding_sizes[-1]
        self.hidden_dim = output_sizes[0]
        self.embedding_net = mlp([input_dim] + list(embedding_sizes), activation)
        self.output_net = mlp( list(output_sizes) + [output_dim], activation )
        self.cell = nn.GRUCell(self.embedding_dim, self.hidden_dim)

    def initial_hidden(self, batch_size, **kwargs):
        return CUDA(torch.zeros(batch_size, self.hidden_dim, **kwargs))
        

    def forward(self, x, h=None):
        '''
        @param x: input data at a single timestep, [tensor, (batch, input_dim)]
        @param h: hidden state at last timestep, [tensor, (batch, input_dim)]
        '''
        if h is None:
            h = self.initial_hidden(x.shape[0])
        embedding = self.embedding_net(x)
        h_next = self.cell(embedding, h)
        output = self.output_net(h_next)
        return output, h_next