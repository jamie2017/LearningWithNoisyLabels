#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Project: "A Study of Learning with Noisy Labels"

Copyright (c) 2017 Jianmei Ye

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import collections,random
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import pandas as pd
import os

# Tasks:
# Create synthetic dataset similar with paper and random 2 class dataset,
# split it into training and testing parts,
# and add some label noise to the training set.

class GenerateData(object):
    def __init__(self,n):
        self.data_size = n
        self.label_map = collections.defaultdict(set)

    def init_lable_fun(self,n,p = 0.5):
        size = 100
        while len(self.label_map[1]) < n * p:
            x = random.randrange(-size,size)
            y = random.randrange(-size,size)
            if x - 0.6 * size >= y:
                self.label_map[1].add((x,y))
        while len(self.label_map[-1]) < n * (1-p):
            x = random.randrange(-size,size)
            y = random.randrange(-size,size)
            if x <= y :
                self.label_map[-1].add((x,y))

    def original_data(self):
        self.init_lable_fun(self.data_size)
        class0 = [[-1,xy[0],xy[1]] for xy in self.label_map[-1]]
        class1 = [[1,xy[0],xy[1]] for xy in self.label_map[1]]
        # print len(class0),len(class1)
        data = class0 + class1
        print "1. Data Generated!"
        return data

    def random_data(self,dim = 3,ws = (0.5,0.5)):
        xy, label = make_classification(n_samples=self.data_size, n_features=dim, n_redundant=1, n_informative=2,
                                        n_clusters_per_class=2,flip_y = 0.0001,weights= ws)
        data = []
        n1,n2 = 0,0
        for d in zip(label, xy[:, 0], xy[:, 1]):
            if d[0] == 0:
                data.append([-1,d[1],d[2]])
                n2 += 1
            else:
                data.append(d)
                n1 += 1
        print "1. Noise Free Data Generated!"
        return data,n1,n2

    def add_noise(self, data,po1,po2):
        noise_data = [list(d) for d in data]
        flip0 = int(sum(1 for d in noise_data if d[0] == -1) * po1)
        flip1 = int(sum(1 for d in noise_data if d[0] == 1) * po2)
        while flip0 or flip1:
            for i,d in enumerate(noise_data):
                if flip0 == 0 and flip1 == 0:
                    break
                ran_flip = random.choice([-1,1])
                if flip1 == 0 and ran_flip == 1:
                    ran_flip = -1
                elif flip0 == 0 and ran_flip == -1:
                    ran_flip = 1
                if ran_flip == d[0]:
                    noise_data[i][0] = - d[0]
                    if ran_flip == 1:
                        flip1 -= 1
                    elif ran_flip == -1:
                        flip0 -= 1
        print "2. Noised Data Generated!"
        return noise_data


    def split_data(self,data):
        train, test = train_test_split(data,shuffle=True)
        print "3. The dataset has been splited into TrainSize and TestSize: {} vs {}".format(len(train), len(test))
        return train, test

    def plot_data(self,all_data,train_data, test_data):
        # Three subplots sharing both x/y axes
        f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
        # plot1
        # plot1
        x_a = [d[1] for d in all_data]
        y_a = [d[2] for d in all_data]
        label_a = [d[0] for d in all_data]
        ax1.scatter(x_a, y_a, marker='+', c=label_a,
                    s=20, edgecolor='k')
        ax1.set_title('All data', horizontalalignment='center')

        x_o = [d[1] for d in train_data]
        y_o = [d[2] for d in train_data]
        label_o = [d[0] for d in train_data]
        ax2.scatter(x_o, y_o, marker='+', c=label_o,
                    s=20, edgecolor='k')
        ax2.set_title('Train data',horizontalalignment='center')
        # plot2
        x_n = [d[1] for d in test_data]
        y_n = [d[2] for d in test_data]
        label_n = [d[0] for d in test_data]
        ax3.scatter(x_n, y_n, marker='+', c=label_n,
                    s=20, edgecolor='k')
        ax3.set_title('Test data',horizontalalignment='center')

        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.show()




def modular_test():
    cur_path = os.curdir
    os.chdir(cur_path)
    os.chdir("..")
    data_size, is_random, po1, po2 = 5000,True,0.2,0.2
    generate = GenerateData(data_size)
    if is_random:
        noise_free_data,_,_ = generate.random_data()
    else:
        noise_free_data = generate.original_data()
    # Generate Noise free data
    data_map1 = {(x,y):z for z,x,y in noise_free_data}
    noise_free_df =  pd.DataFrame.from_dict(data_map1,"index")
    noise_free_df.to_csv("src/SyntheticDataSet/Sample_Noise_free_data.csv",header=False)
    # Generate Noised data
    noised_data = generate.add_noise(noise_free_data,po1, po2)
    data_map2 = {(x,y):z for z,x,y in noised_data}
    noised_data_df = pd.DataFrame.from_dict(data_map2, "index")
    noised_data_df.to_csv("src/SyntheticDataSet/Sample_Noised_data.csv",header=False)
    # Generate Noised train set and test set data
    generate.noised_train_set, generate.noised_test_set = generate.split_data(noised_data)
    data_map3 = {(x, y): z for z, x, y in generate.noised_train_set}
    noised_train_df = pd.DataFrame.from_dict(data_map3, "index")
    noised_train_df.to_csv("src/SyntheticDataSet/Sample_Noised_Train_data.csv", header=False)
    data_map4 = {(x, y): z for z, x, y in generate.noised_test_set}
    noised_test_df = pd.DataFrame.from_dict(data_map4, "index")
    noised_test_df.to_csv("src/SyntheticDataSet/Sample_Noised_Test_data.csv", header=False)


if __name__ == "__main__":
    modular_test()