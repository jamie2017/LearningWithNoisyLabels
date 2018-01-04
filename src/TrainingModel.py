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

# Tasks:
# Train your classier on the synthetic noisy dataset , and generate prediction results

from sklearn import svm
from GenerateData import GenerateData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import  KFold
import os


COLOR = {-1:"r",1:"b"}
class TrainingModel(object):
    def __init__(self,data_size,is_random,po1, po2,dim = 3,ws = (0.5,0.5)):
        self.data_maker = GenerateData(data_size)
        self.true_data_map = {}
        if is_random:
            noise_free_data,self.n1,self.n2 = self.data_maker.random_data(dim,ws)
            self.set_random = True
        else:
            noise_free_data = self.data_maker.original_data()
            self.n1 = self.n2 = self.data_maker.data_size/2
            self.set_random = False
        self.init_true_data_map(noise_free_data)
        noised_data = self.data_maker.add_noise(noise_free_data,po1, po2)
        self.noised_train_set, self.noised_test_set = self.data_maker.split_data(noised_data)
        self.nosiy_test_map = {(x, y): label for label, x, y in self.noised_test_set}
        self.unbiased_loss_pred_map = {}


    def init_true_data_map(self,data):
        for d in data:
            self.true_data_map[(d[1],d[2])] = d[0]

    def trainByNormalSVM(self,train_set):
        train_X = [(d[1],d[2]) for d in train_set]
        train_y = [ d[0] for d in train_set]
        clf = svm.SVC()
        clf.fit(train_X,train_y)
        test_X = [(d[1],d[2]) for d in self.noised_test_set]
        pred_y = clf.predict(test_X)
        self.unbiased_loss_pred_map = {(xy[0],xy[1]):int(label) for label,xy in zip(pred_y,test_X)}
        print "5. Classifer has been trained!"
        return clf


    def selectClfByKFold(self,po1,po2):
        min_Rlf = float('inf')
        target_dataset = None
        data = np.array(self.noised_train_set)
        kf = KFold(n_splits=2)
        for train, test in kf.split(data):
            size = len(train)
            tr_data = data[train]
            p_y = 1.0*sum(1 for d in tr_data if d[0] == -1)/size
            py  = 1.0*sum(1 for d in tr_data if d[0] == 1)/size
            Rlf = []
            for d in tr_data:
                Rlf.append(self.estLossFunction(self.true_data_map[d[1],d[2]],d[0],py,p_y,po1,po2))
            if np.mean(Rlf) < min_Rlf:
                min_Rlf = np.mean(Rlf)
                target_dataset= tr_data
        print "4. Cross-validation finished!"
        return self.trainByNormalSVM(target_dataset)


    def lossFunction(self,fx,y):
        return 0 if fx == y else 1

    def estLossFunction(self,x,y,py,p_y,po1,po2):
        p1,p2 = po1,po2
        return ((1 - p_y) * self.lossFunction(x, y) - py * self.lossFunction(x, -y)) / (1 - p1 - p2)


    def accuracy(self,pred,rst):
        match_cnt, all_cnt = 0,0
        for a,b in zip(pred,rst):
            all_cnt += 1
            if a == b:
                match_cnt += 1
        rst = round(1.0 * match_cnt/all_cnt,4)
        print "The Accuracy of the prediction is : {}".format(rst)
        return rst



    def comparison_plot(self,clf,po1,po2,show_plot = False):
        print " =======================================================> \n"
        # Three subplots sharing both x/y axes
        f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
        # plot1
        x_o = [d[1] for d in self.noised_test_set]
        y_o = [d[2] for d in self.noised_test_set]
        label_o = [self.true_data_map[(x,y)] for x,y in zip(x_o,y_o)]
        color_o = [COLOR[d] for d in label_o]
        ax1.scatter(x_o, y_o, marker='+', c=color_o,
                    s=20, edgecolor='y')
        ax1.set_title('Noise-free')
        # plot2
        x_n = x_o
        y_n = y_o
        label_n = [d[0] for d in self.noised_test_set]
        color_n = [COLOR[d] for d in label_n]
        # rst0 = self.accuracy(label_o,label_n)
        ax2.scatter(x_n, y_n, marker='+', c=color_n,
                    s=20, edgecolor='y')
        ax2.set_title('Noise rate:' +str(po1) + " and " + str(po2))

        # plot3
        x_t1 = x_o
        y_t1 = y_o
        nosiy_test_X1 = zip(x_t1,y_t1)
        pred_label1 = clf.predict(nosiy_test_X1)
        label_p1 = [COLOR[d] for d in pred_label1]
        rst1 = self.accuracy(label_o,pred_label1)
        # target_names = ['class label = -1', 'class label = 1']
        # print(classification_report(label_o, pred_label1, target_names=target_names))
        # y_score1 = clf.decision_function(nosiy_test_X1)

        # average_precision1 = average_precision_score(label_o, y_score1)
        # print('Average precision-recall score: {0:0.2f}'.format(
        #     average_precision1))
        ax3.scatter(x_t1, y_t1, marker='+', c=label_p1,
                    s=20, edgecolor='y')
        ax3.set_title('Accuracy:' + str(rst1))

        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        if show_plot:
            plt.show()
        self.save_test_data(f,po1,po2)
        return rst1


    def save_test_data(self,f,po1,po2):
        table = []
        headers = ["Positions","Noise-free data","Noisy Data","Unbiased_loss_predition"]
        for x,y in self.nosiy_test_map.keys():
            tmp = (x,y),\
                  self.true_data_map[(x,y)],self.nosiy_test_map[(x,y)],\
                  self.unbiased_loss_pred_map[(x,y)]
            table.append(tmp)
        df = pd.DataFrame(table, columns=headers)
        df.set_index("Positions")
        if self.set_random:
            filename = "random_test_data_" + str(self.data_maker.data_size) +"_" +str(po1) + "_" + str(po2)
        else:
            filename = "linearly_test_data_" + str(self.data_maker.data_size) +"_" +str(po1) + "_" + str(po2)

        df.to_csv("src/PredictOutput/" + filename +".csv",index=False)
        print "Data has been saved as CSV file!"
        f.savefig("src/Figures/" + filename+".png")
        print "Figure has been saved!"




def modular_test():
    cur_path = os.curdir
    os.chdir(cur_path)
    os.chdir("..")
    n, is_random, po1, po2 = 10000, True, 0.3, 0.3
    run = TrainingModel(n, is_random, po1, po2)
    clf = run.selectClfByKFold(po1, po2)
    run.comparison_plot(clf, po1, po2,show_plot=True)

if __name__ == "__main__":
    modular_test()