import pandas as pd
import numpy as np
import time
import os
import csv
from bisect import bisect_left, bisect_right
import matplotlib.pyplot as plt
from copy import deepcopy


class NodeData:

    def __init__(self, file_name):
        """将一个节点的原始数据转为npy数据。每行为分钟时间和气压。
        剔除压力<0的,按时间排序"""
        data = pd.read_csv(file_name)
        n = len(data)
        result = np.empty((n*60,2), dtype='float')
        for i in range(n):
            result[i*60:(i+1)*60] = NodeData._expand_minute(data.iloc[i]['datTime'], data.iloc[i]['YL1'])
        result = result[result[:,1] >= 0]
        idx = np.argsort(result, axis=0)[:,0]
        self.data = result[idx]
    
    @staticmethod
    def _expand_minute(dt, yl1):
        """将一行的60分钟数据拆成60条数据。缺失不补
        返回：60*2矩阵，两列分别为时间、气压"""
        t0 = time.mktime(time.strptime(dt, '%Y-%m-%d %X'))
        yl1 = yl1.split(',')
        if len(yl1) == 1: #空的，填充为0
            yl1 = ['%02d:0'%i for i in range(60)]
        result = -np.ones((60,2),dtype='float')
        result[:,0] = t0 + np.arange(60)*60
        for s in yl1:
            i = int(s[:2])
            if result[i,1] == -1 and s[3:] != '#':
                result[i,1] = float(s[3:])
        return result
        
    def select(self, start='', end=''):
        """start: 形如 yyyy-mm-dd (h:m)"""
        if start == '':
            s = 0
        else:
            t = time.mktime(time.strptime(start, '%Y-%m-%d'+ ' %X' if ':' in start else ''))
            s = bisect_left(self.data[:,0], t)
        if end == '':
            e = -1
        else:
            t = time.mktime(time.strptime(end, '%Y-%m-%d'+ ' %X' if ':' in start else ''))
            s = bisect_right(self.data[:,0], t)
        return self.data[s:e]
    
    @staticmethod
    def to_series(d):
        return pd.Series(d[:, 1], index=pd.to_datetime(d[:, 0], unit='s'))
        
    def plot_day(self, day, ax=None):
        """day：yyyy-mm-dd"""
        t = time.mktime(time.strptime(day, '%Y-%m-%d'))
        idx = bisect_left(self.data[:,0], t)
        d = deepcopy(self.data[idx:idx+24*60])
        d[:,0] -= t
        s = NodeData.to_series(d)
        s.plot(ax=ax)
        
    def plot(self, ax=None):
        d = self.to_series(self.data)
        d.plot(ax=ax)
        
    def check_order(self):
        """检查时间是否有序，返回无序的位置"""
        return np.where(self.data[1:,0] < self.data[:-1,0])
        
    def check_continuous(self):
        """检查相邻2个时间查是否为60s"""
        return np.where((self.data[1:,0] - self.data[:-1,0]) != 60) 
            
    def outlier_pos(self):
        idx_miss = (self.data[1:,0] - self.data[:-1,0]) != 60 #不连续
        mu, std = np.mean(self.data[:,1]), np.std(self.data[:,1])
        idx_jump = np.abs(self.data[1:,1]-self.data[:-1,1]) > 3*std #跳跃不连续
        idx_outlier = np.bitwise_or(self.data[:-1,1] > 4*mu, self.data[:-1,1] < mu/5) #太大或大小
        idx = np.bitwise_or(np.bitwise_or(idx_miss, idx_jump), idx_outlier)
        return np.where(idx)[0]
        
    def slide_cut(self, p=6):
        """数据裁剪成每p天+1分钟一条样本。一条样本中的分钟必须是连续的
        返回: m*(1440*p + 1)数组，若无返回None"""
        T = 1440*p + 1
        result = []
        ind = self.outlier_pos()
        ind = np.pad(ind, (1,0), 'constant', constant_values=0)
        for i in range(len(ind)-1):
            if ind[i+1]-ind[i] >= T:
                slide = np.lib.stride_tricks.sliding_window_view(self.data[ind[i]:ind[i+1], 1], T)
                m = slide.shape[0]
                idx = np.random.choice(np.arange(m), int(0.002*m))
                result.append(slide[idx].astype('float32'))
        if result:
            return np.concatenate(result, axis=0)
        else:
            return None


def csv2npy_all(folder_name):
    f = open('node_mu_std.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['节点名', '均值', '标准差'])
    for file_name in os.listdir(folder_name):
        print(file_name)
        total_name = os.path.join(folder_name, file_name)
        d = NodeData(total_name)
        mu,sigma = np.mean(d.data[:,1]), np.std(d.data[:,1])
        p = a[:,1] - mu
        np.save('npy/'+file_name[:-3]+'npy',p)
        writer.writerow([file_name[:-4], mu, sigma])    
    
    
def npy_cut_all(folder_name, p=6):
    """对指定文件夹内所有文件调用slide_cut"""
    f = open('node_mu_std_train.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['节点名', '均值', '标准差'])
    results = []
    for file_name in os.listdir(folder_name):
        if os.path.splitext(file_name)[0].endswith('9'):
            continue
        print(file_name, end=', ')
        total_name = os.path.join(folder_name, file_name)
        d = NodeData(total_name)
        mu,sigma = np.mean(d.data[:,1]), np.std(d.data[:,1])
        writer.writerow([file_name[:-4], mu, sigma]) 
        #d.data[:,1] -= np.mean(d.data[:,1]) #均值归一化
        data = d.slide_cut(p)
        if data is not None:
            results.append(data)
            print('std=',np.std(data))
        else:
            print()
    data = np.concatenate(results, axis=0)
    np.save('slide_cut6_original_train.npy', data) 

    
def npy_cut_normal_all(folder_name, p=6):
    """对指定文件夹内所有文件调用slide_cut。存储mu,std"""
    #f = open('node_mu_std_train.csv', 'w', newline='', encoding='utf-8')
    #writer = csv.writer(f)
    #writer.writerow(['节点名', '均值', '标准差'])
    day6_list, mu_list, sigma_list = [], [], []
    for file_name in os.listdir(folder_name):
        if os.path.splitext(file_name)[0].endswith('9'):
            continue
        print(file_name, end=', ')
        total_name = os.path.join(folder_name, file_name)
        d = NodeData(total_name)
        mu,sigma = np.mean(d.data[:,1]), np.std(d.data[:,1])
        #writer.writerow([file_name[:-4], mu, sigma]) 
        data = d.slide_cut(p)
        if data is not None:
            day6_list.append(data)
            one = np.ones((data.shape[0],), dtype='float')
            mu_list.append(one*mu)
            sigma_list.append(one*sigma)
            print('std=',np.std(data))
        else:
            print()
    day6 = np.concatenate(day6_list, axis=0)
    mean = np.concatenate(mu_list, axis=0)
    std = np.concatenate(sigma_list, axis=0)
    np.savez('slide_cut6ms_train.npz', day6=day6, mean=mean, std=std)
    
    
if __name__ == '__main__': 
    npy_cut_normal_all('./drop_pressure')
    #node = NodeData('sorted_bj0241.csv')
    #node.slide_cut()
    #node.plot_day('2022-3-9')
    #node.plot()
    #plt.show()
