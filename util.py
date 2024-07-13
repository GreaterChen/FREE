#######################
#author: Shiming Chen
#FREE
#######################

#import h5py
import json
import os
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import pdb
import h5py

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(len(classes)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_turmor(opt)
        # self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        
    def read_turmor(self, opt):
        self.train_class = [1, 2]

        # 加载训练特征和标签并转换为 Tensor
        self.train_feature = np.load(os.path.join(opt.dataroot, 'resnet101', 'train_features.npy'))  # (606, 2048)
        self.train_label = np.load(os.path.join(opt.dataroot, 'resnet101', 'train_targets.npy'))
        self.train_feature = torch.tensor(self.train_feature, dtype=torch.float32)
        self.train_label = torch.tensor(self.train_label, dtype=torch.long)

        self.ntrain = self.train_feature.shape[0]
        self.ntrain_class = 2
        self.ntest_class = 1

        # 加载测试特征和标签并转换为 Tensor
        self.test_feature = np.load(os.path.join(opt.dataroot, 'resnet101', 'valid_features.npy'))  # (171, 2048)
        self.test_label = np.load(os.path.join(opt.dataroot, 'resnet101', 'valid_targets.npy'))
        self.test_feature = torch.tensor(self.test_feature, dtype=torch.float32)
        self.test_label = torch.tensor(self.test_label, dtype=torch.long)

        # 加载属性嵌入并转换为 Tensor
        file_path = os.path.join(opt.dataroot, 'att', 'embeddings.json')
        with open(file_path, 'r') as f:
            data = json.load(f)

        attribute = {}
        for key, value in data.items():
            attribute[key] = np.array(value)

        categories = list(attribute.keys())
        embedding_list = [attribute[category] for category in categories]
        self.attribute = torch.tensor(np.array(embedding_list), dtype=torch.float32)

        self.allclasses = [0, 1, 2]
        self.seenclasses = [1, 2]
        self.unseenclasses = [0]
        self.attribute_seen = self.attribute[self.seenclasses, :]

        # 提取标签为 1 和 2 的测试数据
        indices_seen = (self.test_label == 1) | (self.test_label == 2)
        self.test_seen_feature = self.test_feature[indices_seen]
        self.test_seen_label = self.test_label[indices_seen]

        # 提取标签为 0 的测试数据
        indices_unseen = self.test_label == 0
        self.test_unseen_feature = self.test_feature[indices_unseen]
        self.test_unseen_label = self.test_label[indices_unseen]

        # 计算每个类别的样本数量
        self.train_samples_class_index = torch.tensor([self.train_label.eq(i_class).sum().float() for i_class in self.train_class])

        # 打印结果以确认
        print("训练集特征形状:", self.train_feature.shape)
        print("训练集标签形状:", self.train_label.shape)
        print("测试集特征形状:", self.test_feature.shape)
        print("测试集标签形状:", self.test_label.shape)
        print("提取的测试集特征形状 (seen):", self.test_seen_feature.shape)
        print("提取的测试集标签形状 (seen):", self.test_seen_label.shape)
        print("提取的测试集特征形状 (unseen):", self.test_unseen_feature.shape)
        print("提取的测试集标签形状 (unseen):", self.test_unseen_label.shape)
        print("每个类别的样本数量:", self.train_samples_class_index)
     

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1    

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
                
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 
            
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        
        #####few-shot setting
        # ph = [[] for _ in range(250)]
        # for fi in range(len(self.train_feature)):
            # ph[self.train_label[fi]].append(fi)
        # ph = [i for i in ph if i !=[]]
        # training=True
        # ph = ph[0:self.seenclasses.size(0)]

        # feature = []
        
        # for fi in range(len(np.unique(self.train_label))):
            # g = ph[fi][0:10]
            # feature = np.concatenate((feature, g))
        # print("feature:", feature.shape)
        # self.train_feature_new = np.concatenate(np.expand_dims(self.train_feature[feature.astype(int)], axis=1))
        # self.train_feature = torch.from_numpy(self.train_feature_new)
        # self.train_label = self.train_label[feature.astype(int)]

        
        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        #print("***********",self.ntrain_class,self.ntest_class)
        
        
        # 打印结果以确认
        print("训练集特征形状:", self.train_feature.shape)
        print("训练集标签形状:", self.train_label.shape)
        # print("测试集特征形状:", self.test_feature.shape)
        # print("测试集标签形状:", self.test_label.shape)
        print("提取的测试集特征形状 (seen):", self.test_seen_feature.shape)
        print("提取的测试集标签形状 (seen):", self.test_seen_label.shape)
        print("提取的测试集特征形状 (unseen):", self.test_unseen_feature.shape)
        print("提取的测试集标签形状 (unseen):", self.test_unseen_label.shape)
        # print("每个类别的样本数量:", self.train_samples_class_index)
    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att