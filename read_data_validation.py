#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import torch.utils.data as data
import pickle
import numpy as np

path = './data_set/'


class WebKBValidationWithID(data.Dataset):
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'webkb/'
        self.train_labeled_file = 'train_labeled.pkl'
        self.train_unlabeled_file = 'train_unlabeled.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train_labeled':
            with open(self.processed_folder + self.train_labeled_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)
        elif self.set_name == 'train_unlabeled':
            with open(self.processed_folder + self.train_unlabeled_file, 'rb') as fp:
                self.train_page_data, self.train_link_data = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)
        self.id_list = np.linspace(0, length - 1, num=length, endpoint=True, dtype=np.int32)

    def my_init(self):
        length = self.__len__()
        self.id_list = np.linspace(0, length - 1, num=length, endpoint=True, dtype=np.int32)

    def __getitem__(self, index):
        if self.set_name == 'train_unlabeled':
            item_id, page, link = self.id_list[index], self.train_page_data[index], self.train_link_data[index]
            return item_id, page, link
        else:
            item_id, page, link, target = self.id_list[index], self.train_page_data[index], \
                                          self.train_link_data[index], self.labels[index]
            return item_id, page, link, target

    def __len__(self):
        return len(self.train_link_data)


class ADValidationWithID(data.Dataset):
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'ad/'
        self.train_labeled_file = 'train_labeled.pkl'
        self.train_unlabeled_file = 'train_unlabeled.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train_labeled':
            with open(self.processed_folder + self.train_labeled_file, 'rb') as fp:
                self.view1_data, self.view2_data, self.view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'train_unlabeled':
            with open(self.processed_folder + self.train_unlabeled_file, 'rb') as fp:
                self.view1_data, self.view2_data, self.view3_data, = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.view1_data, self.view2_data, self.view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.view1_data, self.view2_data, self.view3_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)
        self.id_list = np.linspace(0, length - 1, num=length, endpoint=True, dtype=np.int32)

    def my_init(self):
        length = self.__len__()
        self.id_list = np.linspace(0, length - 1, num=length, endpoint=True, dtype=np.int32)

    def __getitem__(self, index):
        if self.set_name == 'train_unlabeled':
            item_id, view1_data, view2_data, view3_data = self.id_list[index], self.view1_data[index], self.view2_data[
                index], self.view3_data[index]
            return item_id, view1_data, view2_data, view3_data
        else:
            item_id, view1_data, view2_data, view3_data, target = self.id_list[index], self.view1_data[index], \
                                                                  self.view2_data[index], self.view3_data[index], \
                                                                  self.labels[
                                                                      index]
            return item_id, view1_data, view2_data, view3_data, target

    def __len__(self):
        return len(self.view2_data)
