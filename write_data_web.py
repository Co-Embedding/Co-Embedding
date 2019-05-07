#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function

import pickle as pickle
import scipy.io as scio
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

path = './data_set/'


def read_web_kb():
    data = scio.loadmat(path + 'WebKB.mat')
    page = data['page']
    link = data['link']
    class_label = data['class_label']
    return page.T, link.T, class_label.T - 1


def write_web(data_set_name='webkb', train_label_size=0.1,
              validation_size=0.1,
              train_unlabeled_size=0.3, seed=3):
    if train_label_size + validation_size + train_unlabeled_size != 1:
        print("Error !!! The sum of train_label_size, validation_size and train_unlabeled_size should be 1")
        return
    print("Start write data, train labeled rate (train labeled + validation) = ",
          (train_label_size + validation_size) * 100, "%")

    view1, view2, label = read_web_kb()
    size_list = []
    class_size = 2
    for i in range(class_size):
        indexes = np.where(label == i)[0]
        size_list.append(len(indexes))
    print("Sample number in each class: ", size_list)
    view1 = np.asarray(view1, dtype=np.float32)
    view2 = np.asarray(view2, dtype=np.float32)
    label = np.asarray(label, dtype=np.int32)
    seed = seed
    # Split half of the data as test set ===========================================================================
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, train_size=0.5, random_state=seed)
    for train_idx, test_idx in stratified_split.split(view1, label):
        view1_train, view1_test = view1[train_idx], view1[test_idx]
        view2_train, view2_test = view2[train_idx], view2[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

    size_list = []
    for i in range(class_size):
        indexes = np.where(y_test == i)[0]
        size_list.append(len(indexes))
    print(size_list, "Test size")
    with open(path + data_set_name + '/test.pkl', 'wb') as f_test:
        pickle.dump((view1_test, view2_test, y_test), f_test, -1)

    # Train unlabeled ===========================================================================
    labeled_validation_unlabeled_sum = train_label_size + validation_size + train_unlabeled_size
    stratified_split = StratifiedShuffleSplit(n_splits=1,
                                              test_size=train_unlabeled_size / labeled_validation_unlabeled_sum,
                                              random_state=seed)
    for train_idx, test_idx in stratified_split.split(view1_train, y_train):
        view1_train_validation, view1_train_unlabeled = view1_train[train_idx], view1[test_idx]
        view2_train_validation, view2_train_unlabeled = view2_train[train_idx], view2_train[test_idx]
        train_validation_label, use_less_label = y_train[train_idx], y_train[test_idx]

    with open(path + data_set_name + '/train_unlabeled.pkl', 'wb') as f_train:
        pickle.dump((view1_train_unlabeled, view2_train_unlabeled), f_train, -1)
    size_list = []
    for i in range(class_size):
        indexes = np.where(use_less_label == i)[0]
        size_list.append(len(indexes))
    print(size_list, "Unlabeled size")
    # Train label and validation ===========================================================================

    train_validation_sum = (train_label_size + validation_size)
    stratified_split = StratifiedShuffleSplit(n_splits=1,
                                              test_size=validation_size / train_validation_sum,
                                              random_state=seed)
    for train_idx, test_idx in stratified_split.split(view1_train_validation, train_validation_label):
        view1_train_label, view1_validation = view1_train_validation[train_idx], view1_train_validation[test_idx]
        view2_train_label, view2_validation = view2_train_validation[train_idx], view2_train_validation[test_idx]
        train_label, validation_label = train_validation_label[train_idx], train_validation_label[test_idx]

    with open(path + data_set_name + '/train_labeled.pkl', 'wb') as f_train:
        pickle.dump((view1_train_label, view2_train_label, train_label), f_train, -1)
    with open(path + data_set_name + '/validation.pkl', 'wb') as f_train:
        pickle.dump((view1_validation, view2_validation, validation_label), f_train, -1)
    size_list = []
    for i in range(class_size):
        indexes = np.where(train_label == i)[0]
        size_list.append(len(indexes))
    print(size_list, "labeled size")
    size_list = []
    for i in range(class_size):
        indexes = np.where(validation_label == i)[0]
        size_list.append(len(indexes))
    print(size_list, "validation size")


def test_write(label_rate=0.4):
    name = 'webkb'
    print("=============================== Label rate = ", label_rate, "=========================================")
    unlabeled_size = 1 - label_rate
    label_size = label_rate / 2
    write_web(train_label_size=label_size, validation_size=label_size, train_unlabeled_size=unlabeled_size)
    print("\n\n\nRead data ===========================================================================")
    with open(path + name + '/' + 'test.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
    print("test size = ", len(train_labels), train_page_data.shape, train_link_data.shape)
    with open(path + name + '/' + '/train_labeled.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
        print("train_labeled size = ", len(train_labels), train_page_data.shape, train_link_data.shape)
    with open(path + name + '/' + '/train_unlabeled.pkl', 'rb') as fp:
        train_page_data, train_link_data = pickle.load(fp)
    print("train_unlabeled size = ", len(train_page_data), train_page_data.shape, train_link_data.shape)
    with open(path + name + '/' + '/validation.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
    print("validation size = ", len(train_labels), train_page_data.shape, train_link_data.shape)


if __name__ == "__main__":
    # Test
    test_write(label_rate=0.1)
