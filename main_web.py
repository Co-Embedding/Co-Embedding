from __future__ import print_function

import copy

import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim
from sklearn.metrics import classification_report
from torch import nn
from torch.autograd import Variable
import networkx as nx

import coordinate_alignment as co_embedding
from read_data_validation import WebKBValidationWithID
import write_data_web as write_web


# The loss function
class ContrastiveAndClassificationLoss(nn.Module):
    def __init__(self, large_margin=1.0):
        super(ContrastiveAndClassificationLoss, self).__init__()
        self.margin = large_margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def contrastive_loss(self, x0, x1, t, view_weight):
        self.check_type_forward((x0, x1, t))

        # Euclidean distance
        diff = torch.abs(x0 - x1) + 1e-6
        dist_squared = torch.sum(torch.pow(diff, 2), 1)
        euclidean_distance = torch.sqrt(dist_squared)

        similar_distance = dist_squared
        dissimilar_distance = torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = t * similar_distance + (1 - t) * dissimilar_distance
        weighted_loss_contrastive = view_weight * loss_contrastive
        loss = torch.mean(weighted_loss_contrastive) / 2.0
        return loss

    def forward(self, sample1_view1_code, sample1_view2_code, y_hat1, sample2_view1_code, sample2_view2_code,
                y_hat2, target1, target2, t, view1_similarity_weight, view2_similarity_weight):
        # Classification Loss
        classification_loss = F.cross_entropy(y_hat1, target1) + F.cross_entropy(y_hat2, target2)
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(sample1_view1_code, sample2_view1_code, t, view1_similarity_weight) + \
                           self.contrastive_loss(sample1_view2_code, sample2_view2_code, t, view2_similarity_weight)
        loss = classification_loss + contrastive_loss
        return loss


# Weight deep metric learning model
class WeightedSiameseClassifier(nn.Module):
    def __init__(self, view_size=[3000, 1840], n_units=[128, 64, 64], out_size=32, c_n_units=[64, 32], class_num=2):
        super(WeightedSiameseClassifier, self).__init__()

        # View1
        self.view1_l1 = nn.Linear(view_size[0], n_units[0])
        self.view1_l2 = nn.Linear(n_units[0], n_units[1])
        self.view1_l3 = nn.Linear(n_units[1], n_units[2])
        self.view1_l4 = nn.Linear(n_units[2], out_size)

        # View2
        self.view2_l1 = nn.Linear(view_size[1], n_units[0])
        self.view2_l2 = nn.Linear(n_units[0], n_units[1])
        self.view2_l3 = nn.Linear(n_units[1], n_units[2])
        self.view2_l4 = nn.Linear(n_units[2], out_size)

        # Classification
        self.classification_l1 = nn.Linear(out_size * 2, c_n_units[0])
        self.classification_l2 = nn.Linear(c_n_units[0], c_n_units[1])
        self.classification_l3 = nn.Linear(c_n_units[1], class_num)

        # init
        self.init_params()

    def init_params(self):
        init.kaiming_normal_(self.view1_l1.weight)
        init.kaiming_normal_(self.view1_l2.weight)
        init.kaiming_normal_(self.view1_l3.weight)
        init.kaiming_normal_(self.view1_l4.weight)
        init.kaiming_normal_(self.view2_l1.weight)
        init.kaiming_normal_(self.view2_l2.weight)
        init.kaiming_normal_(self.view2_l3.weight)
        init.kaiming_normal_(self.view2_l4.weight)

        init.kaiming_normal_(self.classification_l1.weight)
        init.kaiming_normal_(self.classification_l2.weight)
        init.kaiming_normal_(self.classification_l3.weight)

    def encode(self, view1_input, view2_input):
        # View 1
        view1_view = F.relu(self.view1_l1(view1_input))
        view1_view = F.relu(self.view1_l2(view1_view))
        view1_view = F.relu(self.view1_l3(view1_view))
        view1_code = self.view1_l4(view1_view)

        # View 2
        view2_view = F.relu(self.view2_l1(view2_input))
        view2_view = F.relu(self.view2_l2(view2_view))
        view2_view = F.relu(self.view2_l3(view2_view))
        view2_code = self.view2_l4(view2_view)
        return view1_code, view2_code

    def forward_once(self, view1_input, view2_input):
        # Encode
        view1_code, view2_code = self.encode(view1_input, view2_input)
        # Classification
        classification_input = torch.cat([view1_code, view2_code], dim=1)
        classification_output = F.relu(self.classification_l1(F.dropout(classification_input)))
        classification_output = F.relu(self.classification_l2(F.dropout(classification_output)))
        classification_output = self.classification_l3(classification_output)
        return view1_code, view2_code, classification_output

    def forward(self, x1_view1, x1_view2, x2_view1, x2_view2):
        sample1_view1_code, sample1_view2_code, y_hat1 = self.forward_once(x1_view1, x1_view2)
        sample2_view1_code, sample2_view2_code, y_hat2 = self.forward_once(x2_view1, x2_view2)
        return sample1_view1_code, sample1_view2_code, y_hat1, sample2_view1_code, sample2_view2_code, y_hat2


# Hyper Parameters
MAX_EPOCH = 150
BATCH_SIZE = 40
USE_GPU = False
CLASS_NUM = 2

# Read data
train_labeled = WebKBValidationWithID(set_name='train_labeled')
train_unlabeled = WebKBValidationWithID(set_name='train_unlabeled')
test_data = WebKBValidationWithID(set_name='test')
validation_data = WebKBValidationWithID(set_name='validation')

# View weight matrix
view1_weight = np.ones((len(train_labeled), len(train_labeled)), dtype=np.float32)
view2_weight = np.ones((len(train_labeled), len(train_labeled)), dtype=np.float32)

# Build Model
model = WeightedSiameseClassifier(view_size=[3000, 1840], n_units=[128, 64, 64], out_size=16, c_n_units=[128, 64],
                                  class_num=2)
# print(model)
if USE_GPU:
    model = model.cuda()

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
loss_function = ContrastiveAndClassificationLoss(large_margin=0.8)
validation_criterion = torch.nn.CrossEntropyLoss()

# Data Loader for easy mini-batch
train_loader = torch.utils.data.DataLoader(dataset=train_labeled, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=True)


def train(item_id, train_page, train_link, train_labels):
    x0_batch_id = item_id.numpy()
    x0_batch_page = train_page.numpy()
    x0_batch_link = train_link.numpy()
    y0_train_labels = train_labels.numpy()

    x1_batch_id = copy.deepcopy(x0_batch_id[::-1])
    x1_batch_page = copy.deepcopy(x0_batch_page[::-1])
    x1_batch_link = copy.deepcopy(x0_batch_link[::-1])
    y1_train_labels = copy.deepcopy(y0_train_labels[::-1])

    t = np.array(y0_train_labels == y1_train_labels, dtype=np.long)

    view1_batch_weight_array = []
    view2_batch_weight_array = []
    for i in range(len(x0_batch_id)):
        view1_batch_weight_array.append(view1_weight[x0_batch_id[i]][x1_batch_id[i]])
        view2_batch_weight_array.append(view2_weight[x0_batch_id[i]][x1_batch_id[i]])

    view1_batch_weight_array = np.asarray(view1_batch_weight_array, dtype=np.float32)
    view2_batch_weight_array = np.asarray(view2_batch_weight_array, dtype=np.float32)

    if USE_GPU:
        x0_batch_page = Variable(torch.from_numpy(x0_batch_page)).type(torch.cuda.FloatTensor)
        x0_batch_link = Variable(torch.from_numpy(x0_batch_link)).type(torch.cuda.FloatTensor)
        x1_batch_page = Variable(torch.from_numpy(x1_batch_page)).type(torch.cuda.FloatTensor)
        x1_batch_link = Variable(torch.from_numpy(x1_batch_link)).type(torch.cuda.FloatTensor)
        y0_train_labels = y0_train_labels.cuda()
        y1_train_labels = y1_train_labels.cuda()
        view1_batch_weight_array = Variable(torch.from_numpy(view1_batch_weight_array)).type(torch.cuda.FloatTensor)
        view2_batch_weight_array = Variable(torch.from_numpy(view2_batch_weight_array)).type(torch.cuda.FloatTensor)
        t = Variable(torch.from_numpy(t)).type(torch.cuda.FloatTensor)
    else:
        x0_batch_page = Variable(torch.from_numpy(x0_batch_page)).type(torch.FloatTensor)
        x0_batch_link = Variable(torch.from_numpy(x0_batch_link)).type(torch.FloatTensor)
        x1_batch_page = Variable(torch.from_numpy(x1_batch_page)).type(torch.FloatTensor)
        x1_batch_link = Variable(torch.from_numpy(x1_batch_link)).type(torch.FloatTensor)
        y0_train_labels = Variable(torch.from_numpy(y0_train_labels)).type(torch.LongTensor)
        y1_train_labels = Variable(torch.from_numpy(y1_train_labels)).type(torch.LongTensor)

        view1_batch_weight_array = Variable(torch.from_numpy(view1_batch_weight_array)).type(torch.FloatTensor)
        view2_batch_weight_array = Variable(torch.from_numpy(view2_batch_weight_array)).type(torch.FloatTensor)
        t = Variable(torch.from_numpy(t)).type(torch.FloatTensor)

    optimizer.zero_grad()

    sample1_view1_code, sample1_view2_code, y_hat1, sample2_view1_code, sample2_view2_code, y_hat2 = \
        model(x1_view1=x0_batch_page, x1_view2=x0_batch_link, x2_view1=x1_batch_page, x2_view2=x1_batch_link)
    loss = loss_function(sample1_view1_code, sample1_view2_code, y_hat1, sample2_view1_code, sample2_view2_code, y_hat2,
                         y0_train_labels, y1_train_labels, t, view1_batch_weight_array, view2_batch_weight_array)
    loss.backward()
    optimizer.step()
    return loss, y_hat1


def validation():
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    for iteration_num, batch_validation_data in enumerate(validation_loader):
        item_id, validation_page_inputs, validation_link_inputs, validation_labels = batch_validation_data
        validation_labels = torch.squeeze(validation_labels)

        if USE_GPU:
            validation_page_inputs, validation_link_inputs, validation_labels = \
                Variable(validation_page_inputs.cuda()), Variable(
                    validation_link_inputs.cuda()), validation_labels.cuda()
        else:
            validation_page_inputs = Variable(validation_page_inputs).type(torch.FloatTensor)
            validation_link_inputs = Variable(validation_link_inputs).type(torch.FloatTensor)
            validation_labels = Variable(validation_labels).type(torch.LongTensor)

        view1_code, view2_code, classification_output = model.forward_once(validation_page_inputs,
                                                                           validation_link_inputs)
        loss = validation_criterion(classification_output, validation_labels)

        # calc validation acc
        _, predicted = torch.max(classification_output.data, 1)
        total_acc += (predicted.data.numpy() == validation_labels.data.numpy()).sum()
        total += len(validation_labels)
        total_loss += loss.item()

    validation_loss = total_loss / total
    validation_acc = total_acc / total
    return validation_loss, validation_acc


def test():
    test_page_inputs = torch.from_numpy(test_data.train_page_data)
    test_link_inputs = torch.from_numpy(test_data.train_link_data)

    if USE_GPU:
        test_page_inputs, test_link_inputs = Variable(test_page_inputs.cuda()), Variable(test_link_inputs.cuda())

    else:
        test_page_inputs = Variable(test_page_inputs).type(torch.FloatTensor)
        test_link_inputs = Variable(test_link_inputs).type(torch.FloatTensor)

    view1_code, view2_code, classification_output = model.forward_once(test_page_inputs, test_link_inputs)
    _, predicted = torch.max(classification_output.data, 1)
    predicted_numpy = predicted.data.numpy()
    classification_result = classification_report(test_data.labels, predicted_numpy, digits=5, output_dict=True)
    test_acc = classification_result['weighted avg']['precision']
    return test_acc


def test_report():
    test_page_inputs = torch.from_numpy(test_data.train_page_data)
    test_link_inputs = torch.from_numpy(test_data.train_link_data)
    test_labels = torch.from_numpy(test_data.labels)

    if USE_GPU:
        test_page_inputs, test_link_inputs = Variable(test_page_inputs.cuda()), Variable(test_link_inputs.cuda())

    else:
        test_page_inputs = Variable(test_page_inputs).type(torch.FloatTensor)
        test_link_inputs = Variable(test_link_inputs).type(torch.FloatTensor)

        view1_code, view2_code, classification_output = model.forward_once(test_page_inputs, test_link_inputs)
    _, predicted = torch.max(classification_output.data, 1)
    print("\n\n--------------------------- Test Result ------------------------------\n")
    print(classification_report(test_labels, predicted, digits=4))


def train_network():
    print("Network Training...")
    train_loss_ = []
    validation_loss_ = []
    validation_acc_ = []
    train_acc_ = []
    validation_acc_max = 0.0
    validation_patience_original = 3
    validation_patience = validation_patience_original
    for epoch in range(MAX_EPOCH):
        # validation =====================================================
        validation_patience -= 1
        if validation_patience == 0:
            if validation_acc_[len(validation_acc_) - 1] < 0.85:
                train_loss_.append(train_loss_[len(train_loss_) - 1])
                train_acc_.append(train_acc_[len(train_acc_) - 1])
                validation_loss_.append(validation_loss_[len(validation_loss_) - 1])
                validation_acc_.append(validation_acc_[len(validation_acc_) - 1])
                continue
            else:
                break
        # Train ==========================================================
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iteration_num, batch_train_labeled in enumerate(train_loader):
            item_id, page_input, link_input, train_labels = batch_train_labeled
            train_labels = torch.squeeze(train_labels)
            train_loss, classification_output = train(item_id, page_input, link_input, train_labels)

            # calc training acc
            _, predicted = torch.max(classification_output.data, 1)
            total_acc += (predicted.numpy() == train_labels.numpy()).sum()
            total += len(train_labels)
            total_loss += train_loss.item()

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)

        # validation ====================================================
        validation_loss, validation_acc = validation()
        validation_loss_.append(validation_loss)
        validation_acc_.append(validation_acc)
        # print("validation_acc: ", validation_acc)
        if validation_acc >= validation_acc_max:
            if validation_acc > validation_acc_max:
                # Take a snapshot
                torch.save(model.state_dict(), "./model/best_validation_acc_model.model")
            validation_acc_max = validation_acc
            validation_patience = validation_patience_original
        print('[Epoch: %3d/%3d] Training Loss: %.3f, Validation Loss: %.3f, Training Acc: %.3f, Validation Acc: %.4f' %
              (epoch, MAX_EPOCH, train_loss_[epoch], validation_loss_[epoch], train_acc_[epoch] * 100,
               validation_acc_[epoch] * 100))
    # Load the best model
    model.load_state_dict(torch.load("./model/best_validation_acc_model.model"))
    test_report()


def encode_data(data):
    view1_inputs = torch.from_numpy(data.train_page_data)
    view2_inputs = torch.from_numpy(data.train_link_data)

    if USE_GPU:
        view1_inputs, view2_inputs = Variable(view1_inputs.cuda()), Variable(view2_inputs.cuda())
    else:
        view1_inputs = Variable(view1_inputs).type(torch.FloatTensor)
        view2_inputs = Variable(view2_inputs).type(torch.FloatTensor)

    view1_code, view2_code = model.encode(view1_inputs, view2_inputs)
    view1_code = view1_code.data.numpy()
    view2_code = view2_code.data.numpy()
    return view1_code, view2_code


def coordinate_align(k=15, m=9):
    print("Coordinate Alignment...")
    view1_code, view2_code = encode_data(train_labeled)
    view_list = [view1_code, view2_code]
    view_weight_list = [view1_weight, view2_weight]
    id_list = np.linspace(0, len(train_labeled) - 1, num=len(train_labeled), endpoint=True, dtype=np.int32)

    similarity_matrix = co_embedding.align(id_list=id_list,
                                           code_of_all_views=view_list,
                                           view_weight_list=view_weight_list,
                                           k=k, m=m)
    return similarity_matrix


"""
Semi Start ==================================================================
"""


def init_view_weight():
    print("Update view weight...")
    global view1_weight
    global view2_weight
    view1_weight = np.ones((len(train_labeled), len(train_labeled)), dtype=np.float32)
    view2_weight = np.ones((len(train_labeled), len(train_labeled)), dtype=np.float32)


def match_between_components(components_list, class_id_list, class_num=CLASS_NUM):
    # all results
    predicted_set_for_all_classes = []
    class_max_match_length_list = []
    matched_set = []
    for i in range(class_num):
        a_predicted_set = set()
        predicted_set_for_all_classes.append(a_predicted_set)
        class_max_match_length_list.append(0)
        matched_set.append(-1)

    for component_index, a_component in enumerate(components_list):
        for class_index in range(class_num):
            # match between the component and each class
            a_matched_set = a_component & set(class_id_list[class_index])
            if len(a_matched_set) > class_max_match_length_list[class_index]:
                class_max_match_length_list[class_index] = len(a_matched_set)
                predicted_set_for_all_classes[class_index] = copy.deepcopy(a_component)
                matched_set[class_index] = component_index

    if len(set(matched_set)) != class_num:
        # print("Wrong, match to same set!!!")
        return -1, -1, None
    else:
        # Calculate the accuracy
        # Class 0
        predicted_zero_set = predicted_set_for_all_classes[0]
        class_zero_id_set = class_id_list[0]
        # Class 1
        predicted_one_set = predicted_set_for_all_classes[1]
        class_one_id_set = class_id_list[1]

        tp = len(set(class_one_id_set) & predicted_one_set)
        fp = len(set(class_zero_id_set) & predicted_one_set)
        fn = len(set(class_one_id_set) & predicted_zero_set)
        tn = len(set(class_zero_id_set) & predicted_zero_set)

        acc = (tp + tn) * 1.0 / (tp + tn + fp + fn)
        acc_length = tp + tn
        return acc, acc_length, predicted_set_for_all_classes


def discriminative_regularization(similarity_matrix, labels, class_num=CLASS_NUM):
    print("Discriminative regularization...")
    perfect_weight = -1
    max_matched_acc = 0.0
    max_matched_length = -1
    min_weight = np.min(similarity_matrix)
    max_weight = np.max(similarity_matrix)
    print("Mini weight = ", min_weight, " Max weight = ", max_weight)
    class_id_list = []
    best_predicted_list = []
    for class_index in range(class_num):
        a_id_list = np.where(labels == class_index)[0]
        class_id_list.append(a_id_list)
        best_predicted_list.append(set())
    for a_weight in range(min_weight, max_weight):
        # Build the Graph
        graph = nx.Graph()
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix)):
                if similarity_matrix[i][j] >= a_weight:
                    graph.add_edge(i, j, weight=similarity_matrix[i][j])

        # Check the Graph
        all_connected_components = sorted(nx.connected_components(graph), key=len, reverse=True)
        if len(all_connected_components) < class_num:
            continue

        # Calculate the accuracy
        match_acc, match_length, predicted_set_for_all_classes = match_between_components(
            components_list=all_connected_components, class_id_list=class_id_list, class_num=class_num)
        if match_acc > 0 and match_acc >= max_matched_acc:
            print("Weight = ", a_weight, " Match Acc = ", match_acc * 100, "%. Match length = ", match_length)
            if match_acc > max_matched_acc:
                max_matched_acc = match_acc
                max_matched_length = match_length
                best_predicted_list = copy.deepcopy(predicted_set_for_all_classes)
                perfect_weight = a_weight
            elif match_acc == max_matched_acc:
                if match_length > max_matched_length:
                    max_matched_acc = match_acc
                    max_matched_length = match_length
                    best_predicted_list = copy.deepcopy(predicted_set_for_all_classes)
                    perfect_weight = a_weight
    print("Perfect weight = ", perfect_weight, " Match Acc = ", max_matched_acc * 100, "%. Match length = ",
          max_matched_length)
    return perfect_weight, best_predicted_list


def semi_supervised_learning(k=15, m=9, class_num=CLASS_NUM):
    print("Start label propagation...")

    # Step1: Encode the labeled and unlabeled training data===================================
    label_data = validation_data
    unlabeled_data = train_unlabeled
    page_input = np.vstack((label_data.train_page_data, unlabeled_data.train_page_data))
    link_input = np.vstack((label_data.train_link_data, unlabeled_data.train_link_data))
    labels = label_data.labels

    code_page_input = torch.from_numpy(page_input)
    code_link_input = torch.from_numpy(link_input)

    if USE_GPU:
        code_page_input, code_link_input = Variable(code_page_input.cuda()), Variable(
            code_link_input.cuda())
    else:
        code_page_input = Variable(code_page_input).type(torch.FloatTensor)
        code_link_input = Variable(code_link_input).type(torch.FloatTensor)

    view1_code, view2_code = model.encode(code_page_input, code_link_input)
    view1_code = view1_code.data.numpy()
    view2_code = view2_code.data.numpy()
    code_list = [view1_code, view2_code]
    all_id_list = np.linspace(0, len(page_input) - 1, num=len(page_input), endpoint=True, dtype=np.int32)
    unlabeled_id_list = np.linspace(len(label_data), len(label_data) + len(unlabeled_data) - 1, num=len(unlabeled_data),
                                    endpoint=True, dtype=np.int32)
    labeled_id_list = np.linspace(0, len(labels) - 1, num=len(labels), endpoint=True, dtype=np.int32)
    labeled_id_set = set(labeled_id_list)
    similarity_matrix = co_embedding.build_similarity_matrix(id_list=all_id_list, code_of_all_views=code_list, k=k, m=m)
    # check the similarity_matrix, and find the perfect similarity weight on validation data
    perfect_weight, best_predicted_list = discriminative_regularization(
        similarity_matrix=similarity_matrix, labels=labels, class_num=CLASS_NUM)

    # label the unlabeled
    new_labeled_list = []
    new_labeled_label = []
    for class_index in range(class_num):
        a_labeled_set = best_predicted_list[class_index] - labeled_id_set
        new_labeled_list.extend(list(a_labeled_set))
        new_labeled_label.extend(np.ones(len(a_labeled_set), dtype=np.int32) * class_index)

    new_labeled_page = page_input[new_labeled_list]
    new_labeled_link = link_input[new_labeled_list]

    new_labeled_page_numpy = np.vstack((train_labeled.train_page_data, new_labeled_page))
    new_labeled_link_numpy = np.vstack((train_labeled.train_link_data, new_labeled_link))
    original_labels = np.asarray(train_labeled.labels, dtype=np.int32).reshape(len(train_labeled.labels))
    new_labeled_label = np.hstack((original_labels, new_labeled_label))

    new_unlabeled_id_list = set(unlabeled_id_list) - set(new_labeled_list)
    new_unlabeled_page_numpy = page_input[list(new_unlabeled_id_list)]
    new_unlabeled_link_numpy = link_input[list(new_unlabeled_id_list)]

    # check
    if len(new_unlabeled_id_list) + len(new_labeled_list) + len(labels) != len(page_input):
        print("Wrong!!!! Lost data")
    del train_labeled.train_page_data
    del train_labeled.train_link_data
    del train_labeled.labels
    train_labeled.train_page_data = copy.deepcopy(new_labeled_page_numpy)
    train_labeled.train_link_data = copy.deepcopy(new_labeled_link_numpy)
    train_labeled.labels = copy.deepcopy(new_labeled_label)
    train_labeled.my_init()

    del train_unlabeled.train_page_data
    del train_unlabeled.train_link_data
    train_unlabeled.train_page_data = copy.deepcopy(new_unlabeled_page_numpy)
    train_unlabeled.train_link_data = copy.deepcopy(new_unlabeled_link_numpy)
    train_unlabeled.my_init()

    # init weight again
    init_view_weight()


"""
Semi End ==================================================================
"""


def init_data():
    global train_labeled
    global train_unlabeled
    global test_data
    global validation_data

    global train_loader
    global test_loader
    global validation_loader

    train_labeled = WebKBValidationWithID(set_name='train_labeled')
    train_unlabeled = WebKBValidationWithID(set_name='train_unlabeled')
    test_data = WebKBValidationWithID(set_name='test')
    validation_data = WebKBValidationWithID(set_name='validation')

    train_loader = torch.utils.data.DataLoader(dataset=train_labeled, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=True)


def main(iter_num=20):
    labels = train_labeled.labels
    class_num = CLASS_NUM
    # setting the K and M
    mini_class_size = 1e6
    for i in range(class_num):
        a_class_size = len(np.where(labels == i)[0])
        if a_class_size < mini_class_size:
            mini_class_size = a_class_size
    print("Mini class size = ", mini_class_size)
    k = int(0.8 * mini_class_size)
    m = int(0.8 * k)
    # train the model
    for i in range(1, iter_num + 1):
        print("Iter = ", i)
        train_network()
        if i == iter_num - 1:
            break
        coordinate_align(k=k, m=m)
        if i % 5 == 0:
            semi_supervised_learning(k=k, m=m, class_num=CLASS_NUM)
    # Test the model
    final_test_acc = test()
    return final_test_acc


if __name__ == "__main__":
    labeled_size = 0.2
    results = []
    for seed in range(20):
        unlabeled_size = 1 - labeled_size
        train_size = labeled_size / 2
        write_web.write_web(train_label_size=train_size, validation_size=train_size,
                            train_unlabeled_size=unlabeled_size, seed=seed)
        init_data()
        acc = main(iter_num=30)
        print(acc)
        results.append(acc)

    average = np.mean(results)
    print("Results: ", results)
    print("Average: ", average)
