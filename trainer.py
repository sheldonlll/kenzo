# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import parser
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pairsampler import *
from metric.utils import recall
from metric.batchsampler import NPairs
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer
from model.embedding import LinearEmbedding
import model.backbone as backbone
import metric.pairsampler as pair

from dataset import Dataset
from params import Params
from de_distmult import DE_DistMult
from de_transe import DE_TransE
from de_simple import DE_SimplE
from tester import Tester





class Trainer:
    def __init__(self, dataset, params, model_name):
        instance_gen = globals()[model_name]
        self.model_name = model_name
        self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params))
        self.dataset = dataset
        self.params = params

    def train(self, early_stop=False):
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.reg_lambda
        )  # weight_decay corresponds to L2 regularization

        loss_f = nn.CrossEntropyLoss()

        for epoch in range(1, self.params.ne + 1):
            last_batch = False
            total_loss = 0.0
            start = time.time()

            while not last_batch:
                optimizer.zero_grad()

                heads, rels, tails, years, months, days = self.dataset.nextBatch(self.params.bsize,
                                                                                 neg_ratio=self.params.neg_ratio)
                last_batch = self.dataset.wasLastBatch()

                scores = self.model(heads, rels, tails, years, months, days)

                ###Added for softmax####
                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                scores_reshaped = scores.view(num_examples, self.params.neg_ratio + 1) # 这一步将模型预测的结果进行reshape操作，是为了对softmax进行计算。具体来说，softmax的输入需要是一个二维的tensor，其中每行是一个样本的类别分数，每列对应一个类别。因此，需要将模型预测的结果进行reshape操作，将每个样本的类别分数排列在一起，形成一个二维的tensor。而其中，num_examples是批次中正样本的数量，self.params.neg_ratio + 1是因为每个正样本会对应n个负样本，所以一个批次中的样本总数就是num_examples * (self.params.neg_ratio + 1)。因此，需要将原本一维的scores tensor reshape成num_examples行，每行有self.params.neg_ratio + 1列。
                l = torch.zeros(num_examples).long().cuda() # 这一步是为了创建一个全零tensor，用于存储每个样本的标签。其中标签都是0，因为这里使用的是交叉熵损失函数(nn.CrossEntropyLoss())，并且在PyTorch中，交叉熵损失函数的输入参数是一个tensor，其中每个元素都表示该样本所属的类别标签。在这个任务中，是一个多类别分类问题，每个样本只有一个类别标签，因此所有样本的标签都是0。这个全零tensor的长度是num_examples，即批次中正样本的数量，因为对于每个正样本，需要对应n个负样本，所以总共的样本数是num_examples * (self.params.neg_ratio + 1)。在softmax计算中，这个标签tensor会作为交叉熵损失函数的输入参数，用于计算模型的损失。
                loss = loss_f(scores_reshaped, l)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print(time.time() - start)
            print("Loss in iteration " + str(epoch) + ": " + str(
                total_loss) + "(" + self.model_name + "," + self.dataset.name + ")")

            if epoch % self.params.save_each == 0:
                self.saveModel(epoch)

    def saveModel(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.model_name + "/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.model, directory + self.params.str_() + "_" + str(chkpnt) + ".chkpnt")


class KDTrainer:

    def __init__(self, dataset, params, model_name,teacher_model):
        instance_gen = globals()[model_name]
        self.model_name = model_name
        self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params))
        self.dataset = dataset
        self.params = params
        self.teacher_model=teacher_model

    def train(self, early_stop=False):
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.reg_lambda
        )  # weight_decay corresponds to L2 regularization

        loss_f = nn.CrossEntropyLoss()

        def knowledge_distillation_loss(y_pred, y_true, teacher_scores, temperature, alpha):
            """
            y_pred: 学生模型的输出
            y_true: 真实标签
            teacher_scores: 教师模型的输出
            temperature: 温度参数
            alpha: 两个损失函数的权重
            """
            # KL 散度损失
            kd_loss = F.kl_div(F.log_softmax(y_pred / temperature, dim=1),
                               F.softmax(teacher_scores / temperature, dim=1),
                               reduction='batchmean') * (temperature ** 2) * alpha

            # 交叉熵损失
            corss_loss = F.cross_entropy(y_pred, y_true) * (1 - alpha)
            # # 均方误差损失
            # mse_loss = F.mse_loss(y_pred, y_true.unsqueeze(1)) * (1 - alpha)

            return kd_loss + corss_loss


        for epoch in range(1, self.params.ne + 1):
            last_batch = False
            total_loss = 0.0
            start = time.time()
            loss_f = nn.CrossEntropyLoss()

            while not last_batch: #我先不切断传播
                optimizer.zero_grad()

                heads, rels, tails, years, months, days = self.dataset.nextBatch(self.params.bsize,
                                                                                 neg_ratio=self.params.neg_ratio)
                last_batch = self.dataset.wasLastBatch()

                scores_stu = self.model(heads, rels, tails, years, months, days)
                scores_tea = self.teacher_model(heads, rels, tails, years, months, days)
                scores_tea = scores_tea.detach() # 切断教师网络的反向传播


                ###Added for softmax####
                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                scores_reshaped_stu = scores_stu.view(num_examples, self.params.neg_ratio + 1)
                scores_reshaped_tea = scores_tea.view(num_examples, self.params.neg_ratio + 1)
                l = torch.zeros(num_examples).long().cuda()
                loss=knowledge_distillation_loss(scores_reshaped_stu,l,scores_reshaped_tea,5,0.5)

                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print(time.time() - start)
            print("Loss in iteration " + str(epoch) + ": " + str(
                total_loss) + "(" + self.model_name + "," + self.dataset.name + ")")

            if epoch % self.params.save_each == 0:
                self.saveModel(epoch)

    def saveModel(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.model_name + "/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.model, directory + self.params.str_() + "_" + str(chkpnt) + ".chkpnt")




class RKDTrainer:

    def __init__(self, dataset, params, model_name, teacher_model):
        instance_gen = globals()[model_name]
        self.model_name = model_name
        self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params))
        self.dataset = dataset
        self.params = params
        self.teacher_model = teacher_model

    def train(self, early_stop=False):

        dist_loss_all = []  # 记录距离损失的值（一个列表）
        angle_loss_all = []  # 记录角度损失的值（一个列表）
        dark_loss_all = []  # 记录暗度损失的值（一个列表）
        triplet_loss_all = []  # 记录Triplet Loss的值（一个列表）
        at_loss_all = []  # 记录注意力损失的值（一个列表）
        loss_all = []  # 记录总损失的值（一个列表）
        triplet_ratio=1
        dist_ratio=1
        angle_ratio=2
        dark_ratio=0.001
        dist_criterion = RkdDistance()
        angle_criterion = RKdAngle()
        dark_criterion = HardDarkRank(alpha=2, beta=3)
        triplet_criterion = L2Triplet(sampler=DistanceWeighted(), margin=0.2)
        at_loss=0



        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.reg_lambda
        )  # weight_decay corresponds to L2 regularization

        loss_f = nn.CrossEntropyLoss()



        for epoch in range(1, self.params.ne + 1):
            last_batch = False
            total_loss = 0.0
            start = time.time()

            while not last_batch:  # 我先不切断传播
                optimizer.zero_grad()

                heads, rels, tails, years, months, days = self.dataset.nextBatch(self.params.bsize,
                                                                                 neg_ratio=self.params.neg_ratio)
                last_batch = self.dataset.wasLastBatch()

                scores_stu = self.model(heads, rels, tails, years, months, days)
                scores_tea = self.teacher_model(heads, rels, tails, years, months, days)
                scores_tea = scores_tea.detach()  # 切断教师网络的反向传播

                ###Added for softmax####
                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                scores_reshaped_stu = scores_stu.view(num_examples, self.params.neg_ratio + 1)
                scores_reshaped_tea = scores_tea.view(num_examples, self.params.neg_ratio + 1)
                l = torch.zeros(num_examples).long().cuda()
                triplet_loss=0


                triplet_loss = triplet_ratio * triplet_criterion(scores_reshaped_stu,l)  # 通过计算三元组损失函数triplet_criterion，得到三元组损失triplet_loss的值
                dist_loss = dist_ratio * dist_criterion(scores_reshaped_stu, scores_reshaped_tea)  # 计算距离损失dist_loss的值，同时乘上权重系数
                angle_loss = angle_ratio * angle_criterion(scores_reshaped_stu, scores_reshaped_tea)  # 计算角度损失angle_loss的值，同时乘上权重系数
                # dark_loss = dark_ratio * dark_criterion(scores_reshaped_stu, scores_reshaped_tea)  # 计算暗度损失dark_loss的值，同时乘上权重系数
                loss = triplet_loss + dist_loss + angle_loss  + at_loss  # 计算总损失loss的值
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print(time.time() - start)
            print("Loss in iteration " + str(epoch) + ": " + str(
                total_loss) + "(" + self.model_name + "," + self.dataset.name + ")")

            if epoch % self.params.save_each == 0:
                self.saveModel(epoch)

    def saveModel(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.model_name + "/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.model, directory + self.params.str_() + "_" + str(chkpnt) + ".chkpnt")

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, is_ca=False):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        if is_ca:
            loss = (nn.KLDivLoss(reduction='none')(p_s, p_t) * (self.T**2)).sum(-1)
        else:
            loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss



class multiKDTrainer:

    def __init__(self, dataset, params, model_name,teacher_models):
        instance_gen = globals()[model_name]
        self.model_name = model_name
        self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params))
        self.dataset = dataset
        self.params = params
        self.teacher_models=teacher_models

    def train(self, early_stop=False):
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.reg_lambda
        )  # weight_decay corresponds to L2 regularization

        loss_f = nn.CrossEntropyLoss()

        def knowledge_distillation_loss(y_pred, y_true, teacher_scores, temperature, alpha):
            """
            y_pred: 学生模型的输出
            y_true: 真实标签
            teacher_scores: 教师模型的输出
            temperature: 温度参数
            alpha: 两个损失函数的权重
            """
            # KL 散度损失
            kd_loss = F.kl_div(F.log_softmax(y_pred / temperature, dim=1),
                               F.softmax(teacher_scores / temperature, dim=1),
                               reduction='batchmean') * (temperature ** 2) * alpha

            # 交叉熵损失
            corss_loss = F.cross_entropy(y_pred, y_true) * (1 - alpha)
            # # 均方误差损失
            # mse_loss = F.mse_loss(y_pred, y_true.unsqueeze(1)) * (1 - alpha)

            return kd_loss + corss_loss



        for epoch in range(1, self.params.ne + 1):
            last_batch = False
            total_loss = 0.0
            start = time.time()

            while not last_batch: #我先不切断传播
                optimizer.zero_grad()

                heads, rels, tails, years, months, days = self.dataset.nextBatch(self.params.bsize,
                                                                                 neg_ratio=self.params.neg_ratio)
                last_batch = self.dataset.wasLastBatch()

                scores_stu = self.model(heads, rels, tails, years, months, days)
                # logit_t_list=[scores_reshaped_tea1,scores_reshaped_tea2,scores_reshaped_tea3]
                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                logit_t_list=[]
                for teacher_model in self.teacher_models:
                    scores_teai = teacher_model(heads, rels, tails, years, months, days)
                    scores_teai = scores_teai.detach() # 切断教师网络的反向传播
                    scores_reshaped_teai= scores_teai.view(num_examples, self.params.neg_ratio + 1)
                    logit_t_list.append(scores_reshaped_teai)


                ###Added for softmax####

                scores_reshaped_stu = scores_stu.view(num_examples, self.params.neg_ratio + 1)

                l = torch.zeros(num_examples).long().cuda()

                criterion_cls_lc = nn.CrossEntropyLoss(reduction='none')

                loss_t_list = [criterion_cls_lc(logit_t, l) for logit_t in logit_t_list]
                loss_t = torch.stack(loss_t_list, dim=0)
                attention = (1.0 - F.softmax(loss_t, dim=0)) / (3 - 1)
                criterion_div=DistillKL(5)

                loss_div_list = [criterion_div(scores_reshaped_stu, logit_t, is_ca=True)
                                 for logit_t in logit_t_list]
                loss_div = torch.stack(loss_div_list, dim=0)
                bsz = loss_div.shape[1]
                loss = (torch.mul(attention, loss_div).sum()) / (1.0 * bsz * 1)
                loss_1 = loss_f(scores_reshaped_stu, l)
                loss=loss*0.3+loss_1*0.7
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print(time.time() - start)
            print("Loss in iteration " + str(epoch) + ": " + str(
                total_loss) + "(" + self.model_name + "," + self.dataset.name + ")")

            if epoch % self.params.save_each == 0:
                self.saveModel(epoch)

    def saveModel(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.model_name + "/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.model, directory + self.params.str_() + "_" + str(chkpnt) + ".chkpnt")
