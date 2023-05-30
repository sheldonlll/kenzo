# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import math
import copy
import time
import numpy as np
from random import shuffle
from scripts import shredFacts


class Dataset:
    """Implements the specified dataloader"""

    def __init__(self,
                 ds_name):
        """
        Params:
                ds_name : name of the dataset 
        """
        self.name = ds_name
        # self.ds_path = "<path-to-dataset>" + ds_name.lower() + "/"
        self.ds_path = "datasets/" + ds_name.lower() + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.data = {"train": self.readFile(self.ds_path + "train.txt"),
                     "valid": self.readFile(self.ds_path + "valid.txt"),
                     "test": self.readFile(self.ds_path + "test.txt")}

        self.start_batch = 0
        self.all_facts_as_tuples = None

        self.convertTimes()

        self.all_facts_as_tuples = set([tuple(d) for d in self.data["train"] + self.data["valid"] + self.data[
            "test"]])  # 将读取到的训练、验证、测试集中的事实保存为元组的形式用于后续操作。

        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl])

    def readFile(self,
                 filename):

        with open(filename, "r", encoding="utf-8") as f:
            data = f.readlines()

        facts = []
        for line in data:
            elements = line.strip().split("\t")

            head_id = self.getEntID(elements[0])
            rel_id = self.getRelID(elements[1])
            tail_id = self.getEntID(elements[2])
            timestamp = elements[3]

            facts.append([head_id, rel_id, tail_id, timestamp])

        return facts

    def convertTimes(self):  # 用于将时间戳转换为时间、日期和时间等不同形式的数据。
        """
        This function spits the timestamp in the day,date and time.
        """
        for split in ["train", "valid", "test"]:
            for i, fact in enumerate(self.data[split]):
                fact_date = fact[-1]
                self.data[split][i] = self.data[split][i][:-1]
                date = list(map(float, fact_date.split("-")))
                self.data[split][i] += date

    def numEnt(self):

        return len(self.ent2id)

    def numRel(self):

        return len(self.rel2id)

    def getEntID(self,
                 ent_name):

        if ent_name in self.ent2id:
            return self.ent2id[ent_name]
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]

    def getRelID(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name]
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]

    def nextPosBatch(self, batch_size):  #  函数用于返回下一个正样本批次。首先按照当前的起始位置和给定的批次大小，从训练集中选择相应数量的事实。如果选择完毕之后已经遍历了整个训练集，则重新从头开始选择事实；否则，当前位置向后移动。最后，返回选中的批次里的事实。
        if self.start_batch + batch_size > len(self.data["train"]):
            ret_facts = self.data["train"][self.start_batch:]
            self.start_batch = 0
        else:
            ret_facts = self.data["train"][self.start_batch: self.start_batch + batch_size]
            self.start_batch += batch_size
        return ret_facts

    def addNegFacts(self, bp_facts, neg_ratio): #用于给定一个事实批次，增加负样本。对于每个正样本，该函数会生成num_neg负样本，并将它们添加到原有批次中。具体实现方式是：首先将原始的批次复制多份，并在复制品上进行修改，以生成新的批次；然后，为个正样本生成num_neg个实体的随机数，并将其加到头实体和尾实体中，在模型ID的范围内进行取模，得到负样本的头实体和尾实体ID。最后，将新生成的正负样本批次列表合并，并返回。
        ex_per_pos = 2 * neg_ratio + 2
        facts = np.repeat(np.copy(bp_facts), ex_per_pos, axis=0)
        for i in range(bp_facts.shape[0]):
            s1 = i * ex_per_pos + 1
            e1 = s1 + neg_ratio
            s2 = e1 + 1
            e2 = s2 + neg_ratio

            facts[s1:e1, 0] = (facts[s1:e1, 0] + np.random.randint(low=1, high=self.numEnt(),
                                                                   size=neg_ratio)) % self.numEnt()
            facts[s2:e2, 2] = (facts[s2:e2, 2] + np.random.randint(low=1, high=self.numEnt(),
                                                                   size=neg_ratio)) % self.numEnt()

        return facts

    def addNegFacts2(self, bp_facts, neg_ratio): # 函数也用于增加负样本，但与addNegFacts()的实现方式略有不同。该函数首对原始批次中的每个正样本生成num_neg+1个组成对，其中一个为正样本，剩下的为负样本。对每个负样本，生成1个实体随机数，并加到头实体和尾实体中，同样进行ID取模操作得到负样本头实体和尾实体ID。最后将新生成的正负样本批次列表合并，并返回
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.numEnt(), size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.numEnt(), size=facts2.shape[0])

        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0

        facts1[:, 0] = (facts1[:, 0] + rand_nums1) % self.numEnt()
        facts2[:, 2] = (facts2[:, 2] + rand_nums2) % self.numEnt()
        return np.concatenate((facts1, facts2), axis=0)

    def nextBatch(self, batch_size, neg_ratio=1): # 用于生成下一个次。首先使用nextPosBatch()函数获取下一个正样本批次；然后调用addNegFacts2()函数，为个批次增加neg_ratio个负样本；最后使用shredFacts()函数，将这个新的正负样本批次列表进行处理，可供模型使用的批次（即将每个事实的不同部分分开存储的数组）
        bp_facts = self.nextPosBatch(batch_size)
        batch = shredFacts(self.addNegFacts2(bp_facts, neg_ratio))
        return batch

    def wasLastBatch(self): # 函数用于判断批次是否为最后一个批次，即当前是否已经遍历了整个训练集。具体实现方式是判断当前位置是否为0。
        return (self.start_batch == 0)
