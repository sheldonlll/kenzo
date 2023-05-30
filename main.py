# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import  torch
import argparse
from dataset import Dataset
from trainer import *
from tester import Tester
from params import Params

desc = 'Temporal KG Completion methods'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-dataset', help='Dataset', type=str, default='icews14', choices = ['icews14', 'icews05-15', 'gdelt'])
parser.add_argument('-model', help='Model', type=str, default='DE_DistMult', choices = ['DE_DistMult', 'DE_TransE', 'DE_SimplE'])
parser.add_argument('-ne', help='Number of epochs', type=int, default=500, choices = [500])
parser.add_argument('-bsize', help='Batch size', type=int, default=256,choices = [256])
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001, choices = [0.001])
parser.add_argument('-reg_lambda', help='L2 regularization parameter', type=float, default=0.0, choices = [0.0])
parser.add_argument('-emb_dim', help='Embedding dimension', type=int, default=500, choices = [50])  #  嵌入维度，小的可以选50# -100；大的可以选200/300 之前那个普通的知识图谱蒸馏中的大模型选的是512
parser.add_argument('-neg_ratio', help='Negative ratio', type=int, default=500, choices = [500])  #通常情况下，neg_ratio 取值越高，负样本的数量就会增加，对于缓解类别不平衡问题和提高模型性能有一定的帮助。
parser.add_argument('-dropout', help='Dropout probability', type=float, default=0.4, choices = [0.0, 0.2, 0.4]) #在Dropout技术中，我们需要设置一个dropout probability参数，表示每个神经元失活的概率。例如，如果设置dropout probability为0.5，那么在每一次训练中，网络的每个神经元都有50%的概率被失活。
parser.add_argument('-save_each', help='Save model and validate each K epochs', type=int, default=20, choices = [20])
parser.add_argument('-se_prop', help='Static embedding proportion', type=float, default=0.36) # 具体地说，如果模型在训练集上表现良好但在测试集上表现不佳，可能是因为模型出现了过拟合的问题，这时可以适当增加Static embedding proportion参数的值，减少静态嵌入向量的比例，增加动态嵌入向量的比例，从而提高模型的泛化能力。但是，如果Static embedding proportion参数设置过高，可能导致模型欠拟合，需要根据具体情况进行调整。
parser.add_argument('-KD',help='1 or 0',type=float,default=0)

args = parser.parse_args()

dataset = Dataset(args.dataset)

params = Params(
    ne=args.ne,
    bsize=args.bsize,
    lr=args.lr,
    reg_lambda=args.reg_lambda,
    emb_dim=args.emb_dim,
    neg_ratio=args.neg_ratio,
    dropout=args.dropout,
    save_each=args.save_each,
    se_prop=args.se_prop,


)
trainer = Trainer(dataset, params, args.model)
trainer.train()

# validating the trained models. we seect the model that has the best validation performance as the fina model 配置训练过程中的模型验证序列，并且将保存的路径用列表变量储存。
validation_idx = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
best_mrr = -1.0 #初始化最好的平均排名（MRR，Mean Reciprocal Rank）为负无穷。
best_index = '0'
model_prefix = "models/" + args.model + "/" + args.dataset + "/" + params.str_() + "_" # 迭代验证序列列表。

for idx in validation_idx: # 迭代验证序列列表。
    model_path = model_prefix + idx + ".chkpnt" # 构建训练好的模型的路径。
    tester = Tester(dataset, model_path, "valid") # 创建Tester对象，用于在验证数据上测试模型的性能。
    mrr = tester.test() # 运行Tester对象的方法来测试模型并计算评估指标MRR。
    if mrr > best_mrr: # 若当前MRR值优于已记录的最好值，则更新最好的MRR记录以及其所在的模型。
        best_mrr = mrr
        best_index = idx # 将当前索引记录为新的最佳索引。

# testing the best chosen model on the test set
print("Best epoch: " + best_index)
model_path = model_prefix + best_index + ".chkpnt"  # 选中最佳的模型权重保存路径。
tester = Tester(dataset, model_path, "test") # 创建Tester对象，用于在测试数据上测试模型的性能。
tester.test() # 运行Tester对象的方法来测试最佳模型并计算评估指标MRR，最终结果将被打印出来。




#KD

model_path="models/DE_DistMult/icews14/500_512_0.001_0.0_36_500_0.4_64_20_0.36_500.chkpnt"
teacher_model = torch.load(model_path)
trainer =KDTrainer(dataset, params, args.model,teacher_model)
trainer.train()

# validating the trained models. we seect the model that has the best validation performance as the fina model 配置训练过程中的模型验证序列，并且将保存的路径用列表变量储存。
validation_idx = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
best_mrr = -1.0 #初始化最好的平均排名（MRR，Mean Reciprocal Rank）为负无穷。
best_index = '0'
model_prefix = "models/" + args.model + "/" + args.dataset + "/" + params.str_() + "_" # 迭代验证序列列表。

for idx in validation_idx: # 迭代验证序列列表。
    model_path = model_prefix + idx + ".chkpnt" # 构建训练好的模型的路径。
    tester = Tester(dataset, model_path, "valid") # 创建Tester对象，用于在验证数据上测试模型的性能。
    mrr = tester.test() # 运行Tester对象的方法来测试模型并计算评估指标MRR。
    if mrr > best_mrr: # 若当前MRR值优于已记录的最好值，则更新最好的MRR记录以及其所在的模型。
        best_mrr = mrr
        best_index = idx # 将当前索引记录为新的最佳索引。

# testing the best chosen model on the test set
print("Best epoch: " + best_index)
model_path = model_prefix + best_index + ".chkpnt"  # 选中最佳的模型权重保存路径。
tester = Tester(dataset, model_path, "test") # 创建Tester对象，用于在测试数据上测试模型的性能。
tester.test() # 运行Tester对象的方法来测试最佳模型并计算评估指标MRR，最终结果将被打印出来。




#KD

model_path1="models/DE_DistMult/icews14/500_512_0.001_0.0_36_500_0.4_64_20_0.36_500.chkpnt"
model_path2="models/DE_DistMult/icews14/500_512_0.001_0.0_36_500_0.4_64_20_0.36_500.chkpnt"
model_path3="models/DE_DistMult/icews14/500_512_0.001_0.0_36_500_0.4_64_20_0.36_500.chkpnt"
teacher_models=[model_path1,model_path2,model_path3]
teacher_models = [torch.load(teacher_model) for teacher_model in teacher_models]
trainer =multiKDTrainer(dataset, params, args.model,teacher_models)
trainer.train()

# validating the trained models. we seect the model that has the best validation performance as the fina model 配置训练过程中的模型验证序列，并且将保存的路径用列表变量储存。
validation_idx = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
best_mrr = -1.0 #初始化最好的平均排名（MRR，Mean Reciprocal Rank）为负无穷。
best_index = '0'
model_prefix = "models/" + args.model + "/" + args.dataset + "/" + params.str_() + "_" # 迭代验证序列列表。

for idx in validation_idx: # 迭代验证序列列表。
    model_path = model_prefix + idx + ".chkpnt" # 构建训练好的模型的路径。
    tester = Tester(dataset, model_path, "valid") # 创建Tester对象，用于在验证数据上测试模型的性能。
    mrr = tester.test() # 运行Tester对象的方法来测试模型并计算评估指标MRR。
    if mrr > best_mrr: # 若当前MRR值优于已记录的最好值，则更新最好的MRR记录以及其所在的模型。
        best_mrr = mrr
        best_index = idx # 将当前索引记录为新的最佳索引。

# testing the best chosen model on the test set
print("Best epoch: " + best_index)
model_path = model_prefix + best_index + ".chkpnt"  # 选中最佳的模型权重保存路径。
tester = Tester(dataset, model_path, "test") # 创建Tester对象，用于在测试数据上测试模型的性能。
tester.test() # 运行Tester对象的方法来测试最佳模型并计算评估指标MRR，最终结果将被打印出来。



