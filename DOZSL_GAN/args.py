import argparse
import os

import random
def loadArgums():
    # 创建了一个 ArgumentParser 对象，用于定义程序可以接受的命令行参数，以及如何解析这些参数
    parser = argparse.ArgumentParser()
    # 接下来通过 add_argument() 方法，可以向 ArgumentParser 对象添加各种类型的参数
    '''
    Data loading
    '''
    parser.add_argument('--DATADIR', default='../../data', help='path to dataset')
    parser.add_argument('--Workers', default=2, help='number of data loading workers')
    # parser.add_argument('--DATASET', default='AwA2', help='for awa')
    parser.add_argument('--DATASET', default='ImNet_A', help='for imagenet')
    # 分别为用于训练的可见类别样本文件路径、用于测试的可见类别样本文件路径、用于测试的不可见类别样本文件路径和数据集分割文件的路径
    parser.add_argument('--SeenFeaFile', default='', help='imagenet seen samples for training model')
    parser.add_argument('--SeenTestFeaFile', default='', help='imagenet seen samples for testing model')
    parser.add_argument('--UnseenFeaFile', default='', help='')
    parser.add_argument('--SplitFile', default='', help='')

    # 语义嵌入的方法/存储 类别嵌入的文件路径
    parser.add_argument('--SemEmbed', default='', type=str, help='{TransE, DisenE, DisenKGAT, DOZSL_RD, DOZSL_AGG, DOZSL_AGG_sub}')
    parser.add_argument('--SemFile', default='', help='the file to store class embedding')
    # 分别表示语义特征的大小、噪声特征的大小和视觉特征的大小
    parser.add_argument('--SemSize', type=int, help='size of semantic features')
    parser.add_argument('--NoiseSize', type=int, default=100, help='size of semantic features')
    parser.add_argument('--FeaSize', default=2048, help='size of visual features')
    # 提取用于测试模型的不可见类别样本的子集大小
    parser.add_argument('--Unseen_NSample', type=int, help='extract the subset of unseen samples, for testing model')
    parser.add_argument('--PerClassAcc', action='store_true', default=False, help='testing the accuracy of each class')

    '''
    Generator and Discriminator
    '''
    # 生成器模型、判别器模型和预训练分类器模型的路径
    parser.add_argument('--NetG_Path', default='', help='path to netG (to continue training)')
    parser.add_argument('--NetD_Path', default='', help='path to netD (to continue training)')
    parser.add_argument('--Pretrained_Classifier', default='', help='path to pretrain classifier (to continue training)')

    parser.add_argument('--NetG_Name', default='MLP_G', help='')
    parser.add_argument('--NetD_Name', default='MLP_CRITIC', help='')
    # 生成器隐藏层单元数、判别器隐藏层单元数
    parser.add_argument('--NGH', default=4096, help='size of the hidden units in generator')
    parser.add_argument('--NDH', default=4096, help='size of the hidden units in discriminator')
    # 判别器迭代次数
    parser.add_argument('--Critic_Iter', default=5, help='critic iteration of discriminator, default=5, following WGAN-GP setting')
    # 梯度惩罚权重
    parser.add_argument('--GP_Weight', type=float, default=10, help='gradient penalty regularizer, default=10, the completion of Lipschitz Constraint in WGAN-GP')
    # 分类器损失权重
    parser.add_argument('--Cls_Weight', default=0.01, help='loss weight for the supervised classification loss')
    # 每个不可见类别生成的特征数量
    parser.add_argument('--SynNum', default=300, type=int, help='number of features generating for each unseen class; awa_default = 300')
    # 测试时训练可见类别分类器时的特征数量
    parser.add_argument('--SeenSynNum', default=300, help='number of features for training seen classifier when testing')

    '''
    Training Parameter
    '''
    parser.add_argument('--GZSL', action='store_true', default=False, help='enable generalized zero-shot learning')
    # 是否进行数据预处理和数据标准化
    parser.add_argument('--PreProcess', default=True, help='enbale MinMaxScaler on visual features, default=True')
    parser.add_argument('--Standardization', default=False, help='')
    # 是否启用交叉验证模式
    parser.add_argument('--Cross_Validation', default=False, help='enable cross validation mode')
    parser.add_argument('--Cuda', default=True, help='')
    parser.add_argument('--NGPU', default=1, help='number of GPUs to use')
    parser.add_argument('--device', default='0', help='')
    parser.add_argument('--ManualSeed', default=9416, type=int, help='')  #
    # 分别为批量大小、迭代轮数、学习率、分类器学习率、简单样本比例和 Adam 优化器的 beta 参数
    parser.add_argument('--BatchSize', default=4096, type=int, help='')
    parser.add_argument('--Epoch', default=100, help='')
    parser.add_argument('--LR', default=0.0001, type=float, help='learning rate to train GAN')
    parser.add_argument('--Cls_LR', default=0.001, help='after generating unseen features, the learning rate for training softmax classifier')
    parser.add_argument('--Ratio', default=0.1, help='ratio of easy samples')
    parser.add_argument('--Beta', default=0.5, help='beta for adam, default=0.5')
    # 分别为输出文件夹、输出文件名称、保存模型的频率、打印日志的频率、验证模型的频率和开始验证的轮数
    parser.add_argument('--OutFolder', default='./checkpoint/', help='folder to output data and model checkpoints')
    parser.add_argument('--OutName', default='imagenet', help='folder to output data and model checkpoints')
    parser.add_argument('--SaveEvery', default=100, help='')
    parser.add_argument('--PrintEvery', default=1, help='')
    parser.add_argument('--ValEvery', default=1, help='')
    parser.add_argument('--StartEvery', default=0, help='')
    # 定义了所有的参数，parse_args() 方法解析命令行参数
    # 并返回一个包含参数值的 Namespace 对象，通过该对象可以轻松地访问各个参数的值
    args = parser.parse_args()
    # 判断数据集
    if args.DATASET == 'AwA2':
        # 特征文件和分割文件
        args.FeaFile = 'res101.mat'
        args.SplitFile = 'binaryAtt_splits.mat'
        # 判断嵌入方法，赋值对应的嵌入后文件和语义特征的大小
        if args.SemEmbed == 'TransE':
            args.SemFile = os.path.join('embeddings', 'TransE_65000.mat')
            args.SemSize = 100
        if args.SemEmbed == 'RGAT':
            args.SemFile = os.path.join('embeddings', 'RGAT_9200_9191.mat')
            args.SemSize = 100
        if args.SemEmbed == 'DisenE':
            args.SemFile = os.path.join('embeddings', 'DisenE_5600_5583.mat')
            args.SemSize = 200
        if args.SemEmbed == 'DisenKGAT':
            args.SemFile = os.path.join('embeddings', 'DisenKGAT_9800_9667.mat')
            args.SemSize = 400
        if args.SemEmbed == 'DOZSL_RD':
            args.SemFile = os.path.join('embeddings', 'DOZSL_RD_4800_4666.mat')
            args.SemSize = 200
        if args.SemEmbed == 'DOZSL_AGG':
            args.SemFile = os.path.join('embeddings', 'DOZSL_AGG_5200_5125.mat')
            args.SemSize = 500
        if args.SemEmbed == 'DOZSL_AGG_sub':
            args.SemFile = os.path.join('embeddings', 'DOZSL_AGG_sub_5400_5291.mat')
            args.SemSize = 400

    else:

        args.SplitFile = 'split.mat'
        args.SeenFeaFile = 'Res101_Features/ILSVRC2012_train'
        args.SeenTestFeaFile = 'Res101_Features/ILSVRC2012_val'
        args.UnseenFeaFile = 'Res101_Features/ILSVRC2011'

        if args.DATASET == 'ImNet_A':

            if args.SemEmbed == 'TransE':
                args.SemFile = os.path.join('embeddings', 'TransE_65000.mat')
                args.SemSize = 100
            if args.SemEmbed == 'RGAT':
                args.SemFile = os.path.join('embeddings', 'RGAT_6200_6164.mat')
                args.SemSize = 100
            if args.SemEmbed == 'DisenE':
                args.SemFile = os.path.join('embeddings', 'DisenE_5000_4553.mat')
                args.SemSize = 200
            if args.SemEmbed == 'DisenKGAT':
                args.SemFile = os.path.join('embeddings', 'DisenKGAT_2400_2356.mat')
                args.SemSize = 200
            if args.SemEmbed == 'DOZSL_RD':
                args.SemFile = os.path.join('embeddings', 'DOZSL_RD_6000_5550.mat')
                args.SemSize = 200
            if args.SemEmbed == 'DOZSL_AGG':
                args.SemFile = os.path.join('embeddings', 'DOZSL_AGG_2200_2191.mat')
                args.SemSize = 400
            if args.SemEmbed == 'DOZSL_AGG_sub':
                args.SemFile = os.path.join('embeddings', 'DOZSL_AGG_sub_2000_1894.mat')
                args.SemSize = 400


        if args.DATASET == 'ImNet_O':


            if args.SemEmbed == 'TransE':
                args.SemFile = os.path.join('embeddings', 'TransE_55000.mat')
                args.SemSize = 100
            if args.SemEmbed == 'RGAT':
                args.SemFile = os.path.join('embeddings', 'RGAT_3000_2869.mat')
                args.SemSize = 100
            if args.SemEmbed == 'DisenE':
                args.SemFile = os.path.join('embeddings', 'DisenE_2800_2342.mat')
                args.SemSize = 200
            if args.SemEmbed == 'DisenKGAT':
                args.SemFile = os.path.join('embeddings', 'DisenKGAT_3000_2980.mat')
                args.SemSize = 200
            if args.SemEmbed == 'DOZSL_RD':
                args.SemFile = os.path.join('embeddings', 'DOZSL_RD_3600_3598.mat')
                args.SemSize = 200
            if args.SemEmbed == 'DOZSL_AGG':
                args.SemFile = os.path.join('embeddings', 'DOZSL_AGG_2000_1753.mat')
                args.SemSize = 400
            if args.SemEmbed == 'DOZSL_AGG_sub':
                args.SemFile = os.path.join('embeddings', 'DOZSL_AGG_sub_4200_4037.mat')
                args.SemSize = 400






    if args.ManualSeed is None:
        args.ManualSeed = random.randint(1, 10000)

    print("Random Seed: ", args.ManualSeed)

    return args
