from __future__ import print_function
import argparse # 命令行参数解析模块
import os       # os 模块提供了许多与操作系统交互的功能，可能被用来处理文件路径和执行系统命令
import random   # 生成随机数等操作
import torch    # pytorch深度学习库的核心模块，用于构建神经网络和执行张量运算等
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np  # numpy是科学计算的一个库，提供了多维数组和许多用于操作数组的函数，可能被用来处理数据和执行数值计算
import time         # 时间库

import sys          # 提供了与python解释器交互的功能，被用来修改解释器的行为或获取系统相关信息
sys.path.append('../../..')
from ZS_IMGC_all.models.DOZSL_GAN import args       # 参数集
from ZS_IMGC_all.models.DOZSL_GAN import model      # 模型集
from ZS_IMGC_all.models.DOZSL_GAN import classifier_cls         # 进行类别分类的模型
from ZS_IMGC_all.models.DOZSL_GAN import classifier_pretrain    # 模型预训练的模块
from ZS_IMGC_all.models.DOZSL_GAN import util       # 辅助功能


# 确保cuda可用
import torch
torch.cuda.current_device() # 获取当前 CUDA 设备的索引
torch.cuda._initialized = True  # 确保 CUDA 被正确初始化并可用

# 返回当前时间
def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

print(GetNowTime())
print('Begin run!!!')
# 记录开始时间 since
since = time.time()
# 调用args.py中定义的参数，并用args来表示
args = args.loadArgums()
print("Params:")
# 嵌入方法/数据集/特征数量/是否使用广义零样本学习/随机数种子
print('SemEmbed:{:s}, Dataset: {:s}, SynNum={:d}, GZSL:{:s}, ManualSeed:{:d}'.format(
        args.SemEmbed, args.DATASET, args.SynNum, str(args.GZSL), args.ManualSeed))
print(args.SemFile)     # 用于指定语义嵌入文件的路径或名称
# 用于强制刷新标准输出缓冲区，以便及时将输出内容显示出来
sys.stdout.flush()
# default=0 "0" 表示使用第一个 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# set random seed
# 设置随机数 生成器的种子为manualseed，保证结果可以复现
random.seed(args.ManualSeed)
# 可以确保使用 PyTorch 的随机操作（如参数初始化、数据加载等）时，生成的随机数也是可复现的
torch.manual_seed(args.ManualSeed)
# 确保若使用 CUDA 加速时，生成的随机数同样是可复现的
if args.Cuda:
    torch.cuda.manual_seed_all(args.ManualSeed)
# 启用 CuDNN 的自动调优功能，动态选择最优的卷积算法
cudnn.benchmark = True


if torch.cuda.is_available() and not args.Cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# load data     加载数据
data = util.DATA_LOADER(args)
print("Training samples: ", data.ntrain)  # number of training samples

# initialize generator and discriminator
# 创建一个 MLP_G 类的对象 netG，即生成器模型
netG = model.MLP_G(args)
if args.NetG_Path != '':
    netG.load_state_dict(torch.load(args.NetG_Path))  # load the trained model: model.load_state_dict(torch.load(PATH))
# 创建一个 MLP_CRITIC 类的对象 netD，即判别器模型
netD = model.MLP_CRITIC(args)
if args.NetD_Path != '':
    netD.load_state_dict(torch.load(args.NetD_Path))


# classification loss, Equation (4) of the paper
# 创建一个负对数似然损失函数，用于计算分类损失
cls_criterion = nn.NLLLoss()  # cross entropy loss
# 输入特征/语义
input_fea = torch.FloatTensor(args.BatchSize, args.FeaSize)  # (64, 2048)
input_sem = torch.FloatTensor(args.BatchSize, args.SemSize)  # (64, 500)
# 噪声向量
noise = torch.FloatTensor(args.BatchSize, args.NoiseSize)  # (64, 500)
# one = torch.FloatTensor([1])
one = torch.tensor(1, dtype=torch.float)
mone = one * -1

# 创建一个大小为 batch_size 的长整型张量，用于存储输入的标签
input_label = torch.LongTensor(args.BatchSize)

if args.Cuda:
    netD.cuda()
    netG.cuda()
    input_fea = input_fea.cuda()
    noise, input_sem = noise.cuda(), input_sem.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

# 生成样例
def sample():
    batch_feature, batch_label, batch_sem = data.next_batch(args.BatchSize)

    input_fea.copy_(batch_feature)
    input_sem.copy_(batch_sem)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

# 生成特征
def generate_syn_feature(netG, classes, semantic, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, args.FeaSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_sem = torch.FloatTensor(num, args.SemSize)
    syn_noise = torch.FloatTensor(num, args.NoiseSize)
    if args.Cuda:
        syn_sem = syn_sem.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_sem = semantic[iclass]
        syn_sem.copy_(iclass_sem.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_sem, volatile=True))
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label

def generate_syn_feature_with_grad(netG, classes, semantic, num):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor(nclass*num, args.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_sem = torch.FloatTensor(nclass * num, args.SemSize)
    syn_noise = torch.FloatTensor(nclass * num, args.NoiseSize)
    if args.Cuda:
        syn_sem = syn_sem.cuda()
        syn_noise = syn_noise.cuda()
        syn_label = syn_label.cuda()
    syn_noise.normal_(0, 1)
    for i in range(nclass):
        iclass = classes[i]
        iclass_sem = semantic[iclass]
        syn_sem.narrow(0, i * num, num).copy_(iclass_sem.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_feature = netG(Variable(syn_noise), Variable(syn_sem))
    return syn_feature, syn_label.cpu()


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an configional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

# setup optimizer
# 优化器
optimizerD = optim.Adam(netD.parameters(), lr=args.LR, betas=(args.Beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.LR, betas=(args.Beta, 0.999))

# the last item of equation (2)
def calc_gradient_penalty(netD, real_data, fake_data, input_sem):
    # print real_data.size()
    alpha = torch.rand(args.BatchSize, 1)
    alpha = alpha.expand(real_data.size())
    if args.Cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if args.Cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_sem))

    ones = torch.ones(disc_interpolates.size())
    if args.Cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    # args.GP_Weight = 10
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.GP_Weight
    return gradient_penalty



# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier_pretrain.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), args.FeaSize, args.Cuda, 0.001, 0.5, 100, 2*args.BatchSize,
                                     args.Pretrained_Classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False


#### 正式开始迭代
for epoch in range(args.Epoch):
    FP = 0
    mean_lossD = 0
    mean_lossG = 0

    for i in range(0, data.ntrain, args.BatchSize):
        # print("batch...", i)
        # iteratively train the generator and discriminator
        for p in netD.parameters():
            p.requires_grad = True
        # 鉴别器
        # DISCRIMINATOR
        # args.critic_iter = 5, following WGAN-GP
        for iter_d in range(args.Critic_Iter):
            sample()  # sample by batch
            netD.zero_grad()
            # torch.gt: compare the 'input_res[1]' and '0' element by element
            sparse_real = args.FeaSize - input_fea[1].gt(0).sum()  # non sparse number
            input_feav = Variable(input_fea)
            input_semv = Variable(input_sem)

            # loss of real data
            criticD_real = netD(input_feav, input_semv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)
            # loss of generated data
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_semv)   # generate samples
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            # detach(): return a new variable, do not compute gradient for it
            criticD_fake = netD(fake.detach(), input_semv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # loss with Lipschitz constraint
            gradient_penalty = calc_gradient_penalty(netD, input_fea, fake.data, input_sem)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            # Final Loss of Discriminator
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        # GENERATOR
        netG.zero_grad()
        input_semv = Variable(input_sem)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_semv)

        criticG_fake = netD(fake, input_semv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

        errG = G_cost + args.Cls_Weight * c_errG

        errG.backward()
        optimizerG.step()

    print('EP[%d/%d]******************************************************' % (epoch, args.Epoch))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    # train_X: input features (of unseen or seen) for training classifier2 in testing stage
    # train_Y: training labels
    # Generalized zero-shot learning
    # if epoch >= 18:
    if args.GZSL:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.semantic, args.SynNum)
        if args.DATASET == 'AwA2':
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)
            classes = torch.cat((data.seenclasses, data.unseenclasses), 0)
            nclass = classes.size(0)
            cls = classifier_cls.CLASSIFIER(args, train_X, util.map_label(train_Y, classes), data, nclass, args.Cuda,
                                            args.Cls_LR, 0.5, 50, 2 * args.SynNum,
                                            True)
        else:
            train_X = torch.cat((data.train_feature1, syn_feature), 0)
            train_Y = torch.cat((data.train_label1, syn_label), 0)
            classes = torch.cat((data.seenclasses, data.unseenclasses), 0)
            nclass = classes.size(0)
            cls = classifier_cls.CLASSIFIER(args, train_X, util.map_label(train_Y, classes), data, nclass, args.Cuda,
                                                args.Cls_LR, 0.5, 50, 2 * args.BatchSize,
                                                True)

    # Zero-shot learning
    else:
        # synthesize samples of unseen classes, for training classifier2 in testing stage
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.semantic, args.SynNum)
        cls = classifier_cls.CLASSIFIER(args, syn_feature, util.map_label(syn_label, data.unseenclasses), data,
                                         data.unseenclasses.size(0), args.Cuda, args.Cls_LR, 0.5, 50, 10*args.SynNum,
                                         False, args.Ratio, epoch)
    del cls
    cls = None
    # reset G to training mode
    netG.train()
    sys.stdout.flush()

print('End run!!!')
print(GetNowTime())

