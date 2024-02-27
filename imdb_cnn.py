import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm


from sklearn.metrics import accuracy_score


#test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\testData.tsv", header=0, delimiter="\t", quoting=3)
#定义全局变量
num_epochs = 10
embed_size = 300
num_filter = 128
filter_size = 3
bidirectional = True
batch_size = 64
labels = 2
lr = 0.8
device = torch.device('cuda:0')#调用gpu进行计算，这个方式的集成度很高，对于不同的gpu支持，可以换位不同的硬件
use_gpu = True


class SentimentNet(nn.Module):
    #这个地方定义的参数多一个use_gpu，而不是像书上一样，将数据全部转移到gou上再计算
    def __init__(self, embed_size, num_filter, filter_size, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.use_gpu = use_gpu
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False#表明不自己训练embeding，即词向量
        self.conv1d = nn.Conv1d(embed_size, num_filter, filter_size, padding=1)#一个卷积
        self.activate = F.relu#使用relu作为激活函数
        self.decoder = nn.Linear(num_filter, labels)#然后采用线性输出


    def forward(self, inputs):
        embeddings = self.embedding(inputs)

        convolution = self.activate(self.conv1d(embeddings.permute([0, 2, 1])))#在这个地方就形成了一个结果
        #这个地方有一个直接调用原函数，而没有将这个函数直接定义出来，但是有部分函数不行
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])#这个地方是将参数缩小，采用取最大值的方式，其size为2
        #在这个地方，输出了一个pooling，依旧为一个tenser格式
        outputs = self.decoder(pooling.squeeze(dim=2))#最后采用线性输出
        # print(outputs)
        return outputs

#总结，cnn就是采用了一层卷积结构的
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    #这个地方，是给文件加入一些可视化的设置，让处理过程不完全摸黑
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    #将其改为绝对地址，后将文件名由测试，变为实际执行
    #pickle_file = os.path.join('pickle', 'imdb_demo_glove.pickle3')
    #读取词向量，vec的
    pickle_file = os.path.join('F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\pickle', 'imdb_glove.pickle3')
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
            vocab] = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')
    #具体加载参数，来构建出将用的神经网络
    net = SentimentNet(embed_size=embed_size, num_filter=num_filter, filter_size=filter_size,
                       weight=weight, labels=labels, use_gpu=use_gpu)
    net.to(device)#将参数加载到gpu中，但是为什么没有将文件加载到gpu的过程呢？
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)#使用sgd优化器
    #定义训练集，和验证集，和测试集
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )
    #定义成可处理的格式，将train打乱
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    #开始进行处理
    for epoch in range(num_epochs):
        #初始化空间
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        #打开过程展示
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, label in train_iter:
                n += 1
                #初始化导数
                net.zero_grad()
                feature = Variable(feature.cuda())
                label = Variable(label.cuda())
                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()
                train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                         dim=1), label.cpu())
                train_loss += loss
                #这个地方完成对模型的训练
                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)
                                  })
                pbar.update(1)
            #在这个地方，有一个对于模型的验证处理
            with torch.no_grad():
                for val_feature, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.cuda()
                    val_label = val_label.cuda()
                    val_score = net(val_feature)
                    val_loss = loss_function(val_score, val_label)
                    val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                    val_losses += val_loss
            end = time.time()
            runtime = end - start
            pbar.set_postfix({'epoch': '%d' % (epoch),
                              'train loss': '%.4f' % (train_loss.data / n),
                              'train acc': '%.2f' % (train_acc / n),
                              'val loss': '%.4f' % (val_losses.data / m),
                              'val acc': '%.2f' % (val_acc / m),
                              'time': '%.2f' % (runtime)})
            #下面注释掉的是一些调试代码，有一个疑问，在这个地方并没有使用验证集
            # tqdm.write('{epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f}' %
            #       (epoch, train_loss.data / n, train_acc / n, val_losses.data / m, val_acc / m, runtime))
    #进行预测，先初始化了一个位置
    test_pred = []
    with torch.no_grad():
        #加上显示的过程
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.cuda()
                test_score = net(test_feature)
                # test_pred.extent
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())
                #加上显示的过程
                pbar.update(1)
    #重组输出格式
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\result\\cnn.csv", index=False, quoting=3)
    logging.info('result saved!')
#完成一次，在result中生成了cnn.csv文件
