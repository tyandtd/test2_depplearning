import logging
import os
import re
import sys
from itertools import chain
from gensim.models import KeyedVectors
import gensim
import pandas as pd
import torch
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split

import pickle

embed_size = 300
max_len = 512

# Read data from files ps该文件用的上实验的训练文件
 #此处改为文件的绝对路径ps未将文件的相对路径录入
train = pd.read_csv("F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\testData.tsv", header=0,
                   delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\unlabeledTrainData.tsv", header=0,
                              delimiter="\t", quoting=3)

#将评价转化为单词列表
def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #移除网页标识符号
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #移除非字母符号
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #变成小写，并且分离出单词
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #移除禁用词，但是不知道为什么，在这个地方并没有使用，可能是因为要使用词向量的缘故，词所在的位置，本身也具有较为重要的意义
    #而不是仅仅单词具有意义
    # 4. Optionally remove stop words (false by default)
    # if remove_stopwords:
    #     stops = set(stopwords.words("english"))
    #     words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return (words)

#编码例子将形成好的词，对应到新的此表上
def encode_samples(tokenized_samples):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:#python真的是一门很神奇的语言，这个地方的定义很奇幻
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features

#找到词在新形成词向量表中的位置,这个地方的max_len为固定的512，把所有的评价长度缩减到512
def pad_samples(features, maxlen=max_len, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while len(padded_feature) < maxlen:
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))
    #记录相关进展的日志
    clean_train_reviews, train_labels = [], []
    #初始化数据
    for i, review in enumerate(train["review"]):
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=False))
        #这个地方已经将停用词注释，也可以加快速度
        train_labels.append(train["sentiment"][i])
    #将train文件中的各个量变为单词
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=False))
    #将test文件中的各个量变为单词
    vocab = set(chain(*clean_train_reviews)) | set(chain(*clean_test_reviews))
    #
    vocab_size = len(vocab)
    #分离出两类文件，train和用于优化的val，这也是为什么cnn明明val没有用却还是用了
    train_reviews, val_reviews, train_labels, val_labels = train_test_split(clean_train_reviews, train_labels,
                                                                            test_size=0.2, random_state=0)

    #此处改为文件的绝对路径ps未将文件的相对路径录入
    wvmodel =  KeyedVectors.load_word2vec_format("F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\gensim_glove_vectors.txt", binary=False)
    #这个地方，是将要用的单词全部标记出来，形成新的一个表，这个表有一个顺序和单词
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    #使用单词进行索引
    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<unk>'
    #使用所在位置进行索引
    
    #在这个地方定义了两者，word_to_idx，和idx_to_word，
    train_features = torch.tensor(pad_samples(encode_samples(train_reviews)))
    val_features = torch.tensor(pad_samples(encode_samples(val_reviews)))
    test_features = torch.tensor(pad_samples(encode_samples(clean_test_reviews)))
    #在这个地方，有的都是先编码再使用pad，三个文件都是一样
    #在这个地方输出了每一个句子在新的表中对应的一个固定在512个句子的组
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)

    weight = torch.zeros(vocab_size + 1, embed_size)
    #这个循环最后得到了，应该是一个有着对应新表顺序的词向量
    for i in range(len(wvmodel.index_to_key)):
        try:
            index = word_to_idx[wvmodel.index_to_key[i]]
            #在这个地方得到了单词
            print(i)
        except:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(
            idx_to_word[word_to_idx[wvmodel.index_to_key[i]]]))
     #在这个文件的最后，成功输出了一个文件imdb——glove.pickle3即词向量文件   
     #
    pickle_file = os.path.join('F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\pickle', 'imdb_glove.pickle3')
    
    pickle.dump(
        [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word, vocab],
        open(pickle_file, 'wb'))
    #最后输出
    print('data dumped!')
