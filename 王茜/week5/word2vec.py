import fasttext
import re
import jieba
import numpy as np
import os

def preprocess_text(text):
    """中文文本预处理流程"""
    # 去除特殊字符
    text = re.sub(r'[^\p{Han}a-zA-Z0-9]', ' ', text)
    # 分词处理
    words = jieba.lcut(text)
    # 过滤停用词（示例列表，需扩展）
    stopwords = set(['的', '了', '是', '在', '和'])
    return ' '.join([w for w in words if w not in stopwords])


def print_analogy(model, a, b, c, k=3):
    """可视化类比关系"""
    print(f"类比推理：{a} → {b} 相当于 {c} → ?")
    results = model.get_analogies(a, b, c)
    for score, word in results[:k]:
        print(f"{word}: {score:.3f}")

# # 数据读取
# with open('New_1.txt', 'r', encoding='utf-8') as f:
#     raw_texts = f.readlines()
#     with open('New_2.txt', 'w', encoding='utf-8') as f1:
#         for text in raw_texts:
#             processed = preprocess_text(text)
#             f1.write(processed + '\n')
def train_model():
    # ====================
    # 2. 模型训练与调优
    # ====================
    # 推荐参数配置
    params = {
        'model': 'skipgram',  # 对罕见词效果更好
        'epoch': 50,  # 适当降低防止过拟合
        'lr': 0.1,  # 学习率调整
        'dim': 300,  # 高维空间适合复杂语义
        'minCount': 2,  # 过滤低频噪声词
        'wordNgrams': 2,  # 捕捉二元短语特征
        'thread': 4  # 多核加速
    }

    model = fasttext.train_unsupervised(
        input='New_1.txt',
        **params
    )


    # 获取词汇
    words = model.get_words()
    print('词汇共有', len(words))
    labels = model.get_labels()
    # 获取词向量的最近邻
    word = '顾里'
    nearest_words = model.get_nearest_neighbors(word)
    print(word)
    print(nearest_words)

    # 词汇间类比
    word_list = []
    print_analogy(model, "顾里", "唐宛如", "南湘")
    model.save_model('word2vec.bin')


def tensorboard():
    import torch
    import torch.nn as nn
    from torch.utils.tensorboard import SummaryWriter
    # 加载预训练模型
    wv_file = 'word2vec.bin'
    word_vectors = fasttext.load_model(wv_file)
    # 获取词汇大小和词向量维度
    word_size = len(word_vectors.words)
    embedding_dim = word_vectors.get_dimension()

    # 创建一个矩阵， 每一行都是一个词的词向量
    embedding_matrix = np.zeros((word_size, embedding_dim))
    for i, word in enumerate(word_vectors.words):
        embedding_vector = word_vectors[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 使用预训练的fasttext词向量
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)

    writer = SummaryWriter()
    meta = []
    while len(meta) < 100:
        i = len(meta)
        if i >= word_size:
            break
        meta = meta + [word_vectors.words[i]]
    meta = meta[:100]
    writer.add_embedding(embedding.weight[:100], metadata=meta)
    writer.close()

if __name__ == '__main__':
    train_model()
    tensorboard()
