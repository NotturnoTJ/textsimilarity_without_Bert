import jieba
import pandas as pd
from torchtext import data, vocab
from torchtext.data import BucketIterator, Iterator
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn

stop_words = []
with open(r'E:\2019ATEC\MyCode\JupyterPro\data\stop_words.txt', 'r', encoding='UTF-8') as f:
    for l in f.readlines():
        stop_words.append(l.strip())
print(stop_words)

DEVICE = torch.device("cuda")


def tokenizer(text):
    res = [w for w in jieba.cut(text)]
    return res


qlength = 30
BATCHSIZE = 100
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True, fix_length=qlength,
                  stop_words=stop_words)
LABEL = data.Field(sequential=False, use_vocab=False)

train, val = data.TabularDataset.splits(
    path=r'E:\2019ATEC\MyCode\JupyterPro\data',
    train='train_balance.csv',
    validation='dev_balance.csv',
    format='csv',
    skip_header=True,
    fields=[('qid', None), ('text1', TEXT), ('text2', TEXT), ('label', LABEL)]
)

"""
构建词汇表，加载词向量，TEXT是Field类实例，train中与TEXT绑定的列构成词汇表映射和词向量映射
"""
vectors = vocab.Vectors(name=r'D:\FromIAO\server\server\server\word2vec\baike_26g_news_13g_novel_229g.txt', cache=r'D:\FromIAO\server\torch_torchtext_变长lstm实例\torch_torchtext_变长lstm实例\vector_cache')
print(vectors)
TEXT.build_vocab(train, vectors=vectors, min_freq=3)

# 同时对训练集和验证集进行迭代器的构建
train_iter, val_iter = BucketIterator.splits(
        (train, val), # 构建数据集所需的数据集
        batch_sizes=(BATCHSIZE, 5000),
        device='cuda', # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.text1)+len(x.text2), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=True,
        repeat=False, # we pass repeat=False because we want to wrap this Iterator layer.
)

train_acc_data = Iterator(train, batch_size=5000, device='cuda',repeat=False,sort=False,sort_within_batch=False)
weight_matrix = TEXT.vocab.vectors

'''定义model2'''


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 128)
        # 若使用预训练的词向量，需在此处指定预训练的权重
        self.word_embeddings.weight.data.copy_(weight_matrix)
        self.word_embeddings.weight = nn.Parameter(self.word_embeddings.weight, requires_grad=False)
        self.lstm = nn.LSTM(128, 200, batch_first=True)
        self.gru = nn.GRU(328, 150, batch_first=True)
        self.conv1 = nn.Conv2d(1, 100, kernel_size=(1, 328), stride=1)  # params: 输入通道数，输出通道数（filter个数），核视野（H,W）,步长
        self.conv2 = nn.Conv2d(1, 100, kernel_size=(2, 328), stride=1)
        self.conv3 = nn.Conv2d(1, 100, kernel_size=(3, 328), stride=1)

        self.bn = nn.BatchNorm1d(3800)

        self.dense_100 = nn.Linear(500, 100)
        self.dense_200 = nn.Linear(qlength * qlength, 200)

        self.dense1 = nn.Linear(3800, 2000)
        self.dense2 = nn.Linear(2000, 1000)
        self.dense3 = nn.Linear(1000, 400)
        self.dense4 = nn.Linear(400, 50)
        self.dense5 = nn.Linear(50, 2)

        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout()

    def get_encoder_matrix1_new_embedding(self, sentence):
        lengths = []
        for row in range(sentence.size(0)):
            x = sentence[row, :]
            x = x[x != 1]
            lengths.append(len(x))

        embedding = self.word_embeddings(sentence)
        embedding_packed = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        lstm_out_packed, (h_out, c_out) = self.lstm(embedding_packed)
        lstm_out_padded_tuple = nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)
        lstm_out_padded_tensor = lstm_out_padded_tuple[0]
        add_dim_num = qlength - lstm_out_padded_tuple[1].max()
        zeros2pad = torch.zeros(lstm_out_padded_tensor.size(0), add_dim_num, 200).to(DEVICE)
        matrix1 = torch.cat([lstm_out_padded_tensor, zeros2pad], 1)
        new_embedding = torch.cat([embedding, matrix1], 2)
        lstm_time_step_maxpooling = nn.MaxPool2d((lstm_out_padded_tensor.size(1), 1), stride=1)(
            lstm_out_padded_tensor)  # 在lstm_out_padded_tensor最后2个维度上做pooling。pooling视野范围高lstm_out_padded_tensor.size(1)，宽1；步长高1，宽1
        lstm_time_step_maxpooling.squeeze_()  # batch*200
        return matrix1, new_embedding, lengths, lstm_time_step_maxpooling  # matrix1:batch*qlength*200   new_embedding:batch*qlength*328

    def get_encoder_matrix2(self, new_embedding, lengths):
        embedding_packed = nn.utils.rnn.pack_padded_sequence(new_embedding, lengths, batch_first=True,
                                                             enforce_sorted=False)
        gru_out_packed, hn = self.gru(embedding_packed)
        gru_out_padded_tuple = nn.utils.rnn.pad_packed_sequence(gru_out_packed, batch_first=True)
        gru_out_padded_tensor = gru_out_padded_tuple[0]
        add_dim_num = qlength - gru_out_padded_tuple[1].max()
        zeros2pad = torch.zeros(gru_out_padded_tensor.size(0), add_dim_num, 150).to(DEVICE)
        matrix2 = torch.cat([gru_out_padded_tensor, zeros2pad], 1)
        return matrix2  # batch*qlength*150

    def get_cnn_encoder_vector(self, new_embedding):
        new_embedding.unsqueeze_(1)  # new_embedding:batch*1*qlength*328  nn.Cov2d()的输入参数是4-D：batch，输入通道数，H，W
        conv1_out = self.conv1(new_embedding)  # conv1_out:batch*100*H*1 , H=qlength-kernel(H)+1
        conv1_out.squeeze_()  # conv1_out:batch*100*W  , W=H
        conv1_encoder_vector = nn.MaxPool2d((1, conv1_out.size(2)), stride=(1, 1))(
            conv1_out)  # conv1_encoder_vector：batch*100*1
        conv1_encoder_vector.squeeze_()  # conv1_encoder_vector：batch*100

        conv2_out = self.conv2(new_embedding)
        conv2_out.squeeze_()
        conv2_encoder_vector = nn.MaxPool2d((1, conv2_out.size(2)), stride=(1, 1))(conv2_out)
        conv2_encoder_vector.squeeze_()  # conv2_encoder_vector：batch*100

        conv3_out = self.conv3(new_embedding)
        conv3_out.squeeze_()
        conv3_encoder_vector = nn.MaxPool2d((1, conv3_out.size(2)), stride=(1, 1))(conv3_out)
        conv3_encoder_vector.squeeze_()  # conv3_encoder_vector：batch*100

        cnn_encoder_vector = torch.cat([conv1_encoder_vector, conv2_encoder_vector, conv3_encoder_vector],
                                       1)  # cnn_encoder_vector: batch*300
        return cnn_encoder_vector

    def get_encoder_layer_out(self, sentence):
        (matrix1, new_embedding, lengths, lstm_time_step_maxpooling) = self.get_encoder_matrix1_new_embedding(sentence)
        matrix2 = self.get_encoder_matrix2(new_embedding, lengths)
        cnn_encoder_vector = self.get_cnn_encoder_vector(new_embedding)
        vector = torch.cat([cnn_encoder_vector, lstm_time_step_maxpooling], 1)  # batch*500
        #         vector = self.dropout5(vector)
        #         vector = self.dense_100(vector)
        #         vector = F.relu(vector) # batch*100
        encoder_layer_out = [vector, matrix1, matrix2]
        return encoder_layer_out

    def interaction_layer(self, encoder_layer_out1, encoder_layer_out2):
        vec1 = encoder_layer_out1[0] - encoder_layer_out2[0]  # batch*100
        vec2 = encoder_layer_out1[0] * encoder_layer_out2[0]  # batch*100
        matrix1_1 = F.normalize(encoder_layer_out1[1], dim=2)  # matrix1_1每行元素单位化后的第一句话的第一个编码矩阵 b*ql*200
        matrix1_2 = F.normalize(encoder_layer_out1[2], dim=2)  # matrix1_2每行元素单位化后的第一句话的第二个编码矩阵 b*ql*150
        matrix2_1 = F.normalize(encoder_layer_out2[1], dim=2)
        matrix2_2 = F.normalize(encoder_layer_out2[2], dim=2)

        matrix1_mul = torch.bmm(matrix1_1, torch.transpose(matrix2_1, 1,
                                                           2))  # matrix2_1的1和2维度转置，再matrix1_1与之做矩阵乘法。matrix1_mul：b*ql*ql
        matrix1_mul_flatten = torch.flatten(matrix1_mul,
                                            start_dim=1)  # 将matrix1_mul从第1维开始展平，展平后可保留batch的形状（第0维的形状） matrix1_mul_flatten：b*（ql*ql）
        #         matrix1_mul_flatten = self.dropout5(matrix1_mul_flatten)
        #         matrix1_mul_flatten = self.dense_200(matrix1_mul_flatten)
        #         matrix1_mul_flatten = F.relu(matrix1_mul_flatten) # b*200

        matrix2_mul = torch.bmm(matrix1_2, torch.transpose(matrix2_2, 1, 2))
        matrix2_mul_flatten = torch.flatten(matrix2_mul, start_dim=1)
        #         matrix2_mul_flatten = self.dropout5(matrix2_mul_flatten)
        #         matrix2_mul_flatten = self.dense_200(matrix2_mul_flatten)
        #         matrix2_mul_flatten = F.relu(matrix2_mul_flatten) # b*200

        interaction_layer_out = torch.cat(
            [vec1, vec2, encoder_layer_out1[0], encoder_layer_out2[0], matrix1_mul_flatten, matrix2_mul_flatten], 1)
        return interaction_layer_out  # b*3800

    def forward(self, sentence1, sentence2):
        encoder_layer_out1 = self.get_encoder_layer_out(sentence1)
        encoder_layer_out2 = self.get_encoder_layer_out(sentence2)

        interaction_layer_out = self.interaction_layer(encoder_layer_out1, encoder_layer_out2)

        #         interaction_layer_out = self.bn(interaction_layer_out)
        out = F.relu(self.dense1(interaction_layer_out))
        out = self.dropout5(out)

        out = F.relu(self.dense2(out))
        out = self.dropout5(out)

        out = F.relu(self.dense3(out))
        out = self.dropout5(out)
        out = F.relu(self.dense4(out))
        out = self.dropout5(out)

        out = self.dense5(out)
        out = F.log_softmax(out, dim=1)

        return out


model2 = Model2().to(DEVICE)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=0.0001)
# optimizer = optim.Adam([{'params':filter(lambda p: p.requires_grad, encoder.parameters())},{'params':interaction.parameters()}])
loss_funtion = F.nll_loss
# weight = torch.FloatTensor([1.0,4.0]).to(DEVICE)

model2.eval()
train_correct = 0
train_loss = 0
with torch.no_grad():
    torch.cuda.empty_cache()  # 清除gpu缓存
    for i, batchgroup in enumerate(train_acc_data):
        output = model2(batchgroup.text1, batchgroup.text2)
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        train_loss += F.nll_loss(output, batchgroup.label, reduction='sum').item()  # 将一批的损失相加
        train_correct += pred.eq(batchgroup.label.view_as(pred)).sum().item()
    print('train_acc:', train_correct / len(train), '\t', 'tarin_loss:', train_loss / len(train))
val_correct = 0
val_loss = 0
with torch.no_grad():
    torch.cuda.empty_cache()  # 清除gpu缓存
    for i, batchgroup in enumerate(val_iter):
        output = model2(batchgroup.text1, batchgroup.text2)
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        val_loss += F.nll_loss(output, batchgroup.label, reduction='sum').item()  # 将一批的损失相加
        val_correct += pred.eq(batchgroup.label.view_as(pred)).sum().item()
    print('val_acc:', val_correct / len(val), '\t', 'val_loss:', val_loss / len(val))
    print('\n')

for epoch in range(60):
    model2.train()
    for i, batchgroup in enumerate(train_iter):
        torch.cuda.empty_cache()  # 清除gpu缓存
        predicted = model2(batchgroup.text1, batchgroup.text2)

        optimizer.zero_grad()
        #         loss = loss_funtion(predicted, batchgroup.label,weight=weight,reduction='sum')
        loss = loss_funtion(predicted, batchgroup.label)

        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(epoch, i, loss)

    model2.eval()
    train_correct = 0
    train_loss = 0
    with torch.no_grad():
        torch.cuda.empty_cache()  # 清除gpu缓存
        for i, batchgroup in enumerate(train_acc_data):
            output = model2(batchgroup.text1, batchgroup.text2)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            train_loss += F.nll_loss(output, batchgroup.label, reduction='sum').item()  # 将一批的损失相加
            train_correct += pred.eq(batchgroup.label.view_as(pred)).sum().item()
        print('train_acc:', train_correct / len(train), '\t', 'tarin_loss:', train_loss / len(train))
    val_correct = 0
    val_loss = 0
    with torch.no_grad():
        torch.cuda.empty_cache()  # 清除gpu缓存
        for i, batchgroup in enumerate(val_iter):
            output = model2(batchgroup.text1, batchgroup.text2)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            val_loss += F.nll_loss(output, batchgroup.label, reduction='sum').item()  # 将一批的损失相加
            val_correct += pred.eq(batchgroup.label.view_as(pred)).sum().item()
        print('val_acc:', val_correct / len(val), '\t', 'val_loss:', val_loss / len(val))
        print('\n')