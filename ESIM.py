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

DEVICE = torch.device("cuda")


def tokenizer(text):
    res = [w for w in jieba.cut(text)]
    return res


qlength = 55
BATCHSIZE = 128
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True,
                  fix_length=qlength)  # , stop_words=stop_words)
LABEL = data.Field(sequential=False, use_vocab=False)

# 加载新的csv文本
train, val = data.TabularDataset.splits(
    path=r'E:\2019ATEC\MyCode\JupyterPro\data',
    train='train_balance.csv',
    validation='dev_balance.csv',
    format='csv',
    skip_header=True,
    fields=[('qid', None), ('text1', TEXT), ('text2', TEXT), ('label', LABEL)]
)

vectors = vocab.Vectors(name=r'D:\FromIAO\server\server\server\word2vec\baike_26g_news_13g_novel_229g.txt', cache=r'D:\FromIAO\server\torch_torchtext_变长lstm实例\torch_torchtext_变长lstm实例\vector_cache')
print(vectors)
TEXT.build_vocab(train, vectors=vectors, min_freq=3)

# 同时对训练集和验证集进行迭代器的构建
train_iter, val_iter = BucketIterator.splits(
        (train, val), # 构建数据集所需的数据集
        batch_sizes=(BATCHSIZE, 1000),
        device='cuda', # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.text1)+len(x.text2), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=True,
        repeat=False, # we pass repeat=False because we want to wrap this Iterator layer.
)

train_acc_data = Iterator(train, batch_size=1000, device='cuda',repeat=False,sort=False,sort_within_batch=False)
weight_matrix = TEXT.vocab.vectors

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


'''定义model3'''


class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()

        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 128)
        # 若使用预训练的词向量，需在此处指定预训练的权重
        self.word_embeddings.weight.data.copy_(weight_matrix)
        self.word_embeddings.weight = nn.Parameter(self.word_embeddings.weight, requires_grad=False)
        self.bilstm = nn.LSTM(128, 200, batch_first=True, bidirectional=True)
        self.bilstm_2 = nn.LSTM(1600, 200, batch_first=True, bidirectional=True)

        self.fc800_400 = nn.Linear(800, 400)
        self.fc400_200 = nn.Linear(400, 200)
        self.fc200_100 = nn.Linear(200, 100)
        self.fc100_50 = nn.Linear(100, 50)
        self.fc50_2 = nn.Linear(50, 2)

        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout()

    def forward_encoder(self, sentence):
        lengths = []
        for row in range(sentence.size(0)):
            x = sentence[row, :]
            x = x[x != 1]
            aa = len(x) if len(x) != 0 else 1
            lengths.append(aa)

        embedding = self.word_embeddings(sentence)
        embedding_packed = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        lstm_out_packed, (h_out, c_out) = self.bilstm(embedding_packed)
        lstm_out_padded_tuple = nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)
        lstm_out_padded_tensor = lstm_out_padded_tuple[0]  # b*sq*(2*200)
        return lstm_out_padded_tensor, lengths

    def soft_align_attention(self, x1, x2):  # 输入lstm_out_padded_tensor1，lstm_out_padded_tensor2
        attention1_2 = torch.bmm(x1, torch.transpose(x2, 1, 2))  # attention1_2  b*sq1*sq2
        attention2_1 = torch.bmm(x2, torch.transpose(x1, 1, 2))  # attention2_1  b*sq2*sq1
        softmax_attention1_2 = F.softmax(attention1_2, dim=2)
        softmax_attention2_1 = F.softmax(attention2_1, dim=2)

        x1_align = torch.bmm(softmax_attention1_2, x2)  # x1_align  b*sq1*400
        x2_align = torch.bmm(softmax_attention2_1, x1)  # x2_align  b*sq2*400
        return x1_align, x2_align

    def fix_encoder(self, x1, x2, x1_align,
                    x2_align):  # 输入lstm_out_padded_tensor1，lstm_out_padded_tensor2, x1_align, x2_align
        x1_mul = x1 * x1_align
        x1_sub = x1 - x1_align
        x2_mul = x2 * x2_align
        x2_sub = x2 - x2_align
        x1_combined = torch.cat([x1, x1_align, x1_sub, x1_mul], dim=2)  # b*sq1*1600
        x2_combined = torch.cat([x2, x2_align, x2_sub, x2_mul], dim=2)  # b*sq2*1600
        return x1_combined, x2_combined

    def forward(self, sentence1, sentence2):
        x1, lengths1 = self.forward_encoder(sentence1)
        x2, lengths2 = self.forward_encoder(sentence2)
        x1_align, x2_align = self.soft_align_attention(x1, x2)
        x1_combined, x2_combined = self.fix_encoder(x1, x2, x1_align, x2_align)

        x1_combined_packed = nn.utils.rnn.pack_padded_sequence(x1_combined, lengths1, batch_first=True,
                                                               enforce_sorted=False)
        lstm_out_packed1, (h_out1, c_out1) = self.bilstm_2(x1_combined_packed)
        lstm_out_padded_1_tuple = nn.utils.rnn.pad_packed_sequence(lstm_out_packed1, batch_first=True)
        lstm_out_padded_tensor1 = lstm_out_padded_1_tuple[0]  # b*sq1*(2*200)

        x2_combined_packed = nn.utils.rnn.pack_padded_sequence(x2_combined, lengths2, batch_first=True,
                                                               enforce_sorted=False)
        lstm_out_packed2, (h_out2, c_out2) = self.bilstm_2(x2_combined_packed)
        lstm_out_padded_2_tuple = nn.utils.rnn.pad_packed_sequence(lstm_out_packed2, batch_first=True)
        lstm_out_padded_tensor2 = lstm_out_padded_2_tuple[0]  # b*sq2*(2*200)

        p1 = nn.MaxPool2d((lstm_out_padded_tensor1.size(1), 1), stride=1)(
            lstm_out_padded_tensor1)  # 在lstm_out_padded_tensor最后2个维度上做pooling。pooling视野范围高lstm_out_padded_tensor.size(1)，宽1；步长高1，宽1
        p1.squeeze_()
        p2 = nn.MaxPool2d((lstm_out_padded_tensor2.size(1), 1), stride=1)(lstm_out_padded_tensor2)  # b*1*(2*200)
        p2.squeeze_()  # b*(2*200)

        p_cat = torch.cat([p1, p2], dim=1)  # b*800

        out = F.relu(self.fc800_400(p_cat))
        #       out = self.dropout5(out)
        out = F.relu(self.fc400_200(out))
        #         out = self.dropout5(out)
        out = F.relu(self.fc200_100(out))
        #         out = self.dropout5(out)
        out = F.relu(self.fc100_50(out))
        #         out = self.dropout5(out)
        out = self.fc50_2(out)

        out = F.softmax(out, dim=1)
        return out


model3 = Model3().to(DEVICE)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model3.parameters()), lr=0.0001)
# optimizer = optim.Adam([{'params':filter(lambda p: p.requires_grad, encoder.parameters())},{'params':interaction.parameters()}])
loss_funtion = FocalLoss(2)
# weight = torch.FloatTensor([1.0,4.0]).to(DEVICE)

model3.eval()
train_correct = 0
train_loss = 0
with torch.no_grad():
    torch.cuda.empty_cache()  # 清除gpu缓存
    for i, batchgroup in enumerate(train_acc_data):
        output = model3(batchgroup.text1, batchgroup.text2)
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        #         train_loss += F.nll_loss(output, batchgroup.label, reduction='sum').item() # 将一批的损失相加
        train_correct += pred.eq(batchgroup.label.view_as(pred)).sum().item()
    print('train_acc:', train_correct / len(train))  # ,'\t','tarin_loss:',train_loss/len(train))
val_correct = 0
val_loss = 0
with torch.no_grad():
    torch.cuda.empty_cache()  # 清除gpu缓存
    for i, batchgroup in enumerate(val_iter):
        output = model3(batchgroup.text1, batchgroup.text2)
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        #         val_loss += F.nll_loss(output, batchgroup.label, reduction='sum').item() # 将一批的损失相加
        val_correct += pred.eq(batchgroup.label.view_as(pred)).sum().item()
    print('val_acc:', val_correct / len(val))  # ,'\t','val_loss:',val_loss/len(val))
    print('\n')

for epoch in range(20):
    model3.train()
    for i, batchgroup in enumerate(train_iter):
        torch.cuda.empty_cache()  # 清除gpu缓存
        predicted = model3(batchgroup.text1, batchgroup.text2)

        optimizer.zero_grad()
        #         loss = loss_funtion(predicted, batchgroup.label,weight=weight,reduction='sum')
        loss = loss_funtion(predicted, batchgroup.label)

        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(epoch, i, loss)

    model3.eval()
    train_correct = 0
    train_loss = 0
    with torch.no_grad():
        torch.cuda.empty_cache()  # 清除gpu缓存
        for i, batchgroup in enumerate(train_acc_data):
            output = model3(batchgroup.text1, batchgroup.text2)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            #             train_loss += F.nll_loss(output, batchgroup.label, reduction='sum').item() # 将一批的损失相加
            train_correct += pred.eq(batchgroup.label.view_as(pred)).sum().item()
        print('train_acc:', train_correct / len(train))  # ,'\t','tarin_loss:',train_loss/len(train))
    val_correct = 0
    val_loss = 0
    with torch.no_grad():
        torch.cuda.empty_cache()  # 清除gpu缓存
        for i, batchgroup in enumerate(val_iter):
            output = model3(batchgroup.text1, batchgroup.text2)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            #             val_loss += F.nll_loss(output, batchgroup.label, reduction='sum').item() # 将一批的损失相加
            val_correct += pred.eq(batchgroup.label.view_as(pred)).sum().item()
        print('val_acc:', val_correct / len(val))  # ,'\t','val_loss:',val_loss/len(val))
        print('\n')