# coding=gb2312
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gensim
import matplotlib.pyplot as plt
#����һ���Զ������ݼ��࣬���ڼ��ز�Ԥ�������ݼ���
class QNLIDataset(Dataset):
    def __init__(self, file_path, max_length, embedding_model):
        self.sentences = []
        self.labels = []
        self.max_length = max_length
        self.embedding_dim = len(embedding_model[next(iter(embedding_model.keys()))])

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines[1:]:
                # ��ȡÿһ�����ݣ���ʽΪ��premise sentence  hypothesis sentence  label
                items = line.strip().split('\t')
                premise = items[1].split()
                hypothesis = items[2].split()
                # ��premise��hypothesis�ϲ���һ������
                sentence = premise + hypothesis
                # ���ƾ�����󳤶�Ϊmax_length
                sentence = sentence[:self.max_length]
                # �������е�ÿ����ת��Ϊ������
               
                sentence_vectors = []
                for word in sentence:
                    if word in embedding_model:
                        sentence_vectors.append(embedding_model[word])
                    if len(sentence_vectors) == 0:
                        sentence_vectors.append(np.zeros(self.embedding_dim))
                # �����ӵĳ��Ȳ��뵽max_length����������������Ϊ����
                while len(sentence_vectors) < self.max_length:
                    sentence_vectors.append(np.zeros_like(sentence_vectors[0]))
                self.sentences.append(np.array(sentence_vectors))
                # ����ǩת��Ϊ0��1
                self.labels.append(int(items[3] == 'entailment'))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sentences[idx]).float(), torch.tensor(self.labels[idx])




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
'''
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out
'''
# ����ѵ����������֤����
def train(model, data_loader):
    model.train()
    running_loss = 0.
    correct = 0
    total = 0
    for inputs, labels in data_loader:
        inputs = inputs
        labels = labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, data_loader):
    model.eval()
    running_loss = 0.
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs
            labels = labels
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



# ����GloVe������ģ��

embedding_model = {}
with open('E13\glove.6B.50d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embedding_model[word] = vector
embedding_dim = len(embedding_model[next(iter(embedding_model.keys()))])



# ����ѵ��������֤���ļ�·��
train_file = 'E13/QNLI/train_40.tsv'
valid_file = 'E13/QNLI/dev_40.tsv'

# ���������󳤶Ⱥ����δ�С
max_length = 50
batch_size = 32

# ����ѵ��������֤��
train_dataset = QNLIDataset(train_file, max_length, embedding_model)
valid_dataset = QNLIDataset(valid_file, max_length, embedding_model)

# ����ѵ��������֤�������ݼ�����
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# ������ݼ��е�������
print('Number of training samples:', len(train_dataset))
print('Number of validation samples:', len(valid_dataset))



# ����ģ�͵ĳ�����
input_size = embedding_dim
hidden_size = 64
num_layers = 1
output_size = 2


# ʵ����ģ�͡���ʧ�������Ż���
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)




loss=[]
accuracy=[]

# ��ʼѵ������֤
num_epochs = 20
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss, train_acc = train(model, train_loader)
    valid_loss, valid_acc = evaluate(model, valid_loader)
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f}'.format(epoch+1, num_epochs, train_loss, train_acc, valid_loss, valid_acc))
    end_time = time.time()
    loss.append(valid_loss)
    accuracy.append(valid_acc)
    elapsed_time = end_time - start_time
    print('Elapsed Time:', elapsed_time,'s')

plt.plot([i+1 for i in range(len(accuracy))],accuracy,color="g")
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy') 
plt.show()

plt.plot([i+1 for i in range(len(loss))],loss,color="b")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel('average loss') 
plt.show()


