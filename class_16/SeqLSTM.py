
# coding: utf-8

# In[1]:

import numpy as np
from matplotlib import pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import pandas as pd
import seaborn as sns
import datetime
import spacy
import sklearn

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Load Spacy word embeddings
word_embeddings = spacy.load('en', vectors='glove.6B.300d.txt')


# In[38]:

# Create a function to get vector format data for a sequence
def sequence_to_data(seq, max_len=None):
    seq = unicode(seq)
    data = [word_embeddings(ix).vector for ix in seq.split()]
    
    if max_len is None:
        max_len = len(data)
    
    data_mat = np.zeros((1, max_len, 300))
    
    for ix in range(min(len(data), max_len)):
        data_mat[:, ix, :] = data[ix]
    
    return data_mat

def seq_data_matrix(seq_data, max_len=None):
    n_seq = len(seq_data)
    data = np.concatenate([sequence_to_data(ix, max_len) for ix in seq_data], axis=0)
    
    return data

df = pd.read_csv('./data/sentiment/dataset.csv', sep='|', index_col=0)

# In[8]:

df['len'] = df['text'].str.split().apply(lambda x: len(x))

bucket_sizes = [[0, 10], [10, 15], [15, 20], [20, 25], [25, 45]]

def assign_bucket(x):
    for bucket in bucket_sizes:
        if x > bucket[0] and x <= bucket[1]:
            return bucket_sizes.index(bucket)
    return len(bucket_sizes)-1


df['bucket'] = df.len.apply(assign_bucket)
df = df.sort(columns=['bucket'])



def make_batch(data, batch_size=10, gpu=True):
    for bx in range(len(bucket_sizes)):
        bucket_data = df[(df.bucket == bx)].reset_index(drop=True)
        # print bx, bucket_sizes[bx][1], bucket_data.shape
        
        start = 0
        stop = start + batch_size
        
        while start < bucket_data.shape[0]:
            seq_length = bucket_sizes[bx][1]
            section = bucket_data[start:stop]
            X_data = seq_data_matrix(section.text, max_len=seq_length)
            y_data = section.score
            
            if gpu:
                yield Variable(torch.FloatTensor(X_data).cuda(), requires_grad=True), Variable(torch.LongTensor(y_data)).cuda()
            else:
                yield Variable(torch.FloatTensor(X_data), requires_grad=True), Variable(torch.LongTensor(y_data))
            
            start = stop
            stop = start + batch_size



class SeqModel(nn.Module):
    def __init__(self, in_shape=None, out_shape=None, hidden_shape=None):
        super(SeqModel, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_shape = hidden_shape
        self.n_layers = 1
        
        self.rnn = nn.LSTM(
            input_size=self.in_shape,
            hidden_size=self.hidden_shape,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.lin = nn.Linear(self.hidden_shape, 64)
        self.dropout = nn.Dropout(0.42)
        self.out = nn.Linear(64, self.out_shape)
    
    def forward(self, x, h):
        r_out, h_state = self.rnn(x, h)
        last_out = r_out[:, -1, :]
        y = F.tanh(self.lin(last_out))
        y = self.dropout(y)
        y = F.softmax(self.out(y))
        return y
    
    def predict(self, x):
        h_state = self.init_hidden(1, gpu=False)
        
        x = sequence_to_data(x)
        pred = self.forward(torch.FloatTensor(x), h_state)
        
        return pred
    
    def get_embedding(self, x):
        h_state = self.init_hidden(1, gpu=False)
        
        x = sequence_to_data(x)
        r_out, h = self.rnn(torch.FloatTensor(x), h_state)
        last_out = r_out[:, -1, :]
        
        return last_out.data.numpy()
            
    def init_hidden(self, batch_size, gpu=True):
        if gpu:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_shape).cuda()),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_shape)).cuda())
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_shape)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_shape)))


model = SeqModel(in_shape=300, hidden_shape=256, out_shape=2)
print model
model.cuda()


# Load the model
# model.load_state_dict(torch.load('/home/shubham/all_projects/CB/Summer_2018/data/checkpoints/seq_lstm/model_256h_epoch_240.ckpt'))


optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()


# In[35]:

# Set to train mode
# model.cuda()
model.train()

for epoch in range(500):
    total_loss = 0
    N = 0
    for step, (b_x, b_y) in enumerate(make_batch(df, batch_size=200)):
        # print step, b_x.shape, b_y.shape
        bsize = b_x.size(0)
        lol = b_x
        h_state = model.init_hidden(bsize, gpu=True)

        pred = model(b_x, h_state)
        loss = criterion(pred, b_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        N += 1.0
        if step%50 == 0:
            print 'Loss: {} at Epoch: {} | Step: {}'.format(loss, epoch, step)
        
    print "Overall Average Loss: {} at Epoch: {}".format(total_loss / float(N), epoch)
    
    # Save model checkpoints
    if epoch % 10 == 0:
        torch.save(model.state_dict(), "/home/shubham/all_projects/CB/Summer_2018/data/checkpoints/seq_lstm_bucket/model_256h_epoch_{}.ckpt".format(epoch))

