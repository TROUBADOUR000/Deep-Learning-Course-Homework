import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

import math

def weights_init(m):
    classname = m.__class__.__name__  #   obtain the class name
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        # print("inital  linear weight ")


class word_embedding(nn.Module):
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))
    def forward(self,input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.Wi = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.Wh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bi = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.bh = nn.Parameter(torch.Tensor(hidden_size * 4))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        # Initialize hidden and cell states if not provided
        if hx is None:
            hx = (torch.zeros(self.num_layers, input.size(0), self.hidden_size),
                  torch.zeros(self.num_layers, input.size(0), self.hidden_size))

        h, c = hx
        output = []

        # Iterate through each time step
        for i in range(input.size(1)):
            # LSTM computation
            gates = input[:, i, :].mm(self.Wi) + h[-1].mm(self.Wh) + self.bi + self.bh
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            c = (forgetgate * c) + (ingate * cellgate)
            h = outgate * torch.tanh(c)

            output.append(h.unsqueeze(1))

        output = torch.cat(output, dim=1)
        return output, (h, c)


class RNN_model(nn.Module):
    def __init__(self, batch_sz ,vocab_len ,word_embedding,embedding_dim, lstm_hidden_dim):
        super(RNN_model,self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        #########################################
        # here you need to define the "self.rnn_lstm"  the input size is "embedding_dim" and the output size is "lstm_hidden_dim"
        # the lstm should have two layers, and the  input and output tensors are provided as (batch, seq, feature)
        # ???

        self.my_lstm = LSTM(embedding_dim, lstm_hidden_dim, num_layers=1)

        ##########################################
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len )
        self.apply(weights_init) # call the weights initial function.

        self.softmax = nn.LogSoftmax(dim=1) # the activation function.
        # self.tanh = nn.Tanh()
    def forward(self,sentence,is_test = False):
        batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
        # print(batch_input.size()) # print the size of the input
        ################################################
        # here you need to put the "batch_input"  input the self.lstm which is defined before.
        # the hidden output should be named as output, the initial hidden state and cell state set to zero.
        # ???

        output, _ = self.my_lstm(batch_input)

        ################################################
        out = output.contiguous().view(-1,self.lstm_dim)

        out =  F.relu(self.fc(out))

        out = self.softmax(out)

        if is_test:
            prediction = out[ -1, : ].view(1,-1)
            output = prediction
        else:
           output = out
        # print(out)
        return output

