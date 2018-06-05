import re
import sys
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from keras.preprocessing.text import text_to_word_sequence

cuda = torch.cuda.is_available()


def load_any(filename):
    with open(filename, 'rb') as handle:
        anything = pickle.load(handle)
    return anything

def trim_word(word):
    trimmed = ''
    i = 0
    while i < len(word):
        trimmed += word[i]
        if i < len(word) - 2 and word[i] == word[i + 1] == word[i + 2]:
            j = i + 1
            while j < len(word) and word[j] == word[i]:
                j += 1
            i = j
            continue
        i += 1
    return trimmed

def tokenize(sen):
    ori_filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'  # not really used
    words = text_to_word_sequence(sen, filters='"#&()*+-./;<=>@[\\]^_{|}~', lower=True, split=' ')
    words = [trim_word(word) for word in words]
    words = ['<NUM>' if re.match(r'^[0-9]+[a,p,A,P]?[m,M]?$', word) else word for word in words]
    
    tmp = []
    i = 0
    while i < len(words):
        if i < len(words) - 2 and ( words[i+1] == '\'' or words[i+1] == '`' ) and words[i+2] == 't':
            tmp.append(words[i] + '\'' + words[i+2])
            i += 3
            continue
        tmp.append(words[i])
        i += 1
    words = tmp
        
    return words


PAD = 0
UNK = 1
class Lang:
    def __init__(self):
        self.word2count = {'<PAD>':1000, '<UNK>':1000}
        self.word2index = {'<PAD>':0, '<UNK>':1}
        self.index2word = {0:'<PAD>', 1:'<UNK>'}
        self.n_useful = 2
        self.n_words = 2
        
    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def trimmed(self, threshold):
        tmp = copy.deepcopy(self)
        for word in list(tmp.word2count):
            if tmp.word2count[word] <= threshold:
                tmp.word2count.pop(word)
                tmp.n_words -= 1
        return tmp
    
    def indexing(self):
        index = self.n_useful
        for word in list(self.word2count)[self.n_useful:]:
            self.word2index[word] = index
            self.index2word[index] = word
            index += 1


def numpy2sen(lang, array):
    sen = ''
    for token in array:
        token = int(token)
        sen = sen + lang.index2word[token] + ' '
    return sen

def sen2numpy(lang, sen, max_length):
    sen_numpy = []
    for word in sen:
        if word in lang.word2count:
            sen_numpy.append(lang.word2index[word])
        else:
            sen_numpy.append(UNK)

    if len(sen_numpy) < max_length:
        sen_numpy += [PAD] * (max_length - len(sen_numpy))
    else:
        sen_numpy = sen_numpy[:max_length]
    return np.array(sen_numpy, dtype=np.int32)

def unpack_pair_get_numpy(pair, maxlen):
    datalist = []
    datanumpy = np.empty((len(pair), maxlen), dtype=np.int32)
    anses = np.empty((len(pair)), dtype=np.int32)
    for i, (data, ans) in enumerate(pair):
        tmp = []
        for char in data:
            if char in lang_trim.word2count:
                tmp.append(char)
            else:
                tmp.append('<UNK>')
        datalist.append(tmp)
        datanumpy[i] = sen2numpy(lang_trim, tmp, maxlen)
        anses[i] = int(ans)
    
    return datalist, datanumpy, anses


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, embedded_matrix, hidden_size, n_layer, bidir):
        super(EncoderRNN, self).__init__()
        self.dropout = nn.Dropout(0.5)
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)    
        self.embedding.weight.data.copy_(torch.Tensor(embedded_matrix))
        
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=n_layer, bidirectional=bidir)
        
        n_direct = 2 if bidir else 1
        self.linear1 = nn.Linear(hidden_size * n_layer * n_direct, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 2)
        
    # TRAIN WHOLE SEQUENCE
    def forward(self, batch_tokens, batch_lengths):
        # batch_tokens: (seqlen, batch)
        # batch_lengths: (batch,)
        batch_size = len(batch_lengths)
        
        embedded = self.embedding(batch_tokens)
        embedded = self.dropout(embedded)
        
        packed = pack_padded_sequence(embedded, batch_lengths)
        encoded, hidden = self.gru(packed)
        unpacked, lengths = pad_packed_sequence(encoded)
        
        hidden = hidden.transpose(0,1).contiguous()
        hidden = hidden.view(batch_size, -1)
        
        reducted = self.linear1(hidden)
        reducted = self.dropout(F.relu(reducted))
        reducted = self.linear2(reducted)
        reducted = self.dropout(F.relu(reducted))
        reducted = self.linear3(reducted)
        probs = F.log_softmax(reducted, dim=1)
        
        return probs

def propogate_batch(mode, batch_tokens, batch_lengths, encoder, labels=None, enopt=None):
    # modes: train, eval, infer
    # batch_tokens: (batch, seqlen)
    # labels: (batch,)
    batch_size = len(batch_tokens)
    
    if mode == 'train':
        encoder.train()
        enopt.zero_grad()
    else:
        encoder.eval()
    
    batch_tokens = torch.LongTensor(batch_tokens.transpose())
    if cuda:
        batch_tokens = batch_tokens.cuda()
    batch_tokens = Variable(batch_tokens) if mode == 'train' else Variable(batch_tokens, volatile=True)
    
    probs = encoder(batch_tokens, batch_lengths)
    probs_np = probs.data.cpu().numpy()
    predict_np = (probs_np[:, 1] > probs_np[:, 0]).astype(np.int32)
    
    if mode == 'infer':
        return probs_np, predict_np
    
    labels = torch.LongTensor(labels)
    if cuda:
        labels = labels.cuda()
    labels = Variable(labels) if mode == 'train' else Variable(labels, volatile=True)
    
    criterion = nn.NLLLoss(size_average=False)
    loss = criterion(probs, labels)
    loss_np = loss.data.cpu().numpy()[0]
    
    label_np = labels.data.cpu().numpy()
    correct_np = (predict_np == label_np).sum()
    
    if mode == 'eval':
        return loss_np, correct_np
    
    loss.backward()
    enopt.step()
    
    return loss_np, correct_np

def infer_iter(testing, encoder, batches):
    test_list, test_np, test_id = testing
    predictions = []
    probs = []
    for start in range(0, len(test_list), batches):
        end = min(start + batches, len(test_list))
        X = test_np[start:end]
        Xlens = np.array([len(x) for x in test_list[start:end]], dtype=np.int32)
        probs_np, predict_np = propogate_batch('infer', X, Xlens, encoder)
        predictions.append(predict_np)
        probs.append(probs_np)
    
    return np.concatenate(predictions), np.vstack(probs)


print('Defining and loading model')
encoder1 = EncoderRNN(
    
    vocab_size = 8859, 
    embedding_size = 100,
    embedded_matrix = np.empty((8859, 100)),
    hidden_size = 128, 
    n_layer = 3,
    bidir = True
    
)

checkpoint = torch.load('model1')
encoder1.load_state_dict(checkpoint['encoder'])

encoder2 = EncoderRNN(
    
    vocab_size = 8884, 
    embedding_size = 100,
    embedded_matrix = np.empty((8884, 100)),
    hidden_size = 128, 
    n_layer = 3,
    bidir = True
    
)

checkpoint = torch.load('model2')
encoder2.load_state_dict(checkpoint['encoder'])

if cuda:
    encoder1 = encoder1.cuda()
    encoder2 = encoder2.cuda()


print('Loading and Preprocessing testing data')
test_path = sys.argv[1]
test_pair = []
with open(test_path, 'r', encoding='utf-8') as handle:
    lines = handle.read().split('\n')[:-1]
    for index, line in enumerate(lines[1:]):
        line = line[line.find(',') + 1:]
        data = tokenize(line)
        test_pair.append((data, index))

print('Predicting')
lang_trim = load_any('lang_trim1')
test_pair_sorted = sorted(test_pair, key=lambda x: len(x[0]), reverse=True)
testing = unpack_pair_get_numpy(test_pair_sorted, 40)
_, probs = infer_iter(testing, encoder1, 100)
concat_id = np.concatenate((probs, testing[2].reshape(-1,1)), axis=1)
sort_probs1 = np.stack(sorted(concat_id, key=lambda x: x[2]))[:, :2]

lang_trim = load_any('lang_trim2')
test_pair_sorted = sorted(test_pair, key=lambda x: len(x[0]), reverse=True)
testing = unpack_pair_get_numpy(test_pair_sorted, 40)
_, probs = infer_iter(testing, encoder2, 100)
concat_id = np.concatenate((probs, testing[2].reshape(-1,1)), axis=1)
sort_probs2 = np.stack(sorted(concat_id, key=lambda x: x[2]))[:, :2]

final_probs = (sort_probs1 + sort_probs2)
predictions = (final_probs[:, 1] > final_probs[:, 0]).astype(np.int32)
pd.DataFrame(data={'id':np.arange(len(predictions)), 'label':predictions}).to_csv(sys.argv[2], columns=['id', 'label'], index=False)
print('Done')