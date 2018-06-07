import re
import sys
import copy
import time
import pickle

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from keras.preprocessing.text import text_to_word_sequence


def save_any(anything, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(anything, handle)
        
def load_any(filename):
    with open(filename, 'rb') as handle:
        anything = pickle.load(handle)
    return anything

def save_checkpoint(items, names, filename):
    state = {}
    for item, name in zip(items, names):
        state[name] = item.state_dict()
    
    torch.save(state, filename)


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
    words = text_to_word_sequence(sen, filters='@', lower=True, split=' ')
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


print('Loading and preprocessing training data')
labeled_path = sys.argv[1]
labeled_pair = []
with open(labeled_path, 'r', encoding='utf-8') as handle:
    lines = handle.read().split('\n')[:-1]
    for line in lines:
        ans = int(line.split('+++$+++')[0])
        data = tokenize(line.split('+++$+++')[1])
        labeled_pair.append((data, ans))

np.random.seed(5)  # or 5
np.random.shuffle(labeled_pair)

split = 20000
valid_pair = labeled_pair[:split]
train_pair = labeled_pair[split:]


print('Building dictionary')
lang = Lang()
for data, ans in train_pair:
    lang.addSentence(data)

lang_trim = lang.trimmed(10)
lang_trim.n_words

# finalize the dictionary with indexing
lang_trim.indexing()
save_any(lang_trim, 'dictionary_tmp')


print('Parsing training data according to dictionary')
valid_pair = sorted(valid_pair, key=lambda x: len(x[0]), reverse=True)
train_pair = sorted(train_pair, key=lambda x: len(x[0]), reverse=True)

valid_list, valid_np, valid_ans = unpack_pair_get_numpy(valid_pair, 40)
train_list, train_np, train_ans = unpack_pair_get_numpy(train_pair, 40)



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
    
    batch_tokens = torch.LongTensor(batch_tokens.transpose()).cuda()
    batch_tokens = Variable(batch_tokens) if mode == 'train' else Variable(batch_tokens, volatile=True)
    
    probs = encoder(batch_tokens, batch_lengths)
    probs_np = probs.data.cpu().numpy()
    predict_np = (probs_np[:, 1] > probs_np[:, 0]).astype(np.int32)
    
    if mode == 'infer':
        return probs_np, predict_np
    
    labels = torch.LongTensor(labels).cuda()
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

def train_iter(training, validating, encoder, enopt, batches, iterations, print_freq, hist=[], folder='tmp'):
    train_list, train_np, train_ans = training
    loss_sum = 0
    correct_sum = 0
    n_sample = 0
    for iteration in range(iterations):
        start = np.random.randint(len(train_list) - batches)
        X = train_np[start : start + batches]
        Y = train_ans[start : start + batches]
        Xlens = np.array([len(x) for x in train_list[start : start + batches]], dtype=np.int32)
        
        loss_np, correct_np = propogate_batch('train', X, Xlens, encoder, labels=Y, enopt=enopt)
        loss_sum += loss_np
        correct_sum += correct_np
        n_sample += batches
        
        if iteration % print_freq == 0:
            train_loss_avg = loss_sum / n_sample
            train_accuracy = correct_sum / n_sample
            loss_sum = 0
            correct_sum = 0
            n_sample = 0
            
            valid_loss_avg, valid_accuracy = eval_iter(validating, encoder, batches)
            record = np.array([train_loss_avg, valid_loss_avg, train_accuracy, valid_accuracy])
            hist.append(record)
            
            print('ITERATION ' + str(iteration))
            print(record)
            
    return hist

def eval_iter(validating, encoder, batches):
    valid_list, valid_np, valid_ans = validating
    loss_sum = 0
    correct_sum = 0
    for start in range(0, len(valid_list), batches):
        end = min(start + batches, len(valid_list))
        X = valid_np[start:end]
        Y = valid_ans[start:end]
        Xlens = np.array([len(x) for x in valid_list[start:end]], dtype=np.int32)
        
        loss_np, correct_np = propogate_batch('eval', X, Xlens, encoder, labels=Y)
        loss_sum += loss_np
        correct_sum += correct_np
    loss_avg = loss_sum / len(valid_list)
    accuracy = correct_sum / len(valid_list)
    return loss_avg, accuracy

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


print('Defining model')
embedded_matrix = np.load('embedding.npy')
encoder = EncoderRNN(
    
    vocab_size = lang_trim.n_words, 
    embedding_size = 100,
    embedded_matrix = embedded_matrix,
    hidden_size = 128, 
    n_layer = 3,
    bidir = True
    
).cuda()

enopt = optim.Adam(encoder.parameters())


print('Start training')
training = (train_list, train_np, train_ans)
validating = (valid_list, valid_np, valid_ans)

history = []
folder = './'
train_iter(training, validating, encoder, enopt, batches=100, iterations=5000, print_freq=100, hist=history, folder=folder)

save_checkpoint([encoder, enopt], ['encoder', 'enopt'], 'model_tmp')
print('Trained dictionary saved as: dictionary_tmp')
print('Trained model saved as: model_tmp')
