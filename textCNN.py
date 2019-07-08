import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, optim, nn

import numpy as np
import random
import argparse

import io
import gensim

# [Utils]
def divide(x, train_prop):
    random.shuffle(x)
    x_train = x[:round(train_prop * len(x))]
    x_test = x[round(train_prop * len(x)):]
    return x_train, x_test


def create_set(corpus, vocab, w2i, mode):
    set = []
    for i, (sentence, label) in enumerate(corpus):
        activated = []
        words = sentence.split()
        if mode == 'rand':
            for word in words:
                if word not in vocab:
                    continue
                activated.append(w2i[word])
            set.append((activated, label))
        elif mode == 'pretrained':
            set.append((words, label))
    return set


# [TextCNN neural network]
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_size, mode, option, num_filter=100, window_sizes=(3, 4, 5)):
        super(TextCNN, self).__init__()

        if mode == 'rand':
            # Initialize random embeddings
            self.embedding = nn.Embedding(vocab_size+1, emb_size)
        elif mode == 'pretrained':
            print("loading the pretrained model...")
            self.embedding = gensim.models.KeyedVectors.load_word2vec_format('pretrained_word2vec/GoogleNews-vectors-negative300.bin', binary=True)

        self.convs = nn.ModuleList([nn.Conv1d(1, 100, [window_size, emb_size], padding=(window_size - 1, 0))
                                    for window_size in window_sizes])

        self.fc = nn.Linear(num_filter * len(window_sizes), 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.flag = False

        # Handle options
        if option == 'static':
            self.fc.weight.requires_grad = False
        elif option == 'non-static':
            self.fc.weight.requires_grad = True
        elif option == 'multichannel':
            self.flag = True
            self.fc1 = nn.Linear(num_filter * len(window_sizes), 2)
            self.fc2 = nn.Linear(num_filter * len(window_sizes), 2)
            self.fc1.weight.requires_grad = False
            self.fc2.weight.requires_grad = True

        self.loss_function = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)

    def forward(self, x, mode):  # x = activated index
        #######################################################################################
        # (Matrix Size Info)
        # B = batch size
        # C = channel dimension
        # L = this batch's max sentence length
        # E = embedding dimension
        #######################################################################################

        if mode == 'rand':  # x = activated index
            x = torch.LongTensor(x)
            x = self.embedding(x)
        elif mode == 'pretrained':  # x = words lists
            # Tricky partIf a word is OOV, should be initialized first.
            xs = []
            for batch_element in x:
                xw = []
                for word in batch_element:
                    if word in self.embedding:
                        xw.append(self.embedding[word])
                    else:
                        rand_vec = np.random.normal(0, 0.1, 300)
                        xw.append(rand_vec)
                xs.append(xw)
            x = torch.DoubleTensor(xs)
        else:
            print("Unknown mode. Terminated")
            return mode

        x = torch.unsqueeze(x, 1)  # [B, C, T, E] Add a channel dim.
        temp = []
        for conv in self.convs:
            x2 = self.relu(conv(x))  # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            temp.append(x2)
        x = torch.cat(temp, 2)  # [B, F, window]

        # Drop & FC
        x = self.drop(x)
        flatten = x.view(x.size(0), -1)  # [B, F * window]
        if self.flag:  # Multichannel
            logit1 = self.fc1(flatten)
            logit2 = self.fc2(flatten)
            logits = logit1 + logit2
        else:
            logits = self.fc(flatten)  # [B, class]

        # Regularization
        norm = self.fc.weight.norm()
        if norm > 3:
            rescaled = self.fc.weight * 3 / norm
            self.fc.weight = nn.Parameter(rescaled)

        # Prediction
        probs = F.softmax(logits)  # [B, class]
        classes = torch.max(probs, dim=1)[1]  # [B]

        return probs, classes

# [Trainer and Tester]
def trainer(train_set, num_epochs, num_batch, model, mode):
    print("Start training with %d num of train data" %(len(train_set)))

    total_step = int(len(train_set) / num_batch)
    for epoch in range(num_epochs):
        end = 0
        for step in range(total_step):
            # Batch organization
            start = end
            end = start + num_batch
            wordlists, labels = list(zip(*train_set[start:end]))
            wordlists = list(wordlists)

            # Add padding to resize the data
            max_len = 0
            for wordlist in wordlists:
                temp = len(wordlist)
                if temp > max_len:
                    max_len = temp  # max length for 'this' batch
            for i in range(num_batch):
                if mode == 'rand':
                    wordlists[i] = wordlists[i] + [0] * (max_len - len(wordlists[i]))
                elif mode == 'pretrained':
                    wordlists[i] = wordlists[i] + [' '] * (max_len - len(wordlists[i]))

            if mode == 'rand':
                inputs = torch.LongTensor(wordlists)
            elif mode == 'pretrained':
                inputs = wordlists
            labels = torch.tensor(labels)

            # Forward pass
            probs, classes = model(inputs, mode)

            # Backpropagation
            model.optimizer.zero_grad() ### TODO : search that zero_grad() is suitable for SGD optimizer 내일
            losses = model.loss_function(probs, labels)
            losses.backward()
            model.optimizer.step()

            if (step + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, step + 1, total_step,
                                                                         losses.item()))
    print("Training Done.")
    return model


def tester(test_set, model, mode):
    num_epochs = 1
    num_batch = 10
    test_size = len(test_set)
    print("Start test with %d num of test data" % (test_size))

    total_step = int(test_size / num_batch)
    corrects = 0
    for epoch in range(num_epochs):
        end = 0
        for step in range(total_step):
            # Batch organization
            start = end
            end = start + num_batch
            wordlists, labels = list(zip(*test_set[start:end]))
            wordlists = list(wordlists)

            # Add padding to resize the data
            max_len = 0
            for wordlist in wordlists:
                temp = len(wordlist)
                if temp > max_len:
                    max_len = temp  # max length for 'this' batch
            for i in range(num_batch):
                if mode == 'rand':
                    wordlists[i] = wordlists[i] + [0] * (max_len - len(wordlists[i]))
                elif mode == 'pretrained':
                    wordlists[i] = wordlists[i] + [' '] * (max_len - len(wordlists[i]))

            # Forward pass
            with torch.no_grad():
                model.eval()
                probs, classes = model(wordlists, mode)

            classes = classes.tolist()

            for i in range(num_batch):
                if classes[i] == labels[i]:
                    corrects += 1

    return corrects / test_size

# [Pipeline]
def main():
    parser = argparse.ArgumentParser(description='cnntext')
    parser.add_argument('mode', metavar='mode', type=str)
    parser.add_argument('option', metavar='option', type=str)
    args = parser.parse_args()
    mode = args.mode
    option =args.option
    torch.set_default_tensor_type(torch.DoubleTensor)

    # 1. Load the corpus
    neg_text = open('rt-polaritydata/rt-polaritydata/rt-polarity.neg',mode='r', encoding='utf-16').readlines()
    pos_text = open('rt-polaritydata/rt-polaritydata/rt-polarity.pos',mode='r', encoding='utf-16').readlines()

    corpus = []
    for sentence in neg_text:
        corpus.append((sentence.rstrip(), 0))
    for sentence in pos_text:
        corpus.append((sentence.rstrip(), 1))

    # 2. Create Vocabulary Dicts
    print("preprocessing...")
    words = []
    text = neg_text + pos_text
    for sentence in text:
        words = words + sentence.split()

    vocab = set(words)
    w2i = {}
    w2i[' '] = 0
    i = 1
    for word in vocab:
        w2i[word] = i
        i += 1
    i2w = {}
    for k, v in w2i.items():
        i2w[v] = k

    corpus_size = len(corpus)
    vocab_size = len(vocab)
    words_size = len(words)
    print("Done. total_sentences: %d, total_words: %d, vocab: %d" % (corpus_size, words_size, vocab_size))

    # 3. Create Training set & Test set
    corpus_train, corpus_test = divide(corpus, 0.9)
    train_set = create_set(corpus_train, vocab, w2i, mode)
    test_set = create_set(corpus_test, vocab, w2i, mode)
    print("Created %d size of train_set and %d of test_set." % (len(train_set), len(test_set)))

    # 4. Train model
    num_epochs = 20
    num_batch = 50

    model = TextCNN(vocab_size, 300, mode=mode, option=option)

    trained_model = trainer(train_set, num_epochs, num_batch, model, mode)

    # 5. Test model
    accuracy = tester(test_set, trained_model, mode)
    print("Accuracy: ", accuracy)

main()
