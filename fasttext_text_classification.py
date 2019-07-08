import torch
import numpy as np
import random
import re
import argparse
from collections import Counter

##################################### Options  ######################################
# Statistics Option & Subsampling & Removal of stop words(Not defined as a function #
#####################################################################################
def statistics(corpus, incorrect_dict):

    # Frequency counting
    keys = list(incorrect_dict.keys())
    test_freq_dict = {keys[0]: 0, keys[1]: 0, keys[2]: 0, keys[3]: 0}
    for class_num, n_words in corpus:
        test_freq_dict[keys[int(class_num)-1]] += 1

    # Prediction Accuracy for each class
    for class_name, values in incorrect_dict.items():
        test_freq_dict[class_name] = round((1 - (values/test_freq_dict[class_name]))*100, 2)

    return test_freq_dict

def subsampling(word_seq):

    words_count = Counter(word_seq)
    total_count = len(word_seq)
    words_freq = {word: count/total_count for word, count in words_count.items()}

    prob = {}

    for word in words_freq:
        prob[word] = 1 - np.sqrt(0.00001/words_freq[word])

    subsampled = [word for word in word_seq if random.random() < (1 - prob[word])]
    return subsampled

############################### Ngram Hashing  ######################################
# Ngram and Hashing functions
#####################################################################################
def NGram_Hashing(str):

    hval = 0x811c9dc5
    fnv_32_prime = 0x01000193
    max_bucket = 2100000
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * fnv_32_prime) % max_bucket

    return hval

def NGram_Indicing(words, ngram):

    grams = zip(*[words[i:] for i in range(ngram)])

    return set(grams)


def get_hash_idx(set_ngrams, ngram):
    idx_set =[]

    for word in set_ngrams:
        combined_word = ''
        for i in range(ngram-1):
            combined_word += word[i] + " "
        combined_word += word[ngram-1]
        idx_set.append(NGram_Hashing(combined_word))

    # Output: index of ngrams. size= torch.tensor((N, )) when N = num of a ngram set.
    return torch.tensor(idx_set).view(-1,)

###############################  Neural Model  ######################################
# Forward&Backward, fasttext trainer and tester of Neural Model
#####################################################################################

def ForBackward(inputs, answer, inputMatrix, outputMatrix):
# inputs : Index Set of N-grams (type: torch.tensor(P,))
# inputMatrix : Weight matrix of input = Hash table (type:torch.tesnor(K,D))
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(4,D))

    answer = int(answer) - 1
    _, D = inputMatrix.size()
    inputVector = (inputMatrix[inputs].sum(0))/(inputs.size())[0]
    out = outputMatrix.mm(inputVector.view(D,1))

    expout = torch.exp(out)
    softmax = expout / expout.sum()

    loss = -torch.log(softmax[answer])

    grad = softmax
    grad[answer] -= 1

    grad_in = grad.view(1,-1).mm(outputMatrix)
    grad_out = grad.mm(inputVector.view(1,-1))

    return loss, grad_in, grad_out


def fasttext_trainer(corpus, numclass, ngram=2, dimension=10000, learning_rate=0.025, epoch=3):

    max_bucket = 2100000
    W_in = torch.randn(max_bucket, dimension) / (dimension**0.5)
    W_out = torch.randn(numclass, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples", len(corpus))

    for _ in range(epoch):

        for class_num, n_words in corpus:
            i+=1

            inputs = get_hash_idx(n_words, ngram)

            L, G_in, G_out = ForBackward(inputs, class_num, W_in, W_out)
            W_in[inputs] -= learning_rate * G_in.squeeze()
            W_out -= learning_rate * G_out

            losses.append(L.item())
            if i%1000==0:
                avg_loss=sum(losses)/len(losses)
                print("Iteration:", i, "Loss : %f" %(avg_loss,))
                losses=[]

    return W_in, W_out

def fasttext_tester(corpus, emb, decoder, statistics_option, ngram=2, dimension=10000):

    # Bring the class names
    class_name = open('ag_news_csv/classes.txt', mode='r').read().split()

    incorrect_dict = {class_name[0]: 0, class_name[1]: 0, class_name[2]: 0, class_name[3]: 0}
    corrects = 0
    data = []

    for class_num, n_words in corpus:

        D = dimension
        inputs = get_hash_idx(n_words, ngram)
        inputVector = (emb[inputs].sum(0)) / (inputs.size())[0]
        out = decoder.mm(inputVector.view(D, 1))
        prediction = out.max(0)[1] #Predicted class number
        class_num = int(class_num) - 1

        if prediction == class_num:
            data.append("[O] Prediction:%s Answer:%s\n" %(class_name[prediction], class_name[class_num]))
            corrects += 1

        else:
            data.append("[X] Prediction:%s Answer:%s\n" % (class_name[prediction], class_name[class_num]))
            if statistics_option:
                incorrect_dict[class_name[class_num]] += 1

    return data, corrects, incorrect_dict

#########################  Data processing & Pipeline  ##############################
# Preprocessing of train & test data, Main pipeline and Processing of an output file
#####################################################################################

def main():
    parser = argparse.ArgumentParser(description='Fasttext')
    parser.add_argument('ngram', metavar='n-grams', type=int,
                        help='n-gram number')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    parser.add_argument('remove_stop', metavar='stopwords', type=str,
                        help='Option for removal of stop words')
    parser.add_argument('statistics_option', metavar='statistics_option', type=str,
                        help='Option for statistics')

    args = parser.parse_args()
    part = args.part
    ngram = args.ngram
    remove_stop = args.remove_stop
    statistics_option = args.statistics_option

    # Load and preprocess TRAIN corpus
    print("Data Reading...")
    if part == "part":
        text = open('ag_news_csv/train.csv', mode='r').readlines()[:10000]  # Load a part of corpus for debugging
    elif part == "full":
        text = open('ag_news_csv/train.csv', mode='r').readlines()  # Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("Preprocessing...")
    corpus = []
    stops = open('stop_words.txt', mode='r').read()
    stops = stops.split("\n")

    #text = subsampling(text)

    for element in text:
        element = re.sub('["=.#?!/:$&;,]', '', element[1:])
        class_num = element[0]
        temp = element[1:].split()
        if remove_stop:
            temp = [w for w in temp if not w in stops]
        corpus.append((class_num, NGram_Indicing(temp, ngram)))

    # TRAIN section
    emb, decoder = fasttext_trainer(corpus, 4, ngram, dimension=64, epoch=1, learning_rate=0.01)

    # Load and preprocess TEST corpus
    print("Test Data Reading...")
    text = open('ag_news_csv/test.csv', mode='r').readlines()

    print("Preprocessing...")
    test_corpus = []

    #text = subsampling(text)

    for element in text:
        element = re.sub('["=.#?!/:$&;,]', '', element[1:])
        class_num = element[0]
        temp = element[1:].split()
        if remove_stop:
            temp = [w for w in temp if not w in stops]
        test_corpus.append((class_num, NGram_Indicing(temp, ngram)))

    # TEST Section
    data, correct_answers, incorrect_dict = fasttext_tester(test_corpus, emb, decoder, statistics_option, ngram, dimension=64)

    # Processing File output
    print("Creating Result File...")
    f = open("Test_set_result.txt", 'w')
    f.writelines("Correct %d in total %d test articles. [Accuracy: %0.2f%%]\n" %(correct_answers, len(test_corpus), (correct_answers / len(test_corpus))*100))
    if statistics_option:
        temp = statistics(test_corpus, incorrect_dict)
        f.writelines("Accuracy of each class in percentage \n%s \n\n" % temp.items())
    f.writelines(data)
    f.close()

    print("Completed.")

main()
