import torch
import io
import numpy as np
import random
from random import shuffle
from collections import Counter
import argparse


def CBOW_NS(contextWords, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    V, D = inputMatrix.size()
    inputVector = inputMatrix[contextWords].sum(0)
    out = outputMatrix.mm(inputVector.view(D, 1))

    out_for_loss = -out
    out_for_loss[0] = -out_for_loss[0]

    loss = -torch.log(torch.sigmoid(out_for_loss)).sum()

    grad = torch.sigmoid(out)
    grad[0] -= 1

    grad_in = grad.view(1,-1).mm(outputMatrix)
    grad_out = grad.mm(inputVector.view(1,-1))

    return loss, grad_in, grad_out


def word2vec_trainer(input_seq, target_seq, numwords, codes, stats, mode="CBOW", NS=20, dimension=10000, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples", len(input_seq))
    stats = torch.LongTensor(stats)

    # Build a corresponding Huffman Tree
    tree = HuffmanTree(codes.values())

    for _ in range(epoch):
        #Training word2vec using SGD(Batch size : 1)
        for inputs, output in zip(input_seq,target_seq):
            i+=1
            if mode=="CBOW":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    output_path = FindPath(codes[output], tree)
                    activated = torch.tensor(output_path).view(-1, )

                    L, G_in, G_out = CBOW_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    random_idx = torch.randint(0, len(stats), size=(NS,))
                    neg_sample = (stats.view(-1, ))[random_idx]
                    activated = torch.cat([torch.tensor([output]), neg_sample], 0)

                    L, G_in, G_out = CBOW_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out

            elif mode=="SG":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    output_path = FindPath(codes[output], tree)
                    activated = torch.tensor(output_path).view(-1, )

                    L, G_in, G_out = skipgram_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    random_idx = torch.randint(0, len(stats), size=(NS, ))
                    neg_sample = (stats.view(-1,))[random_idx]
                    activated = torch.cat([torch.tensor([output]), neg_sample], 0)

                    L, G_in, G_out = skipgram_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out

                
            else:
                print("Unkwnown mode : "+mode)
                exit()
            losses.append(L.item())
            if i%100000==0:
            	avg_loss=sum(losses)/len(losses)
            	print("Iteration:", i, "Loss : %f" %(avg_loss,))
            	losses=[]

    return W_in, W_out


def main():
    # Part and Full options
    parser = argparse.ArgumentParser(description='cnntext')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    part = args.part


    # Load the corpus
    print("loading...")
    if part=="part":
        text = open('rt-polaritydata/rt-polaritydata/rt-polarity.pos', mode='r', encoding='utf-16').readlines()[0][:100000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('rt-polaritydata/rt-polaritydata/rt-polarity.pos',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    # Preprocessing
    print("preprocessing...")
    corpus = text.split()
    vocab = set(corpus)

    # Creating Vocabulary Dicts
    stats = Counter(corpus)
    words = []
    w2i = {}
    i = 1
    for word in vocab:
        w2i[word] = i
        i += 1
    i2w = {}
    for k, v in w2i.items():
        i2w[v] = k

    #

    #Training section
    emb,_ = word2vec_trainer(input_set, target_set, len(w2i), codedict, freqtable, mode=mode, NS=ns, dimension=64, epoch=1, learning_rate=0.01)

    #Creating embedding dictionary
    emb_dict = {}
    for i in range(len(w2i)):
        emb_dict[i2w[i]] = emb[i]
    #Starting Analogical task
    Analogical_Reasoning_Task(emb_dict)

main()