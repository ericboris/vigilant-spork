# -*- coding: utf-8 -*-
"""Project 2 RNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N6U11iuRq4-ihD5jDRT71iuYQzSxIFKs

Built following this tutorial:
https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb

TODO: Handle all unicode chars, currently convereted to ascii
TODO: Handle UNK tokens when outputting the top 3
TODO: save the trained model so it can just be referenced
TODO: note that currently i am omitting the new line from the dictionary, when evaluating new lines are stripped away
 but i dont know how the training data is loaded in and dealt with much i have only been working with evaluate

"""

# Import from drive contents.
#from google.colab import drive
#drive.mount('/content/drive')

# unidecode not installed by default.
# !pip install unidecode

import unidecode
import string
import random
import torch
import torch.nn as nn
from torch.autograd import Variable

# NOTE: UNK is not actually a character, it will just be known as the last index in the characters by
# adding 1 to the overall n_charcters

#TODO: Eric this should theoretically be all known characters by the traiing data correct?
all_characters = ''
with open('../../data/cleaned_data/train.txt') as train:
    for line in train.readlines():
        for i in line:
            if i not in all_characters and i != '\n':
                all_characters += i

# add one for UNK
n_characters = len(all_characters) + 1
UNK_INDEX = len(all_characters)

print(str(all_characters))

file = unidecode.unidecode(open('../../data/cleaned_data/train.txt').read())
file_len = len(file)
print(f'Top 10 Lines (First 323 characters):\n{file[:323]}\n')
print(f'file len (in characters) = {file_len}')

chunk_len = 200

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

#print(random_chunk())

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        if string[c] in all_characters:
            tensor[c] = all_characters.index(string[c])
        else:
            tensor[c] = UNK_INDEX
    return Variable(tensor)

#print(char_tensor('abcDEF'))

def random_training_set():    
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

# TODO: Eric I modified this to take in a string and then predict top 3 and return a string of the top 3 characters
def evaluate(prime_str='A', temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]

    output, hidden = decoder(inp, hidden)

    # Sample from the network as a multinomial distribution
    output_dist = output.data.view(-1).div(temperature).exp()
    top_3 = torch.multinomial(output_dist, 4)
    #TODO: modify to get top 4 and select ther other if there is an UNK as a top 3, for now it is a star

    # Add predicted character to string and use as next input
    # * = UNK for now...
    predicted_char_1 = '*'
    predicted_char_2 = '*'
    predicted_char_3 = '*'
    if top_3[0] != UNK_INDEX:
        predicted_char_1 = all_characters[top_3[0]]
    if top_3[1] != UNK_INDEX:
        predicted_char_2 = all_characters[top_3[1]]
    if top_3[2] != UNK_INDEX:
        predicted_char_3 = all_characters[top_3[2]]
    predicted = predicted_char_1 + predicted_char_2 + predicted_char_3
    return predicted

import time, math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(inp, target):
    # print('p1')
    hidden = decoder.init_hidden()
    # print('p2')
    decoder.zero_grad()
    # print('p3')
    loss = 0
    # print('p4')

    for c in range(chunk_len):
        # print('p5')
        output, hidden = decoder(inp[c], hidden)
        # print('p6')
        # print(f'out shape={output.shape}')
        
        # print(f'tar shape={target.shape}, tar[0]={target[0]}')
        # Added this reshape to fix "tensor has no dimension" error.
        target = torch.reshape(target, (len(target), 1))
        # print(f'tar shape={target.shape}, tar[0]={target[0]}')

        loss += criterion(output, target[c])

    # print('p7')
    loss.backward()
    # print('p8')
    decoder_optimizer.step()
    # print('p9')
    return loss.data.item() / chunk_len
    # Removed the following because error. 
    # return loss.data[0] / chunk_len

n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    #print(f'Epoch: {epoch}')
    inp, target = random_training_set()
    # print(f'inp shape={inp.shape}, inp[0]={inp[0]}')
    # print(f'tar shape={target.shape}, tar[0]={target[0]}')
    loss = train(inp, target)     
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate('Wh'), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

# Commented out IPython magic to ensure Python compatibility.
#import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %matplotlib inline

#plt.figure()
#plt.plot(all_losses)

answers = open('../../data/dev_predict', 'w')
with open('../../data/cleaned_data/dev_input.txt') as input:
    for line in input.readlines():
        print(str(line))
        top_3 = evaluate(line.rstrip())
        print(top_3 + '\n')
        answers.write(top_3 + '\n')
input.close()
answers.close()

