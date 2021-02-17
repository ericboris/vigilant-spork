# -*- coding: utf-8 -*-
"""Project 2 RNN.ipynb
Built following this tutorial:
https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb
"""

import os
from os.path import join
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import pickle
import time, math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_vocabulary(data):
	vocabulary = '' 
	for char in data:
		if char.isalpha() and char not in vocabulary:
			vocabulary += char
	return vocabulary

def random_chunk(data, chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return data[start_index:end_index]

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

	def train(self, data):
		self.data = data	
		start = time.time()
		all_losses = []
		loss_avg = 0

		def inner(inp, target):
			hidden = self.init_hidden()
			self.zero_grad()
			loss = 0

			for c in range(chunk_len):
				output, hidden = self(inp[c], hidden)
				target = torch.reshape(target, (len(target), 1))
				loss += criterion(output, target[c])

			loss.backward()
			decoder_optimizer.step()
			return loss.data.item() / chunk_len

		for epoch in range(1, n_epochs + 1):
			chunk = random_chunk(data, chunk_len)

			inp, target = random_training_set(chunk)
			loss = inner(inp, target)     
			loss_avg += loss

			if epoch % print_every == 0:
				print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
				print('INPUT:\n{}'.format(chunk)+'\nPREDICTION:\n{}'.format(evaluate("Wh"))+'\n')

			if epoch % plot_every == 0:
				all_losses.append(loss_avg / plot_every)
				loss_avg = 0
		

# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        if string[c] in all_characters:
            tensor[c] = all_characters.index(string[c])
        else:
            tensor[c] = UNK_INDEX
    return Variable(tensor)

def random_training_set(chunk):
	inp = char_tensor(chunk[:-1])
	target = char_tensor(chunk[1:])
	return inp, target

# Returns the top 3 next characters for a given string
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
    top_4 = torch.multinomial(output_dist, 4)

    # this should go through and skip the UNK character if it is a top 3 character
    top_3 = []
    for i in range(4):
        if top_4[i] != UNK_INDEX:
            top_3.append(all_characters[top_4[i]])

    predicted_char_1 = top_3[0]
    predicted_char_2 = top_3[1]
    predicted_char_3 = top_3[2]
    predicted = predicted_char_1 + predicted_char_2 + predicted_char_3
    return predicted

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def write(file_name, obj):
	''' Write the object to a file with the given file name. '''
	with open(file_name, 'wb') as f:
		pickle.dump(obj, f)

def read(file_name):
	''' Return the read object. '''
	with open(file_name, 'rb') as f:
		return pickle.load(f)

def write_pred(preds, fname):
	with open(fname, 'wt') as f:
		for p in preds:
			f.write('{}\n'.format(p))

def load_test_data(fname):
	# your code here
	data = []
	with open(fname) as f:
		for line in f:
			inp = line[:-1]  # the last character is a newline
			data.append(inp)
	return data

if __name__ == '__main__':
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('mode', choices=('train', 'test'), help='what to run')
	parser.add_argument('--work_dir', help='where to save', default='work')
	parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
	parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
	args = parser.parse_args()

	if args.mode == 'train':	
		if not os.path.isdir(args.work_dir):
			print(f'Making working directory {args.work_dir}')
			os.makedirs(args.work_dir)
		
		if os.path.isfile(path=join(args.work_dir, 'model.checkpoint')):
			print('Trained model already exists')
			# decoder = read('../work/trained_model')
		else:
			print('Instantiating model')

			n_epochs = 2000
			decoder = None
			print_every = 20
			plot_every = 10
			hidden_size = 100
			n_layers = 2
			lr = 0.005
			chunk_len = 100

			print('Loading training data')
			
			data = open('data/cleaned_data/train.txt').read()
			file_len = len(data)
	
			all_characters = get_vocabulary(data)

			n_characters = len(all_characters) + 1
			UNK_INDEX = len(all_characters)

			print('Training')
			decoder = RNN(n_characters, hidden_size, n_characters, n_layers)

			decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
			criterion = nn.CrossEntropyLoss()
			decoder.train(data)

			print('Saving model')
			write(file_name=join(args.work_dir, 'model.checkpoint'), obj=decoder)
	elif args.mode == 'test':
		print('Loading model')
		decoder = read(join(args.work_dir, 'model.checkpoint'))
		
		print(f'Loading test data from {args.test_data}')
		test_data = load_test_data(args.test_data)
		all_characters = get_vocabulary(decoder.data)
		UNK_INDEX = len(all_characters)

		print('Making predictions')
		results = []
		for line in test_data:
			results.append(evaluate(line))

		print(f'Writing predictions to {args.test_output}')
		assert len(results) == len(test_data), f'Expected {len(test_data)} but got {len(results)}'
		#print(f'ARGS OUT={args.test_output}')
		write_pred(results, args.test_output)
	else:
		raise NotImplementedError(f'Unknown mode {args.mode}')
