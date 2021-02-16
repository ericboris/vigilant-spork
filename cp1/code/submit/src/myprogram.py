#!/usr/bin/env python
import os
import pickle
import random
import string 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from nltk.corpus import words
from MLE import MLE
from constants import START, STOP, UNK
from copy import deepcopy

class MyModel:
	@classmethod
	def load_training_data(cls):
		''' Return the individual-letter tokenized nltk words corpus. '''
		tokens = [[t.lower() for t in word] for word in words.words()]
		return tokens

	@classmethod
	def load_test_data(cls, fname):
		# your code here
		data = []
		with open(fname) as f:
			for line in f:
				data.append(line.rstrip())
		return data

	def run_train(self, data, work_dir):
		''' Train a LI trigram model. '''
		n = 3
		mle = MLE(data)
		self.mle = mle.ngram(n)

	
	def run_pred(self, data):
		''' Use the given model to make predictions of the next letter in each line 
			of the given data. '''	

		# Let preds be the list of predicted next letters to return.
		preds = []
		#TODO: get the proper file path
		#Figure out how to configure the file to have
		with open('../../data/cleaned_data/dev_input.txt') as input:
			for line in input.readlines():
				top_3 = evaluate(line.rstrip())
				# print(f'{line} :: {top_3}')
				preds.append(top_3)
		input.close()
		return preds

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

	@classmethod
	def write_pred(cls, preds, fname):
		with open(fname, 'wt') as f:
			for p in preds:
				f.write('{}\n'.format(p))

	def save(self, work_dir):
		''' Save the trained model to a file. '''
		# Pickle the entire model.
		with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
			pickle.dump(self, f)

	@classmethod
	def load(cls, work_dir):
		''' Return a trained model from the given directory. '''
		# Load the pickled model.
		with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
			model = pickle.load(f)
		return model


if __name__ == '__main__':
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('mode', choices=('train', 'test'), help='what to run')
	parser.add_argument('--work_dir', help='where to save', default='work')
	parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
	parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
	args = parser.parse_args()

	if args.mode == 'train':
		if not os.path.isdir(args.work_dir):
			print('Making working directory {}'.format(args.work_dir))
			os.makedirs(args.work_dir)
		print('Instatiating model')
		model = MyModel()
		print('Loading training data')
		train_data = MyModel.load_training_data()
		print('Training')
		model.run_train(train_data, args.work_dir)
		print('Saving model')
		model.save(args.work_dir)
	elif args.mode == 'test':
		print('Loading model')
		model = MyModel.load(args.work_dir)
		print('Loading test data from {}'.format(args.test_data))
		test_data = MyModel.load_test_data(args.test_data)
		print('Making predictions')
		pred = model.run_pred(test_data)
		print('Writing predictions to {}'.format(args.test_output))
		assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
		model.write_pred(pred, args.test_output)
	else:
		raise NotImplementedError('Unknown mode {}'.format(args.mode))
