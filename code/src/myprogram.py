#!/usr/bin/env python
import os
import string
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from nltk.corpus import words
from MLE import MLE
from constants import START, UNK

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
				inp = line[:-1]  # the last character is a newline
				data.append(inp)
		return data

	@classmethod
	def write_pred(cls, preds, fname):
		with open(fname, 'wt') as f:
			for p in preds:
				f.write('{}\n'.format(p))

	def run_train(self, data, work_dir):
		''' Train a LI trigram model. '''
		n = 3
		mle = MLE(data)
		self.mle = mle.ngram(n)
	
	def run_pred(self, data):
		''' Use the given model to make predictions of the next letter in each line 
			of the given data. '''	
		n = 3
		# Let preds be the list of predicted next letters to return.
		preds = []

		if self.mle:
			for line in data:
				line = [t for t in line]
				print(f'LINE {line}')
				# Pad the line. 
				for _ in range(n - 1):
					line.insert(0, START)

				# Let hist be the last n-1 characters of ngram.
				hist = tuple(line[-(n-1): ])
				
				# Let yhat be the most likely letter predicted by the model.
				yhat = max(self.mle[hist], key=self.mle[hist].get) if hist in self.mle else None

				preds.append(yhat)
		return preds

	def save(self, work_dir):
		''' Save the trained model to a file. '''
		with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
			pickle.dump(self, f)

	@classmethod
	def load(cls, work_dir):
		''' Return a trained model from the given directory. '''
		# this particular model has nothing to load, but for demonstration purposes we will load a blank file
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
