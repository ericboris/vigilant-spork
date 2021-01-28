#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.corpus import words
from MLE import MLE
from constants import START

# TODO REMOVE WITH GENERATE_SENT
from nltk.tokenize.treebank import TreebankWordDetokenizer

# TODO REMOVE THIS FUNCTION
def generate_sent(model, num_chars, random_seed=42):
	detokenize = TreebankWordDetokenizer().detokenize
	content = []
	for token in model.generate(num_chars, random_seed=random_seed):
		if token == '<s>':
			continue
		elif token == '</s>':
			break
		content.append(token)
	return detokenize(content)

class MyModel:
	"""
	This is a starter model to get you started. Feel free to modify this file.
	"""

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
		return mle.ngram(n)
	
	def run_pred(self, data, model):
		# your code here
		''' Predict the next letter in the line. '''
		"""
		preds = []
		all_chars = string.ascii_letters
		for inp in data:
			# this model just predicts a random character each time
			top_guesses = [random.choice(all_chars) for _ in range(3)]
			preds.append(''.join(top_guesses))
		return preds
		"""
		n = 3
		preds = []
		for line in data:
			# add n-1 padding symbols
			for _ in range(n - 1):
				line.insert(0, START)
			# the the last n-1 characters of ngram
			prec = tuple(line[-(n-1): ])

			p = float('-inf')
			c = None

			print(f'prec: {prec}')
			# use prec as a key to find the value in ngram model with highest probability
			# basically argmax p for c
			for v in model[prec]:
				print(f'MP: {v} {model[prec][v]}')
				if model[prec][v] > p:
					c = v
					p = model[prec][v]
			preds.append(c)
		return preds

	def save(self, work_dir):
		# your code here
		# this particular model has nothing to save, but for demonstration purposes we will save a blank file
		with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
			f.write('dummy save')

	@classmethod
	def load(cls, work_dir):
		# your code here
		# this particular model has nothing to load, but for demonstration purposes we will load a blank file
		with open(os.path.join(work_dir, 'model.checkpoint')) as f:
			dummy_save = f.read()
		return MyModel()


if __name__ == '__main__':

	m = MyModel()
	data = m.load_training_data()
	mle = m.run_train(data, None)
	lines = [
		'Happ',
		'Happy Ne',
		'Happy New Yea',
		'That’s one small ste',
		'That’s one sm',
		'That’',
		'Th',
		'one giant leap for mankin',
		'one giant leap fo',
		'one giant lea',
		'one giant l',
		'one gia',
		'on']

	test_data = [[t.lower() for t in word] for word in lines]
	preds = m.run_pred(test_data, mle)
	print(f'PRED: {preds}')
	
	"""
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('mode', choices=('train', 'test'), help='what to run')
	parser.add_argument('--work_dir', help='where to save', default='work')
	parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
	parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
	args = parser.parse_args()

	random.seed(0)

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
	"""
