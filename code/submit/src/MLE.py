from collections import defaultdict, Counter
from functools import partial
from constants import UNK, START, STOP
from copy import deepcopy
import math

def get_data(file_name):
	''' Return the file name as a tokenized list of strings. '''
	data = []
	with open(file_name) as f:
		for line in f:
			data.append(line.split())
	return data 

class MLE:
	def __init__(self, data, unk_threshold=3):
		self.data = data
		
		# Build a vocabulary of word counts for determining out of voculabulary words.
		# Let vocabulary be a mapping of token -> token_count.
		self.vocabulary = Counter([token for line in data for token in line])

		# Let the following represent the minimum number of times that a token appears
		# in the vocabulary without being replaced by UNK.
		self.unk_threshold = unk_threshold
	
		# Let the cache map n -> ngram model, i.e. 1 -> unigram	model, 2 -> bigram model, etc.
		self.cache = {}
	
	def unigram(self, verbose=False):
		''' Train a unigram model on the data. '''
		unigram_model = defaultdict(int)							

		if verbose:
			print('Building 1-gram model\nGetting 1-gram model counts')

		# Temporarily fill the model with token emission counts.
		for line in self.data:
			line = self._unk(line)

			for gram in self._ngrams(line, n=1):
				w1 = gram[-1]
				unigram_model[w1] += 1

		if verbose:
			print('Getting 1-gram model probabilities')

		# Convert the token counts to probabilities.
		unigram_total = float(sum(unigram_model.values()))
		for w1 in unigram_model:
			unigram_model[w1] /= unigram_total
		
		# Cache and return the result.
		self.cache[1] = unigram_model
		return unigram_model

	def bigram(self, verbose=False):
		''' Train a bigram model on the data. '''
		bigram_model = self.ngram(2, verbose)
		return bigram_model

	def trigram(self, verbose=False):
		''' Train a trigram model on the data. '''
		trigram_model = self.ngram(3, verbose)
		return trigram_model

	def ngram(self, n, verbose=False):
		''' Train a ngram model on the data where n>1. '''
		ngram_model = defaultdict(partial(defaultdict, int))
		
		if verbose:
			print(f'Building {n}-gram model\nGetting {n}-gram model counts')

		# Temporarily fill the model with token emission counts.
		for line in self.data:
			line = self._unk(line)
		
			for gram in self._ngrams(line, n):
				# Let prec be the n-1 tokens preceeding the current curr token.
				# Store prec as a tuple of strings and curr as a single string.
				prec, curr = tuple(gram[:-1]), gram[-1:][0]
				ngram_model[prec][curr] += 1

		if verbose:
			print(f'Getting {n}-gram model probabilities')

		# Convert the token counts to probabilities.
		ngram_model = self._probabilities(ngram_model)

		# Cache and return the result.
		self.cache[n] = ngram_model
		return ngram_model

	def li_ngram(self, n, lambdas, verbose=False):
		''' Return a ngram model with linearly interpolated probabilities. '''
		if verbose:
			print(f'Getting {n}-gram linearly interpolated model')
		# Deepcopy to prevent side effects.
		lambdas = deepcopy(lambdas)

		# Let igrams be a list of ngram models s.t. 1<=i<=n.
		igrams = self._igrams(n, verbose)

		# For convenience and consistency.
		igrams.reverse()
		lambdas.reverse()

		# Let the following be the linear interpolation model that will be returned.
		li_ngram_model = deepcopy(igrams[0])

		if verbose:
			print(f'Computing {n}-gram LI probabilities')
	
		# Iterate over every key value_1 -> value_2 -> probability triplet in the ngram li model.
		# For each triplet, compute the new probability using the weighted sum of the igram models
		# from lambda_i * (some subset of value_1) -> value_2 -> probability)

		# Let prec be the j-(n-1) to j-1 tokens preceeding the jth token.
		# Let index and total be used only for verbose progress output.
		total = len(li_ngram_model)
		for index, prec in enumerate(li_ngram_model):
			if verbose:
				# Output progress every 10%.
				if index % int((total + 1) * 0.1) == 0:
					print(f'{math.ceil(index / total * 100)}%')
	
			# Let curr be the jth token.
			for curr in li_ngram_model[prec]:
				new_prob = 0
				k = len(prec)	

				# Compute the smoothed probability of the current token on each igram model s.t. 1<i<=n.
				for i in range(k):
					# Let the following be a subset of the preceeding tokens, diminishing as i increases.
					sub_prec = prec[i : k]

					# Let the following be the existing probability of the current token from th ith model.
					old_prob = igrams[i][sub_prec][curr]

					# Increment the new probabilty by the weighted amount.
					new_prob += lambdas[i] * old_prob

				# Handle the probability from the igram model for i=1 separately since the model is different. 
				new_prob += lambdas[-1] * igrams[-1][curr]
			
				# Set the smoothed probability.
				li_ngram_model[prec][curr] = new_prob

		return li_ngram_model
			
	def _igrams(self, n, verbose=False):	
		''' Return a list of i-gram models for 1<=i<=n. '''
		gram_list = []
		
		# Add unigram separately since it's instantiated differently.
		if 1 in self.cache:
			if verbose:
				print(f'1-gram cache hit')
			gram_list.append(self.cache[1])
		else:
			if verbose:
				print(f'1-gram cache miss')
			unigram = self.unigram(verbose)
			self.cache[1] = unigram
			gram_list.append(unigram)

		# Add the remaining 1<i<=n ngrams.	
		k = len(gram_list)
		while k < n:
			k += 1
			if k in self.cache:
				if verbose:
					print(f'{k}-gram cache hit')
				gram_list.append(self.cache[k])
			else:
				if verbose:
					print(f'{k}-gram cache miss')
				ngram_model = self.ngram(k, verbose) 
				self.cache[k] = ngram_model
				gram_list.append(ngram_model)

		return gram_list
	
	def _ngrams(self, line, n, padding=True):
		''' Yield the n length ngrams from the given line. '''
		# Insert START and STOP padding symbols.
		if padding:
			# Unigram padding.
			if n == 1:
				line.append(STOP)
			# ngram padding for n>1.
			else:
				for _ in range(n - 1):
					line.insert(0, START)
					line.append(STOP)

		# Split the line into n length ngrams.
		for i in range(len(line) - n + 1):
			#yield tuple([line[j] for j in range(i, i+n)])
			yield [line[j] for j in range(i, i+n)]
	
	def _probabilities(self, model):
		''' For ngram models s.t. n>1, return a ngram_model of the form tokens -> token -> probability. '''
		# Deepcopy the model to prevent side effects.
		new_model = deepcopy(model)
		for key_1 in new_model:
			total = float(sum(new_model[key_1].values()))
			for key_2 in new_model[key_1]:
				new_model[key_1][key_2] /= total
		return new_model

	def _unk(self, line):
		''' Return the given line with appropriate tokens changed to UNK. '''
		# Deepcopy the line to prevent side effects.
		new_line = deepcopy(line)
		if self.unk_threshold > 0:
			for i, t in enumerate(new_line):
				if t not in self.vocabulary or self.vocabulary[t] < self.unk_threshold: 
					new_line[i] = UNK
		return new_line

	def perplexity(self, data, ngram_model, n, verbose=False):
		''' Compute the perplexity of the data using the given ngram model. 
			Note that the ngram_model must match n, i.e. if unigram then n=1, if bigram then n=2, etc. '''
		if verbose:
			print('Computing perplexity')
		cross_entropy, zero_percentage = self.cross_entropy(data, ngram_model, n, verbose)
		perplexity = math.pow(2, cross_entropy) if cross_entropy != 0 else float('inf')	
		return perplexity, zero_percentage
	
	def cross_entropy(self, data, ngram_model, n, verbose=False):
		''' Return the cross entropy of the data using the given n sized ngram model.
			Note that the ngram_model must match n, i.e. if unigram then n=1, if bigram then n=2, etc. '''
		# Let the following be the sum of log losses over the data set.	
		log_loss_sum = 0
		# Let the following be the sum of tokens in the data set with non-zero log losses.
		token_sum = 0	
		# Let the following be the sum of tokens in the data set with zero log losses.
		zero_sum = 0

		if verbose:
				print('Summing cross entropy')

		for line in data:
			# Prepare the line by replacing unknown tokens.
			line = self._unk(line)
			
			# Get the probabilities of each ngram in the line.
			for ngram in self._ngrams(line, n):
				# Let the following be a flag indicating whether or not the
				# ngram appeared in the training data or not.			
				ngram_in_vocab = False

				# For a unigram model.
				if n == 1:
					curr = ngram[-1]
					if curr in ngram_model:
						ngram_in_vocab = True 
					prob = ngram_model[curr]
				# For a n>1 ngram model.
				else:
					# Let curr be the ith token and prec be the 0<=j<n tokens preceeding curr.
					# prec is stored as a tuple of strings and curr as a single string.
					prec, curr = tuple(ngram[:-1]), ngram[-1:][0]
					if prec in ngram_model and curr in ngram_model[prec]:
						ngram_in_vocab = True
					prob = ngram_model[prec][curr] 
				
				# And compute the log loss from the probability.
				log_loss_sum += math.log2(prob) if prob != 0 else 0

				# Use a flag to indicate if the ngram appeared in any model or not
				# and update the appropriate sum.
				if ngram_in_vocab:
					token_sum += 1
				else:
					zero_sum += 1
	
		# Let the following be the percentage of observations that had zero logs and thus, are
		# not included in the cross entropy calculation.	
		zero_percentage = zero_sum / (zero_sum + token_sum) * 100

		# Let cross entropy = - sum_i^k \frac{1}{k} log_2 p(t_i, t_{i-j}) 
		# where k is the total number of tokens in the data
		# and p(t_i, t_{i-j}) is the probability of token i begin preceeded by tokens j for 0<=j<n.
		cross_entropy = -log_loss_sum / token_sum if token_sum > 0 else 0.0

		return cross_entropy, zero_percentage

