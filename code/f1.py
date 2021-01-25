# Model
# Get and return top 3 letters.
# Manage a timer.
# Manage quit/send.
# Need to determine standard data transfer format. Are we using dicts? or np.array? or something else?
# Handle load data - train, valid, test
# Train a model
# Make a prediction
# Update from prediction and make next prediction
class Model:
	''' Provide a consistent interface for getting letters from a Predictor. '''
	def next_letters():
		''' Return the next 3 letters as predicted by the model. '''
		pass

	def timer():
		''' Maintain the amount of time that has elapsed since displaying
			the predicted letters. '''
		pass

	def run():
		''' Run the model. Get predicted letters and maintain the timer. '''
		pass

	def quit():
		''' Convenience method to formally shut down the model. '''
		pass
		
	def send():
		''' "Send" the missive being written by the model.
			Trigger generating performance and timing results. '''
		pass

# Sends and receives data from the model.
class Predictor:
	''' Provide an abstract parent class for predictors to implement
		so that we can swap in different predictors for different results. '''
	def __init__():
		''' Perform any steps (ex: training) necessary to prepare the 
			predictor to predict. '''
		pass

	def train():
		''' Train a model. '''
		pass
	
	def predict():
		''' Return the predicted results
			based on the model (AND current state of the missive?). '''
		pass

""" Example Predictor class implementation. """
"""
class MLE(Predictor):
	''' Implementing a Maximum Likelihood Estimator as a Predictor. '''
	def __init__():
		# Implementation
		pass
	def train():
		# Implementation
		pass
	def predict():
		# Implementation
		pass
	def something_specific_to_MLE():
		# Implementation
		pass
"""

# View
# Handle how the predicted letters are presented.
# Probably as minimal as print(f'{char1}\t{char2}\t{char3}')
class View:
	''' Provide a consistent interface for displaying results from the Model. '''
	def show():
		''' Display the top 3 predicted letters
			as well as "Quit?" and "Send?" options. '''
		pass

# Controller
# Accept inputs and return the results to the model. 
class Controller
	''' Provide a consistent interface for sending input to the Model. '''
	def get_input():
		''' Listen for input. '''
		pass
		
	def validate():
		''' Clean up the input (ie. upper->lowercase ?),
			confirm that it's valid (ie. one of 3 letters or Quit or Send),
			and handle if not valid. '''
		pass
		
	def request():
		''' Request next action from model based on validated input. '''
		pass

