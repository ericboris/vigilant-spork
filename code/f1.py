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
	def next_letter():
		pass

	def timer():
		pass

	def run():
		pass

	def quit():
		pass
		
	def send():
		pass

# Sends and receives data from the model.
class Predictor:
	''' Provide an abstract parent class for predictors to implement
		so that we can swap in different predictors for different results. '''
	def train():
		pass
	
	def predict():
		pass

# View
# Handle how the predicted letters are presented.
# Probably as minimal as print(f'{char1}\t{char2}\t{char3}')
class View:
	''' Provide a consistent interface for displaying results from the Model. '''
	def show():
		pass

# Controller
# Accept inputs and return the results to the model. 
class Controller
	''' Provide a consistent interface for sending input to the Model. '''
	def get_input():
		pass
		
	def validate():
		pass
		
	def request():
		pass

