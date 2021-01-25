import random
import string

def get_letters(n=3):
	return [random.choice(string.ascii_lowercase) for _ in range(n)]

def is_valid(letters, choice):
	return choice.lower() in letters

def should_quit(choice):
	return choice.lower() == 'quit'


if __name__ == '__main__':
	missive = ''
	while True:
		print(f'Missive: {missive}')

		letters = get_letters()
		print(f'Letters: {letters}')

		choice = input('Choose a letter: ')

		if should_quit(choice):
			break

		if is_valid(letters, choice):
			missive += choice
		
