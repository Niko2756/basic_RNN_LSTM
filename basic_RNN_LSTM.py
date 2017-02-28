#LSTM neural network to generate city names
from __future__ import absolute_import,division,print_function

from tflearn.data_utils import *
from six import moves
import os
import ssl
import tflearn

def main():
	#Step 1 - Retrieve the data
	path = "US_Cities.txt"
	#if not os.path.isfile(path):
		#get dataset
		#moves.urllib.request.urlretrieve("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_Cities.txt", path, ssl._create_unverified_context())

	#city name max length
	maxlen = 17

	#vectorize the text file
	X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen = maxlen, redun_step = 3)

	#CREATE LSTM NETWORK
	g = tflearn.input_data(shape = [None, maxlen, len(char_idx)])
	g = tflearn.lstm(g, 512, return_seq = True, restore = True)
	g = tflearn.dropout(g, 0.5)
	g = tflearn.lstm(g, 512, restore = True)
	g = tflearn.dropout(g, 0.5)
	g = tflearn.fully_connected(g, len(char_idx), activation = "softmax", restore = True)
	#g = tflearn.regression(g, optimizer = "rmsprop", loss = "binary_crossentropy", learning_rate = 0.001, restore = True)
	g = tflearn.regression(g, optimizer = "adam", loss = "categorical_crossentropy", learning_rate = 0.001, restore = True)
	#g = tflearn.regression(g, optimizer = "rmsprop", loss = "categorical_crossentropy", learning_rate = 0.001, restore = True)

	#generate cities
	#m = tflearn.SequenceGenerator(g, dictionary = char_idx, seq_maxlen = maxlen, clip_gradients = 5.0, checkpoint_path = "model_us_cities")
	m = tflearn.SequenceGenerator(g, dictionary = char_idx, seq_maxlen = maxlen, clip_gradients = 5.0)
	
	#m.save("model_us_cities")
	if os.path.isfile("model_us_cities.index"):
		m.load("./model_us_cities")
		print()
		print()
		print("Loading Previous Weights")
		print("(0)(0)\n (='.'=)\n (')__(')")
		retrainOrNo = maybeRetrain()
		if retrainOrNo in ("yes","y"):
			#Retraining
			for i in range(40):
				m.fit(X, Y, validation_set = 0.1, batch_size = 128, n_epoch = 5, run_id = "US Cities")
			m.save("model_us_cities")
		else:
			pass
	else:
		#training
		for i in range(60):
			m.fit(X, Y, validation_set = 0.1, batch_size = 128, n_epoch = 5, run_id = "US Cities")
		m.save("model_us_cities")

	seq_length = 200
	nameOfFile = inputNameOfFile()
	with open("""{}.txt""".format(nameOfFile),"w") as inFile:
		for item in range(40):
			seed = random_sequence_from_textfile(path, maxlen)
			inFile.write(str("TESTING at temperature 0.2") + "\n")
			inFile.write(str(m.generate(seq_length, temperature = 0.2, seq_seed = seed)) + "\n")
			inFile.write("\n")
			inFile.write(str("TESTING at temperature 1.0") + "\n")
			inFile.write(str(m.generate(seq_length, temperature = 1.0, seq_seed = seed)) + "\n")
			inFile.write("\n")
			inFile.write(str("TESTING at temperature 0.5") + "\n")
			inFile.write(str(m.generate(seq_length, temperature = 0.5, seq_seed = seed)) + "\n")
	inFile.close()
	print("Your list has been saved under: " """{}.txt""".format(nameOfFile))

	#print("TESTING at temperature 0.2")
	#print(m.generate(seq_length, temperature = 0.2, seq_seed = seed, display = True))
	#print()
	#print("TESTING at temperature 1.0")
	#print(m.generate(seq_length, temperature = 1.0, seq_seed = seed, display = True))
	#print()
	#print("TESTING at temperature 0.5")
	#print(m.generate(seq_length, temperature = 0.5, seq_seed = seed, display = True))
def inputNameOfFile():
	while True:
		nameOfFile = input("Enter title of save file: ") or "Final list"
		if not nameOfFile.isalpha() and not nameOfFile.isprintable():
			print("Come on that's not what I asked for.\nPlease just give me a name in English please.\n")
			continue
		else:
			return nameOfFile.capitalize()

def maybeRetrain():
	while True:
		retrainOrNo = input("\nDo you want to retrain? (Enter yes or no, just produce the list): ") or "yes"
		if retrainOrNo.lower() not in ("yes", "no", "y", "n"):
			print("Come on that's not what I asked for.\nPlease just give me a affirmative yes or a no in English please.")
			continue
		else:
			return retrainOrNo.lower()

main()