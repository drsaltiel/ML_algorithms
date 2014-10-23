'''
This file contains an implimentation of k-Fold cross validation
'''

import random

def kFoldCross(k, algorithm, data, random_seed = None):
	'''
	implimentation of k-fold cross validation
	inputs: k - number of folds 
			algorithm - classification algorithm
			data - data for classification
			random_seed: seed for initial division
	outputs: average test error for all folds
	'''
	#initial divide data
	random.seed = random_seed
	random.shuffle(data)
	testPortion = len(data)/k

	#test error on each train/test divide
	errors = [0]*k
	for i in range(1, k+1):
		train = data[(i-1)*testPortion:i*testPortion]
		test = data[:(i-1)*testPortion]
		test.append(data[i*testPortion:])

		#train on data


		#test error
		errors[i] = error

	#average error and output
	ave_error = sum(errors)/float(k)
	return ave_error