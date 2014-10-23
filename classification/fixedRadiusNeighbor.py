'''
This is a machine learning classification algorithim that is similar to
the k-Nearest-Neighbor classification.  For a test point, it takes all of
the neighbors within a certain radius as the nearest neighbors and assigns
the test point the most common classification of these neighbors.  In the
case of a tie it picks from the most common classifications at random.
'''

import random as random

def fRN(radius
		train_data_X, 
		train_data_y,
		test_point,
		random_seed = None):
	