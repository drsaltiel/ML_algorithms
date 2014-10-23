'''
This is an implementation of the k-Nearest Neighbor classification algorithm.
For a given test point, it finds the k nearest neighbors and assigns to the 
test point the most common classification of those neighbors.  In the case of a 
tie it picks randomly from the classifications most represented.
'''

import random as random

def kNN1(k = int, 
    train_data_X, 
    train_data_y,
    test_point,
    random_seed = None):
    '''
    to be added: weighting, distance metric?
    inputs: number of neighbors (integer), 
            training data (array),
            training classifications (list), 
            data point to classify (array)
            random seed (for tie breaker)
    outputs: classifier for test point based on k nearest neighbors
    '''
    if len(train_data_X) != len(train_data_y):
        raise ValueError('training data inconsistent length')
    if k < 1 :
        raise ValueError('k less than 1')
    if k > len(train_data_X):
        raise ValueError('k greater than total neighbors')
    nearestNeighbors_indicies = [None]*k
    nearestNeighbors_distances = [None]*k
    for i in range(len(train_data_X)):      
        distance = distance(test_point, train_data_X[i])
        minimum = min(nearestNeighbors)
        if minimum is None:
            nearestNeighbors_indicies[nearestNeighbors_indicies.index(minimum)] = i
            nearestNeighbors_distances[nearestNeighbors_indicies.index(minimum)] = distance
            continue
        distance = distance(test_point, train_data_X[i])
        maximum = max(nearestNeighbors_diestances)
        if maximum > distance:
            nearestNeighbors_indicies[nearestNeighbors_indicies.index(maximum)] = i
            nearestNeighbors_distances[nearestNeighbors_indicies.index(maximum)] = distance
    nearest_classifiers = {}
    for classifier in train_data_y[nearestNeighbors]:
        if classifier in nearest_classifiers.keys():
            nearest_classifiers[classifier] +=1
        else:
            nearest_classifiers[classifier] = 1
    nearest_neighbors_classifier = []
    greatest_nNeighbors = 0
    for classifier, nNeighbors in nearest_classifiers.iteritems():
        if nNeighbors > greatest_nNeighbors:
            nearest_neighbors_classifier = [classifier]
            greatest_nNeighbors = nNeighbors
        else if nNeighbors == greatest_nNeighbors:
            nearest_neighbors_classifier.append(classifier)
    random.seed(random_seed)
    NN = nearest_neighbors_classifier[random.randrange(len(nearest_neighbors_classifier))]
    return NN

def kNN2(k = int, 
    train_data_X, 
    train_data_y,
    test_point,
    random_seed = None):
    '''
    shomik's version
    '''
    if len(train_data_X) != len(train_data_y):
        raise ValueError('training data inconsistent length')
    if k < 1 :
        raise ValueError('k less than 1')
    try:
        closest = sorted(train_data_X, key=lambda point: distance(test_point, point))[:k] 
    except IndexError:
        raise ValueError('Not enough neighbors ({0} requested, {1} exist)'.format(
            k, len(train_data_X)))
    categories = [train_data_y[train_data_X.index(close)] for close in closest]
    winner = max(set(categories), key=categories.count)
    return winner


def distance(point1, point2):
    '''
    finds the euclidian distance between two points for an arbitrary number of dimensions
    inputs: two points, lists
    outputs: distance between points, float
    '''
    nDimensions = len(point1)
    if nDimensions != len(point2):
        raise ValueError('Inputs have different dimensions')
    distanceSquared = 0
    for i in range(nDimensions):
        distanceSquared += (point1[i]-point2[i])**2
    distance = distanceSquared ** 0.5
    return distance








class kNN(object):
    def __init__(self, k):
        self.k = k
    
    def predict(self, train_data_X, 
        train_data_y,
        test_point):
        if len(train_data_X) != len(train_data_y):
            raise ValueError('training data inconsistent length')
        if self.k < 1 :
            raise ValueError('k less than 1')
        try:
            closest = sorted(train_data_X, key=lambda point: kNN.distance(test_point, point))[:self.k] 
        except IndexError:
            raise ValueError('Not enough neighbors ({0} requested, {1} exist)'.format(
                self.k, len(train_data_X)))
        categories = [train_data_y[train_data_X.index(close)] for close in closest]
        winner = max(set(categories), key=categories.count)
        return winner
    
    def test_error(self, X_train_data, y_train_data, X_test_data, y_test_data):
        if len(test_data_X) != len(test_data_y):
            raise ValueError('testing data inconsistent length')
        predicted = [False]*len(X_test_data)
        for i in range(len(X_test_data)):
            pred = kNN.predict(X_train_data, y_train_data, X_test_data[i])
            if pred == y_test_data[i]:
                predicted[i]=True
        error = predicted.count(True)/float(len(predicted))
        return error
    
    @staticmethod
    def distance(point1, point2):
        '''
        finds the euclidian distance between two points for an arbitrary number of dimensions
        inputs: two points, lists
        outputs: distance between points, float
        '''
        nDimensions = len(point1)
        if nDimensions != len(point2):
            raise ValueError('Inputs have different dimensions')
        distanceSquared = 0
        for i in range(nDimensions):
            distanceSquared += (point1[i]-point2[i])**2
        distance = distanceSquared ** 0.5
        return distance
