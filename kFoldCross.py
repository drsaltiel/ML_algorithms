'''
This file contains an implimentation of k-Fold cross validation
'''


from sklearn.cross_validation import KFold
def crossValidate(X, y, classifier, k_folds) :
    '''
    only works for the form classifier.fit()
    '''
    # get train and test indexes
    indicies = KFold(len(X), n_folds=k_folds, shuffle = True)
    
    #get scores
    scores = []
    for train, test in indicies:
        knn = classifier.fit(X[train],y[train])
        score = knn.score(X[test], y[test])
        scores.append(score)

    # return the average accuracy
    return sum(scores)/float(k_folds)







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
    indicies = random.shuffle(range(0,len(data)))
    test_i = len(data)/k

    #test error on each train/test divide
    errors = [0]*k
    for i in range(1, k+1):
        train = data[(i-1)*test_i:i*test_i]
        test = data[:(i-1)*test_i]
        test.append(data[i*_i:])

        #train on data


        #test error
        errors[i] = error

    #average error and output
    ave_error = sum(errors)/float(k)
    return ave_error