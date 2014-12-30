'''
This is a machine learning classification algorithim that is similar to
the k-Nearest-Neighbor classification.  For a test point, it takes all of
the neighbors within a certain radius as the nearest neighbors and assigns
the test point the most common classification of these neighbors. 
Only makes sense to use if all your distances are normalized.
'''

def fRN(radius,
    train_data_X, 
    train_data_y,
    test_point,
    random_seed = None):
    '''

    '''
    if len(train_data_X) != len(train_data_y):
        raise ValueError('training data inconsistent length')
    closest_neighbor = kNN2(1, train_data_X, train_data_y, test_point)
    if radius < closest_neighbor:
        raise ValueError('no neighbors within radius')
    sort_by_dist = sorted(train_data_X, key=lambda point: distance(test_point, point))
    for dist in sort_by_dist:
        if dist < radius:
            continue
        else:
            i_first_over = sort_by_dist.index(dist)

    categories = [train_data_y[train_data_X.index(close)] for close in sort_by_dist[:i_first_over-1]]
    winner = max(set(categories), key=categories.count)
    return winner

    

def kNN2(k, 
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