'''
This is an implimentation of Principal Component Analysis.
The algorithm takes a set of data and calculates a covariance matrix.
It then finds the eigenvectors and respective eigenvalues of this 
covariance matrix.  It then translates it to the new basis of these 
eigenvectors for which the variance in the direction of each 
eigenvector is in desending order - i.e. the earlier the component,
the more information is contained in it.
Additionally, a measure for the amount of information contained in
each variable - the ratio of the eigenvector for one component against
the sum of all the eigenvectors, is present. 
'''

