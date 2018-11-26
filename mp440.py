import inspect
import sys
from collections import Counter
import pdb
import numpy as np
import math

f_count = {}
prior = {}
k = 3

'''
Raise a "not defined" exception as a reminder 
'''
def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)


'''
Extract 'basic' features, i.e., whether a pixel is background or
forground (part of the digit) 
'''
def extract_basic_features(digit_data, width, height):
    features=[]
    for row in range(height):
        for col in range(width):
            if(digit_data[row][col] == 0):
                features.append(False)
            else:
                features.append(True)
    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here 
    #_raise_not_defined()
    #print "Inside first mp function printing features \n" + str(features) + ""
    return features

'''
Extract advanced features that you will come up with 
'''
def extract_advanced_features(digit_data, width, height):
    features=[]
    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here 
    _raise_not_defined()
    return features

'''
Extract the final features that you would like to use
'''
def extract_final_features(digit_data, width, height):
    features=[]
    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here 
    _raise_not_defined()
    return features

'''
Compupte the parameters including the prior and and all the P(x_i|y). Note
that the features to be used must be computed using the passed in method
feature_extractor, which takes in a single digit data along with the width
and height of the image. For example, the method extract_basic_features
defined above is a function than be passed in as a feature_extractor
implementation.

The percentage parameter controls what percentage of the example data
should be used for training. 
'''
def compute_statistics(data, label, width, height, feature_extractor, percentage=100.0):
    num_labels = len(label)*percentage/100
    global prior
    global f_count
    #Get frequency of labels Ysub(i)
    prior = Counter(label)
    #Make values P(Y) instead of frequency
    for key, value in prior.iteritems():
        prior[key] = float(value)/num_labels
       # print "key: " + str(key)+ " value: "+ str(prior[key])+ "\n"
    #Cond prob
    f_count = {}
    for i in range(int(num_labels)):
        #if label[i] == 1:
        #    pdb.set_trace()
        if label[i] not in f_count:
            f_count[label[i]] = feature_counter(feature_extractor(data[i], width, height), None)
        else:
            f_count[label[i]] = feature_counter(feature_extractor(data[i], width, height), f_count[label[i]])
    #^^^ Now have all values necessary for cond prob calc
    span = (width*height)
    for key in f_count: #check ya sanwhich
        for i in range(2):
            for j in range(span):
                try:
                    f_count[key][i][j] = math.log(float(k+f_count[key][i][j])/(k+f_count[key][1][j] + k+f_count[key][0][j]) )
                except:
                    print "feature " + str(f_count[label[i]][1][j]) + ", at label " + str(label[i]) + ", at index " + str(j)
                    print "pos " + str(f_count[label[i]][1][j]) + ", neg " + str(f_count[label[i]][0][j])
                    print "k " + str(k) + "\n"

def feature_counter(feature_extractor, value = None):
    if value is None:
        value = np.zeros(2, len(feature_extractor))
    span = len(value[0])
    for i in range(span):
        if(feature_extractor[i] == False):
            value[0][i] += 1
        else:
            value[1][i] += 1
    return value


'''
For the given features for a single digit image, compute the class 
'''
def compute_class(features):
    global f_count
    global prior
    max_prob_label = (-1, -1)
    summ = 0.0
    for label in f_count:
        # Sum over the features that match per image
        # set max_prob_label = max(max_prob_label, the newly calculated features sum)
        summ += math.log(prior[label])
        for idx, val in enumerate(f_count[label][1]):
            if features[idx]:
                summ += val
            else:
                summ += 1-val
        if summ > max_prob_label[0]:
            max_prob_label = (summ, label)
        summ = 0.0
    return max_prob_label[1]

'''
Compute joint probaility for all the classes and make predictions for a list
of data
'''
def classify(data, width, height, feature_extractor):
    predicted=[]
    for image in data:
        predicted.append(compute_class(feature_extractor(image, width, height)))
    return predicted







        
    
