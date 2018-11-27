import inspect
import sys
from collections import Counter
import pdb
import numpy as np
import math


k = math.e


'''
Raise a "not defined" exception as a reminder 
'''
def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)

def _value_to_pixel(value):
    if(value == 0):
        return ' '
    elif(value == 1):
        return '#'
    elif(value == 2):
        return '+'


def _print_digit_image(data):
    for row in range(len(data)):
        print ''.join(map(_value_to_pixel, data[row]))

'''
Eundorxtract 'basic' features, i.e., whether a pixel is backgro 
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
    #Cond prob
    f_count = {}
    for i in range(int(num_labels)):
        if label[i] not in f_count:
            f_count[label[i]] = feature_counter(feature_extractor(data[i], width, height), None)
        else:
            f_count[label[i]] = feature_counter(feature_extractor(data[i], width, height), f_count[label[i]])
    #^^^ Now have all values necessary for cond prob calc
    span = (width*height)
    for key in f_count: 
        for i in range(2):
            for j in range(span):
                f_count[key][i][j] = k + math.log(.0000001 + float((f_count[key][i][j])/(prior[key]*num_labels)))
                

def feature_counter(feature_extractor, value = None):
    span = len(feature_extractor)
    if value is None:
        value = [[0.0 for i in range(span)] for i in range(2)]
        #value = np.zeros((2, len(feature_extractor))
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
    span = len(features) 
    #image_sum = [[0.0 for i in range(span)] for j in range(2)] #
   # global saved_image
    max_match = [0.0, None]
    for key in f_count:
        _sum = 0.0
        _sum += math.log(prior[key])
        for i in range(span):
            if features[i]:
       #         image_sum[1][i] += f_count[key][1][i]
                _sum += f_count[key][1][i]
            else:
      #          image_sum[0][i] += f_count[key][0][i]
                _sum += f_count[key][0][i]
        if _sum > max_match[0]:
     #       saved_image = image_sum
            max_match = [_sum, key]
    #print_data(saved_image, max_match[1])
    return max_match[1]

'''
Compute joint probaility for all the classes and make predictions for a list
of data
'''
def classify(data, width, height, feature_extractor):
    predicted=[]
    for image in data:
        #print "Printing Image: "
        #_print_digit_image(image)
        predicted.append(compute_class(feature_extractor(image, width, height)))
    return predicted



def print_data(shit, key):
    span = (28*28)
    count = 0
    for i in range(2):
        if i == 0:
            print "\n\n Printing false for label " + str(key) + " \n"
        else:
            print "\n\n Printing true for label " + str(key) + " \n"
        for j in range(span):
            if count == 0 or 28%count != 0:
                count += 1
                print( str(shit[i][j]) + " ,"),
            else:
                count += 1
                print str(shit[i][j]) + " \n"



        
    
