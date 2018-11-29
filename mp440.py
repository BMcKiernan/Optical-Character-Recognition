import inspect
import sys
from collections import Counter
import pdb
import numpy as np
import math
import random


f_count = {}
prior = {}
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
    #Feature 1:  Fix Image roatation <-----------------------------------------Advanced Feature 1
    x1 = -1
    x2 = -1
    x3 = -1
    x4 = -1
    top = 0
    bottom = height - 1
    c_idx = 0
    centers = [ 0 ] * width
    for row in digit_data:
        if 1 in row or 2 in row:
            for i, val in enumerate(row):
                if val > 0:
                    if x1 == -1:
                        x1 = i
                    x2 = i
            centers[c_idx] = float(x2 + x1)/2
            c_idx += 1
            x1 = -1
        else:
            if c_idx < 1:
                top += 1
    for idx in range(height):
        if 1 in digit_data[-idx] or 2 in digit_data[-idx]:
            bottom -= idx
            break
    cnt = []
    for e in centers:
        if e > 0:
            cnt.append(e)
    centers = cnt
    # average out the slope
    dy = float(top - bottom)
    slopes = [0] * c_idx
    # pdb.set_trace()
    for i in range(c_idx):
        if i > 0:
            if centers[i] - centers[i-1] != 0:
                slopes[i] = abs(dy/(centers[i] - centers[i-1]))
            else:
                slopes[i] = abs(dy/centers[i])
    del slopes[0]
    slope = abs(sum(slopes)/len(slopes))
    l = 14-14*slope
    if slope < 0.2 or slope > 1.5:
        slope = None
    #print "dx: " + str(dx) + ", dy: " + str(dy)
    dc = 0
    ft = [[False for x in range(width)] for y in range(height) ]
    for r in range(height):
        for c in range(width):
            if digit_data[r][c] > 0:
                dc = bmx(slope, r, l)
                try:
                    ft[r][c - dc] = True
                except:
                    ft[r][c] = True
                    #pdb.set_trace()
    features = []
    for row in ft:
        for col in row:
            features.append(col)

    #Feature 2 Magnification of image by 4 <---------------------------------------- Advanced Feature 2
    features2 = [ False ] * (4*width*height)
    feature_len2 = len(features)
    for i in range(feature_len2):
            if features[i] > 0:
                for j in range(4):
                    features2[i+j] = True


    #Basic features included for testing 
   # for i in range(feature_len2):
    #    if features2[i] == True:
     #       features2[i] == False
      #  else:
       #     features2[i] == True '''

    
    #Feature 3 invert boolean  <---------------------------------------------------- Advanced Feature 3
    feature_len3 = len(features2)
    features3 = [True] * feature_len3
    for i in range(feature_len3):
        if features2[i]:
            features3[i] = False
        else:
            features3[i] = True

    return features3

    

def bmx(slope, y, l):
    if slope is None:
        return 0
    else:
        return int((y-l)/slope)

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
    num_labels = int(len(label)*percentage/100)
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
    span = len(f_count[random.choice(f_count.keys())][1])
    total = 0
    for key in f_count:
        total = num_labels * prior[key]
        for j in range(span):
            f_count[key][1][j] = f_count[key][1][j]/total

def feature_counter(feature_extractor, value = None):
    span = len(feature_extractor)
    if value is None:
        value = [[0.0 for i in range(span)] for i in range(2)]
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
        summ = 0.0
        summ += math.log(prior[label])
        for idx, val in enumerate(f_count[label][1]):
            if features[idx]:
                summ += k+math.log(0.00000001+val)
            else:
                summ += k+math.log(0.00000001+1-val)
        if summ > max_prob_label[0]:
            max_prob_label = (summ, label)
    return max_prob_label[1]
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



def print_data(image_data, key):
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
                print( str(image_data[i][j]) + " ,"),
            else:
                count += 1
                print str(image_data[i][j]) + " \n"



        
    
