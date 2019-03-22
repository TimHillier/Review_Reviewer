from numpy import load
import numpy as np
'''
The first step is to obvious read the .npz files
It seems like if its a 1 its good, if its 0 its bad
'''
def readFiles(f):
    data = load(f)
    lst = data.files
    reviews = []
    alignment = []
    for item in lst:
        reviews = list(data[item][0])
        alignment = list(data[item][1])
    return alignment,reviews

'''
Encodes the numpy data to unicode
'''
def decode(lst):
    newlist = []
    for item in lst:
        newlist.append(item.encode('utf8'))
    return newlist

'''
use words instead of numbers as labels
'''
def describeLabels(lst):
    labels = []
    for label in lst:
        if label == '1.0':
            labels.append("positive")
        if label == '0.0':
            labels.append("negitive")
    return labels

