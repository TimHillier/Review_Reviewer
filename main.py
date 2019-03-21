from numpy import load
import numpy as np
import analyze


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


'''
main method
'''
def main():
    alignment,reviews = readFiles("train.npz")
    alignment = decode(alignment)
    alignment = describeLabels(alignment)
    reviews = decode(reviews)

    test_align,test_reviews = readFiles("test.npz")
    test_align = decode(test_align)
    test_align = describeLabels(test_align)
    test_reviews = decode(test_reviews)

    model = analyze.train(reviews,alignment)
    analyze.predict(test_reviews,model[0],model[1],model[2],model[3])

main()
