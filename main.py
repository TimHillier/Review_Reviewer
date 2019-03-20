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
main method
'''
def main():
    alignment,reviews = readFiles("train.npz")
    alignment = decode(alignment)
    reviews = decode(reviews)

    model = analyze.train(reviews,alignment)
    analyze.predict(reviews,model[0],model[1],model[2],model[3])

main()
