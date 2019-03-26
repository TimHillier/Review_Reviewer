from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

'''
This just gie positive for everything its dumb
'''


# def sortData(text,labels):

'''
Train the data using the training input
returns an array [x_train,x_test,y_train,y_test]
'''
def train(data,final_data,labels):
    # print("L",labels)
    countVec = CountVectorizer(binary=True)
    countVec.fit(data)
    X = countVec.transform(data)
    X_test = countVec.transform(final_data)
    X_train,X_val,y_train,y_val = train_test_split(X,labels,train_size=0.75)
    features = [X_train,y_train,y_val,X_val,X_test,X] #This might need to be checked
    return features


'''
predict the data, based on the training.
might have to pass it the model? (the train/test ^^^
'''
def predict(data,labels,model):
    # print("L",labels)
    X_train = model[0]
    y_train = model[1]
    y_val   = model[2]
    X_val   = model[3]
    X_test  = model[4]

    # #for testing different values of c
    # for c in [0.01,0.05,0.25,0.5,1]:
    #     lr = LogisticRegression(C=c)
    #     lr.fit(X_train,y_train)
    #     print("accuracy for C=%s: %s" %(c,accuracy_score(y_val,lr.predict(X_val))))
    #
    #
    #
    #     # accuracy for C=0.01: 0.7745803357314148
    #     # accuracy for C=0.05: 0.7937649880095923
    #     # accuracy for C=0.25: 0.8081534772182254
    #     # accuracy for C=0.5:  0.7985611510791367
    #     # accuracy for C=1:    0.8009592326139089
    #     # best accuracy is c = 0.25

    final_model = LogisticRegression(C=0.25)
    final_model.fit(X_test,labels)
    print("Final Acuracy: %s"%accuracy_score(labels,final_model.predict(X_test)))


