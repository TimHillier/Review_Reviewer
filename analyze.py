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
# def train(data,labels):
#     vectorizer = CountVectorizer(analyzer='word',lowercase=False)
#     features = vectorizer.fit_transform(data)
#     features_array = features.toarray()
#     X_train,X_test,Y_train,Y_test = train_test_split(features_array,labels,train_size=.8,random_state=1234)
#     features = [X_train,Y_train,X_test,features_array,Y_test]
#     return features

def train(data,final_data,labels):
    # print("L",labels)
    countVec = CountVectorizer(binary=True)
    countVec.fit(data)
    X = countVec.transform(data)
    X_test = countVec.transform(final_data)
    X_train,X_val,y_train,y_val = train_test_split(X,labels,train_size=0.75)
    features = [X_train,y_train,y_val,X_val,X_test,X] #This might need to be checked
    return  features


def predict(data,labels,model):
    # print("L",labels)
    X_train = model[0]
    y_train = model[1]
    y_val   = model[2]
    X_val   = model[3]
    X_test  = model[4]

    #for testing different values
    for c in [0.01,0.05,0.25,0.5,1]:
        lr = LogisticRegression(C=c)
        lr.fit(X_train,y_train)
        print("accuracy for C=%s: %s" %(c,accuracy_score(y_val,lr.predict(X_val))))



        # accuracy for C=0.01: 0.7745803357314148
        # accuracy for C=0.05: 0.7937649880095923
        # accuracy for C=0.25: 0.8081534772182254
        # accuracy for C=0.5:  0.7985611510791367
        # accuracy for C=1:    0.8009592326139089
        # best accuracy is c = 0.25

    final_model = LogisticRegression(C=0.25)
    final_model.fit(X_test,labels)
    print("Final Acuracy: %s"%accuracy_score(labels,final_model.predict(X_test)))

'''
predict the data, based on the training.
might have to pass it the model? (the train/test ^^^
'''
# def predict(data,model):
#     # print(data)
#     X_train = model[0]
#     Y_train = model[1]
#     X_test = model [2]
#     feature_array = model[3]
#     y_test = model[4]
#
#     print(X_test)
#     log_model = LogisticRegression()
#     log_model = log_model.fit(X=X_train,y=Y_train)
#     y_predict = log_model.predict(X_test)
#     # y_predict = log_model.predict(data)
#     j = random.randint(0, len(X_test) - 7)
#     # for i in range(j, j + 7):
#     for i in range(0,3):
#         # print(len(y_predict))
#         print(y_predict[0])
#         ind = feature_array.tolist().index(X_test[i].tolist())
#         # print("ind",ind)
#         print(data[ind].strip())
#     # print(accuracy_score(y_test,y_predict))

