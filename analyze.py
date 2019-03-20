from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random



# def sortData(text,labels):

'''
Train the data using the training input
returns an array [x_train,x_test,y_train,y_test]
'''
def train(data,labels):
    vectorizer = CountVectorizer(analyzer='word',lowercase=False)
    features = vectorizer.fit_transform(data)
    features_array = features.toarray()
    X_train,X_test,Y_train,Y_test = train_test_split(features_array,labels,train_size=.8,random_state=1234)
    features = [X_train,Y_train,X_test,features_array]
    return features

'''
predict the data, based on the training.
might have to pass it the model? (the train/test ^^^
'''
def predict(data,X_train,Y_train,X_test,features_array):
    log_model = LogisticRegression()
    log_model = log_model.fit(X=X_train,y=Y_train)
    y_predict = log_model.predict(X_test)
    j = random.randint(0, len(X_test) - 7)
    for i in range(j, j + 7):
        print(y_predict[0])
        ind = features_array.tolist().index(X_test[i].tolist())
        print(data[ind].strip())


