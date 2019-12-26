# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:10:08 2019

@author: linhnh
"""

import pandas as pd
import numpy as np 
import math
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

training_data_path = 'D:/University/Senior year/Doan3/adult/adult.data.csv'
test_data_path = 'D:/University/Senior year/Doan3/adult/adult.test.csv'

training_data_all = pd.read_csv(training_data_path)
test_data = pd.read_csv(test_data_path)

label_encoder = preprocessing.LabelEncoder()

#==================================================================================== this part is for Random Forest, KNN
training_data_all['class'] = label_encoder.fit_transform(training_data_all['class'])

training_data = pd.concat([training_data_all.loc[training_data_all['class'] == 0].head(5841), training_data_all.loc[training_data_all['class'] == 1]])
#print(training_data)
test_data['class'] = label_encoder.fit_transform(test_data['class'])
training_data_one_hot_encode = pd.get_dummies(training_data)
test_data_one_hot_encode = pd.get_dummies(test_data)
#=====================================================================================

# this part is for Naive Bayes
training_data_NB = training_data.copy()
test_data_NB = test_data.copy()
training_data_NB['workclass'] = label_encoder.fit_transform(training_data_NB['workclass'])
training_data_NB['education'] = label_encoder.fit_transform(training_data_NB['education'])
training_data_NB['marital-status'] = label_encoder.fit_transform(training_data_NB['marital-status'])
training_data_NB['occupation'] = label_encoder.fit_transform(training_data_NB['occupation'])
training_data_NB['relationship'] = label_encoder.fit_transform(training_data_NB['relationship'])
training_data_NB['race'] = label_encoder.fit_transform(training_data_NB['race'])
training_data_NB['sex'] = label_encoder.fit_transform(training_data_NB['sex'])
training_data_NB['native-country'] = label_encoder.fit_transform(training_data_NB['native-country'])
training_data_NB['class'] = label_encoder.fit_transform(training_data_NB['class'])

test_data_NB['workclass'] = label_encoder.fit_transform(test_data_NB['workclass'])
test_data_NB['education'] = label_encoder.fit_transform(test_data_NB['education'])
test_data_NB['marital-status'] = label_encoder.fit_transform(test_data_NB['marital-status'])
test_data_NB['occupation'] = label_encoder.fit_transform(test_data_NB['occupation'])
test_data_NB['relationship'] = label_encoder.fit_transform(test_data_NB['relationship'])
test_data_NB['race'] = label_encoder.fit_transform(test_data_NB['race'])
test_data_NB['sex'] = label_encoder.fit_transform(test_data_NB['sex'])
test_data_NB['native-country'] = label_encoder.fit_transform(test_data_NB['native-country'])
test_data_NB['class'] = label_encoder.fit_transform(test_data_NB['class'])

del training_data_NB['class']
del test_data_NB['class']
del training_data_NB['fnlwgt']
del test_data_NB['fnlwgt']
bins_age = np.arange(0, training_data_NB['age'].max()+1, 10)
bins_edunum = np.arange(0, training_data_NB['education-num'].max()+1, 5)
bins_capital_gain = np.arange(-1, training_data_NB['capital-gain'].max()+1, 10000)
bins_capital_loss = np.arange(-1, training_data_NB['capital-loss'].max()+1, 1000)
bins_hours = np.arange(0, training_data_NB['hours-per-week'].max()+1, 10)
training_data_NB['age'] = pd.cut(training_data_NB['age'], bins=bins_age, labels=[i for i in range(len(bins_age)-1)])
training_data_NB['education-num'] = pd.cut(training_data_NB['education-num'], bins=bins_edunum, labels=[i for i in range(len(bins_edunum)-1)])
training_data_NB['capital-gain'] = pd.cut(training_data_NB['capital-gain'], bins=bins_capital_gain, labels=[i for i in range(len(bins_capital_gain)-1)])
training_data_NB['capital-loss'] = pd.cut(training_data_NB['capital-loss'], bins=bins_capital_loss, labels=[i for i in range(len(bins_capital_loss)-1)])
training_data_NB['hours-per-week'] = pd.cut(training_data_NB['hours-per-week'], bins=bins_hours, labels=[i for i in range(len(bins_hours)-1)])

test_data_NB['age'] = pd.cut(test_data_NB['age'], bins=bins_age, labels=[i for i in range(len(bins_age)-1)])
test_data_NB['education-num'] = pd.cut(test_data_NB['education-num'], bins=bins_edunum, labels=[i for i in range(len(bins_edunum)-1)])
test_data_NB['capital-gain'] = pd.cut(test_data_NB['capital-gain'], bins=bins_capital_gain, labels=[i for i in range(len(bins_capital_gain)-1)])
test_data_NB['capital-loss'] = pd.cut(test_data_NB['capital-loss'], bins=bins_capital_loss, labels=[i for i in range(len(bins_capital_loss)-1)])
test_data_NB['hours-per-week'] = pd.cut(test_data_NB['hours-per-week'], bins=bins_hours, labels=[i for i in range(len(bins_hours)-1)])
#=====================================================================================
# this part is for QDA
training_data_QDA = training_data_NB.copy()
test_data_QDA = test_data_NB.copy()
del training_data_QDA['workclass']
del training_data_QDA['education']
del training_data_QDA['marital-status']
del training_data_QDA['occupation']
del training_data_QDA['relationship']
del training_data_QDA['sex']
del training_data_QDA['native-country']
#=====================================================================================
train_label = training_data['class']
test_label = test_data['class']

#del test_data['class']
#del training_data['class']


naive_bayes = GaussianNB()
naive_bayes.fit(training_data_NB.fillna(0), train_label)
#nb_prediction = naive_bayes.predict(test_data_NB)
nb_prediction = naive_bayes.predict(training_data_NB.fillna(0))
#print(classification_report(train_label, nb_prediction))

rf = RandomForestClassifier(n_estimators=50)
rf.fit(training_data_NB.fillna(0), train_label)
#nb_prediction = naive_bayes.predict(test_data_NB)
rf_prediction = rf.predict(training_data_NB.fillna(0))
#print(classification_report(train_label, rf_prediction))

#qda = QuadraticDiscriminantAnalysis()
#qda.fit(training_data_QDA, train_label)
#qda_prediction = qda.predict(training_data_QDA)
#print(classification_report(train_label, qda_prediction))

kNN = KNeighborsClassifier(n_neighbors=5)
kNN.fit(training_data_one_hot_encode, train_label)
#kNN_prediction = kNN.predict(test_data_one_hot_encode)
kNN_prediction = kNN.predict(training_data_one_hot_encode)
#print(classification_report(train_label, kNN_prediction))

# to map shuffled minibatch to prediction result

#training_data.insert(0, 'id', range(len(training_data)))
#ids = train_data_pruning['id']

confidence_nb = []
confidence_kNN = []
confidence_rf = []

nb_prediction_proba = naive_bayes.predict_proba(training_data_NB.fillna(0))
rf_prediction_proba = rf.predict_proba(training_data_NB.fillna(0))
kNN_prediction_proba = kNN.predict_proba(training_data_one_hot_encode)
for proba in nb_prediction_proba:
    confidence_nb.append(float(abs(proba[0] - proba[1])))
for proba in kNN_prediction_proba:
    confidence_kNN.append(float(abs(proba[0] - proba[1])))
for proba in kNN_prediction_proba:
    confidence_rf.append(float(abs(proba[0] - proba[1])))


def entropy_loss(x): # x =  threshold vector
    loss = 0
    threshold_nb = x[0]
    threshold_kNN = x[1]
    threshold_rf = x[2]
    for i in range(len(nb_prediction_proba)):
        conf_nb = confidence_nb[i]
        conf_kNN = confidence_kNN[i]
        conf_rf = confidence_rf[i]
        nb_proba_label_0 = nb_prediction_proba[i][0]
        nb_proba_label_1 = nb_prediction_proba[i][1]
        kNN_proba_label_0 = kNN_prediction_proba[i][0]
        kNN_proba_label_1 = kNN_prediction_proba[i][1]
        
        rf_proba_label_0 = rf_prediction_proba[i][0]
        rf_proba_label_1 = rf_prediction_proba[i][1]
        classification_combination = [max(0, conf_nb - x[0])*nb_proba_label_0 + max(0, conf_kNN - x[1])*kNN_proba_label_0 + max(0, conf_rf - x[2])*rf_proba_label_0,
                                      max(0, conf_nb - x[0])*nb_proba_label_1 + max(0, conf_kNN - x[1])*kNN_proba_label_1 + max(0, conf_rf - x[2])*rf_proba_label_1]
#        classification_combination = [max(0, conf_nb - x[0])*nb_proba_label_0 + max(0, conf_kNN - x[1])*kNN_proba_label_0,
#                                      max(0, conf_nb - x[0])*nb_proba_label_1 + max(0, conf_kNN - x[1])*kNN_proba_label_1]
        classification_combination = softmax(classification_combination)
        accurate_label = list(train_label)[i]
        loss -= math.log(classification_combination[accurate_label])
    print(x)
    loss = loss/len(nb_prediction_proba)
    print(loss)
    return loss

x0 = [0.5, 0.5, 0.5]
b = (0, 1)
bounds = (b, b, b)
    
sol = minimize(entropy_loss, x0, method = 'L-BFGS-B', bounds=bounds)
print(sol)
#
threshold = sol.x
#threshold = x0
# output ensemble result
nb_test_proba = naive_bayes.predict_proba(test_data_NB.fillna(0))
rf_test_proba = rf.predict_proba(test_data_NB.fillna(0))
kNN_test_proba = kNN.predict_proba(test_data_one_hot_encode)
ensemble_prediction = []
for i in range(len(nb_test_proba)):
    conf_nb = float(abs(nb_test_proba[i][0] - nb_test_proba[i][1]))
    conf_rf = float(abs(rf_test_proba[i][0] - rf_test_proba[i][1]))
    conf_kNN = float(abs(kNN_test_proba[i][0] - kNN_test_proba[i][1]))
    
    nb_proba_label_0 = nb_test_proba[i][0]
    nb_proba_label_1 = nb_test_proba[i][1]
    
    kNN_proba_label_0 = kNN_test_proba[i][0]
    kNN_proba_label_1 = kNN_test_proba[i][1]
    
    rf_proba_label_0 = rf_test_proba[i][0]
    rf_proba_label_1 = rf_test_proba[i][1]
    classification_combination = [max(0, conf_nb - threshold[0])*nb_proba_label_0 + max(0, conf_kNN - threshold[1])*kNN_proba_label_0 + max(0, conf_rf - threshold[2])*rf_proba_label_0,
                                      max(0, conf_nb - threshold[0])*nb_proba_label_1 + max(0, conf_kNN - threshold[1])*kNN_proba_label_1 + max(0, conf_rf - threshold[2])*rf_proba_label_1]
#    classification_combination = [max(0, conf_nb - threshold[0])*nb_proba_label_0 + max(0, conf_kNN - threshold[1])*kNN_proba_label_0,
#                                      max(0, conf_nb - threshold[0])*nb_proba_label_1 + max(0, conf_kNN - threshold[1])*kNN_proba_label_1]

    classification_combination = softmax(classification_combination)
    ensemble_prediction.append(np.argmax(classification_combination))

simple_averaging = []
for i in range(len(nb_test_proba)):
    nb_proba_label_0 = nb_test_proba[i][0]
    nb_proba_label_1 = nb_test_proba[i][1]
    
    kNN_proba_label_0 = kNN_test_proba[i][0]
    kNN_proba_label_1 = kNN_test_proba[i][1]
    
    rf_proba_label_0 = rf_test_proba[i][0]
    rf_proba_label_1 = rf_test_proba[i][1]
    classification_combination = [1/3*nb_proba_label_0 + 1/3*kNN_proba_label_0 + 1/3*rf_proba_label_0,
                                      1/3*nb_proba_label_1 + 1/3*kNN_proba_label_1 + 1/3*rf_proba_label_1]
#    classification_combination = [1/2*nb_proba_label_0 + 1/2*kNN_proba_label_0,
#                                      1/2*nb_proba_label_1 + 1/2*kNN_proba_label_1]

    simple_averaging.append(np.argmax(classification_combination))
    
print('New result - pruning ensemble')
print(classification_report(test_label, ensemble_prediction))

print('Simple averaging')
print(classification_report(test_label, simple_averaging))

nb_test = naive_bayes.predict(test_data_NB.fillna(0))
#rf_test= rf.predict(test_data_NB.fillna(0))
kNN_test= kNN.predict(test_data_one_hot_encode)
print('Old result')
print(classification_report(test_label, nb_test))
#print(classification_report(test_label, rf_test))
print(classification_report(test_label, kNN_test))

# compare accuracy on test set

