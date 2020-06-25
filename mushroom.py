import pandas as pd
import numpy as np 
import math
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from scipy.special import softmax
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data_path = 'D:/University/Senior year/Doan3/mushroom/agaricus-lepiota.data'
data_all = pd.read_csv(data_path)
data_all = data_all[data_all['stalk-root']!= '?']
label_encoder = preprocessing.LabelEncoder()
data_all['label'] = label_encoder.fit_transform(data_all['label'])

label = data_all['label']
del data_all['label']
data_all = pd.get_dummies(data_all)

data_all['label'] = label

train, test = train_test_split(data_all, test_size=0.99)

label_train = train['label']
label_test = test['label']

del train['label']
del test['label']

rf = RandomForestClassifier(n_estimators=2)
rf.fit(train, label_train)
rf_prediction = rf.predict(test)
print(classification_report(label_test, rf_prediction))

kNN = KNeighborsClassifier(n_neighbors=20)
kNN.fit(train, label_train)
kNN_prediction = kNN.predict(test)
print(classification_report(label_test, kNN_prediction))

confidence_kNN = []
confidence_rf = []

rf_prediction_proba = rf.predict_proba(train)
kNN_prediction_proba = kNN.predict_proba(train)

for proba in kNN_prediction_proba:
    confidence_kNN.append(float(abs(proba[0] - proba[1])))
for proba in kNN_prediction_proba:
    confidence_rf.append(float(abs(proba[0] - proba[1])))

def entropy_loss(x): # x =  threshold vector
    loss = 0
    for i in range(len(label_train)):
        conf_kNN = confidence_kNN[i]
        conf_rf = confidence_rf[i]
        kNN_proba_label_0 = kNN_prediction_proba[i][0]
        kNN_proba_label_1 = kNN_prediction_proba[i][1]
        
        rf_proba_label_0 = rf_prediction_proba[i][0]
        rf_proba_label_1 = rf_prediction_proba[i][1]
        
        classification_combination = [max(0, conf_kNN - x[0])*kNN_proba_label_0 + max(0, conf_rf - x[1])*rf_proba_label_0,
                                      max(0, conf_kNN - x[0])*kNN_proba_label_1 + max(0, conf_rf - x[1])*rf_proba_label_1]
        classification_combination = softmax(classification_combination)

        accurate_label = list(label_train)[i]
        loss -= math.log(classification_combination[accurate_label])
#    print(x)
    loss = loss/len(label_train)
#    print(loss)
    return loss
            

x0 = [0, 0.75]
b = (0, 1)
bounds = (b, b)
    

sol = minimize(entropy_loss, x0, method = 'L-BFGS-B', bounds=bounds, options = {'iprint':100})
print(sol)
threshold = sol.x

rf_test_proba = rf.predict_proba(test)
kNN_test_proba = kNN.predict_proba(test)
ensemble_prediction = []
for i in range(len(label_test)):
    conf_rf = float(abs(rf_test_proba[i][0] - rf_test_proba[i][1]))
    conf_kNN = float(abs(kNN_test_proba[i][0] - kNN_test_proba[i][1]))

    kNN_proba_label_0 = kNN_test_proba[i][0]
    kNN_proba_label_1 = kNN_test_proba[i][1]
    
    rf_proba_label_0 = rf_test_proba[i][0]
    rf_proba_label_1 = rf_test_proba[i][1]
    classification_combination = [max(0, conf_kNN - threshold[0])*kNN_proba_label_0 + max(0, conf_rf - threshold[1])*rf_proba_label_0,
                                  max(0, conf_kNN - threshold[0])*kNN_proba_label_1 + max(0, conf_rf - threshold[1])*rf_proba_label_1]

#    classification_combination = softmax(classification_combination)
    ensemble_prediction.append(np.argmax(classification_combination))
print('New result - pruning ensemble')
#print(len(ensemble_prediction))
print(classification_report(label_test, ensemble_prediction))

rf_test= rf.predict(test)
kNN_test= kNN.predict(test)
print('Old result')
print('Random Forest')
print(classification_report(label_test, rf_test))
print('kNN')
print(classification_report(label_test, kNN_test))
