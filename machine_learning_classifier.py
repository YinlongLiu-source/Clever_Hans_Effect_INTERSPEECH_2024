import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from sklearn.ensemble import VotingClassifier
import xgboost  

data_set = pd.read_csv("35s_pca_5_xxx.csv",header='None') #pca(5)
data = data_set.values[:,:]
print(data.shape)
y = data[:,5]
x = data[:,:5]
print(y.shape)
print(x.shape)

print("==========================================")
print("GBDT(Gradient Boosting Decision Tree) Classifier ")   
from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier(n_estimators=200)

print("==========================================")
print('AdaBoost Classifier')  
from sklearn.ensemble import  AdaBoostClassifier
AdaBoost = AdaBoostClassifier()

print("==========================================")   
print('xgboost')  
from sklearn.naive_bayes import MultinomialNB
XGBoost= xgboost.XGBClassifier()

print("==========================================")     
print('voting_classify')  
clf1 = GradientBoostingClassifier(n_estimators=200)
clf2 = AdaBoostClassifier()
clf3 = xgboost.XGBClassifier()
clf_multi_voting = VotingClassifier(estimators=[
    ('gbdt',clf1),
    ('AdaBoost',clf2),
    ('xgboost',clf3),
    ],
    voting='soft')

model_dict={
            'GBDT':GBDT,':AdaBoost':AdaBoost,'XGBoost':XGBoost,'voting_classify':clf_multi_voting}
for k in model_dict.keys():
        print(k)   
        # accuracy
        score = ms.cross_val_score(model_dict[k], x, y, cv=5, scoring='accuracy')
        print('accuracy score=', score)
        print('accuracy mean=', score.mean())
        # f1
        score = ms.cross_val_score(model_dict[k], x, y, cv=5, scoring='f1_weighted')
        print('f1_weighted score=', score)
        print('f1_weighted mean=', score.mean())
        # precision
        score = ms.cross_val_score(model_dict[k], x, y, cv=5, scoring='precision_weighted')
        print('precision_weighted score=', score)
        print('precision_weighted mean=', score.mean())
        # recall
        score = ms.cross_val_score(model_dict[k], x, y, cv=5, scoring='recall_weighted')
        print('recall_weighted score=', score)
        print('recall_weighted mean=', score.mean())
