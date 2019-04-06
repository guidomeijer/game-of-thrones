# -*- coding: utf-8 -*-
"""
Train a random forest classifier to predict whether characters of Game of Thrones
will die based on a large dataset derived from A Song of Ice and Fire wiki
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np

# Load in data
got = pd.read_csv('character-predictions.csv')

# Select characters to predict
chars = ['Jon Snow', 'Sansa Stark', 'Arya Stark', 'Bran Stark', 'Cersei Lannister', 'Jaime Lannister', 
         'Tyrion Lannister', 'Daenerys Targaryen', 'Asha Greyjoy', 'Theon Greyjoy', 'Melisandre', 
         'Jorah Mormont', 'Sandor Clegane', 'Samwell Tarly', 'Gilly', 'Varys', 'Davos Seaworth',
         'Bronn', 'Podrick Payne', 'Tormund', 'Grey Worm', 'Missandei', 'Gendry', 'Beric Dondarrion',
         'Euron Greyjoy', 'Qyburn', 'Gregor Clegane']

# Select features to use in prediction
got_pred = got[['book1','book2','book3','book4','book5','male','isMarried','isNoble','numDeadRelations','isPopular','popularity']]
alive = got['isAlive']

# Perform n-fold cross-validated prediction to determine model accuracy 
n_fold = 5
ground_truth = []
pred = []
clf = RandomForestClassifier(n_estimators = 100)
kf = KFold(n_splits = n_fold, shuffle = True)
for train_index, test_index in kf.split(got_pred):
    clf.fit(got_pred.loc[train_index], alive[train_index])
    pred = np.append(pred, clf.predict(got_pred.loc[test_index]))
    ground_truth = np.append(ground_truth, alive[test_index])
print('Random Forest model F1 score: %s'%str(round(f1_score(ground_truth, pred),2)))

# Predict single characters using leave-one-out cross validation
got_result = pd.DataFrame(index=chars, columns=['predictedAlive'])
for char in chars:
    got_leave = got_pred.loc[got['name'] != char]
    alive_leave = alive.loc[got['name'] != char]
    clf.fit(got_leave, alive_leave)
    got_result.loc[char,'predictedAlive'] = clf.predict(got_pred.loc[got['name'] == char])
print(got_result)     


    




