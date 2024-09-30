"""
EnviFormer a transformer based method for the prediction of biodegradation products and pathways
Copyright (C) 2024  Liam Brydon
Contact at: lbry121@aucklanduni.ac.nz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import copy
from sklearn.dummy import DummyClassifier
import numpy as np
from joblib import Parallel, delayed


# Binary Relevance
#  -> filters out missing values for training; this is feasible because the labels are 
#  treated separately.
class BinaryRelevance:
    def __init__(self, baseline_clf, seed=1111):
        self.clf = baseline_clf
        self.seed = seed

    def fit(self, X, Y):
        self.classifiers = []
        for l in range(len(Y[0])):
            X_l = X[~np.isnan(Y[:, l])]
            Y_l = (Y[~np.isnan(Y[:, l]), l])
            if len(X_l) == 0:  # all labels are nan -> predict 0
                clf = DummyClassifier(strategy='constant', constant=0)
                clf.fit([X[0]], [0])
                self.classifiers.append(clf)
                continue
            elif len(np.unique(Y_l)) == 1:  # only one class -> predict that class
                clf = DummyClassifier(strategy='most_frequent')
            else:
                clf = copy.deepcopy(self.clf)
            clf.fit(X_l, Y_l)
            self.classifiers.append(clf)

    def predict(self, X):
        labels = []
        for clf in self.classifiers:
            labels.append(clf.predict(X))
        return np.column_stack(labels)

    def predict_proba(self, X):
        labels = np.empty((len(X), 0))
        for clf in self.classifiers:
            pred = clf.predict_proba(X)
            if pred.shape[1] > 1:
                pred = pred[:, 1]
            #pred = pred[:,int(clf.predict([X[0]])[0] == np.argmax(pred[0]))]
            else:
                pred = pred * clf.predict([X[0]])[0]
            labels = np.column_stack((labels, pred))
        return labels


# Classifier Chains
#  -> During training, train on existing values und use the trained classifier to predict 
#  the missing ones and move on.
class ClassifierChain:
    def __init__(self, baseline_clf, seed=1111):
        self.clf = baseline_clf
        self.seed = seed

    def fit(self, X, Y):
        self.perm = np.random.RandomState(seed=self.seed).permutation(len(Y[0]))
        Y = Y[:, self.perm]
        self.classifiers = []
        for p in range(len(self.perm)):
            X_p = X[~np.isnan(Y[:, p])]
            Y_p = Y[~np.isnan(Y[:, p]), p]
            if len(X_p) == 0:  # all labels are nan -> predict 0
                clf = DummyClassifier(strategy='constant', constant=0)
                self.classifiers.append(clf.fit([X[0]], [0]))
            elif len(np.unique(Y_p)) == 1:  # only one class -> predict that class
                clf = DummyClassifier(strategy='most_frequent')
                self.classifiers.append(clf.fit(X_p, Y_p))
            else:
                clf = copy.deepcopy(self.clf)
                self.classifiers.append(clf.fit(X_p, Y_p))
            newcol = Y[:, p]
            pred = clf.predict(X)
            newcol[np.isnan(newcol)] = pred[np.isnan(newcol)]  # fill in missing values with clf predictions
            X = np.column_stack((X, newcol))
        return self

    def predict(self, X):
        labels = np.empty((len(X), 0))
        for clf in self.classifiers:
            pred = clf.predict(np.column_stack((X, labels)))
            labels = np.column_stack((labels, pred))
        return labels[:, np.argsort(self.perm)]

    def predict_proba(self, X):
        labels = np.empty((len(X), 0))
        for clf in self.classifiers:
            pred = clf.predict_proba(np.column_stack((X, np.round(labels))))
            if pred.shape[1] > 1:
                pred = pred[:, 1]
            else:
                pred = pred * clf.predict(np.column_stack(([X[0]], np.round([labels[0]]))))[0]
            labels = np.column_stack((labels, pred))
        return labels[:, np.argsort(self.perm)]


# Ensemble of Classifier Chains
class EnsembleClassifierChain:
    def __init__(self, baseline_clf, num_classifiers=50, seed=1111, debug=False):
        self.clf = baseline_clf
        self.num = num_classifiers
        self.seed = seed
        self.num_jobs = 20 if not debug else 1

    def fit(self, X, Y):
        self.classifiers = [ClassifierChain(self.clf, self.seed + p) for p in range(self.num)]
        self.num_labels = len(Y[0])
        self.classifiers = Parallel(n_jobs=self.num_jobs, verbose=5)(delayed(clf.fit)(X, Y) for clf in self.classifiers)

    def predict(self, X):
        labels = np.zeros((len(X), self.num_labels))
        results = Parallel(n_jobs=self.num_jobs, verbose=5)(delayed(clf.predict_proba)(X) for clf in self.classifiers)
        for result in results:
            labels += result
        return np.round(labels / self.num)

    def predict_proba(self, X):
        labels = np.zeros((len(X), self.num_labels))
        results = Parallel(n_jobs=self.num_jobs, verbose=5)(delayed(clf.predict_proba)(X) for clf in self.classifiers)
        for result in results:
            labels += result
        return labels / self.num
