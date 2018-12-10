#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import numpy as np
from sklearn.datasets import make_multilabel_classification 
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier


if __name__ == '__main__':
    y = np.load('../ProtsDataset/Train_y.npy')
    x = np.load('../ProtsDataset/Train_X.npy')

    xgb1 = LGBMClassifier(
            objective='binary',
            colsample_bytree=.8,
            subsample=.8
    )
    
    
    classifier = OneVsRestClassifier(xgb1)


    space = [Integer(5, 50, name='estimator__max_depth'),
            Real(1e-3, 1e0, "log-uniform", name='estimator__learning_rate'),
            Real(1e-10, 1e0, "log-uniform", name='estimator__reg_alpha'),     # L1 reg
            Real(1e-10, 1e0, "log-uniform", name='estimator__reg_lambda'),    # L2 reg
            Integer(2, 200, name='estimator__num_leaves'),           # Leaves in every tree
            Integer(50, 1000, name='estimator__n_estimators'),                  # Number of trees
    ]


    @use_named_args(space)
    def objective(**params):
        classifier.set_params(**params)

        return -np.mean(cross_val_score(classifier, x, y, cv=5, n_jobs=-1, scoring=make_scorer(f1_score, average='macro')))

    # for loop do try different space features
    res_gp = gp_minimize(objective, space, n_calls=50, random_state=0, verbose=1)
    
    print("Best score=%.4f" % res_gp.fun)

    print("Best Params:")
    print('estimator_max_depth', res_gp.x[0])
    print('learning_rate', res_gp.x[1])
    print('reg_alpha', res_gp.x[2])
    print('reg_lambda', res_gp.x[3])
    print('num_leaves', res_gp.x[4])
    print('n_estimators', res_gp.x[5])




    