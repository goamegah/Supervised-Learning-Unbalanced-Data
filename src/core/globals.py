from random import randint

import numpy as np

MODEL_HYPERPARAMETERS_DEF = \
    {
        "Logistic Regression": \
            {
                "penalty": "l2",
                "C": 1.,
                "solver": "lbfgs",
                "multi_class": "auto",
                "n_jobs": None,
                "max_iter": 100
            },
        "SVM":
            {
                "kernel": "linear",
                "C": 1.,
                "degree": 3
            },
        "DecisionTreeClassifier":
            {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.,
                "max_features": None,
                "random_state": None,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.,
                "class_weight": None,
                "ccp_alpha": 0.
            },
        "RandomForestClassifier":{
            "n_estimators":100,
            "criterion":"gini",
            "max_depth":None,
            "min_samples_split":2,
            "min_samples_leaf":1,
            "min_weight_fraction_leaf":0.0,
            "max_features":"sqrt",
            "max_leaf_nodes":None,
            "min_impurity_decrease":0.0,
            "bootstrap":True,
            "oob_score":False,
            "n_jobs":None,
            "random_state":None,
            "verbose":0,
            "warm_start":False,
            "class_weight":None,
            "ccp_alpha":0.0,
            "max_samples":None

        }
    }

PARAMS_GRID = \
    {
        "Logistic Regression": \
            {
                "penalty": ["l1", "l2"],
                "C": np.logspace(-3, 3, 7),
                "solver": ["liblinear"],
                "multi_class": ["auto"],
                "n_jobs": [None],
                "max_iter": [100,150]
            },
        "SVM":
            {
                "kernel": ["poly", "rbf", "sigmoid"],
                "C": [0.1, 1, 10, 100],
                "degree": [3],
                "gamma": [1, 0.1, 0.01, 0.001]

            },
        "DecisionTreeClassifier":
            {
                "criterion": ["gini"],
                "splitter": ["best"],
                "max_depth": [None],
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [1],
                "min_weight_fraction_leaf": [0.],
                "max_features": [None],
                "random_state": [None],
                "max_leaf_nodes": [None]+list(range(2, 100)),
                "min_impurity_decrease": [0.],
                "class_weight": [None],
                "ccp_alpha": [0.]
            },
        "RandomForestClassifier":
            {
                "n_estimators":list(np.random.randint(50,500,size=5)),
                "max_depth":list(np.random.randint(1,20,size=5))
            }

    }

