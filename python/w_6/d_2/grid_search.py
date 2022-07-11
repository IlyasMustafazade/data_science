import numpy as np
import pandas as pd
from sklearn import (ensemble, linear_model, metrics, model_selection,
                     neighbors, svm)

from funcs import *


def main():
    file_base = "hpt_small"
    file_ext = ".csv"
    file_name = file_base + file_ext
    f_space = pd.read_csv(file_name)
    f_space = pd.get_dummies(
        f_space, drop_first=True)
    f_space_columns = f_space.columns
    predictor_vector = f_space.iloc[:, :-1]
    dependent_feature = f_space.iloc[:, -1]
    predictor_vector_train, predictor_vector_test, \
        dependent_feature_train, dependent_feature_test = \
        model_selection.train_test_split(
            predictor_vector, dependent_feature, test_size=0.2)
    sv_cl = svm.SVC(random_state=1234)
    sv_cl_param_dict = {'C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
                        'kernel': ['rbf', 'linear'],
                        'gamma': [0.1, 0.25, 0.5, 1, 5]
                        }
    random_forest_cl = ensemble.RandomForestClassifier(
        random_state=1234)
    random_forest_cl_param_dict = {'n_estimators': [10, 15, 20, 100, 200],
                                   'min_samples_split': [8, 16],
                                   'min_samples_leaf': [1, 2, 3, 4, 5]
                                   }
    logistic_cl = linear_model.LogisticRegression(
        random_state=1234, max_iter=400)
    logistic_cl_param_dict = {'C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
                              'penalty': ["l1", "l2", "elasticnet"],
                              'solver': ['liblinear', 'lbfgs', 'saga']
                              }
    algo_arr = [logistic_cl,
                random_forest_cl, sv_cl]
    algo_arr = np.array(algo_arr)
    param_dict_arr = [logistic_cl_param_dict,
                      random_forest_cl_param_dict, sv_cl_param_dict]
    param_dict_arr = np.array(param_dict_arr)
    best_hparams_arr = get_best_params_multi_estimator(
        predictor_vector, dependent_feature, algo_arr, param_dict_arr)
    print("For logistic -> \n",
          best_hparams_arr[0])
    print("For random forest -> \n",
          best_hparams_arr[1])
    print("For SVC -> \n", best_hparams_arr[2])


if __name__ == "__main__":
    main()
