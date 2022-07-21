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
    random_forest_cl = ensemble.RandomForestClassifier(
        random_state=1234)
    random_forest_cl_param_dict = {'n_estimators': [10, 15, 20, 100, 200],
                                   'min_samples_split': [8, 16],
                                   'min_samples_leaf': [1, 2, 3, 4, 5]
                                   }
    print("For random forest -> \n",
          get_best_hparams(predictor_vector, dependent_feature,
                           random_forest_cl, random_forest_cl_param_dict, tuner_class="randomized"))


if __name__ == "__main__":
    main()
