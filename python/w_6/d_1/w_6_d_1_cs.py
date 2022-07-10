import catboost as catb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import (ensemble, linear_model, metrics, model_selection,
                     naive_bayes, preprocessing, svm)
from imblearn import over_sampling, combine


def main():

    f_space = pd.read_csv("data.csv")

    f_space = f_space.drop(labels=[" Net Income Flag", " Liability-Assets Flag"], axis=1)

    f_space_columns = f_space.columns

    for i in np.array(f_space.columns[1:]):

        if not (f_space[i] == 0).any():

            f_space = log_transform(f_space, i)

    f_space = normalize(f_space, np.array(f_space.columns[1:]))

    voting_style = "hard"

    voting_ensemble = ensemble.VotingClassifier(estimators=[
        ("logistic_cl", linear_model.LogisticRegression(penalty="l1", solver="liblinear")),
            ("naive_bayes_cl", naive_bayes.GaussianNB()),
                ("random_forest_cl", ensemble.RandomForestClassifier())],
                    voting=voting_style)

    bagging_ensemble_n_estimators = 10

    bagging_ensemble = ensemble.BaggingClassifier(base_estimator=svm.SVC(),
        n_estimators=bagging_ensemble_n_estimators)

    gboosting_ensemble_n_estimators = 3

    gboosting_ensemble_learning_rate = 1.0

    gboosting_ensemble = ensemble.GradientBoostingClassifier(
        n_estimators=gboosting_ensemble_n_estimators,
            learning_rate=gboosting_ensemble_learning_rate)

    xgboost_ensemble_n_estimators = 3
    
    xgboost_ensemble = xgb.XGBClassifier(
        n_estimators=xgboost_ensemble_n_estimators,
            use_label_encoder=False, eval_metric="logloss")

    lightgboost_ensemble_n_estimators = 3

    lightgboost_ensemble = lgb.LGBMClassifier(n_estimators=lightgboost_ensemble_n_estimators)

    catboost_ensemble_iterations = 10

    catboost_ensemble_learning_rate = 1.0

    catboost_ensemble_depth = 1

    catboost_ensemble = catb.CatBoostClassifier(
        iterations=catboost_ensemble_iterations,
            learning_rate=catboost_ensemble_learning_rate,
                depth=catboost_ensemble_depth, verbose=False)

    ensemble_arr = np.array([linear_model.LogisticRegression(penalty="l1", solver="liblinear"), 
        voting_ensemble, bagging_ensemble,
        gboosting_ensemble, xgboost_ensemble, lightgboost_ensemble, 
            catboost_ensemble])

    columns_without_first = f_space_columns[1:]

    predictor_arr = columns_without_first

    first_col_name = f_space_columns[0]
    
    predictor_arr = np.array(predictor_arr)

    outcome_feature = first_col_name

    for ensemble_algo in ensemble_arr:

        train_eval_model(ensemble_algo, f_space, predictor_arr, outcome_feature)


def log_transform(f_space, col_arr, method=np.log10):

    copy_f_space = f_space

    for i in col_arr:

        elem = copy_f_space[i]

        copy_f_space[i] = method(elem)

    return copy_f_space


def train_eval_model(algo, data, predictor_arr, outcome_feature):

    algo_name = algo.__class__.__name__

    tuple_predictor_arr = tuple(predictor_arr)

    predictor_data = data[[*tuple_predictor_arr]]

    outcome_data = data[outcome_feature]

    algo.fit(predictor_data, outcome_data)

    prediction = algo.predict(predictor_data)

    print("\nAlgorithm name -> ", algo_name)

    print("\nPerformance metrics on training data ->")

    get_metrics(outcome_data, prediction, show_metrics=True)

    stratified_k_fold = model_selection.StratifiedKFold(n_splits=2)

    sum_accuracy = 0

    sum_f1 = 0

    sum_f2 = 0


    sum_confusion = [
                        [0, 0],
                        [0, 0]
                    ]
                
    n_predictions = 0

    for train_index_arr, test_index_arr in stratified_k_fold.split(
        predictor_data, outcome_data):

        predictor_train_set, predictor_test_set = \
            predictor_data.iloc[train_index_arr], predictor_data.iloc[test_index_arr]
    
        outcome_train_set, outcome_test_set = \
            outcome_data.iloc[train_index_arr], outcome_data.iloc[test_index_arr]
        
        algo.fit(predictor_train_set, outcome_train_set)

        prediction = algo.predict(predictor_test_set)

        accuracy, f_1_score, f_2_score, conf_matrix = get_metrics(
            outcome_test_set, prediction)[:4]
        
        sum_accuracy += accuracy

        sum_f1 += f_1_score

        sum_f2 += f_2_score

        sum_confusion += conf_matrix

        n_predictions += 1

    avg_accuracy = sum_accuracy / n_predictions

    avg_f1 = sum_f1 / n_predictions

    avg_f2 = sum_f2 / n_predictions

    avg_confusion = sum_confusion / n_predictions

    print("\nAverage cross-validation accuracy score -> ", avg_accuracy)

    print("\nAverage cross-validation F1 score -> ", avg_f1)

    print("\nAverage cross-validation F2 score -> ", avg_f2)

    print("\nAverage cross-validation confusion matrix -> \n\n", avg_confusion)

    return (algo_name, avg_accuracy, avg_f1)


def get_metrics(outcome_data, prediction, show_metrics=False):

    accuracy = metrics.accuracy_score(outcome_data, prediction)

    f_1_score = metrics.f1_score(outcome_data, prediction, zero_division=1)

    f_2_score = metrics.fbeta_score(outcome_data, prediction, beta=2, zero_division=1)

    conf_matrix = metrics.confusion_matrix(outcome_data, prediction)

    cl_report = metrics.classification_report(outcome_data, prediction, zero_division=1)

    if show_metrics:

        print("\nAccuracy score -> ", accuracy)

        print("\nF1 score -> ", f_1_score)

        print("\nF2 score -> ", f_2_score)

        print("\nConfusion matrix -> \n\n", conf_matrix)

        print("\nClassification report -> \n\n", cl_report)

    as_tuple = (accuracy, f_1_score, f_2_score, conf_matrix, cl_report)

    return as_tuple


def raw_info(f_space):

    print("\nRaw data information -> \n")

    f_space.info()


def normalize(f_space, col_arr):

    copy_f_space = f_space

    for i in col_arr:

        elem = f_space[i]

        copy_f_space[i] = (elem - elem.mean()) / elem.std()
    
    return copy_f_space


def write_xlsx(f_space, file_name):

    new_file_name = "cleaned_" + file_name + ".xlsx"

    with open(new_file_name, "wb") as f:

        f_space.to_excel(f)


if __name__ == "__main__":

    main()
