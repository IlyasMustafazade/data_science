import math as math
import warnings as warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skopt as skopt
from scipy import stats
from sklearn import metrics, model_selection, preprocessing


def bayesian_optimize(X, y, objective_function, model):
    for i in range(100):
        x = optimize_acquisition(X, y, model)
        objective_func_vals = objective_function(
            x, noise=0.1)
        estimated = surrogate(model, [[x]])[0]
        print('>x={0}, f()={1}, actual={2}'.format
              (x, estimated[0][0], objective_func_vals))
        X = np.vstack((X, [[x]]))
        y = np.vstack(
            (y, [[objective_func_vals]]))
        model.fit(X, y)

    plot_real_surrogate(X, y, model)
    best_indices = np.argmax(y)
    print("Best Result: x={0}, y={1}".format
          (X[best_indices], y[best_indices]))


def acquisition(X, X_samples, model):
    surrogate_prediction = surrogate(model, X)
    y_hat = surrogate_prediction[0]
    max_y_hat = max(y_hat)
    mean, std = surrogate(model, X_samples)
    mean = mean[:, 0]
    prob = stats.norm.cdf(
        (mean - max_y_hat) / std+1E-9)
    return prob


def optimize_acquisition(X, y, model):
    X_samples = np.random.random(100)
    X_samples = np.reshape(X_samples, (-1, 1))
    scores = acquisition(X, X_samples, model)
    max_indices = np.argmax(scores)
    return X_samples[max_indices, 0]


def plot_real_surrogate(X, y, model):
    plt.scatter(X, y)
    X_samples = np.asarray(np.arange(0, 1, 0.001))
    X_samples = np.reshape(X_samples, (-1, 1))
    y_samples = surrogate(model, X_samples)[0]
    plt.plot(X_samples, y_samples)


def surrogate(model, X):
    with warn.catch_warnings():
        warn.simplefilter("ignore")
        return model.predict(X, return_std=True)


def get_best_params_multi_estimator(predictor_vector=None, outcome_feature=None,
                                    algo_arr=None, param_dict_arr=None, tuner_class="grid", cv=10, n_iter=10):
    best_hparams_arr = []
    for i in range(len(algo_arr)):
        algo = algo_arr[i]
        param_dict = param_dict_arr[i]
        best_hparams = get_best_hparams(predictor_vector, outcome_feature,
                                        algo, param_dict, tuner_class=tuner_class, cv=cv, n_iter=n_iter)
        best_hparams_arr.append(best_hparams)
    return best_hparams_arr


def get_best_hparams(predictor_vector=None, outcome_feature=None,
                     algo=None, param_dict=None, tuner_class="grid", cv=10, n_iter=10, verbose=True):
    if tuner_class == "grid":
        tuner = model_selection.GridSearchCV(estimator=algo,
                                             param_grid=param_dict, scoring="accuracy", cv=cv, n_jobs=-1,
                                             return_train_score=True)
    elif tuner_class == "randomized":
        tuner = model_selection.RandomizedSearchCV(estimator=algo,
                                                   param_distributions=param_dict,
                                                   scoring='accuracy',
                                                   cv=cv,
                                                   n_iter=n_iter,
                                                   return_train_score=True)
    elif tuner_class == "bayesian":
        tuner = skopt.BayesSearchCV(estimator=algo, search_spaces=param_dict, scoring="accuracy",
                                    cv=cv, n_jobs=-1, return_train_score=True)
    tuner.fit(
        predictor_vector, outcome_feature)
    best_hparams = tuner.best_params_
    algo_name = algo.__class__.__name__
    if verbose is True:
        print("\nBest parameters for ", algo_name,
              " according to ",  tuner_class, " tuner are -> \n", best_hparams)
        print("\nCorresponding score -> ",
              tuner.best_score_)
        results_as_df = pd.DataFrame(
            tuner.cv_results_)
        print(
            "\nResults of best scorers -> \n", results_as_df[results_as_df['rank_test_score'] == 1])
    return best_hparams


def train_eval_model(model, data, predictor_arr, outcome_feature, n_splits=5, verbose=True):
    model_name = model.__class__.__name__
    tuple_predictor_arr = tuple(predictor_arr)
    predictor_data = data[[*tuple_predictor_arr]]
    outcome_data = data[outcome_feature]
    model.fit(predictor_data, outcome_data)
    prediction = model.predict(predictor_data)
    if verbose:
        print("\nModel name -> ", model_name)
        hyperparameter_dict = model.get_params()
        print("\nModel hyperparameters -> \n\n",
              hyperparameter_dict)
        print(
            "\nPerformance metrics on training data ->")
    get_metrics(
        outcome_data, prediction, show_metrics=verbose)
    stratified_k_fold = model_selection.StratifiedKFold(
        n_splits=n_splits)
    sum_accuracy = 0
    sum_f1 = 0
    sum_f2 = 0
    sum_confusion = [[0, 0], [0, 0]]
    n_predictions = 0
    for train_index_arr, test_index_arr in stratified_k_fold.split(
        predictor_data, outcome_data
    ):
        predictor_train_set, predictor_test_set = (
            predictor_data.iloc[train_index_arr],
            predictor_data.iloc[test_index_arr],
        )
        outcome_train_set, outcome_test_set = (
            outcome_data.iloc[train_index_arr],
            outcome_data.iloc[test_index_arr],
        )
        model.fit(predictor_train_set,
                  outcome_train_set)
        prediction = model.predict(
            predictor_test_set)
        accuracy, f_1_score, f_2_score, conf_matrix = get_metrics(
            outcome_test_set, prediction
        )[:4]
        sum_accuracy += accuracy
        sum_f1 += f_1_score
        sum_f2 += f_2_score
        sum_confusion += conf_matrix
        n_predictions += 1
    avg_accuracy = sum_accuracy / n_predictions
    avg_f1 = sum_f1 / n_predictions
    avg_f2 = sum_f2 / n_predictions
    avg_confusion = sum_confusion / n_predictions
    if verbose:
        print(
            "\nAverage cross-validation accuracy score -> ", avg_accuracy)
        print(
            "\nAverage cross-validation F1 score -> ", avg_f1)
        print(
            "\nAverage cross-validation F2 score -> ", avg_f2)
        print(
            "\nAverage cross-validation confusion matrix -> \n\n", avg_confusion)
    as_tuple = (model_name, avg_accuracy,
                avg_f1, avg_f2, avg_confusion)
    return as_tuple


def get_metrics(outcome_data, prediction, show_metrics=False):
    accuracy = metrics.accuracy_score(
        outcome_data, prediction)
    f_1_score = metrics.f1_score(
        outcome_data, prediction, zero_division=1)
    f_2_score = metrics.fbeta_score(
        outcome_data, prediction, beta=2, zero_division=1)
    conf_matrix = metrics.confusion_matrix(
        outcome_data, prediction)
    cl_report = metrics.classification_report(
        outcome_data, prediction, zero_division=1)
    if show_metrics:
        print("\nAccuracy score -> ", accuracy)
        print("\nF1 score -> ", f_1_score)
        print("\nF2 score -> ", f_2_score)
        print("\nConfusion matrix -> \n\n",
              conf_matrix)
        print(
            "\nClassification report -> \n\n", cl_report)
    as_tuple = (accuracy, f_1_score,
                f_2_score, conf_matrix, cl_report)
    return as_tuple


def encode_col(f_space=None, encode_dict=None, col_arr=None, encoder_class=None):
    col_arr_ = []
    if (encode_dict is None) and (col_arr is None) and (encoder_class is None):
        return f_space
    elif not (encode_dict is None):
        if not ((col_arr is None) and (encoder_class is None)):
            raise Exception(
                "When encode_dict is not None, col_arr and encoder_class both must be None"
            )
        col_arr_ = list(encode_dict.keys())
        col_arr_ = np.array(col_arr_)
        encoder_arr = list(encode_dict.values())
        encoder_arr = np.array(encoder_arr)
    elif encode_dict is None:
        if (col_arr is None) or (encoder_class is None):
            raise Exception(
                "When encode_dict is None, col_arr and encoder_class both must be non-None"
            )
        col_arr_ = np.array(col_arr)
        len_col_arr = len(col_arr_)
        encoder_arr = [
            encoder_class] * len_col_arr
        encoder_arr = np.array(encoder_arr)
    copy_f_space = f_space
    for i in range(len(col_arr_)):
        name = col_arr_[i]
        elem = copy_f_space[name]
        elem = np.array(elem)
        class_ = encoder_arr[i]
        encoder = class_()
        encoder.fit(elem)
        transformed = encoder.transform(elem)
        copy_f_space = col_to_df(
            copy_f_space, name, transformed)
    return copy_f_space


def delete_numeric_outliers(f_space, col_arr, stddev_limit=3):
    remove_indices = []
    for column_name in f_space:
        bool_arr = np.abs(stats.zscore(
            f_space[column_name])) > stddev_limit
        bool_arr = np.array(bool_arr)
        for j in range(len(bool_arr)):
            if bool_arr[j]:
                remove_indices.append(j)
    remove_indices = list(set(remove_indices))
    remove_indices = np.array(remove_indices)
    return f_space.drop(f_space.index[remove_indices])


def col_to_df(df, name, sub_df):
    new_df = df
    df_columns = df.columns
    sub_df_shape = sub_df.shape
    sub_df_n_col = sub_df_shape[0]
    if sub_df_n_col == 1:
        elem_series = sub_df.iloc[:, 0]
        new_df[name] = elem_series
        return new_df
    elif sub_df_n_col > 1:
        clone_index = 0
        drop_index = df_columns.get_loc(name)
        new_df = new_df.drop(labels=name, axis=1)
        insert_index = drop_index
        sub_df_columns = sub_df.columns
        for i in sub_df_columns:
            elem_series = sub_df[i]
            elem_array = np.array(elem_series)
            new_col_name = name + \
                "_" + str(clone_index)
            new_df.insert(
                insert_index, new_col_name, elem_array)
            insert_index += 1
            clone_index += 1
    else:
        raise Exception(
            "Number of columns of sub_df must be more than 0")
    return new_df


def vis_missing(f_space=None):
    plt.figure(num="Missing values heatmap")
    sns.heatmap(f_space.isna())
    missing_val_series = f_space.isna().sum()
    n_row = f_space.shape[0]
    print("\nTotal percentage of missing values -> ",
          missing_val_series.sum() / n_row * 100)
    print(
        "\nCounts of missing values per columns ->\n")
    print(missing_val_series)
    print(
        "\nPercentages of missing values per columns ->\n")
    print(missing_val_series / n_row * 100)


def std_fillna(f_space):
    na_dict = {}
    f_space_columns = f_space.columns
    for i in f_space_columns:
        elem = f_space[i]
        if elem.isna().any():
            if elem.dtype != object and len(elem.unique()) > 3:
                na_dict[i] = elem.mean()
            else:
                na_dict[i] = elem.mode()[0]
    return f_space.fillna(na_dict)


def small_increment(arr, increment_val=1):
    return arr + increment_val


def remove_rows_with_val(f_space=None, col_arr=None, val=None):
    copy_f_space = f_space
    for column_name in col_arr:
        boolean_mask = copy_f_space[column_name] != val
        copy_f_space = copy_f_space[boolean_mask]
    return copy_f_space


def merge_columns(f_space, column_pair_arr, new_name, method=np.sum):
    copy_f_space = f_space
    first = column_pair_arr[0]
    second = column_pair_arr[1]
    copy_f_space[first] += copy_f_space[second]
    copy_f_space = copy_f_space.rename(
        columns={first: new_name})
    copy_f_space = copy_f_space.drop(
        labels=second, axis=1)
    return copy_f_space


def apply_transform(f_space, col_arr, method=np.log10):
    copy_f_space = f_space
    for i in col_arr:
        elem = copy_f_space[i]
        copy_f_space[i] = method(elem)
    return copy_f_space


def corr_heatmap(f_space):
    plt.figure(num="Correlation matrix heatmap")
    corr_matrix = f_space.corr()
    sns.heatmap(corr_matrix, annot=True)


def encode_binary_col(f_space):
    copy_f_space = f_space
    f_space_columns = f_space.columns
    category_encoder = preprocessing.LabelEncoder()
    for i in f_space_columns:
        elem = copy_f_space[i]
        count_lim = 3 if elem.isna().any() else 2
        if elem.dtype == object:
            if len(elem.unique()) == count_lim:
                copy_f_space[i] = category_encoder.fit_transform(
                    np.array(elem))
    return copy_f_space


def normalize(f_space=None, col_arr=None):
    copy_f_space = f_space
    for i in col_arr:
        elem = f_space[i]
        copy_f_space[i] = (
            elem - elem.mean()) / elem.std()
    return copy_f_space


def plot_col_freq(f_space):
    f_space_values = np.array(f_space.values)
    f_space_values_shape = np.array(
        f_space_values.shape)
    col_arr = np.reshape(
        f_space_values, f_space_values_shape)
    col_arr = col_arr.T
    last_col = col_arr[-1]
    col_arr = col_arr[:-1]
    col_names = f_space.columns
    last_name = col_names[-1]
    plt.figure(num=last_name,
               figsize=[12.8, 12.8])
    sns.histplot(y=last_name, data=f_space)
    for index, elem in enumerate(col_arr):
        col_name = col_names[index]
        col = f_space[col_name]
        plt.figure(
            num=col_name, figsize=[12.8, 12.8])
        sns.histplot(y=col_name, hue=last_name,
                     multiple="stack", data=f_space, kde=True)


def write_xlsx(f_space, file_name):
    new_file_name = "cleaned_" + file_name + ".xlsx"
    with open(new_file_name, "wb") as f:
        f_space.to_excel(f)


def raw_info(f_space):
    print("\nRaw data information -> \n")
    f_space.info()
