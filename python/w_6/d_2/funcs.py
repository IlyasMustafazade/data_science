import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics, model_selection, preprocessing


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
        print("\nPerformance metrics on training data ->")
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


def encode_col(f_space, encode_dict=None, col_arr=None, encoder_class=None):
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


def vis_missing(f_space):
    plt.figure(num="Missing values heatmap")
    sns.heatmap(f_space.isna())
    missing_val_series = f_space.isna().sum()
    n_row = f_space.shape[0]
    print("\nTotal percentage of missing values -> ",
          missing_val_series.sum() / n_row * 100)
    print("\nCounts of missing values per columns ->\n")
    print(missing_val_series)
    print("\nPercentages of missing values per columns ->\n")
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


def merge_columns(f_space, column_pair_arr, new_name, method=np.sum):
    copy_f_space = f_space
    first = column_pair_arr[0]
    second = column_pair_arr[1]
    copy_f_space[first] += copy_f_space[second]
    copy_f_space = copy_f_space.rename(columns={first: new_name})
    copy_f_space = copy_f_space.drop(labels=second, axis=1)
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


def normalize(f_space, col_arr):
    copy_f_space = f_space
    for i in col_arr:
        elem = f_space[i]
        copy_f_space[i] = (elem - elem.mean()) / elem.std()
    return copy_f_space


def plot_col_freq(f_space):
    f_space_values = np.array(f_space.values)
    f_space_values_shape = np.array(f_space_values.shape)
    col_arr = np.reshape(f_space_values, f_space_values_shape)
    col_arr = col_arr.T
    last_col = col_arr[-1]
    col_arr = col_arr[:-1]
    col_names = f_space.columns
    last_name = col_names[-1]
    plt.figure(num=last_name, figsize=[12.8, 12.8])
    sns.histplot(y=last_name, data=f_space)
    for index, elem in enumerate(col_arr):
        col_name = col_names[index]
        col = f_space[col_name]
        plt.figure(num=col_name, figsize=[12.8, 12.8])
        sns.histplot(y=col_name, hue=last_name,
                     multiple="stack", data=f_space, kde=True)


def write_xlsx(f_space, file_name):
    new_file_name = "cleaned_" + file_name + ".xlsx"
    with open(new_file_name, "wb") as f:
        f_space.to_excel(f)


def raw_info(f_space):
    print("\nRaw data information -> \n")
    f_space.info()
