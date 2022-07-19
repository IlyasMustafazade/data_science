import sys
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from PIL import Image
from category_encoders import BinaryEncoder, OneHotEncoder

sys.path.append("/Users/YTU/source/repos/ai/ai_helper")
from ui_helper import run_page, load_dataset    
from matrix_helper import extract_X_y_df


def main():
    dsa_logo = Image.open("dsa_logo.png")
    web_app = WebApp()

    st.sidebar.image(dsa_logo)
    web_app.chosen_page = st.sidebar.selectbox("", web_app.PAGE_TPL)

    run_page(chosen_page=web_app.chosen_page, page_func_dict=web_app.PAGE_FUNC_DICT)


class WebApp():
    def __init__(self):
        print("init called")
        self.PAGE_TPL = ("Homepage", "EDA", "Modeling")
        self.FUNC_TPL = (self.run_home, self.run_eda, self.run_modeling)
        len_page_tpl = len(self.PAGE_TPL)
        self.PAGE_FUNC_DICT = {self.PAGE_TPL[i]: self.FUNC_TPL[i] for i in range(len_page_tpl)}
        self.SET_TPL = ("Water potability", "Loan prediction")
        self.FILE_TPL = ("water_potability.csv", "loan_prediction.csv")
        self.DF_TPL = [pd.read_csv(f) for f in self.FILE_TPL]
        self.DF_TPL = tuple(self.DF_TPL)
        len_set_tpl = len(self.SET_TPL)
        self.SET_DF_DICT = {
            self.SET_TPL[i]: self.DF_TPL[i] for i in range(len_set_tpl)}
        self.SCALE_MODE_TPL = ("Standard", "Robust", "MinMax")
        self.ENCODE_MODE_TPL = ("Binary", "One-Hot")
        self.CL_NAME_TPL = ("XGBoost", "Naive Bayes",
                          "Logistic Regression", "Random Forest",
                          "Support Vector")
        self.CL_TPL = (XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                       GaussianNB(), LogisticRegression(), RandomForestClassifier(),
                       SVC())
        len_model_tpl = len(self.CL_TPL)
        self.NAME_CL_DICT = {self.CL_NAME_TPL[i]: self.CL_TPL[i] for i in range(len_model_tpl)}
        self.CAT_IMPUTE_MODE_TPL = ("Mode", "Backfill", "Forwardfill")
        self.NUM_IMPUTE_MODE_TPL = ("Mode", "Median")
        self.FEATURE_ENG_MODE_TPL = ("Under Sampling", "Clean Outlier")

        self.chosen_page = self.run_home
        self.chosen_set = self.SET_TPL[0]
        self.chosen_scale_mode = self.SCALE_MODE_TPL[0]
        self.chosen_encode_mode = self.ENCODE_MODE_TPL[0]
        self.chosen_cl = self.CL_NAME_TPL[0]
        self.chosen_cat_impute_mode = self.CAT_IMPUTE_MODE_TPL[0]
        self.chosen_num_impute_mode = self.NUM_IMPUTE_MODE_TPL[0]
        self.chosen_feature_eng_mode = self.FEATURE_ENG_MODE_TPL[0]
        self.chosen_random_state = 0
        self.chosen_test_size = 0.2

        self.df = None

    def run_home(self):
        self.chosen_set = self.make_home_ui()
        st.session_state.chosen_set = self.chosen_set

        self.df = self.make_df()

    def run_eda(self):
        self.chosen_set = st.session_state.chosen_set
        self.make_eda_ui()

    def run_modeling(self):
        self.chosen_set = st.session_state.chosen_set
        self.make_modeling_ui()

    def make_home_ui(self):
        ds_role = Image.open("ds_role.png")

        st.title("Homepage")
        st.image(ds_role)
        chosen_set = st.selectbox("Select dataset", self.SET_TPL)
        st.write("Selected: **" + chosen_set + "** dataset")
        return chosen_set

    def make_eda_ui(self):
        st.title("EDA")
        self.df = self.make_df()
        st.dataframe(self.df.head(100))

    def make_modeling_ui(self):
        st.title("Modeling")

    def make_df(self):
        df = self.SET_DF_DICT[self.chosen_set]
        pipe = (self.apply_feature_eng_mode, self.apply_num_impute_mode,
                self.apply_cat_impute_mode, self.apply_encode_mode, 
                self.apply_scale_mode)

        for process in pipe:
            df = process(df=df)
        return df

    def apply_feature_eng_mode(self, df=None):
        if self.chosen_feature_eng_mode == "Under Sampling":
            X, y = extract_X_y_df(df=df)
            sampler = RandomUnderSampler()
            X, y = sampler.fit_resample(X, y)
            new_df = pd.concat([X, y], axis=1)
        elif self.chosen_feature_eng_mode == "Clean Outlier":
            new_df = new_df.dropna(axis=0)
        return new_df
    
    def apply_num_impute_mode(self, df=None):
        num_df = df.select_dtypes(include="number")
        num_col_idx = num_df.columns
        num_col_tpl = tuple(num_col_idx)
        if len(num_col_tpl) < 1:
            return df
        if self.chosen_num_impute_mode == "Mode":
            imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        elif self.chosen_num_impute_mode == "Median":
            imputer = SimpleImputer(missing_values=np.nan, strategy="median")

        imputer.fit(num_df)
        matrix = imputer.transform(num_df)
        imputed_df = pd.DataFrame(data=matrix, columns=num_col_tpl)
        new_df = df
        new_df[num_col_idx] = imputed_df
        return new_df

    def apply_cat_impute_mode(self, df=None):
        cat_df = df.select_dtypes(include=object)
        cat_col_idx = cat_df.columns
        cat_col_tpl = tuple(cat_col_idx)
        if len(cat_col_tpl) < 1:
            return df
        if self.chosen_cat_impute_mode == "Mode":
            imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
            imputer.fit(cat_df)
            matrix = imputer.transform(cat_df)
            imputed_df = pd.DataFrame(data=matrix, columns=cat_col_tpl)
            new_df = df
            new_df[cat_col_idx] = imputed_df
        elif self.chosen_cat_impute_mode == "Backfill":
            new_df = df.bfill()
        elif self.chosen_cat_impute_mode == "Forwardfill":
            new_df = df.ffill()
        return new_df

    def apply_encode_mode(self, df=None):
        name_encoder_lst = [("Binary", BinaryEncoder()),
                            ("One-Hot", OneHotEncoder())]
        for name_encoder in name_encoder_lst:
            name, encoder = name_encoder
            if self.chosen_scale_mode == name:
                chosen_encoder = encoder
        encoder.fit(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = encoder.transform(df)
        column_arr = encoder.get_feature_names()
        new_df = pd.DataFrame(data=matrix, columns=column_arr)
        return new_df

    def apply_scale_mode(self, df=None):
        name_scaler_lst = [("Standard", StandardScaler()),
                            ("Robust", RobustScaler()),
                            ("MinMax", MinMaxScaler())]
        for name_scaler in name_scaler_lst:
            name, scaler = name_scaler
            if self.chosen_scale_mode == name:
                chosen_scaler = scaler
        scaler.fit(df)
        matrix = scaler.transform(df)
        column_arr = scaler.get_feature_names_out()
        new_df = pd.DataFrame(data=matrix, columns=column_arr)
        return new_df
 

if __name__ == "__main__":
    main()
