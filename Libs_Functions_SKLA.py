# Tailored functions for skla app

# libraries

import streamlit as st
from streamlit_option_menu import option_menu 
from PIL import Image
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, classification_report,accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.svm import SVR, SVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import streamlit as st

# ------------- Settings --------------

# Cache
time_to_live_cache = 3600 # Cache data for 1 hour (=3600 seconds)


# ------------- Get data in and show it --------------

## Loading data with specific separator

@st.cache_data(ttl = time_to_live_cache)  # Add the caching decorator
def load_data(url, sep):
    df = pd.read_csv(url, sep)
    return df


## show col info

def show_info(df):
    colnames = []
    is_na = []
    is_not_na = []
    is_unique = []
    is_type = []
    for i in df.columns:
        colnames.append(i)
        is_type.append(df[i].dtype)
        n_na = df[i].isna().sum()
        is_na.append(n_na)
        is_not_na.append(len(df[i]) - n_na)
        is_unique.append(len(df[i].unique()))
    df_col_info = pd.DataFrame({'columns' : colnames, 'n_non_null': is_not_na, 'n_null': is_na, 'n_unique': is_unique, 'type': is_type})
    return df_col_info


# ------------- Data Preprocessinng --------------

## Split train test on test_size

@st.cache_data(ttl = time_to_live_cache)
def split_testsize(df, testsize):
    trainsize = (1-testsize)
    # reshuffel df
    df = df.sample(frac= 1)
    # first trainsize percent e.g. 0.8
    train_df = df[:int((len(df.index))*trainsize)]
    train_df = train_df.reset_index(drop = True)
    # rest for testing
    test_df = df[int((len(df.index))*trainsize):] 
    test_df = test_df.reset_index(drop = True)
    return train_df, test_df


## Split train test on daterange for time series analysis
# Splitting needs dataframe with datetime index

@st.cache_data(ttl = time_to_live_cache) 
def split_timeseries(df_ts, start_date, end_date): # us_test_size
   
    us_start_date = datetime.combine(us_start_date, datetime.min.time())
    us_end_date = datetime.combine(us_end_date, datetime.min.time())

    train_df_ts = df_ts.loc[df_ts.index < us_start_date]
    test_df_ts = df_ts.loc[(df_ts.index >= us_start_date) & (df_ts.index <= us_end_date)]
    
    train_ts_index = train_df_ts.index
    test_ts_index = test_df_ts.index

    return train_df_ts, test_df_ts, train_ts_index, test_ts_index


## numerical and categorical columns

def find_num_cols(df):
    num_cols = df.select_dtypes(include=np.number).columns
    return num_cols

def find_cat_cols(df):
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    return cat_cols

## Fill NA

def fill_na_mean_mode(df):

    num_cols = find_num_cols(df)
    cat_cols = find_cat_cols(df)

    if len(num_cols) > 0 and len(cat_cols) > 0:
        # fill NA
        df_num = df[num_cols].fillna(df[num_cols].mean())
        df_cat = df[cat_cols].fillna(df[cat_cols].mode())
        df = pd.concat([df_num, df_cat], axis = 1)
    # the elif repeat above steps but only for one kind of variables
    elif len(num_cols) > 0:
        df = df[num_cols].fillna(df[num_cols].mean())
    elif len(cat_cols) > 0:
        df = df[cat_cols].fillna(df[cat_cols].mode())

    return df

## PCA Pipeline 

def pca_on_us_col(df, us_pca_var):
    pca_df = df[us_pca_var]

    scaler = StandardScaler().set_output(transform="pandas")
    pca_scaled_df = scaler.fit_transform(pca_df)
    
    pca = PCA(0.95).set_output(transform="pandas") # keep explaning dimensions 95% of the variance 
    pca_transformed_df = pca.fit_transform(pca_scaled_df)
    pca_expl = pca.explained_variance_ratio_ # variance explained by the dimension
    pca_vars = [str(round(i, 2)) for i in pca_expl]
    st.write('dimensions explaining min. 95% of the variance will be kept:')
    for i in range(len(pca_vars)):
        st.write('pca' , i , ': explained variance = ' , pca_vars[i])

    df = df.drop(df[us_pca_var], axis = 1)
    df = pd.concat([df, pca_transformed_df], axis = 1)
    return df