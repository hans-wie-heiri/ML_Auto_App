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
import math as math
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report,accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC
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
@st.cache_data(ttl = time_to_live_cache) 
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


## numerical and categorical columns
@st.cache_data(ttl = time_to_live_cache) 
def find_num_cols(df):
    num_cols = df.select_dtypes(include=np.number).columns
    return num_cols

@st.cache_data(ttl = time_to_live_cache) 
def find_cat_cols(df):
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    return cat_cols

@st.cache_data(ttl = time_to_live_cache) 
def find_date_cat_cols(df):
    date_cat_cols = df.select_dtypes(include=['object', 'bool','datetime64']).columns
    return date_cat_cols


## find candidates for date conversion

@st.cache_data(ttl = time_to_live_cache) 
def datetime_candidate_col(df):
    df_datetime_candid = df.copy()
    for col in df_datetime_candid.columns:
        if df_datetime_candid[col].dtype == 'object':
            # try to convert to date and if it fails leave as is
            try:
                df_datetime_candid[col] = pd.to_datetime(df_datetime_candid[col])
            except (ValueError, TypeError):
                pass
    date_candid_cols = df_datetime_candid.select_dtypes(include=['datetime64']).columns
    return list(date_candid_cols)


## convert chosen var to datetime

@st.cache_data(ttl = time_to_live_cache) 
def datetime_converter(df, us_date_var, datetimeformats, us_datetimeformats):
    converted_date_var = []
    for i in us_date_var:
            before_date = str(df[i][0])
            try:
                df[i] = pd.to_datetime(df[i], format = datetimeformats[us_datetimeformats])
                converted_date_var.append(df[i].name)
                after_date = str(df[i][0])
                st.info(i + ': before = ' + before_date + ' ; new = ' + after_date + ' (y-m-d h;m;s)', icon="ℹ️")
            except (ValueError, TypeError):
                try:
                    df[i] = pd.to_datetime(df[i], format = None) # try converting with automatic date format
                    st.warning(df[i].name + ' was converted with default format because chosen format failed', icon="⚠️")
                    converted_date_var.append(df[i].name)
                    after_date = str(df[i][0])
                    st.write(i + ': before = ' + before_date + ' ; new = ' + after_date + ' ; new type = ' + str(df[i].dtype))
                except (ValueError, TypeError):
                    pass
                    st.warning(df[i].name + ' could not be converted to date format', icon="⚠️")
    return df, converted_date_var

# ------------- Data Splitting and Preprocessinng --------------


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
def split_timeseries(df_ts, us_start_date, us_end_date): # us_test_size
   
    us_start_date = datetime.combine(us_start_date, datetime.min.time())
    us_end_date = datetime.combine(us_end_date, datetime.min.time())

    train_df_ts = df_ts.loc[df_ts.index < us_start_date]
    test_df_ts = df_ts.loc[(df_ts.index >= us_start_date) & (df_ts.index <= us_end_date)]
    
    train_ts_index = train_df_ts.index
    test_ts_index = test_df_ts.index

    return train_df_ts, test_df_ts



## Fill NA

@st.cache_data(ttl = time_to_live_cache) 
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

## Fill NA Simple imputer

@st.cache_data(ttl = time_to_live_cache) 
def fill_na_si_mean_mode(train_df, test_df):

    train_num_cols = train_df[find_num_cols(train_df)]
    train_cat_cols = train_df[find_cat_cols(train_df)]
    test_num_cols = test_df[find_num_cols(test_df)]
    test_cat_cols = test_df[find_cat_cols(test_df)]

    if len(train_num_cols.columns) > 0 and len(train_cat_cols.columns) > 0:
        imp_mean = SimpleImputer(missing_values=pd.NA, strategy='mean').set_output(transform="pandas")
        imp_mean.fit(train_num_cols)
        train_num_imp = imp_mean.transform(train_num_cols)
        test_num_imp = imp_mean.transform(test_num_cols)

        imp_mode = SimpleImputer(missing_values=pd.NA, strategy='most_frequent').set_output(transform="pandas")
        imp_mode.fit(train_cat_cols)
        train_cat_imp = imp_mode.transform(train_cat_cols)
        test_cat_imp = imp_mode.transform(test_cat_cols)

        train_df = pd.concat([train_num_imp, train_cat_imp], axis = 1)
        test_df = pd.concat([test_num_imp, test_cat_imp], axis = 1)
    # the elif repeat above steps but only for one kind of variables
    elif len(train_num_cols.columns) > 0:
        imp_mean = SimpleImputer(missing_values=pd.NA, strategy='mean').set_output(transform="pandas")
        imp_mean.fit(train_num_cols)
        train_df = imp_mean.transform(train_num_cols)
        test_df = imp_mean.transform(test_num_cols)
    elif len(train_cat_cols.columns) > 0:
        imp_mode = SimpleImputer(missing_values=pd.NA, strategy='most_frequent').set_output(transform="pandas")
        imp_mode.fit(train_cat_cols)
        train_df = imp_mode.transform(train_cat_cols)
        test_df = imp_mode.transform(test_cat_cols)

    return train_df, test_df


## PCA on train and test 

@st.cache_data(ttl = time_to_live_cache) 
def pca_on_us_col(train_df, test_df, us_pca_var):
    # columns subset
    pca_train_df = train_df[us_pca_var]
    pca_test_df = test_df[us_pca_var]
    # scaler
    scaler = StandardScaler().set_output(transform="pandas")
    pca_scaled_train_df = scaler.fit_transform(pca_train_df)
    pca_scaled_test_df = scaler.transform(pca_test_df)
    # pca
    pca = PCA(0.95).set_output(transform="pandas") # keep explaning dimensions 95% of the variance 
    pca_transformed_train_df = pca.fit_transform(pca_scaled_train_df)
    pca_transformed_test_df = pca.transform(pca_scaled_test_df)
    # pca explanation
    pca_expl = pca.explained_variance_ratio_ # variance explained by the dimension
    pca_vars = [str(round(i, 2)) for i in pca_expl]
    st.write('dimensions explaining min. 95% of the variance will be kept:')
    for i in range(len(pca_vars)):
        st.write('pca' , i , ': explained variance = ' , pca_vars[i])
    # output df creation
    train_df = train_df.drop(train_df[us_pca_var], axis = 1)
    train_df = pd.concat([train_df, pca_transformed_train_df], axis = 1)
    test_df = test_df.drop(test_df[us_pca_var], axis = 1)
    test_df = pd.concat([test_df, pca_transformed_test_df], axis = 1)

    return train_df, test_df


## extract features from date columns

@st.cache_data(ttl = time_to_live_cache) 
def create_time_features(df, converted_date_var):
        df = df.copy()
        for i in converted_date_var:
            df[i+'_hour'] = df[i].dt.hour
            df[i+'_dayofweek'] = df[i].dt.dayofweek
            df[i+'_quarter'] = df[i].dt.quarter
            df[i+'_month'] = df[i].dt.month
            df[i+'_year'] = df[i].dt.year
            df[i+'_dayofyear'] = df[i].dt.dayofyear
            df[i+'_dayofmonth'] = df[i].dt.day
            df[i+'_weekofyear'] = df[i].dt.isocalendar().week
            df.drop(i , axis = 1, inplace = True)
        return df


## check number of unique values befor dummi coding

@st.cache_data(ttl = time_to_live_cache) 
def col_with_n_uniques(df, col_list, n):
    alotofuniques = []
    for i in col_list:
        if df[i].nunique() >= n:
            alotofuniques.append(i)
    return alotofuniques 


## dummi encoding

@st.cache_data(ttl = time_to_live_cache) 
def dummi_encoding(train_df, test_df, us_dummie_var):
    # create dummies for cat variables
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
    enc.fit(train_df[us_dummie_var])
    train_dum_df = enc.transform(train_df[us_dummie_var])
    test_dum_df = enc.transform(test_df[us_dummie_var])
    # drop original cat column from df
    train_df = train_df.drop(train_df[us_dummie_var], axis = 1)
    test_df = test_df.drop(test_df[us_dummie_var], axis = 1)
    # concat df and cat variables
    train_df = pd.concat([train_df, train_dum_df], axis = 1)
    test_df = pd.concat([test_df, test_dum_df], axis = 1)
    return train_df, test_df


# dataframe to csv converter

@st.cache_data(ttl = time_to_live_cache) 
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv().encode('utf-8')


# function for splitting, normalizing data that outputs arrays
@st.cache_data(ttl = time_to_live_cache) 
def scaling_test_train(X_train_df, X_test_df, y_train_ser, y_test_ser, us_scaler_key):
    
    # creating the dict again is stupid cheating!
    scalers = {
    'MinMaxScaler' : MinMaxScaler(),
    'StandardScaler' : StandardScaler(),
    'RobustScaler' : RobustScaler()
    }

    scaler = scalers[us_scaler_key]
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    y_train = y_train_ser.to_numpy()
    y_test = y_test_ser.to_numpy()
    
    return(X_train, X_test, y_train, y_test)


# remove dupplicate function that keeps last version of index for time series
@st.cache_data(ttl = time_to_live_cache) 
def remove_duplicated_index(df):
    all_dup = df.index.duplicated(keep=False) # all duplicates will be True rest False
    last_dup = df.index.duplicated(keep='last') # last duplicates will again be True rest False
    keep_last = all_dup == last_dup # lose duplicates that are not the last (first True then False = False) 
    df_no_dup = df[keep_last]
    return df_no_dup