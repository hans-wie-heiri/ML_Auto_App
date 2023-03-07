# ------------- ToDo List --------------
# - posiibility to delete NA rows (additional to fill)
# - possibility to delete duplicates
# - Algo selection
# - design
# - crashproove
# - select time space for time series testing 
# - export model ?
# - predict new data?
# - feature importance ?

# ------------- Linbraries --------------

import streamlit as st
from streamlit_option_menu import option_menu 
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


# ------------- Settings --------------

page_title = 'SK Learn Automation'
page_description = 'This App automates a Data Science analysis with the [scikit-learn](https://scikit-learn.org/stable/index.html) API. \
It is functional and automates the process steps for a small data science project or a first shot at model selection. \
The application does not claim to replace a full and comprehensive Data Science Project. \
The application allows regression, classification and time series analysis with machine learning algorithms.'

page_icon = ':eyeglasses:' # emoji : https://www.webfx.com/tools/emoji-cheat-sheet/
layout = 'centered' # derfault but can be chenged to wide

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)
st.write(page_description)

# Cache
time_to_live_cache = 3600 # Cache data for 1 hour (=3600 seconds)

# ------------- hide streamlit style --------------

hide_st_style = """
<style>

footer {visibility: hidden;}
</style>
"""

# header {visibility: hidden;}
# #MainMenu {visibility: hidden;} this would be another option (inkl.# )

st.markdown(hide_st_style, unsafe_allow_html=True)

# ------------- Get data in and show it --------------

st.header("data selection and first glance")
st.write('')
st.subheader("choose the data")


csv_options = {
    'winequality': ['winequality-red.csv', ';'],
    'california housing': ['https://raw.githubusercontent.com/sonarsushant/California-House-Price-Prediction/master/housing.csv', ','],
    'breast cancer': ['breast_cancer.csv', ','], 
    'bioactivity acetylcholinesterase': ['acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv', ','],
    'energy consumption hourly': ['https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv' , ','],
    'Konsumentenpreise Index Mai 2000': ['Konsumentenpreise_Index_Mai_2000.csv', ';']
}

csv_name = [i for i in csv_options]


use_csv_name = st.selectbox('**select dataset from List**', options= csv_name)
uploaded_file = st.file_uploader("**or upload your own**")


@st.cache_data(ttl = time_to_live_cache)  # Add the caching decorator
def load_data(url, sep):
    df = pd.read_csv(url, sep)
    return df

if uploaded_file is None:
    df = load_data(csv_options[use_csv_name][0], sep= csv_options[use_csv_name][1])
    df_name = use_csv_name
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_name = str(uploaded_file.name)

## head the data
st.subheader("first entries of the dataframe - " + df_name)
st.dataframe(df.head(10).style.set_precision(2))
n_row, n_col = df.shape
st.write(n_col, " features, ", n_row, " rows, ", df.size, " total elements")

## show col info function

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

st.subheader("column info")
st.dataframe(show_info(df))

use_cor_matrix = st.button("create correlation matrix of continuous variables")

if use_cor_matrix:
    st.subheader("Correlation matrix of continuous variables")
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale=px.colors.sequential.Blues) # reverse color by adding "_r" (eg. Blues_r) 
    st.plotly_chart(fig)

# Plot features
st.subheader("Plot Selcted Features")

plot_types = ['Scatter Plot', 'Histogramm', 'Line Plot', 'Box Plot', 'Heatmap of count']

us_plot_type = st.selectbox('Select Plot Type', plot_types)
us_x_axis = st.selectbox('select x-Axis', list(df.columns))
if us_plot_type != 'Histogramm':
    us_y_axis = st.selectbox('select y-Axis', list(df.columns), index = (len(list(df.columns))-1))

# plot user selected features

if us_plot_type == 'Scatter Plot':
    fig = px.scatter(df, x = us_x_axis, y = us_y_axis, 
                title= 'Scatter plot of Selected Features').update_layout(
                xaxis_title= us_x_axis, yaxis_title= us_y_axis)
elif us_plot_type == 'Histogramm':
    fig = px.histogram(df, x = us_x_axis, 
                title= 'Histogramm plot of Selected Features').update_layout(
                xaxis_title= us_x_axis, yaxis_title= 'count')
elif us_plot_type == 'Line Plot':
    fig = px.line(df, x = us_x_axis, y = us_y_axis, 
                title= 'Line plot of Selected Features').update_layout(
                xaxis_title= us_x_axis, yaxis_title= us_y_axis)
elif us_plot_type == 'Box Plot':
    fig = px.box(df, x = us_x_axis, y = us_y_axis, 
                title= 'Box plot of Selected Features').update_layout(
                xaxis_title= us_x_axis, yaxis_title= us_y_axis)
elif us_plot_type == 'Heatmap of count':
    heatmap_df = pd.DataFrame(df.groupby([us_x_axis, us_y_axis])[us_x_axis].count())
    heatmap_df.rename(columns={us_x_axis: "count"}, inplace=True)
    heatmap_df.reset_index(inplace = True)
    heatmap_df = heatmap_df.pivot(index=us_y_axis, columns=us_x_axis)['count'].fillna(0)
    fig = px.imshow(heatmap_df, x=heatmap_df.columns, y=heatmap_df.index, title= 'Heatmap of Count for Selected Features', text_auto=True, color_continuous_scale=px.colors.sequential.Blues)
    
st.plotly_chart(fig)

st.markdown("""---""")

# ------------- Data Preprocessinng --------------

st.header('Data Preprocessing')
st.write("NA-values will automatically be filled (numerical variables by their mean and categorical by their mode) \
         alternatively all rows with NA-values can droped.")

# find column type

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=['object', 'bool']).columns


# fill or drop NA

NA_handling = {
    'fill NA (mean/mode)' : 'fill_na',
    'drop rows with NA' : 'drop_na'
}

us_na_handling = st.radio('Do you want to fill or drop the NA?', NA_handling.keys(), horizontal=True, index=0)

if us_na_handling == 'fill NA (mean/mode)':
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
elif us_na_handling == 'drop rows with NA':
    df = df.dropna()
    df = df.reset_index()

# PCA
# info_PCA = st.button("ℹ️")
# if info_PCA:
#     st.info('Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.', icon="ℹ️")

us_pca_var = st.multiselect(
    'Do you want to do a PCA on some columns?',
    num_cols, default=None)


def pca_on_us_col(df):
    pca_df = df[us_pca_var]

    scaler = StandardScaler().set_output(transform="pandas")
    pca_scaled_df = scaler.fit_transform(pca_df)
    
    pca = PCA(0.95).set_output(transform="pandas") # keep explaning dimensions 95% of the variance 
    pca_transformed_df = pca.fit_transform(pca_scaled_df)
    pca_expl = pca.explained_variance_ratio_ # variance explained by the dimension
    pca_vars = [str(round(i, 2)) for i in pca_expl]
    for i in range(len(pca_vars)):
        st.write('pca' , i , ': explained variance = ' , pca_vars[i])

    df = df.drop(df[us_pca_var], axis = 1)
    df = pd.concat([df, pca_transformed_df], axis = 1)
    return df

if len(us_pca_var) > 0:
    df = pca_on_us_col(df)
    


# find candidates for datetime conversion

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

date_candid_cols = datetime_candidate_col(df)


# change user selected date columns to datetime 

us_date_var = st.multiselect(
    'Which columns do you want to recode as dates?',
    cat_cols, default=list(date_candid_cols))

if len(us_date_var) > 0:
    cat_cols_no_date = list(cat_cols)
    cat_cols_no_date = set(cat_cols_no_date) - set(us_date_var)
else:
    cat_cols_no_date = list(cat_cols)

datetimeformats = {'automatic': None,
               'day.month.Year': "%d.%m.%Y",
               'day/month/Year': "%d/%m/%Y",
               'day-month-Year': "%d-%m-%Y",
               'Year-month-day': "%Y-%m-%d"}

if len(us_date_var) > 0:
    us_datetimeformats = st.selectbox('Choose a datetime format', list(datetimeformats.keys()))

converted_date_var = []

if len(us_date_var) > 0:

    for i in us_date_var:
        before_date = str(df[i][0])
        try:
            df[i] = pd.to_datetime(df[i], format = datetimeformats[us_datetimeformats])
            converted_date_var.append(df[i].name)
            after_date = str(df[i][0])
            st.write(i + ': before = ' + before_date + ' ; new = ' + after_date + ' ; new type = ' + str(df[i].dtype))
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


# extract features from datetime 

if len(converted_date_var) > 0:
    # save the first var as index for time series analysis
    ts_index = df[converted_date_var[0]]

    def create_time_features(df):
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
    
    df = create_time_features(df)



# safety for cat columns with a lot of uniques

def col_with_n_uniques(df, col_list, n):
    alotofuniques = []
    for i in col_list:
        if df[i].nunique() >= n:
            alotofuniques.append(i)
    return alotofuniques 

# a lot is set to 100
n = 100
list_alotofuniques = col_with_n_uniques(df, cat_cols_no_date, n)

dummy_default_list = set(cat_cols_no_date) - set(list_alotofuniques)

if len(list_alotofuniques) > 0:
    for i in list_alotofuniques:
        st.warning(i + ' with type = object but more than 100 unique values', icon="⚠️")


# dummie code user selected cat columns

us_dummie_var = st.multiselect(
    'Which columns do you want to recode as dummies?',
    cat_cols_no_date, default= list(dummy_default_list))

if len(us_dummie_var) > 0:
    # create dummies for cat variables
    df_us_dummies = df[us_dummie_var]
    empt = [i for i in range(len(df_us_dummies.index))]
    df_dummies = pd.DataFrame({'emp_col': empt})
    for i in df_us_dummies:
        tmp_df = pd.get_dummies(df_us_dummies[i], prefix= i)
        df_dummies = pd.concat([df_dummies, tmp_df], axis = 1)
    df_dummies = df_dummies.drop('emp_col', axis = 1)
    #drop original cat column from df
    df = df.drop(df[us_dummie_var], axis = 1)
    #concat df and cat variables
    df = pd.concat([df, df_dummies], axis = 1)
    df = df.dropna()


st.subheader("dataframe after cleaning")
st.dataframe(df.head().style.set_precision(2))
st.dataframe(show_info(df))

st.markdown("""---""")

# ------------- Data Splitting, Scaling and transform to array for model --------------

st.header('Dependent and Independent Variable Selection')
st.write('')

st.subheader('Choose your Y')

us_y_var = st.selectbox(
    'Which column do you want as dependent variable?',
    df.columns)

x_options_df = df.drop(columns=[us_y_var])

# TODO: split threshold for dummies (% occurence) ans continuous (variance). continuous hase to be scaled before see commented lines
# scaler = MinMaxScaler().set_output(transform="pandas")
# x_options_scal_df = scaler.fit_transform(x_options_df)


st.subheader('Choose your X')

st.write('Do you want to drop Columns with 0 variance?')
use_variance_threshold = st.button("drop columns with 0 variance")

if use_variance_threshold:
    # variancetreshold = st.number_input("Feature Selection with minimal variance within feature", min_value=0.0, max_value=1.0, value=0.0)
    n_col_before = len(x_options_df.columns)
    selection = VarianceThreshold(threshold=(0)).set_output(transform="pandas")
    x_options_df = selection.fit_transform(x_options_df)
    n_col_after = len(x_options_df.columns)
    n_del_col = (n_col_before - n_col_after)
    st.write("Number of deleted columns: ", n_del_col)
    st.write("Number of retained columns: ", n_col_after)

us_x_var = st.multiselect(
    'Which columns do you want as independent variable?',
    list(x_options_df),
    default=list(x_options_df))


y_ser = df[us_y_var]
X_df = df[us_x_var]

st.subheader('See selected variables')

st.write('chosen dependent variable')

col1, col2 = st.columns((0.25, 0.75))
col1.dataframe(y_ser)

fig = px.histogram(df, x= us_y_var )
col2.plotly_chart(fig, use_container_width=True)

st.write('chosen independent variable')
st.dataframe(X_df.head().style.set_precision(2))


st.markdown("""---""")

st.header('Launch the desired models')
descr_model_launch = 'Regression models are only launchable if the chosen dependen variable is a number and \
classification models in turn, only if it is of type bool, object or int. \
Time Series analysis is only callable if at least one feature has been recoded as date.'

st.write(descr_model_launch)

# test size selection

testsizes = {
    '10%' : 0.10,
    '20%' : 0.20,
    '30%' : 0.30,
    '40%' : 0.40,
    '50%' : 0.50,
}

us_test_size_pct = st.radio('What % of data do you want to use as test size?', list(testsizes.keys()), index=1, horizontal = True)
us_test_size = testsizes[us_test_size_pct]

# scaler selection

scalers = {
    'MinMaxScaler' : MinMaxScaler(),
    'StandardScaler' : StandardScaler(),
    'RobustScaler' : RobustScaler()
}

us_scaler_key = st.radio('What scaler do you want to use?', list(scalers.keys()), index=0, horizontal = True)

# ------------- Launch model calculation --------------

# initialize states for x and y selction by user 
# if user x or y selection changes, then we dont want an automatic model rerun below

if "y_var_user" not in st.session_state:
    st.session_state.y_var_user = us_y_var

if "x_var_user" not in st.session_state:
    st.session_state.x_var_user = us_x_var

if "test_size_user" not in st.session_state:
    st.session_state.test_size_user = us_test_size

if "scaler_user" not in st.session_state:
    st.session_state.scaler_user = us_scaler_key

check_y_no_change = st.session_state.y_var_user == us_y_var
check_x_no_change = st.session_state.x_var_user == us_x_var
check_test_size_no_change = st.session_state.test_size_user == us_test_size
check_scaler_no_change = st.session_state.scaler_user == us_scaler_key

# check y for possible models
reg_cols = df.select_dtypes(include=np.number).columns
clas_cols = df.select_dtypes(include=['object', 'bool', 'int']).columns


# function for splitting, normalizing data that outputs arrays
@st.cache_data(ttl = time_to_live_cache) 
def split_normalize(X_df, y_ser, us_test_size, us_scaler_key):
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_ser, random_state=0, test_size= us_test_size)

    scaler = scalers[us_scaler_key]
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    y_train = y_train_df.to_numpy()
    y_test = y_test_df.to_numpy()
    
    return(X_train, X_test, y_train, y_test)

# function for running and comparing regression models

regression_models = {'RandomForestRegressor': RandomForestRegressor(),
          'LinearRegression': LinearRegression(),
          'GradientBoostingRegressor': GradientBoostingRegressor(),
          'HistGradientBoostingRegressor': HistGradientBoostingRegressor(),
          'DummyRegressor': DummyRegressor()}

@st.cache_data(ttl = time_to_live_cache) 
def reg_models_comparison(X_train, X_test, y_train, y_test):

    # init values to be stored
    modelnames = []
    r2_scores = []
    mae_scores = []
    predictions = {}
    residuals = {}
    
    # loop through models
    for i in regression_models:
        m_reg = regression_models[i]
        m_reg.fit(X_train, y_train)

        y_test_predict = m_reg.predict(X_test)
        mae = mean_absolute_error(y_test_predict, y_test)
        r2 = m_reg.score(X_test, y_test)

        modelnames.append(i)
        r2_scores.append(round(r2, 4))
        mae_scores.append(round(mae, 4))
        predictions[i] = y_test_predict
        residuals[i] = (y_test_predict - y_test)
        
    # create score dataframe
    reg_scores_df = pd.DataFrame({'Model': modelnames, 'R2': r2_scores, 'MAE': mae_scores})
    reg_scores_df = reg_scores_df.sort_values(by='R2', ascending = False).reset_index().drop(columns=['index'])
    R2_floor = 0.0
    reg_scores_df['R2_floored_0'] = np.maximum(reg_scores_df['R2'], R2_floor)

    # create prediction dataframe
    reg_pred_y_df = pd.DataFrame(predictions)
    reg_pred_y_df['y_test'] = pd.Series(y_test)
    
    # create residual dataframe
    reg_res_y_df = pd.DataFrame(residuals)
    reg_res_y_df['y_test'] = pd.Series(y_test)
    
    # return the 3 dataframes
    return reg_scores_df, reg_pred_y_df, reg_res_y_df

# function for splitting (last 1 - test size percent of time series), that outputs arrays for time series
@st.cache_data(ttl = time_to_live_cache) 
def split_timeseries(X_df_ts, y_ser_ts, us_test_size, us_scaler_key):
    train_size = (1-us_test_size)
    
    X_train_df_ts = X_df_ts[:round(len(X_df_ts)*train_size)]
    X_test_df_ts = X_df_ts[-(len(X_df_ts)-round(len(X_df_ts)*train_size)):]
    y_train_df_ts = y_ser_ts[:round(len(y_ser_ts)*train_size)]
    y_test_df_ts = y_ser_ts[-(len(y_ser_ts)-round(len(y_ser_ts)*train_size)):]
    
    train_ts_index = y_train_df_ts.index
    test_ts_index = y_test_df_ts.index

    scaler = scalers[us_scaler_key]
    X_train_ts = scaler.fit_transform(X_train_df_ts)
    X_test_ts = scaler.transform(X_test_df_ts)

    y_train_ts = y_train_df_ts.to_numpy()
    y_test_ts = y_test_df_ts.to_numpy()
    
    return(X_train_ts, X_test_ts, y_train_ts, y_test_ts, train_ts_index, test_ts_index)


#--- Regression models

# only if y is a number type

if us_y_var in reg_cols:

    st.subheader("Launch auto regression models")


    start_reg_models = st.button("Start regression analysis")

    # initialise session state - this keeps the analysis open when other widgets are pressed and therefore script is rerun

    if "start_reg_models_state" not in st.session_state :
        st.session_state.start_reg_models_state = False
        

    if (start_reg_models or (st.session_state.start_reg_models_state 
                            and check_y_no_change 
                            and check_x_no_change
                            and check_test_size_no_change
                            and check_scaler_no_change)):
        st.session_state.start_reg_models_state = True
        st.session_state.y_var_user = us_y_var
        st.session_state.x_var_user = us_x_var
        st.session_state.test_size_user = us_test_size
        st.session_state.scaler_user = us_scaler_key

        # run splitting
        X_train, X_test, y_train, y_test = split_normalize(X_df, y_ser, us_test_size, us_scaler_key)

        # run model and compare
        reg_scores_df, reg_pred_y_df, reg_res_y_df = reg_models_comparison(X_train, X_test, y_train, y_test)

        # Titel
        st.subheader("Results for Regression Models on Testset")

        # plot model scores
        fig = px.bar(reg_scores_df, x = 'R2_floored_0', y = 'Model', orientation = 'h', color = 'R2_floored_0',
            title="Model Comparison on R2 (floored at 0)")
        fig['layout']['yaxis']['autorange'] = "reversed"
        st.plotly_chart(fig)

        # show tabel of model scores
        st.dataframe(reg_scores_df.style.set_precision(4))


        use_reg_model = st.radio('show results for:', reg_scores_df.Model) 

        # plot model value vs actual
        fig = px.scatter(reg_pred_y_df, x = use_reg_model, y = 'y_test', 
                    title= 'Model Prediction vs Y Test - ' + use_reg_model).update_layout(
                    xaxis_title="model prediction", yaxis_title="y test")
        fig = fig.add_traces(px.line(reg_pred_y_df, x='y_test', y='y_test', color_discrete_sequence=["yellow"]).data)
        st.plotly_chart(fig)

        # plot histogramm of residuals
        fig = px.histogram(reg_res_y_df, x = use_reg_model,
                    title="Histogramm of Residuals - " + use_reg_model).update_layout(
                    xaxis_title="residuals")

        st.plotly_chart(fig)

        # plot residuals and y test
        fig = px.scatter(reg_res_y_df, x = 'y_test', y = use_reg_model,
                    title="Residuals and Y Test - " + use_reg_model).update_layout(
                    xaxis_title="y test", yaxis_title="residuals")

        st.plotly_chart(fig)



#--- Classification models

# only if y is a number type

if us_y_var in clas_cols:

    classifier_models = {'RandomForestClassifier': RandomForestClassifier(),
                        'LogisticRegression': LogisticRegression(solver='sag'), #https://scikit-learn.org/stable/modules/linear_model.html#solvers
                        'SVC': SVC(),
                        'DummyClassifier': DummyClassifier(),
                        'KNeighborsClassifier': KNeighborsClassifier(),
                        'Perceptron': Perceptron()
                        }

    st.subheader("Launch auto classification models")

    start_clas_models = st.button("Start classification analysis")

    # initialise session state - this keeps the analysis open when other widgets are pressed and therefore script is rerun

    if "start_clas_models_state" not in st.session_state :
        st.session_state.start_clas_models_state = False
        

    if (start_clas_models or (st.session_state.start_clas_models_state 
                            and check_y_no_change 
                            and check_x_no_change 
                            and check_test_size_no_change
                            and check_scaler_no_change)):
        st.session_state.start_clas_models_state = True
        st.session_state.y_var_user = us_y_var
        st.session_state.x_var_user = us_x_var
        st.session_state.test_size_user = us_test_size
        st.session_state.scaler_user = us_scaler_key

        X_train, X_test, y_train, y_test = split_normalize(X_df, y_ser, us_test_size, us_scaler_key)

        @st.cache_data(ttl = time_to_live_cache) 
        def class_models_comparison(X_train, X_test, y_train, y_test):
            
            # init values to be stored
            modelnames = []
            accuracy_scores = []
            w_precision_scores = []
            w_recall_scores = []
            w_f1_scores = []
            predictions = {}
            class_labels = {}

            
            # loop through models
            for i in classifier_models:
                m_clas = classifier_models[i]
                m_clas.fit(X_train, y_train)

                y_test_predict = m_clas.predict(X_test)
                accuracy = accuracy_score(y_test, y_test_predict)
                precision = precision_score(y_test, y_test_predict, average='weighted')
                recall = recall_score(y_test, y_test_predict, average='weighted')
                f1 = f1_score(y_test, y_test_predict, average='weighted')

                modelnames.append(i)
                accuracy_scores.append(round(accuracy, 4))
                w_precision_scores.append(round(precision, 4))
                w_recall_scores.append(round(recall, 4))
                w_f1_scores.append(round(f1, 4))
                predictions[i] = y_test_predict
                class_labels[i] = list(m_clas.classes_)
                
            # create score dataframe
            clas_scores_df = pd.DataFrame({'Model': modelnames, 'Accuracy': accuracy_scores, 'Precision': w_precision_scores, 'Recall': w_recall_scores, 'F1': w_f1_scores})
            clas_scores_df = clas_scores_df.sort_values(by='Accuracy', ascending = False).reset_index().drop(columns=['index'])

            # create prediction dataframe
            clas_pred_y_df = pd.DataFrame(predictions)
            clas_pred_y_df['y_test'] = pd.Series(y_test)
            
            # create class dataframe
            clas_label_df = pd.DataFrame(class_labels)
            
            # return the 3 dataframes
            return clas_scores_df, clas_pred_y_df, clas_label_df

        clas_scores_df, clas_pred_y_df, clas_label_df = class_models_comparison(X_train, X_test, y_train, y_test)

        # Titel
        st.subheader("Results for Classification Models on Testset")
        
        fig = px.bar(clas_scores_df, x = 'Accuracy', y = 'Model', orientation = 'h', color = 'Accuracy')
        fig['layout']['yaxis']['autorange'] = "reversed"
        st.plotly_chart(fig)

        st.dataframe(clas_scores_df.style.set_precision(4))

        use_clas_model = st.radio('show results for:', clas_scores_df.Model)

        cm = confusion_matrix(clas_pred_y_df['y_test'], clas_pred_y_df[use_clas_model])
        xlab = list(clas_label_df[use_clas_model])
        ylab = xlab

        fig = px.imshow(cm, x= xlab, y=ylab, text_auto=True, color_continuous_scale=px.colors.sequential.Blues,
        title= "Confusion Matrix - " + use_clas_model).update_layout(
        xaxis_title="predicted label", yaxis_title="true label")
        st.plotly_chart(fig)

        fig = px.histogram(clas_pred_y_df, x = 'y_test', title = 'Histogram of True Values -' + use_clas_model).update_layout(
        xaxis_title="True Values")
        st.plotly_chart(fig)

        fig = px.histogram(clas_pred_y_df, x = use_clas_model, title = 'Histogram of Predicted Values - ' + use_clas_model).update_layout(
        xaxis_title="Predicted Values")
        st.plotly_chart(fig)

        @st.cache_data(ttl = time_to_live_cache) 
        def clas_score_label(clas_pred_y_df):
            label_list = list(clas_pred_y_df['y_test'].unique())
            precision_list = []
            recall_list = []

            for i in label_list:
                sub_df_prec = clas_pred_y_df[clas_pred_y_df['RandomForestClassifier'] == i]
                precision = precision_score(sub_df_prec['y_test'], sub_df_prec['RandomForestClassifier'], average='micro')
                precision_list.append(precision)

                sub_df_reca = clas_pred_y_df[clas_pred_y_df['y_test'] == i]
                recall = recall_score(sub_df_reca['y_test'], sub_df_reca['RandomForestClassifier'], average='micro')
                recall_list.append(recall)

            score_label_df = pd.DataFrame({'Label': label_list, 'Precision' : precision_list, 'Recall': recall_list})
            score_label_df['F1'] = 2 * (score_label_df['Precision'] * score_label_df['Recall']) / (score_label_df['Precision'] + score_label_df['Recall'])
            return score_label_df

        score_label_df = clas_score_label(clas_pred_y_df)

        st.markdown('**Scores per Category**')
        st.dataframe(score_label_df)


#--- Time Series models

if len(converted_date_var) > 0:

    st.subheader("Launch auto time series models")
    start_ts_models = st.button("Start Time Series Analysis")

    # initialise session state - this keeps the analysis open when other widgets are pressed and therefore script is rerun

    if "start_ts_models_state" not in st.session_state :
        st.session_state.start_ts_models_state = False
        

    if (start_ts_models or (st.session_state.start_ts_models_state 
                            and check_y_no_change 
                            and check_x_no_change
                            and check_test_size_no_change
                            and check_scaler_no_change)):
        st.session_state.start_ts_models_state = True
        st.session_state.y_var_user = us_y_var
        st.session_state.x_var_user = us_x_var
        st.session_state.test_size_user = us_test_size
        st.session_state.scaler_user = us_scaler_key
        

        # remove dupplicate function that keeps last version of index
        def remove_duplicated_index(df):
            all_dup = df.index.duplicated(keep=False) # all duplicates will be True rest False
            last_dup = df.index.duplicated(keep='last') # last duplicates will again be True rest False
            keep_last = all_dup == last_dup # lose duplicates that are not the last (first True then False = False) 
            df_no_dup = df[keep_last]
            return df_no_dup
        
        # create a time series dataframe wit datetime index without duplicates
        df_ts = df.set_index(ts_index)
        df_ts = remove_duplicated_index(df_ts)
        df_ts = df_ts.sort_index()


        y_ser_ts = df_ts[us_y_var]
        X_df_ts = df_ts[us_x_var]

        X_train_ts, X_test_ts, y_train_ts, y_test_ts, train_ts_index, test_ts_index = split_timeseries(X_df_ts, y_ser_ts, us_test_size, us_scaler_key)

        ts_scores_df, ts_pred_y_df, ts_res_y_df = reg_models_comparison(X_train_ts, X_test_ts, y_train_ts, y_test_ts)

        # Titel
        st.subheader("Results for Regression Models on Testset")

        # plot model scores
        fig = px.bar(ts_scores_df, x = 'R2_floored_0', y = 'Model', orientation = 'h', color = 'R2_floored_0',
            title="Model Comparison on R2 (floored at 0)")
        fig['layout']['yaxis']['autorange'] = "reversed"
        st.plotly_chart(fig)

        # show tabel of model scores
        st.dataframe(ts_scores_df.style.set_precision(4))

        use_ts_model = st.radio('show results for:', ts_scores_df.Model)

        ts_pred_y_df.set_index(test_ts_index, inplace=True)
        pred_df_ts = pd.concat([df_ts, ts_pred_y_df], axis = 1)

        pred_df_ts['Datetime_index'] = pred_df_ts.index # two y doesn't work on index appearently
        fig = px.line(pred_df_ts, x='Datetime_index', y=[us_y_var, use_ts_model]).update_layout(
                xaxis_title="Datetime Index", 
                yaxis_title= (us_y_var + " and Prediction"),
                title = 'Time Series and Prediction')
        st.plotly_chart(fig)

        ts_res_y_df.set_index(test_ts_index, inplace=True)
        
        # plot histogramm of residuals
        fig = px.histogram(ts_res_y_df, x = use_ts_model,
                    title="Histogramm of Residuals - " + use_ts_model).update_layout(
                    xaxis_title="residuals")

        st.plotly_chart(fig)

        # plot residuals and y test
        ts_res_y_df['Datetime_index'] = ts_res_y_df.index # two y doesn't work on index appearently
        fig = px.bar(ts_res_y_df, x = 'Datetime_index', y = use_ts_model,
                    title="Residuals over Datetime - " + use_ts_model).update_layout(
                    xaxis_title="Datetime Index", yaxis_title="residuals")

        st.plotly_chart(fig)

