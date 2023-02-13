
# ------------- Linbraries --------------

import streamlit as st
from streamlit_option_menu import option_menu 
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
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


# ------------- Settings --------------

page_title = 'SK Learn Automation'
page_icon = ':eyeglasses:' # emoji : https://www.webfx.com/tools/emoji-cheat-sheet/
layout = 'centered' # derfault but can be chenged to wide

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

# Cache
time_to_live_cache = 3600 # Cache data for 1 hour (=3600 seconds)

# ------------- hide streamlit style --------------

hide_st_style = """
<style>

footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

# #MainMenu {visibility: hidden;} this would be another option (inkl.# )

st.markdown(hide_st_style, unsafe_allow_html=True)

# ------------- Get data in and show it --------------

st.subheader("choose the data")

csv_options = {
    'winequality': ['winequality-red.csv', ';'],
    'california housing': ['https://raw.githubusercontent.com/sonarsushant/California-House-Price-Prediction/master/housing.csv', ','],
    'breast cancer': ['breast_cancer.csv', ','], 
    'bioactivity acetylcholinesterase': ['acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv', ',']
}

csv_name = [i for i in csv_options]
use_csv_name = st.selectbox('select dataset', options= csv_name)

@st.cache_data(ttl = time_to_live_cache)  # Add the caching decorator
def load_data(url, sep):
    df = pd.read_csv(url, sep)
    return df

df = load_data(csv_options[use_csv_name][0], sep= csv_options[use_csv_name][1])


## head the data
st.subheader("first entries of the dataframe")
st.dataframe(df.head(10).style.set_precision(2))
n_row, n_col = df.shape
st.write(n_col, " features, ", n_row, " rows, ", df.size, " total elements")

## show col info function

def show_info(df):
    colnames = []
    is_na = []
    is_not_na = []
    is_type = []
    for i in df.columns:
        colnames.append(i)
        is_type.append(df[i].dtype)
        n_na = df[i].isna().sum()
        is_na.append(n_na)
        is_not_na.append(len(df[i]) - n_na)
    df_col_info = pd.DataFrame({'columns' : colnames, 'n_non_null': is_not_na, 'n_null': is_na, 'type': is_type})
    return df_col_info

st.subheader("column info")
st.dataframe(show_info(df))

use_cor_matrix = st.button("create correlation matrix of continuous variables")

if use_cor_matrix:
    st.subheader("Correlation matrix of continuous variables")
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale=px.colors.sequential.Blues) # reverse color by adding "_r" (eg. Blues_r) 
    st.plotly_chart(fig)


st.markdown("""---""")

# ------------- Data Cleaning --------------

# find column type

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=['object', 'bool']).columns

# fill NA

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

# dummie code user selected cat columns

us_dummie_var = st.multiselect(
    'Which columns do you want to recode as dummies?',
    cat_cols, default=list(cat_cols))

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

st.subheader("dataframe after cleaning")
st.dataframe(df.head().style.set_precision(2))
st.dataframe(show_info(df))

st.markdown("""---""")

# ------------- Data Splitting, Scaling and transform to array for model --------------

st.subheader('Choose your Y')

us_y_var = st.selectbox(
    'Which column do you want as dependent variable?',
    df.columns)

x_options_df = df.drop(columns=[us_y_var])

# TODO: split threshold for dummies (% occurence) ans continuous (variance). continuous hase to be scaled before see commented lines
# scaler = MinMaxScaler().set_output(transform="pandas")
# x_options_scal_df = scaler.fit_transform(x_options_df)


st.subheader('Choose your X')

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


# ------------- Launch model calculation --------------


#--- Regression models


regression_models = {'RandomForestRegressor': RandomForestRegressor(),
          'LinearRegression': LinearRegression(),
          'GradientBoostingRegressor': GradientBoostingRegressor(),
          'HistGradientBoostingRegressor': HistGradientBoostingRegressor(),
          'DummyRegressor': DummyRegressor()}

st.subheader("Launch auto regression models")

# reg_models_ready = False

start_reg_models = st.button("Start regression analysis")

# initialise session state - this keeps the analysis open when other widgets are pressed and therefore script is rerun

if "start_reg_models_state" not in st.session_state :
    st.session_state.start_reg_models_state = False

if start_reg_models or st.session_state.start_reg_models_state:
    st.session_state.start_reg_models_state = True

    @st.cache_data(ttl = time_to_live_cache) 
    def split_normalize(X_df, y_ser):
        X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_ser, random_state=0, test_size=0.25)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_df)
        X_test = scaler.transform(X_test_df)

        y_train = y_train_df.to_numpy()
        y_test = y_test_df.to_numpy()
        
        return(X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = split_normalize(X_df, y_ser)

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
            r2_scores.append(round(r2, 3))
            mae_scores.append(round(mae, 2))
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


    reg_scores_df, reg_pred_y_df, reg_res_y_df = reg_models_comparison(X_train, X_test, y_train, y_test)

    # plot model scores
    fig = px.bar(reg_scores_df, x = 'R2_floored_0', y = 'Model', orientation = 'h', color = 'R2_floored_0')
    fig['layout']['yaxis']['autorange'] = "reversed"
    st.plotly_chart(fig)

    # show tabel of model scores
    st.dataframe(reg_scores_df.style.set_precision(4))


    use_reg_model = st.radio('show results for:', regression_models) #reg_scores_df['Model']

    # plot model value vs actual
    fig = px.scatter(reg_pred_y_df, x = use_reg_model, y = 'y_test', title= 'Prediction vs y_test').update_layout(
                xaxis_title="model prediction", yaxis_title="y test")
    fig = fig.add_traces(px.line(reg_pred_y_df, x='y_test', y='y_test', color_discrete_sequence=["yellow"]).data)
    st.plotly_chart(fig)

    # plot histogramm of residuals
    fig = px.histogram(reg_res_y_df, x = use_reg_model,
                title="Histogramm of Residuals").update_layout(
                xaxis_title="residuals")

    st.plotly_chart(fig)

    # plot residuals and y test
    fig = px.scatter(reg_res_y_df, x = 'y_test', y = use_reg_model,
                title="Residuals and Y").update_layout(
                xaxis_title="y test", yaxis_title="residuals")

    st.plotly_chart(fig)



#--- Classification models

classifier_models = {'RandomForestClassifier': RandomForestClassifier(),
                    'LogisticRegression': LogisticRegression(solver='sag'), #https://scikit-learn.org/stable/modules/linear_model.html#solvers
                     'SVC': SVC(),
                     'DummyClassifier': DummyClassifier(),
                     'KNeighborsClassifier': KNeighborsClassifier(),
                     'Perceptron': Perceptron()
                    }

st.subheader("Launch auto classification models")

start_class_models = st.button("Start classification analysis")

if start_class_models:

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_ser, random_state=0, test_size=0.25)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    y_train = y_train_df.to_numpy()
    y_test = y_test_df.to_numpy()

    modelnames = []
    accuracy_scores = []
    w_precision_scores = []
    w_recall_scores = []
    w_f1_scores = []

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

    clas_scores_df = pd.DataFrame({'Model': modelnames, 'Accuracy': accuracy_scores, 'Precision': w_precision_scores, 'Recall': w_recall_scores, 'F1': w_f1_scores})
    clas_scores_df = clas_scores_df.sort_values(by='Accuracy', ascending = False).reset_index().drop(columns=['index'])
    

    fig = px.bar(clas_scores_df, x = 'Accuracy', y = 'Model', orientation = 'h', color = 'Accuracy')
    fig['layout']['yaxis']['autorange'] = "reversed"
    st.plotly_chart(fig)

    st.dataframe(clas_scores_df.style.set_precision(4))
