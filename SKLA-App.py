
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


# ------------- Settings --------------

page_title = 'SK Learn Automation'
page_icon = ':eyeglasses:' # emoji : https://www.webfx.com/tools/emoji-cheat-sheet/
layout = 'centered' # derfault but can be chenged to wide

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

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

# housing = pd.read_csv('https://raw.githubusercontent.com/sonarsushant/California-House-Price-Prediction/master/housing.csv')
# df = housing
# df = pd.read_csv('winequality-red.csv', sep=';')
df = pd.read_csv('breast_cancer.csv', sep=',')


## head the data
st.subheader("first entries of the dataframe")
st.dataframe(df.head().style.set_precision(2))

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

# TODO: Dummie recoding doesn't work in breastcancer data
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

us_y_var = st.selectbox(
    'Which column do you want as dependent variable?',
    df.columns)

#st.write('You selected:', us_y_var, 'as the dependent variable')

x_options = list(df.columns)
x_options.remove(us_y_var)

us_x_var = st.multiselect(
    'Which columns do you want as independent variable?',
    x_options,
    default=x_options)

#st.write('You selected:', us_x_var, 'as the independent variable')



y_ser = df[us_y_var]
X_df = df[us_x_var]

st.write('chosen dependent variable')
st.dataframe(y_ser.head())

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

start_reg_models = st.button("Start regression analysis")

if start_reg_models:

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_ser, random_state=0, test_size=0.25)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    y_train = y_train_df.to_numpy()
    y_test = y_test_df.to_numpy()

    modelnames = []
    r2_scores = []
    mae_scores = []

    for i in regression_models:
        m_reg = regression_models[i]
        m_reg.fit(X_train, y_train)

        y_test_predict = m_reg.predict(X_test)
        mae = mean_absolute_error(y_test_predict, y_test)
        r2 = m_reg.score(X_test, y_test)
        
        modelnames.append(i)
        r2_scores.append(round(r2, 4))
        mae_scores.append(round(mae, 4))

    reg_scores_df = pd.DataFrame({'Model': modelnames, 'R2': r2_scores, 'MAE': mae_scores})
    reg_scores_df = reg_scores_df.sort_values(by='R2', ascending = False).reset_index().drop(columns=['index'])
    

    fig = px.bar(reg_scores_df, x = 'R2', y = 'Model', orientation = 'h', color = 'R2')
    fig['layout']['yaxis']['autorange'] = "reversed"
    st.plotly_chart(fig)

    st.dataframe(reg_scores_df.style.set_precision(4))

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
