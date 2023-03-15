# ------------- ToDo List --------------
# - crashproove
# - add clustering ?
# - st.experimental_data_editor make dataframe editable ?
# - export model ?
# - predict new data?
# - feature importance - maybe just use random forrest for that?
# - add xg boost ?

# ------------- Libraries & Project Specific Functions --------------


from Libs_Functions_SKLA import *


# ------------- Settings --------------

page_title = 'Machine Learning Automation'
page_description = 'This App automates predictive data analysis with the [scikit-learn](https://scikit-learn.org/stable/index.html) API. \
It is functional and automates the process steps for a small data science project or a first shot at model selection. \
The application does not replace a full and comprehensive Data Science Project. \
The application allows regression, classification and time series analysis with machine learning algorithms.'

page_icon = ':eyeglasses:' # emoji : https://www.webfx.com/tools/emoji-cheat-sheet/
layout = 'centered' # derfault but can be chenged to wide

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
image = Image.open('images/header_ML_no_back_ground_2.png')

st.image(image, use_column_width=True)
# st.title(page_title + " " + page_icon)
st.write(page_description)


# ------------- hide streamlit style --------------

hide_st_style = """
<style>

footer {visibility: hidden;}
</style>
"""

# header {visibility: hidden;}
# #MainMenu {visibility: hidden;} this would be another option (inkl.# )

st.markdown(hide_st_style, unsafe_allow_html=True)

# ------------- Plotly Configurations --------------

config_plotly = {
  'toImageButtonOptions': {
    'format': 'svg', # one of png, svg, jpeg, webp
    'filename': 'custom_image',
    'height': 500,
    'width': 700,
    'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
  },
  'displaylogo': False,
  'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
  'modeBarButtonsToAdd':['drawopenpath', 'drawline', 'drawcircle', 'drawrect', 'eraseshape']
}

drawing_color_plotly = '#f72d4e'

# ------------- Get data in and show it --------------


# --- Data upload

st.header("Raw Data Selection and Visualization")
st.write('')
st.subheader("Choose the Data")


csv_options = {
    'California Housing': ['https://raw.githubusercontent.com/sonarsushant/California-House-Price-Prediction/master/housing.csv', ','],
    'Winequality': ['local_data/winequality-red.csv', ';'],
    'Breast Cancer': ['local_data/breast_cancer.csv', ','], 
    'Bioactivity Acetylcholinesterase': ['local_data/acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv', ','],
    'Energy Consumption Hourly': ['https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv' , ','],
    'Consumer Price Index Switzerland (May 2000 = 100)': ['local_data/Konsumentenpreise_Index_Mai_2000.csv', ';']
}

csv_name = [i for i in csv_options]


use_csv_name = st.selectbox('**select dataset from list**', options= csv_name)
uploaded_file = st.file_uploader("**or upload your own**")

if uploaded_file is None:
    df = load_data(csv_options[use_csv_name][0], sep= csv_options[use_csv_name][1])
    df_name = use_csv_name
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep = None, engine='python')
        df_name = str(uploaded_file.name).removesuffix('.csv')
    except:
        try:
            uploaded_file.seek(0) # the buffering needs to be reset otherwise there is a parsing error
            df = pd.read_csv(uploaded_file, encoding='latin-1', sep = None, engine='python')
            df_name = str(uploaded_file.name).removesuffix('.csv')
        except:
            df = load_data(csv_options[use_csv_name][0], sep= csv_options[use_csv_name][1])
            df_name = use_csv_name
            st.warning('Could not read the file. Dataset from list is loaded.', icon="⚠️")


## head the data
st.subheader("First 1'000 Instances of " + df_name)
st.dataframe(df.head(1000))
#st.dataframe(df.head(1000).style.set_precision(2))
n_row, n_col = df.shape
st.write(n_col, " features, ", n_row, " instances, ", df.size, " total elements")

st.subheader("Column Info")
st.dataframe(show_info(df))


# --- Datetime conversion

st.subheader('Datetime conversion')
st.write('By default, datetime variables are loaded as type object. \
        A conversion is necessary if they are to be used as dates. \
        Time Series anlysis will only be possible if at least one feature has been converted to a date format.')

# find candidates for datetime conversion
date_candid_cols = datetime_candidate_col(df)

# change user selected date columns to datetime 
cat_cols = find_cat_cols(df)
us_date_var = st.multiselect(
    'Are there columns you want to recode as dates?',
    cat_cols, default=list(date_candid_cols))

datetimeformats = {'automatic': None,
               'day.month.Year': "%d.%m.%Y",
               'day/month/Year': "%d/%m/%Y",
               'day-month-Year': "%d-%m-%Y",
               'Year-month-day': "%Y-%m-%d"}

if len(us_date_var) > 0:
    us_datetimeformats = st.selectbox('Choose the input datetime format:', list(datetimeformats.keys()))

if len(us_date_var) > 0:
    df, converted_date_var = datetime_converter(df, us_date_var, datetimeformats, us_datetimeformats)
else:
    converted_date_var = [] 

if len(converted_date_var) > 0:
    cat_cols_no_date = list(cat_cols)
    cat_cols_no_date = set(cat_cols_no_date) - set(converted_date_var)
else:
    cat_cols_no_date = list(cat_cols)

# --- Plot features

st.subheader("Plot Selcted Features")

plot_types = ['Scatter Plot', 'Histogramm', 'Line Plot', 'Bar Plot', 'Box Plot', 'Heatmap of count']
axis_options = list(df.columns)
aggfun_options = {
    'None': '',
    'Sum': np.sum, 
    'Min': np.min,
    'Max': np.max,
    'Mean': np.mean, 
    'Median': np.median}


us_plot_type = st.selectbox('select plot type', plot_types)

col1_plot, col2_plot = st.columns(2)

with col1_plot:
    us_x_axis = st.selectbox('select x-axis', axis_options)


if us_plot_type != 'Histogramm':
    with col2_plot:
        us_y_axis = st.selectbox('select y-axis', axis_options, index = (len(axis_options)-1))

if us_plot_type not in ['Histogramm','Heatmap of count', 'Box Plot'] and us_y_axis in find_num_cols(df) and us_y_axis != us_x_axis:
    with col2_plot:
        us_y_axis_agg = st.selectbox('select y-axis aggregation', aggfun_options.keys(), index = 1)
    if us_y_axis_agg != 'None':
        agg_values = 'yes'
    else:
        agg_values = 'no'
else:
    agg_values = 'no'

if us_plot_type not in ['Histogramm','Heatmap of count']:
    color_options = axis_options.copy()
    color_options.append(None)
    color_options.remove(us_x_axis)
    if us_x_axis != us_y_axis:
        color_options.remove(us_y_axis)
    with col1_plot:
        us_color_group = st.selectbox('select color grouping', color_options, index = (len(color_options)-1))

if agg_values == 'yes':
    pivot_df = np.round(pd.pivot_table(df, values=us_y_axis, 
                                    index=[us_x_axis], 
                                    columns=us_color_group, 
                                    aggfunc=aggfun_options[us_y_axis_agg],
                                    fill_value=0),2)


# titel creation

if agg_values == 'yes':
    title_name = us_plot_type + ' - ' + us_x_axis + ' and ' + us_y_axis_agg + ' of '+ us_y_axis
elif us_plot_type != 'Histogramm': 
    title_name = us_plot_type + ' - ' + us_x_axis + ' and ' +  us_y_axis
else:
    title_name = us_plot_type + ' - ' + us_x_axis

 # plot user selected features   

if us_plot_type == 'Scatter Plot':
    if agg_values == 'yes':
        fig = px.scatter(pivot_df, title= title_name)
    else:
        fig = px.scatter(df, x = us_x_axis, y = us_y_axis, color = us_color_group, color_continuous_scale=px.colors.sequential.Blues_r,
                    title= title_name)
    fig.update_layout(xaxis_title= us_x_axis, yaxis_title= us_y_axis).update_xaxes(
                    categoryorder='category ascending')
elif us_plot_type == 'Histogramm':
    fig = px.histogram(df, x = us_x_axis, 
                title= title_name).update_layout(
                xaxis_title= us_x_axis, yaxis_title= 'count').update_xaxes(
                categoryorder='category ascending')
elif us_plot_type == 'Line Plot':
    if agg_values == 'yes':
        fig = px.line(pivot_df, title= title_name)
    else:
        fig = px.line(df, x = us_x_axis, y = us_y_axis, color = us_color_group,
                    title= title_name)
    fig.update_layout(xaxis_title= us_x_axis, yaxis_title= us_y_axis).update_xaxes(
                    categoryorder='category ascending')
elif us_plot_type == 'Bar Plot':
    if agg_values == 'yes':
        fig = px.bar(pivot_df, title= title_name)
    else:
        fig = px.bar(df, x = us_x_axis, y = us_y_axis, color = us_color_group,
                    title= title_name)
    fig.update_layout(xaxis_title= us_x_axis, yaxis_title= us_y_axis).update_xaxes(
                    categoryorder='category ascending')
elif us_plot_type == 'Box Plot':
    fig = px.box(df, x = us_x_axis, y = us_y_axis, color = us_color_group,
                title= title_name).update_layout(
                xaxis_title= us_x_axis, yaxis_title= us_y_axis).update_xaxes(
                categoryorder='category ascending')
elif us_plot_type == 'Heatmap of count':
    heatmap_df = pd.DataFrame(df.groupby([us_x_axis, us_y_axis])[us_x_axis].count())
    heatmap_df.rename(columns={us_x_axis: "count"}, inplace=True)
    heatmap_df.reset_index(inplace = True)
    heatmap_df = heatmap_df.pivot(index=us_y_axis, columns=us_x_axis)['count'].fillna(0)
    fig = px.imshow(heatmap_df, x=heatmap_df.columns, y=heatmap_df.index, title= title_name, text_auto=True, 
                    color_continuous_scale=px.colors.sequential.Blues_r)

fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
st.plotly_chart(fig, use_container_width=True, config = config_plotly)

use_cor_matrix = st.radio("Do you want to create a correlation matrix of the numeric variables?", ['no', 'yes'], horizontal=True)

if use_cor_matrix == 'yes':
    st.subheader("Correlation Matrix of Continuous Variables")
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale=px.colors.sequential.Blues_r) # reverse color by adding "_r" (eg. Blues_r) 
    fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
    st.plotly_chart(fig, use_container_width=True, config = config_plotly)


st.markdown("""---""")

# ------------- Data Splitting and Preprocessinng --------------

st.header('Data Splitting and Preprocessing')
st.write('The data is first split into a training and a test set. To avoid data leakage by train-test contamination all further \
         preprocessing steps will be executed separately in training (e.g. fit and transform) and test (e.g. only transform).')

st.write("")

st.subheader('Splitting into Training and Test set')

test_basis = ['size', 'date range']

# --- Auto extract features from daytime

# extract features from datetime 
if len(converted_date_var) > 0:
    # save the first var as index for time series analysis
    ts_index = df[converted_date_var[0]]
    # extract features of all dates
    df = create_time_features(df, converted_date_var)
    # user selected test basis
    st.write('If you want to do a regression in form of a time series anlysis, you need to select "date range".\
         The first feature you converted to date format will be used as index. Regardless of your selection\
         for test set splitting, all converted dates will undergo a feature extraction.')
    us_test_basis = st.radio('Do you want to base your test set on size (random instances) or a date range?', test_basis, index=0, horizontal=True)  
    
else:
    us_test_basis = test_basis[0]

# --- Split into train and test on testsize

# info: it is important that splitting is done before preprocessing to avoid target leakage.

if us_test_basis == test_basis[0]:
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

    # split on testsize
    train_df, test_df = split_testsize(df, us_test_size)

    # show split
    st.write('')
    n_row, n_col = train_df.shape
    st.write("Training set: " , n_row, " instances")
    n_row, n_col = test_df.shape
    st.write("Test set: " , n_row, " instances")

# --- Splittrain and test on date

elif us_test_basis == test_basis[1]:

    # Start prediction period
    min_date = min(ts_index)
    b_min_date = min_date.to_pydatetime()
    max_date = max(ts_index)
    max_date = max_date.to_pydatetime()
    b_max_date = max_date - timedelta(days=1)
    days_20pct = ((max_date - min_date) *0.2).days
    defalut_beg_day = max_date - timedelta(days=days_20pct)
    # user selection
    us_start_date = st.date_input("Beginning of prediction period", value = defalut_beg_day, min_value = b_min_date, max_value=b_max_date)
    
    # End prediction period
    e_min_date = us_start_date + timedelta(days=1)
    e_max_date = max_date
    #user selection
    us_end_date = st.date_input("End of prediction period", value = e_max_date, min_value = e_min_date, max_value=e_max_date)

    # Split 
    df_ts = df.set_index(ts_index)
    train_df, test_df = split_timeseries(df_ts, us_start_date, us_end_date)
    
    # Reset index and keep datetime index as col for later
    name_index = train_df.index.name
    new_name_datetimeindex = ("Index_"+name_index)
    train_df = train_df.reset_index(drop=False).rename(columns={name_index:new_name_datetimeindex})
    train_df[new_name_datetimeindex] = train_df[new_name_datetimeindex].astype(str)
    test_df = test_df.reset_index(drop=False).rename(columns={name_index:new_name_datetimeindex})
    test_df[new_name_datetimeindex] = test_df[new_name_datetimeindex].astype(str)

    # show split
    st.write('')
    n_row, n_col = train_df.shape
    st.write("Training set: " , n_row, " instances")
    n_row, n_col = test_df.shape
    st.write("Test set: " , n_row, " instances")


# --- Preprocessing

st.write('')
st.subheader('Data Preprocessing')
st.write("NA-values will automatically be filled (numerical variables by their mean and categorical by their mode) \
         alternatively all instances with NA-values can be droped.")

# --- fill or drop NA

NA_handling = {
    'fill NA (mean/mode)' : 'fill_na',
    'drop instances with NA' : 'drop_na'
}

us_na_handling = st.radio('Do you want to fill or drop the NA?', NA_handling.keys(), horizontal=True, index=0)

if us_na_handling == 'fill NA (mean/mode)':
    train_df, test_df = fill_na_si_mean_mode(train_df, test_df)
elif us_na_handling == 'drop instances with NA':
    train_df = train_df.dropna()
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.dropna()
    test_df = test_df.reset_index(drop=True)


# --- look for duplicates

n_duplicate_rows = (len(train_df[train_df.duplicated()]) + len(test_df[test_df.duplicated()]))

if n_duplicate_rows > 0:


    dup_handling = {
        'leave as is' : 'leave',
        'drop dupplicate instances' : 'drop'
    }

    us_dup_handling = st.radio('How do you want to handle the dupplicate instances?', dup_handling.keys(), horizontal=True, index=0)
    

    if dup_handling[us_dup_handling] == 'drop':
        train_df = train_df.drop_duplicates(train_df)
        test_df = test_df.drop_duplicates(test_df)
        st.info('There were '+ str(n_duplicate_rows) +' dupplicated instances, where duplicates have been removed', icon="ℹ️")
    else:
        st.warning('There are '+ str(n_duplicate_rows) +' dupplicate instances', icon="⚠️")
else: 
    us_dup_handling = st.radio('There are no duplicates in the dataset:', ['no action necessary'], horizontal=True, index=0)


# --- PCA

us_pca_var = st.multiselect(
    'Do you want to do a PCA on some columns?',
    find_num_cols(train_df), default=None)

if len(us_pca_var) > 0:
    train_df, test_df = pca_on_us_col(train_df, test_df, us_pca_var)
    


# --- safety for cat columns with a lot of uniques

# a lot is set to 100
n = 100
list_alotofuniques = col_with_n_uniques(train_df, cat_cols_no_date, n)

dummy_default_list = set(cat_cols_no_date) - set(list_alotofuniques)

if len(list_alotofuniques) > 0:
    for i in list_alotofuniques:
        st.warning(i + ' with type = object but more than 100 unique values', icon="⚠️")


# --- dummie code user selected cat columns

us_dummie_var = st.multiselect(
    'Which columns do you want to recode as dummies?',
    cat_cols_no_date, default= list(dummy_default_list))

if len(us_dummie_var) > 0:
    train_df, test_df = dummi_encoding(train_df, test_df, us_dummie_var)

# --- set the daytime index if it has been chosen

if us_test_basis == test_basis[1]:
    train_df = train_df.set_index(new_name_datetimeindex, drop=True)
    test_df = test_df.set_index(new_name_datetimeindex, drop=True)

## show data after preprocessing

st.subheader("Training Dataframe After Preprocessing")
st.dataframe(train_df.head(1000))
# st.dataframe(train_df.head(1000).style.set_precision(2))

n_row, n_col = train_df.shape
st.write("Training set: " , n_col, " features, ", n_row, " instances, ", train_df.size, " total elements")
n_row, n_col = test_df.shape
st.write("Test set: " , n_col, " features, ", n_row, " instances, ", test_df.size, " total elements")

st.subheader("Column Info")
st.dataframe(show_info(train_df))



st.subheader("Download the Preprocessed Data")

st.download_button(
  label="Download Training Data as CSV",
  data=convert_df_to_csv(train_df),
  file_name= (df_name + 'train_preprocessed.csv'),
  mime='text/csv',
  key='ptrd'
)

st.download_button(
  label="Download Test Data as CSV",
  data=convert_df_to_csv(test_df),
  file_name= (df_name + 'test_preprocessed.csv'),
  mime='text/csv',
  key='pted'
)

st.markdown("""---""")

# ------------- Target and Feature selection --------------

st.header('Target and Feature Selection')
st.write('')


# --- Target selection

st.subheader('Choose your Target ( Y )')

us_y_var = st.selectbox(
    'Which column do you want to predict?',
    train_df.columns)

st.write('')


# --- Feature selection

st.subheader('Choose your Features ( X )')
x_options_df = train_df.drop(columns=[us_y_var])
cat_cols_x = find_cat_cols(x_options_df)
x_options_df = x_options_df.drop(cat_cols_x, axis = 1)
if len(cat_cols_x) > 0:
    st.warning('Models can not be launched with categroical features '+str(list(cat_cols_x))+'.\
                Please recode as dummies otherwise they can only be used as dependent variable.', icon="⚠️")


variance_threshold_options = {
        'no' : False,
        'search and drop' : True
    }

use_variance_threshold = st.radio("Do you want to look for and drop features with 0 variance?", variance_threshold_options.keys(), horizontal= True)

if variance_threshold_options[use_variance_threshold]:
    try:
        # variancetreshold = st.number_input("Feature Selection with minimal variance within feature", min_value=0.0, max_value=1.0, value=0.0)
        n_col_before = len(x_options_df.columns)
        selection = VarianceThreshold(threshold=(0)).set_output(transform="pandas")
        x_options_df = selection.fit_transform(x_options_df)
        n_col_after = len(x_options_df.columns)
        n_del_col = (n_col_before - n_col_after)
        st.write("Number of deleted columns: ", n_del_col)
        st.write("Number of retained columns: ", n_col_after)
    except (ValueError):
        cat_col_names = list(x_options_df.select_dtypes(include=['object', 'bool']).columns)
        st.warning('Could not perform variance reduction. Is there a categorical  variable that has not been recoded as dummy? Maybe: ' + str(cat_col_names), icon="⚠️")
    except:
        st.warning('Could not perform variance reduction.', icon="⚠️")

us_x_var = st.multiselect(
    'Which columns do you want as inputs for the prediction?',
    list(x_options_df),
    default=list(x_options_df))


y_train_ser = train_df[us_y_var]
X_train_df = train_df[us_x_var]

y_test_ser = test_df[us_y_var]
X_test_df = test_df[us_x_var]

st.subheader('Selected Variables')

st.write('Chosen target:')

col1, col2 = st.columns((0.25, 0.75))
col1.dataframe(y_train_ser[0:1000])

fig = px.histogram(train_df, x= us_y_var )
col2.plotly_chart(fig, use_container_width=True)

st.write('Chosen features:')
st.dataframe(X_train_df.head(1000))
#st.dataframe(X_train_df.head(1000).style.set_precision(2))
n_row, n_col = X_train_df.shape
st.write(n_col, " features, ", n_row, " instances, ", X_train_df.size, " total elements")

st.markdown("""---""")

st.header('Launch Model Training and Prediction')
descr_model_launch = 'Regression models are only launchable if the chosen target variable is a number and \
classification models in turn, only if it is of type bool, object or int. \
Time Series analysis is only callable if at least one feature has been recoded as date. \
Unless otherwise indicated, all models are run in the [scikit-learn](https://scikit-learn.org/stable/index.html) -default configuration. \
Crossvalidation, gridsearch and manual hyper parameter tuning are (not yet) implemented.'

st.write(descr_model_launch)
st.write("")

st.subheader('Scaling')


# scaler selection

scalers = {
    'MinMaxScaler' : MinMaxScaler(),
    'StandardScaler' : StandardScaler(),
    'RobustScaler' : RobustScaler()
}

us_scaler_key = st.radio('What scaler do you want to use?', list(scalers.keys()), index=0, horizontal = True)

# ------------- Launch model calculation --------------

# initialize states for x and y selction by user 
# if user model relevant selection for all models changes, then we dont want an automatic model rerun below

if "y_var_user" not in st.session_state:
    st.session_state.y_var_user = us_y_var

if "x_var_user" not in st.session_state:
    st.session_state.x_var_user = us_x_var

if us_test_basis == test_basis[1]:
    d1 = us_start_date.strftime("%d/%m/%Y")
    d2 = us_end_date.strftime("%d/%m/%Y")
    test_size_identifier = (d1+d2)
else:
    test_size_identifier = us_test_size

if "test_size_user" not in st.session_state:
    st.session_state.test_size_user = test_size_identifier

if "scaler_user" not in st.session_state:
    st.session_state.scaler_user = us_scaler_key

check_y_no_change = st.session_state.y_var_user == us_y_var
check_x_no_change = st.session_state.x_var_user == us_x_var
check_test_size_no_change = st.session_state.test_size_user == test_size_identifier
check_scaler_no_change = st.session_state.scaler_user == us_scaler_key

# check y for possible models
reg_cols = find_num_cols(df)
clas_cols = df.select_dtypes(include=['object', 'bool', 'int']).columns
cat_cols_x = find_cat_cols(X_train_df)


# function for running and comparing regression models

regression_models = {
          'LinearRegression': LinearRegression(),
          'Ridge' : Ridge(),
          'RandomForestRegressor': RandomForestRegressor(),
          'LinearSVR' : LinearSVR(),
          'SVR' : SVR(),
          'GradientBoostingRegressor': GradientBoostingRegressor(),
          # 'HistGradientBoostingRegressor': HistGradientBoostingRegressor(), # it is extremely slow on streamlit hosted server
          'DummyRegressor': DummyRegressor()
          }

@st.cache_data(ttl = time_to_live_cache) 
def reg_models_comparison(X_train, X_test, y_train, y_test, us_reg_models):

    # progress bar
    progress_text = us_reg_models.copy()
    progress_text.append('complete')
    my_bar = st.progress(0, text=str(progress_text[0]))
    increment_progress = int(math.ceil(100 / len(us_reg_models)))
    text_counter = 0
    percent_complete = 0

    # init values to be stored
    modelnames = []
    r2_scores = []
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    duration_scores = []
    predictions = {}
    residuals = {}
    
    # loop through models
    for i in us_reg_models:
        m_reg = regression_models[i]

        start_time = time.time()
        m_reg.fit(X_train, y_train)
        y_test_predict = m_reg.predict(X_test)
        end_time = time.time()

        duration = (end_time - start_time)
        mae = mean_absolute_error(y_test, y_test_predict)
        mse = mean_squared_error(y_test, y_test_predict)
        rmse = math.sqrt(mse)
        r2 = m_reg.score(X_test, y_test)

        modelnames.append(i)
        r2_scores.append(round(r2, 4))
        mae_scores.append(round(mae, 4))
        mse_scores.append(round(mse, 4))
        rmse_scores.append(round(rmse, 4))
        duration_scores.append(round(duration, 4))

        predictions[i] = y_test_predict
        residuals[i] = (y_test_predict - y_test)

        text_counter += 1
        percent_complete += increment_progress
        percent_complete = min(100, percent_complete)
        my_bar.progress(percent_complete, text=str(progress_text[text_counter]))
        
        
    # create score dataframe
    reg_scores_df = pd.DataFrame({'Model': modelnames, 'R2': r2_scores, 'MAE': mae_scores, 'MSE': mse_scores, 'RMSE': rmse_scores, 'Duration (sec)': duration_scores})
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


classifier_models = {'RandomForestClassifier': RandomForestClassifier(),
                        'LogisticRegression(solver="sag")': LogisticRegression(solver='sag'), #https://scikit-learn.org/stable/modules/linear_model.html#solvers
                        'SVC': SVC(),
                        'LinearSVC': LinearSVC(),
                        'DummyClassifier': DummyClassifier(),
                        'KNeighborsClassifier': KNeighborsClassifier(),
                        'Perceptron': Perceptron()
                        }

# function for running and comparing classification models

@st.cache_data(ttl = time_to_live_cache) 
def class_models_comparison(X_train, X_test, y_train, y_test, us_clas_models):
    
    # progress bar
    progress_text = us_clas_models.copy()
    progress_text.append('complete')
    my_bar = st.progress(0, text=str(progress_text[0]))
    increment_progress = int(math.ceil(100 / len(us_clas_models)))
    text_counter = 0
    percent_complete = 0
    
    # init values to be stored
    modelnames = []
    accuracy_scores = []
    w_precision_scores = []
    w_recall_scores = []
    w_f1_scores = []
    duration_scores = []
    predictions = {}
    class_labels = {}

    
    # loop through models
    for i in us_clas_models:
        m_clas = classifier_models[i]

        start_time = time.time()
        m_clas.fit(X_train, y_train)
        y_test_predict = m_clas.predict(X_test)
        end_time = time.time()

        duration = (end_time - start_time)
        accuracy = accuracy_score(y_test, y_test_predict)
        precision = precision_score(y_test, y_test_predict, average='weighted')
        recall = recall_score(y_test, y_test_predict, average='weighted')
        f1 = f1_score(y_test, y_test_predict, average='weighted')

        modelnames.append(i)
        accuracy_scores.append(round(accuracy, 4))
        w_precision_scores.append(round(precision, 4))
        w_recall_scores.append(round(recall, 4))
        w_f1_scores.append(round(f1, 4))
        duration_scores.append(round(duration, 4))
        predictions[i] = y_test_predict
        class_labels[i] = list(m_clas.classes_)

        text_counter += 1
        percent_complete += increment_progress
        percent_complete = min(100, percent_complete)
        my_bar.progress(percent_complete, text=str(progress_text[text_counter]))
        
    # create score dataframe
    clas_scores_df = pd.DataFrame({'Model': modelnames, 'Accuracy': accuracy_scores, 'Precision': w_precision_scores, 'Recall': w_recall_scores, 'F1': w_f1_scores, 'Duration (sec)': duration_scores})
    clas_scores_df = clas_scores_df.sort_values(by='Accuracy', ascending = False).reset_index().drop(columns=['index'])

    # create prediction dataframe
    clas_pred_y_df = pd.DataFrame(predictions)
    clas_pred_y_df['y_test'] = pd.Series(y_test)
    
    # create class dataframe
    clas_label_df = pd.DataFrame(class_labels)
    
    # return the 3 dataframes
    return clas_scores_df, clas_pred_y_df, clas_label_df


if len(cat_cols_x) > 0:
    st.warning('Models can not be launched with categroical independent variables '+str(list(cat_cols_x))+'. Please recode as dummies or exclude.', icon="⚠️")

#--- Regression models

# only if y is a number type
if us_y_var in reg_cols and len(cat_cols_x) == 0:

    st.markdown("""---""")
    st.subheader("Regression Models")

    us_reg_models = st.multiselect('What regression models do you want to launch and compare?', regression_models.keys(), default= list(regression_models.keys()))

    # check that at least one model has been selected otherwise don't show launch button
    if len(us_reg_models) > 0:
        start_reg_models = st.button("Start Regression Analysis")
    elif len(us_reg_models) == 0:
        start_reg_models = False
        st.warning('Please select at least one model if you want to do a regression analysis.', icon="⚠️")


    # initialise session state - this keeps the analysis open when other widgets are pressed and therefore script is rerun
    # if user model selection changes, then we dont want an automatic model rerun below

    if "start_reg_models_state" not in st.session_state :
        st.session_state.start_reg_models_state = False
    
    if "reg_model_selection" not in st.session_state:
        st.session_state.reg_model_selection = us_reg_models

    check_reg_models_no_change = st.session_state.reg_model_selection == us_reg_models

    if (start_reg_models or (st.session_state.start_reg_models_state 
                            and check_y_no_change 
                            and check_x_no_change
                            and check_test_size_no_change
                            and check_scaler_no_change
                            and check_reg_models_no_change)):
        st.session_state.start_reg_models_state = True
        st.session_state.y_var_user = us_y_var
        st.session_state.x_var_user = us_x_var
        st.session_state.test_size_user = test_size_identifier
        st.session_state.scaler_user = us_scaler_key
        st.session_state.reg_model_selection = us_reg_models

        # run scaling
        X_train, X_test, y_train, y_test = scaling_test_train(X_train_df, X_test_df, y_train_ser, y_test_ser, us_scaler_key)

        # run model and compare
        reg_scores_df, reg_pred_y_df, reg_res_y_df = reg_models_comparison(X_train, X_test, y_train, y_test, us_reg_models)

        # Titel
        st.subheader("Results for Regression Models on Testset")

        # plot model scores
        fig = px.bar(reg_scores_df, x = 'R2_floored_0', y = 'Model', orientation = 'h', color = 'R2_floored_0',
            title="Model Comparison on R2 (floored at 0)")
        fig['layout']['yaxis']['autorange'] = "reversed"
        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        # show tabel of model scores
        st.dataframe(reg_scores_df.style.set_precision(4))
        

        # download tabel of model scores
        st.download_button(
        label="Download Model Comparison as CSV",
        data=convert_df_to_csv(reg_scores_df),
        file_name= (df_name + '_model_comparison.csv'),
        mime='text/csv',
        key='regmodcomp'
        )

        st.write('')
        st.subheader('Plot Predictions')

        use_reg_model = st.radio('show results for:', reg_scores_df.Model, key='use_reg_model') 

        # plot model value vs actual
        fig = px.scatter(reg_pred_y_df, x = 'y_test', y = use_reg_model, 
                    title= 'True Target Value vs Model Prediction - ' + use_reg_model).update_layout(
                    yaxis_title="Model Prediction", xaxis_title="True Target Value")
        fig = fig.add_traces(px.line(reg_pred_y_df, x='y_test', y='y_test', color_discrete_sequence=["yellow"]).data)
        # legend naming traces and showing in legend
        fig['data'][0]['name']= us_y_var
        fig['data'][0]['showlegend']=True
        fig['data'][1]['name']='f(x) = x'
        fig['data'][1]['showlegend']=True
        # fig = fig.update_traces(selector = dict(name="first_trace"))
        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        # plot histogramm of residuals
        fig = px.histogram(reg_res_y_df, x = use_reg_model,
                    title="Histogramm of Residuals - " + use_reg_model).update_layout(
                    xaxis_title="Residuals")

        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        # plot residuals and True Target Value
        fig = px.scatter(reg_res_y_df, x = 'y_test', y = use_reg_model,
                    title="Residuals and True Target Value - " + use_reg_model).update_layout(
                    xaxis_title="True Target Value", yaxis_title="Residuals")

        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        # download model prediction and true value
        st.download_button(
        label="Download Model Predictions and True Value as CSV",
        data=convert_df_to_csv(reg_pred_y_df),
        file_name= (df_name + '_model_prediction.csv'),
        mime='text/csv',
        key='regmodpred'
        )


#--- Classification models

# only if y is a number type

if us_y_var in clas_cols and len(cat_cols_x) == 0:

    st.markdown("""---""")

    
    st.subheader("Classification Models")

    us_clas_models = st.multiselect('What classification models do you want to launch and compare?', classifier_models.keys(), default= list(classifier_models.keys()))
    
    # check that at least one model has been selected otherwise don't show launch button
    if len(us_clas_models) > 0:
        start_clas_models = st.button("Start Classification Analysis")
    elif len(us_clas_models) == 0:
        start_clas_models = False
        st.warning('Please select at least one model if you want to do a classification analysis.', icon="⚠️")


    # initialise session state - this keeps the analysis open when other widgets are pressed and therefore script is rerun
    # if user model selection changes, then we dont want an automatic model rerun below

    if "start_clas_models_state" not in st.session_state :
        st.session_state.start_clas_models_state = False
    
    if "clas_model_selection" not in st.session_state:
        st.session_state.clas_model_selection = us_clas_models

    check_clas_models_no_change = st.session_state.clas_model_selection == us_clas_models
     

    if (start_clas_models or (st.session_state.start_clas_models_state 
                            and check_y_no_change 
                            and check_x_no_change 
                            and check_test_size_no_change
                            and check_scaler_no_change
                            and check_clas_models_no_change)):
        st.session_state.start_clas_models_state = True
        st.session_state.y_var_user = us_y_var
        st.session_state.x_var_user = us_x_var
        st.session_state.test_size_user = test_size_identifier
        st.session_state.scaler_user = us_scaler_key
        st.session_state.clas_model_selection = us_clas_models

        # run scaling
        X_train, X_test, y_train, y_test = scaling_test_train(X_train_df, X_test_df, y_train_ser, y_test_ser, us_scaler_key)

        # run classification models
        clas_scores_df, clas_pred_y_df, clas_label_df = class_models_comparison(X_train, X_test, y_train, y_test, us_clas_models)

        # Titel
        st.subheader("Results for Classification Models on Testset")
        
        fig = px.bar(clas_scores_df, x = 'Accuracy', y = 'Model', orientation = 'h', color = 'Accuracy')
        fig['layout']['yaxis']['autorange'] = "reversed"
        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        st.dataframe(clas_scores_df.style.set_precision(4))

        # download tabel of model scores
        st.download_button(
        label="Download Model Comparison as CSV",
        data=convert_df_to_csv(clas_scores_df),
        file_name= (df_name + '_model_comparison.csv'),
        mime='text/csv',
        key='clasmodcomp'
        )

        st.write('')
        st.subheader('Plot Predictions')

        us_clas_model_result = st.radio('show results for:', clas_scores_df.Model, key='us_clas_model_result')

        conf_matrix_df = pd.DataFrame(clas_pred_y_df.groupby([us_clas_model_result, 'y_test'])['y_test'].count())
        conf_matrix_df.rename(columns={'y_test': "count"}, inplace=True)
        conf_matrix_df.reset_index(inplace = True)
        conf_matrix_df = conf_matrix_df.pivot(index='y_test', columns=us_clas_model_result)['count'].fillna(0)
        fig = px.imshow(conf_matrix_df, x=conf_matrix_df.columns, y=conf_matrix_df.index, title= 'Confusionmatrix - ' + us_clas_model_result
                        , text_auto=True, color_continuous_scale=px.colors.sequential.Blues_r).update_layout(
                        xaxis_title="predicted label", yaxis_title="true label")
        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        fig = px.histogram(clas_pred_y_df, x = 'y_test', title = 'Histogram of True Values -' + us_clas_model_result).update_layout(
        xaxis_title="True Values")
        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        fig = px.histogram(clas_pred_y_df, x = us_clas_model_result, title = 'Histogram of Predicted Values - ' + us_clas_model_result).update_layout(
        xaxis_title="Predicted Values")
        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        # download model prediction and true value
        st.download_button(
        label="Download Model Predictions and True Value as CSV",
        data=convert_df_to_csv(clas_pred_y_df),
        file_name= (df_name + '_model_prediction.csv'),
        mime='text/csv',
        key='clasmodpred'
        )

        @st.cache_data(ttl = time_to_live_cache) 
        def clas_score_label(clas_pred_y_df):
            label_list = list(clas_pred_y_df['y_test'].unique())
            precision_list = []
            recall_list = []

            for i in label_list:
                sub_df_prec = clas_pred_y_df[clas_pred_y_df[us_clas_model_result] == i]
                precision = precision_score(sub_df_prec['y_test'], sub_df_prec[us_clas_model_result], average='micro')
                precision_list.append(precision)

                sub_df_reca = clas_pred_y_df[clas_pred_y_df['y_test'] == i]
                recall = recall_score(sub_df_reca['y_test'], sub_df_reca[us_clas_model_result], average='micro')
                recall_list.append(recall)

            score_label_df = pd.DataFrame({'Label': label_list, 'Precision' : precision_list, 'Recall': recall_list})
            score_label_df['F1'] = 2 * (score_label_df['Precision'] * score_label_df['Recall']) / (score_label_df['Precision'] + score_label_df['Recall'])
            return score_label_df

        score_label_df = clas_score_label(clas_pred_y_df)

        st.markdown('**Scores per Category**')
        st.dataframe(score_label_df)

        # download model prediction and true value
        st.download_button(
        label="Download Scores Per Category as CSV",
        data=convert_df_to_csv(score_label_df),
        file_name= (df_name + '_scores_category.csv'),
        mime='text/csv',
        key='classcorescateg'
        )





#--- Time Series models

if len(converted_date_var) > 0 and len(cat_cols_x) == 0 and us_test_basis == test_basis[1]:

    st.markdown("""---""")
    st.subheader("Regression Models on Time Series")
 
    # time series regression model selection
    us_ts_models = st.multiselect('What regression models do you want to launch and compare for the time series?', regression_models.keys(), default= list(regression_models.keys()))

    # check that at least one model has been selected otherwise don't show launch button
    if len(us_ts_models) > 0:
        start_ts_models = st.button("Start Time Series Analysis")
    elif len(us_ts_models) == 0:
        start_ts_models = False
        st.warning('Please select at least one model if you want to do a time series analysis.', icon="⚠️")


    # initialise session state - this keeps the analysis open when other widgets are pressed and therefore script is rerun
    # if user model selection changes, then we dont want an automatic model rerun below

    if "start_ts_models_state" not in st.session_state :
        st.session_state.start_ts_models_state = False
    
    if "ts_model_selection" not in st.session_state:
        st.session_state.ts_model_selection = us_ts_models

    check_ts_models_no_change = st.session_state.ts_model_selection == us_ts_models    
        
    if (start_ts_models or (st.session_state.start_ts_models_state 
                            and check_y_no_change 
                            and check_x_no_change
                            and check_scaler_no_change
                            and check_ts_models_no_change
                            and check_test_size_no_change)):
        st.session_state.start_ts_models_state = True
        st.session_state.y_var_user = us_y_var
        st.session_state.x_var_user = us_x_var
        st.session_state.test_size_user = test_size_identifier
        st.session_state.scaler_user = us_scaler_key
        st.session_state.ts_model_selection = us_ts_models
        
       
        # remove duplicates from index and sort
        train_df = remove_duplicated_index(train_df)
        train_df = train_df.sort_index()
        test_df = remove_duplicated_index(test_df)
        test_df = test_df.sort_index()


        y_ser_ts = df_ts[us_y_var]
        X_df_ts = df_ts[us_x_var]

        # run scaling
        X_train, X_test, y_train, y_test = scaling_test_train(X_train_df, X_test_df, y_train_ser, y_test_ser, us_scaler_key)

        # run models
        ts_scores_df, ts_pred_y_df, ts_res_y_df = reg_models_comparison(X_train, X_test, y_train, y_test, us_ts_models)

        # Titel
        st.subheader("Results for Regression Models on Testset")

        # plot model scores
        fig = px.bar(ts_scores_df, x = 'R2_floored_0', y = 'Model', orientation = 'h', color = 'R2_floored_0',
            title="Model Comparison on R2 (floored at 0)")
        fig['layout']['yaxis']['autorange'] = "reversed"
        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        # show tabel of model scores
        st.dataframe(ts_scores_df.style.set_precision(4))

        # download tabel of model scores
        st.download_button(
        label="Download Model Comparison as CSV",
        data=convert_df_to_csv(ts_scores_df),
        file_name= (df_name + '_model_comparison.csv'),
        mime='text/csv',
        key='tsmodcomp'
        )

        st.write('')
        st.subheader('Plot Predictions')

        # user model result selection
        use_ts_model = st.radio('show results for:', ts_scores_df.Model, key='use_ts_model')

        # create a complete dataframe for plotting
        ts_pred_y_df.set_index(test_df.index, inplace=True)
        df_ts = pd.concat([train_df, test_df], axis = 0)
        pred_df_ts = pd.concat([df_ts, ts_pred_y_df], axis = 1)

        pred_df_ts[new_name_datetimeindex] = pred_df_ts.index # two y doesn't work on index appearently
        fig = px.line(pred_df_ts, x=new_name_datetimeindex, y=[us_y_var, use_ts_model]).update_layout(
                xaxis_title=new_name_datetimeindex, 
                yaxis_title= (us_y_var + " and Prediction"),
                title = 'Time Series and Prediction')
        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        ts_res_y_df.set_index(test_df.index, inplace=True)
        
        # plot histogramm of residuals
        fig = px.histogram(ts_res_y_df, x = use_ts_model,
                    title="Histogramm of Residuals - " + use_ts_model).update_layout(
                    xaxis_title="residuals")

        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        # plot residuals and True Target Value
        ts_res_y_df[new_name_datetimeindex] = ts_res_y_df.index # two y doesn't work on index appearently
        fig = px.bar(ts_res_y_df, x = new_name_datetimeindex, y = use_ts_model,
                    title="Residuals over Datetime - " + use_ts_model).update_layout(
                    xaxis_title=new_name_datetimeindex, yaxis_title="residuals")

        fig = fig.update_layout(newshape_line_color = drawing_color_plotly)    
        st.plotly_chart(fig, use_container_width=True, config = config_plotly)

        # download model prediction and true value
        st.download_button(
        label="Download Model Predictions and True Value as CSV",
        data=convert_df_to_csv(ts_res_y_df),
        file_name= (df_name + '_model_prediction.csv'),
        mime='text/csv',
        key='tsmodpred'
        )

