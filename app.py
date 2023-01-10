import streamlit as st
import pandas as pd
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import multiprocessing


header = st.container()
dataset = st.container()
features = st.container()
model_train = st.container()

# ********  Reading File  *********
@st.cache
def read_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # ********  Data Preprocessing  ********
    df.columns=[i.strip() for i in df.columns]
    # Add new features
    df['Diff1']=df['Close'].diff(1)
    df['Diff2']=df['Close'].diff(2)
    df['Diff3']=df['Close'].diff(3)
    df['Diff4']=df['Close'].diff(4)
    df['Diff5']=df['Close'].diff(5)
    # drop null values
    df.dropna(inplace=True)
    return df

# ******  Plot Class Distribution Chart  *******
def ploting(df = None):
    st.write(df.head())
    st.subheader("Target Feature Class Distribution")
    classes_dist = df['Target'].value_counts()
    # convert series to dataframe
    classes_dist_df = pd.DataFrame({'classes':classes_dist.index, 'count':classes_dist.values})

    # add color column
    classes_dist_df['color'] = ['#6F3D86','#FFA600','#228B22'][:len(classes_dist_df)]

    # Create a chart
    chart = alt.Chart(classes_dist_df).mark_bar().encode(
        x='classes',
        y='count',
        color=alt.Color('color:N'),
        size = alt.Size(value=50)
    )

    st.altair_chart(chart, theme="streamlit", use_container_width=True)

# *********  Model Trainig Function  **********
def model_training(df=None, input_features=None, models=[], max_depth=3, n_estimator=100, n_core=1):
    model_scores, model_obj = {}, {}
    standard = StandardScaler()
    X = standard.fit_transform(df[input_features])
    Y = df['Target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    
    if 'Random Forest' in models:
        SRF = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=0, n_jobs=n_core)
        SRF.fit(X_train,Y_train)
        predications = SRF.predict(X_test)
        model_scores['Random Forest'] = accuracy_score(predications, Y_test)
        model_obj['Random Forest'] = SRF
        
    
    if 'AdaBoost' in models:
        base_model = DecisionTreeClassifier(max_depth=2)
        ada_model = AdaBoostClassifier(base_estimator=base_model, n_estimators=n_estimator)
        ada_model.fit(X_train, Y_train)
        predications = ada_model.predict(X_test)
        model_scores['AdaBoost'] = accuracy_score(predications, Y_test)
        model_obj['AdaBoost'] = ada_model

    if 'XGBoost' in models:
        Y_train1 = Y_train.copy()
        Y_test1 = Y_test.copy()
        Y_train1[Y_train1 == -1] = 2
        Y_test1[Y_test1 == -1] = 2
        xg_reg = XGBClassifier(eval_metric='mlogloss', n_estimators=n_estimator)
        xg_reg.fit(X_train, Y_train1)
        predications = xg_reg.predict(X_test)
        model_scores['XGBoost'] = accuracy_score(predications, Y_test1)
        model_obj['XGBoost'] = xg_reg
    
    results = { 'scores' : model_scores,
                'models' : model_obj}

    return results    



# &&&&&&&&&&&  Main Coding Part  &&&&&&&&&&
with header:
    st.title("Stock Market Trend Classification")
    st.text("Stock Market trend classification project. In which we will classify trend of \nmarket like Bullish, Bearish or nothing...")
    st.text("\n")

try:
    with dataset:
        st.header("Nasdaq 30 Second Timeframe Dataset ")
        uploaded_file = st.file_uploader("Select Stocks Data")
        if uploaded_file is not None:
            df = read_file(uploaded_file)
            ploting(df)
        
    with features:
        st.header("Features Engineering")
        st.text("We add some feature in our dataset & did target feature labeling.")
        st.markdown("We have added some **Indicators** & **Patterns** in our dataset. List of those indicators below")
        st.markdown("&emsp;&emsp;&emsp; **ATR &emsp;&emsp;&emsp;&emsp;&emsp; RSI &emsp;&emsp;&emsp;&emsp;&emsp; AD &emsp;&emsp;&emsp;&emsp;&emsp; AROON_DOWN &emsp;&emsp;&emsp; AROON_UP**")
        st.markdown("&emsp;&emsp;&emsp; **ADX &emsp;&emsp;&emsp;&emsp;&emsp; STOCH_K &emsp;&emsp;&nbsp; STOCH_D &emsp;&emsp;&nbsp; MACD &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp; MACD_SIGNAL**")
        st.markdown("&emsp;&emsp;&emsp; **HAMMER &emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp; INVERTED_HAMMER &emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; ENGULFING**")
        st.markdown("We also add percentage difference between **Last 5 Consective Bars** in our dataset. List of those Columns below")
        st.markdown("&emsp;&emsp;&emsp; **Diff1 &emsp;&emsp;&emsp;&emsp;&emsp; Diff2 &emsp;&emsp;&emsp;&emsp;&emsp; Diff3 &emsp;&emsp;&emsp;&emsp;&emsp; Diff4 &emsp;&emsp;&emsp; Diff5**")

    with model_train:
        st.header("Model Training")
        st.text("We will use different models like Random Forest, XGBoost & etc to train model.")
        
        if uploaded_file is not None:
            feature_set = []
            model_set = ['Random Forest', 'XGBoost', 'AdaBoost']
            feature_set = list(df.columns.drop(['Target','Date','Time']))
            sel_col, disp_col = st.columns(2)
            # Filters
            max_depth = sel_col.slider("What should be max depth of Model?", min_value=2, max_value=100, value=4)
            n_cores = sel_col.slider("What should be number of CPU cores to train Model?", min_value=1, max_value=multiprocessing.cpu_count(), value=1)
            n_estimator = sel_col.selectbox("How many tree should be use in Model?",[100,150,200,250,300,'No Limit'])      
            model_set = sel_col.multiselect("Which feature should be use as input features?", model_set, default=['Random Forest'])
            feature_set = sel_col.multiselect("Which feature should be use as input features?", feature_set, default=feature_set)
            # Model Scores Section
            disp_col.subheader("Model Scores")
            if (feature_set) and sel_col.button("Submit"):
                results = model_training(df, feature_set, model_set, max_depth, n_estimator, n_cores)
                for key in results['scores'].keys():
                    disp_col.write(f'{key} : {results["scores"][key]}')
                # Downloading Part
                disp_col.subheader("Download Trained Model")    
                # Download Model
                model_name = list(results['models'].keys())[0]
                filename = f"{model_name}_model.pkl"
                with open(filename, 'wb') as f:
                    joblib.dump(results['models'][model_name], f)

                with open(filename, 'rb') as f:
                    disp_col.download_button(f'Download {filename}', f, 'application/pkl')
except:
    st.write("Dataset not Correct")