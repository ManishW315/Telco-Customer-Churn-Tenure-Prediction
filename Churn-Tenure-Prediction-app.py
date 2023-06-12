# Imports
import streamlit as st
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('Telco-Customer-Churn.csv')

# Background Style
page_bg_img = '''
<style>
[data-testid='stAppViewContainer'] {
background-image: url("https://img.freepik.com/premium-vector/grainy-gradient-background-using-different-colors_606954-9.jpg");
background-size: cover;
}
[data-testid='stSidebar'] {
background-color: rgba(80, 30, 50, 0.5);
}

</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Heading
st.write("""
# Telco Churn and Tenure Prediction
""")
st.text("")
st.text("")
st.text("")


# Sidebar User Inputs
def user_input_features():
    st.sidebar.write('## Demographic Information')
    gender = st.sidebar.selectbox('gender', ('Female', 'Male'))
    SeniorCitizen = st.sidebar.selectbox('SeniorCitizen', ('Yes', 'No'))
    Partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    Dependents = st.sidebar.selectbox('Dependents', ('No','Yes'))
    st.sidebar.write('## Account Information')
    Contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.sidebar.selectbox('PaperlessBilling', ('Yes', 'No'))
    PaymentMethod = st.sidebar.selectbox('PaymentMethod', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    MonthlyCharges = st.sidebar.slider('MonthlyCharges', 0, 150, 64)
    TotalCharges = st.sidebar.slider('TotalCharges', 0, 10000, 2640)
    st.sidebar.write('## Serviecs')
    PhoneService = st.sidebar.selectbox('PhoneService', ('No', 'Yes'))
    MultipleLines = st.sidebar.selectbox('MultiLines', ('No phone service', 'No', 'Yes'))
    InternetService = st.sidebar.selectbox('InternetService', ('DSL', 'Fiber optic', 'No'))
    OnlineSecurity = st.sidebar.selectbox('OnlineSecurity', ('Yes', 'No', 'No internet service'))
    OnlineBackup = st.sidebar.selectbox('OnlineBackup', ('Yes', 'No', 'No internet service'))
    DeviceProtection = st.sidebar.selectbox('DeviceProtection', ('No', 'Yes', 'No internet service'))
    TechSupport = st.sidebar.selectbox('TechSupport', ('No', 'Yes', 'No internet service'))
    StreamingTV = st.sidebar.selectbox('StreamingTV', ('No', 'Yes', 'No internet service'))
    StreamingMovies = st.sidebar.selectbox('StreamingMovies', ('No', 'Yes', 'No internet service'))

    data = {'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges}
    features = pd.DataFrame(data, index=['User Input'])
    return features

# Data Preprocessing before feeding into model
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

df_churn = df.drop('tenure', axis=1)
df_churn = pd.get_dummies(df_churn, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod'])
y_c = df_churn['Churn']
X_c = df_churn.drop(columns=['Churn'])

X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

df_tenure = df.drop('Churn', axis=1)
df_tenure = pd.get_dummies(df_tenure, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod'])
y_r = df_tenure['tenure']
X_r = df_tenure.drop(columns=['tenure'])

X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42, stratify=y_r)
sc_r = StandardScaler()
X_train = sc_r.fit_transform(X_train)

input_df = user_input_features()
input_df['SeniorCitizen'].replace(to_replace='Yes', value=1, inplace=True)
input_df['SeniorCitizen'].replace(to_replace='No',  value=0, inplace=True)
user_input_df = df.drop(columns=['tenure', 'Churn'])
user_input_df = user_input_df.append(input_df, ignore_index=True)
user_input_df = pd.get_dummies(user_input_df, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod'])

user_input_df = user_input_df.astype(int).astype(object)
user_input_df_r = user_input_df.copy()

user_input_df_r = sc_r.transform(user_input_df_r)
user_input_r = user_input_df_r[user_input_df_r.shape[0]-1, :]
user_input_r = pd.DataFrame(user_input_r).T

user_input_df = sc.transform(user_input_df)
user_input = user_input_df[user_input_df.shape[0]-1, :]
user_input = pd.DataFrame(user_input).T

# Loading models
x_r = pickle.load(open('telco_tenure.pkl', 'rb'))
g_clf = pickle.load(open('telco_churn.pkl', 'rb'))

# Predictions
churn = ['No', 'Yes']
tenure_prediction = int(x_r.predict(user_input_r)[0])
churn_prediction = churn[g_clf.predict(user_input)[0]]
churn_prediction_proba = g_clf.predict_proba(user_input)

col1, col2 = st.columns([5, 9], gap='large')
with col1:
    st.subheader('Customer Information')
    st.dataframe(input_df.T, height=320)
with col2:
    if churn_prediction == 'Yes':
        st.subheader('Estimated Tenure')
        st.write('#####', tenure_prediction, 'Months')
        st.text("")
        st.text("")
        st.text("")
    if churn_prediction == 'No':
        st.subheader('The Customer will not churn')
        st.write('##### The probability of the customer not churning is',(churn_prediction_proba[:, 0][0]*100).round(2), '%')
    if churn_prediction == 'Yes':
        st.subheader('The Customer will probably churn')
        st.write('##### The probability of the customer churning is', (churn_prediction_proba[:, 1][0] * 100).round(2), '%')
