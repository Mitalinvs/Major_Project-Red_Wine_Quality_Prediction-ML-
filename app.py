#Type in terminal: py -m streamlit run app.py

import subprocess
# Install Streamlit
subprocess.call(['pip', 'install', 'streamlit'])
# Upgrade scikit-learn, scipy, and matplotlib
subprocess.call(['pip', 'install', '-U', 'scikit-learn', 'scipy', 'matplotlib'])

import numpy as np
import pandas as pd
import streamlit as st

#loading the dataset to a pandas Dataframe
df = pd.read_csv('Wine_Quality.csv')
# converting quality column, if quality is greater than or equal to 7 to be 1 else 0
df['quality']= df['quality'].apply(lambda x: 1 if x>=7 else 0)
df['quality'].head()

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
data_scale = scale.fit_transform(df)
data_scaled = pd.DataFrame(data_scale)
data_scaled.columns = df.columns
data_scaled.head()

# Splitting the dataset into predictor variable(X) and target variable (y).
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Splitting predictor and target variable into training and testing datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)

# printing the accuracy of training and testing data
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,roc_auc_score
y_pred_train = rf_model.predict(x_train)
y_pred_test = rf_model.predict(x_test)
print('Accuracy score of Random Forest with training dataset is : ', accuracy_score(y_pred_train,y_train))
print('Accuracy score of Random Forest with testing dataset is : ', accuracy_score(y_pred_test,y_test))

#Web app
st.title("Red-Wine Quality Prediction Model")
input_data = st.text_input('Enter comma-separated input features here')
if st.button('Predict'):
    input_data_np_array = np.asarray(input_data.split(','), dtype=float)
    reshaped_input = input_data_np_array.reshape(1, -1)
    prediction = rf_model.predict(reshaped_input)
    if prediction[0] == 1:
        st.write('Good Quality Red-Wine')
    else:
        st.write('Bad Quality Red-Wine')
