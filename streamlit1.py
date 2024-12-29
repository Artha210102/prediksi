import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Load the dataset
df = pd.read_csv('/content/kc_house_data.csv', usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])

# Preprocessing the data
df['bathrooms'] = df['bathrooms'].astype('int')
df['bedrooms'] = df['bedrooms'].replace(33, 3)

# Splitting the data into X and y
X = df.drop(columns='price')
y = df['price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Save the trained model using pickle
pickle.dump(lin_reg, open('house_price_model.sav', 'wb'))

# Load the model (in case you need to reload the model later)
model = pickle.load(open('house_price_model.sav', 'rb'))

# Streamlit title
st.title('Prediksi Harga Rumah')

# Add a subtitle for better understanding
st.subheader('Masukkan Data untuk Memprediksi Harga Rumah')

# Create input fields for user to enter house details
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    bedrooms = st.number_input('Jumlah Kamar Tidur', min_value=1, max_value=10, value=3)
with col2:
    bathrooms = st.number_input('Jumlah Kamar Mandi', min_value=1, max_value=10, value=2)
with col3:
    sqft_living = st.number_input('Luas Bangunan (sqft)', min_value=100, max_value=10000, value=1800)
with col4:
    grade = st.number_input('Nilai Grade Rumah', min_value=1, max_value=13, value=7)

col5, col6 = st.columns([2, 2])

with col5:
    yr_built = st.number_input('Tahun Dibangun', min_value=1900, max_value=2024, value=1990)

# Placeholder for the predicted price result
predicted_price = ''

# Create a button to trigger the prediction
if st.button('Prediksi Harga Rumah'):
    input_data = np.array([[bedrooms, bathrooms, sqft_living, grade, yr_built]])
    predicted_price = model.predict(input_data)

    # Display the predicted price
    st.success(f'Harga Rumah yang Diprediksi: ${predicted_price[0]:,.2f}')

# Footer
st.write("---")
st.write("Aplikasi Prediksi Harga Rumah ini dibuat untuk membantu memprediksi harga rumah berdasarkan data yang dimasukkan.")
