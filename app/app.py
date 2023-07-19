# Import dependencies
import streamlit as st
import requests
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# List of all features in the CSV
list = ["Phone Name","Rating ?/5" ,"Number of Ratings","RAM","ROM/Storage","Back/Rare Camera","Front Camera" ,"Battery" ,"Processor" ,"Price in INR","Date of Scraping"]

# Load the trained model
best_model = joblib.load('best_model.pkl')

# Title of the Application
st.title("Mobile Price Prediction")

# Sidebar of the application
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Prediction Options')
    choice = st.radio("Prediction Mode", ["Individual Features", "Uploaded CSV"])

if choice == 'Individual Features':
    # Individual feature prediction
    st.header("Predict Mobile Price for Individual Features")
    num_ratings = st.number_input("Number of Ratings", min_value=1)
    ram = st.number_input("RAM", min_value=1)
    rom = st.number_input("ROM/Storage", min_value=1)
    rear_camera = st.number_input("Back/Rare Camera", min_value=1)
    front_camera = st.number_input("Front Camera", min_value=1)
    battery = st.number_input("Battery", min_value=1)
    processor = st.number_input("Processor", min_value=1)

    # Create a feature list
    new_mobile_features = [[num_ratings, ram, rom, rear_camera, front_camera, battery, processor]]

    if st.button("Predict"):
        # Use the predict_price function to get the individual prediction
        predicted_price = best_model.predict(new_mobile_features)
        st.success(f"Predicted Price: {predicted_price} INR")

if choice == 'Uploaded CSV':
    # Predict from uploaded CSV
    st.header("Predict Mobile Prices from Uploaded CSV")
    st.write("Upload a CSV file with the following columns:")
    st.info("Number of Ratings, RAM, ROM/Storage, Back/Rare Camera, Front Camera, Battery, Processor")
    file = st.file_uploader("Upload your CSV", type=['csv'])
    if file:
        if st.button("Predict from CSV"):
            df = pd.read_csv(file)
            #st.write(df.head())  # Print the first few rows of the DataFrame for debugging
            # Preprocess the CSV data
            le = LabelEncoder()
            for column in list:
                df[column] = le.fit_transform(df[column])
            # Create a feature list
            new_mobile_features = df[list[2:9]].values 
            # Use the predict_prices_from_csv function to get the bulk predictions
            predicted_prices = best_model.predict(new_mobile_features)
            # Add the predicted prices to the DataFrame
            df["Predicted Price"] = predicted_prices
            st.table(df)  # Display the prediction results and original data as a table
