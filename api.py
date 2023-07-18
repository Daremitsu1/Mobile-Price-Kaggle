# Import Depdencies
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flasgger import Swagger
from sklearn.preprocessing import LabelEncoder

# Load the trained model
best_model = joblib.load('best_model.pkl')

# Initialize Flask app and Swagger
app = Flask(__name__)
swagger = Swagger(app)

# List of all features in the CSV
list = ["Phone Name","Rating ?/5" ,"Number of Ratings","RAM","ROM/Storage","Back/Rare Camera","Front Camera" ,"Battery" ,"Processor" ,"Price in INR","Date of Scraping"]

@app.route('/predict', methods=['GET'])
def predict_single():
    """
    Predict mobile price for individual features
    ---
    parameters:
      - name: Number of Ratings
        in: query
        type: number
        required: true
      - name: RAM
        in: query
        type: number
        required: true
      - name: ROM/Storage
        in: query
        type: number
        required: true
      - name: Back/Rare Camera
        in: query
        type: number
        required: true
      - name: Front Camera
        in: query
        type: number
        required: true
      - name: Battery
        in: query
        type: number
        required: true
      - name: Processor
        in: query
        type: number
        required: true
    responses:
        200:
            description: Predicted price
    """
    # Get individual feature values from query parameters
    num_ratings = float(request.args.get('Number of Ratings'))
    ram = float(request.args.get('RAM'))
    rom = float(request.args.get('ROM/Storage'))
    rear_camera = float(request.args.get('Back/Rare Camera'))
    front_camera = float(request.args.get('Front Camera'))
    battery = float(request.args.get('Battery'))
    processor = float(request.args.get('Processor'))

    # Create a feature list
    new_mobile_features = [[num_ratings, ram, rom, rear_camera, front_camera, battery, processor]]

    # Make prediction
    predicted_price = best_model.predict(new_mobile_features)

    return jsonify({'Predicted Price': predicted_price[0]})

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """
    Predict mobile prices for multiple features from CSV upload
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: Predicted prices
    """
    # Get the uploaded CSV file
    csv_file = request.files['file']
    df = pd.read_csv(csv_file)

    # Preprocess the CSV data
    le = LabelEncoder()
    for column in list:
        df[column] = le.fit_transform(df[column])

    # Create a feature list
    new_mobile_features = df[list[2:9]].values  # Exclude 'Phone Name', 'Price in INR', and 'Date of Scraping'

    # Make predictions for each row
    df['Predicted Price'] = best_model.predict(new_mobile_features)

    # Convert the DataFrame to a JSON object
    result = df.to_json(orient='records')

    return result

if __name__ == '__main__':
    app.run(debug=True)