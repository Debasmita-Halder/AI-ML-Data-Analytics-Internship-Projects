from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('house_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the home route
@app.route('/')
def home():
    return render_template('home.html')



# Define the prediction route
@app.route('/prediction', methods=['POST','GET'])
def predict():
    try:
        # Get necessary inputs from the form
        necessary_features = [float(x) for x in request.form.values()]
        
        # Add default values for missing features
        # Ensure the order matches the model's feature order
        default_values = {
            "indus": 0.0, "chas": 0.0, "nox": 0.0, "dis": 0.0,
            "rad": 0.0, "ptratio": 0.0, "b": 0.0  # Add reasonable defaults
        }
        features = {
            "crim": necessary_features[0], 
            "rm": necessary_features[1], 
            "lstat": necessary_features[2], 
            "age": necessary_features[3], 
            "tax": necessary_features[4], 
            "zn": necessary_features[5],
            **default_values
        }

        # Convert to list in the correct order
        feature_list = [features[feature] for feature in ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]]
        
        # Scale and predict
        features_scaled = scaler.transform([feature_list])
        prediction = model.predict(features_scaled)
        price = round(prediction[0], 2)
        return render_template('prediction.html', prediction_text=f'Predicted House Price: ${price}K')
    except Exception as e:
        print(f"Error: {e}")
        return render_template('prediction.html', prediction_text="Error: Invalid input.")
    
@app.route('/listing')
def listing():
    return render_template('listing.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

