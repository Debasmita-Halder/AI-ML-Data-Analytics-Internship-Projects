from flask import Flask, request, jsonify,render_template
import pickle


with open('spam_classifier_nb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('spam_classifier_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

import joblib
import os
app=Flask(__name__)

@app.route("/")
def home():
  return render_template("home.html")
@app.route('/classify', methods=['POST'])
def classify():
    # Get the email content from the form input
    email_content = request.form['email']

    # Preprocess the email content using the vectorizer
    email_vectorized = vectorizer.transform([email_content])

    # Classify the email using the model
    prediction = model.predict(email_vectorized)[0]  # Prediction result (0 or 1)

    # Map the prediction to "Spam" or "Not Spam"
    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    # Return the result to the user
    return render_template('result.html', result=result)
@app.route("/about")
def about():
  return render_template("about.html")
@app.route("/contact")
def contact():
  return render_template("contact.html")


if __name__=="__main__":
  app.run(debug=True)