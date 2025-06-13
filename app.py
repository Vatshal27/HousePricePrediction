from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

# Load both models
with open('house_price_prediction.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', form_values={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        # Handle CHAS input safely
        chas_raw = form.get('CHAS', '').strip()
        chas_value = float(chas_raw) if chas_raw else 0.0

        features = [
            float(form['CRIM']),
            float(form['ZN']),
            float(form['INDUS']),
            chas_value,
            float(form['NOX']),
            float(form['RM']),
            float(form['AGE']),
            float(form['DIS']),
            float(form['RAD']),
            float(form['TAX']),
            float(form['PTRATIO']),
            float(form['B']),
            float(form['LSTAT'])
        ]

        final_features = np.array([features])
        model_choice = form.get('model_choice', 'linear')

        if model_choice == 'xgb':
            prediction = xgb_model.predict(final_features)[0]
            model_name = "XGBoost"
        else:
            prediction = linear_model.predict(final_features)[0]
            model_name = "Linear Regression"

        output = round(prediction, 2)
        return render_template('index.html', prediction_text=f'{model_name} Prediction: ${output:.2f}', form_values=form)

    except Exception as e:
        return render_template('index.html', prediction_text="Invalid input! Please ensure all fields are filled correctly.", form_values=request.form)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', form_values={}), 404

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
