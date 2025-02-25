from flask import Flask, request, jsonify, render_template
import pickle

application = Flask(__name__)
app = application

# Load the model and scaler properly
with open('model.pkl', 'rb') as model_file:
    lasso_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/")
def index():
    # You might want to render home.html directly or an index.html page
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=="POST":
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scale the data
            new_scaled_data=scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result=lasso_model.predict(new_scaled_data)
            return render_template('home.html', result=result[0])


    else:
        return render_template('home.html')
        


if __name__ == '__main__':
    app.run(debug=True)
