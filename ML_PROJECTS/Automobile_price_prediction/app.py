import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import sklearn

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def results():
    
    Drive_wheels= float(request.form["Drive_wheels"])
    Wheel_base= float(request.form["Wheel_base"])
    length= float(request.form["length"])
    width= float(request.form["width"])
    Curb_weight= float(request.form["Curb_weight"])
    Engine_size= float(request.form["Engine_size"])
    Fuel_system= float(request.form["Fuel_system"])
    bore= float(request.form["bore"])
    horsepower= float(request.form["horsepower"])
    City_mpg= float(request.form["City_mpg"])
    Highway_mpg= float(request.form["Highway-mpg"])
   
    


    x = np.array([[Drive_wheels,Wheel_base, length,width,Curb_weight, Engine_size,Fuel_system,bore,horsepower,City_mpg,Highway_mpg]])
    model = pickle.load(open('GBR_TUNED.pkl','rb'))
    Y_predict = model.predict(x)
    return jsonify({'Prediction': float(Y_predict)})


if __name__ == '__main__':
    app.run(debug = True, port = 1010)
