from operator import concat
from flask import Flask, jsonify, request
import processor_local as pc
import boto3
import os
from process_functions import model_loader as ml

application = Flask(__name__)
root_directory = "C:/Users/fedeb/cardiov/"
loaded_model = ml.cargarModelo()

@application.route('/ping')
def ping():
    return 'Pong'

@application.route('/getHeartValues')
def get_heart_values():
    
    file = request.args.get('file')
    print(f"FILE TO PROCESS: {file}")
    path = root_directory + file
    
    type = request.args.get('type')
    
    if type == 'i':
        val = pc.process_image(path= path, model = loaded_model)
    else:
        val = pc.process_video(path= path, model = loaded_model)
    
    return val

@application.route('/showGraph')
def show_graph():
    
    pc.make_a_graph()

if __name__ == '__main__':
    application.run(debug= True, port= 4000)
