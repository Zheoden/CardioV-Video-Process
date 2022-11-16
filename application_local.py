from operator import concat
from flask import Flask, jsonify, request
import processor_local as pc
import boto3
import os
from process_functions import model_loader as ml

application = Flask(__name__)
root_directory = "C:\\Users\\matia\\cardiov\\muestra.avi"
root_directory2 = "C:\\Users\\matia\\cardiov\\muestra2.avi"
root_directory3 = "C:\\Users\\matia\\cardiov\\muestra3.avi"
root_directory4 = "C:\\Users\\matia\\cardiov\\muestra4.avi"
root_directory5 = "C:\\Users\\matia\\cardiov\\test_img2.jpg"
root_directory6 = "C:\\Users\\matia\\cardiov\\bend.jpg" 
root_directory7 = "C:\\Users\\matia\\cardiov\\bens.jpg" 


loaded_model = ml.cargarModelo()

@application.route('/ping')
def ping():
    return 'Pong'

@application.route('/getHeartValues')
def get_heart_values():
    
    file = request.args.get('file')
    print(f"FILE TO PROCESS: {file}")
    #path = root_directory + file
    path = root_directory
    num = request.args.get('num')
    
    type = request.args.get('type')
    if(num == '1'):
        path = root_directory
    elif (num == '2'):
        path = root_directory2
    elif (num == '3'):
        path = root_directory3
    elif (num == '4'):
        path = root_directory4
    elif (num == '5'):
        path = root_directory5
    elif (num == '6'):
        path = root_directory6
    elif (num == '7'):
        path = root_directory7
        
    
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
