from operator import concat
from flask import Flask, jsonify, request
import processor as pc

app = Flask(__name__)
root_directory = "C:/Users/matia/cardiov/"

@app.route('/ping')
def ping():
    return 'Pong'

@app.route('/processImage')
def get_img_info():
    
    path = request.args.get('path')
    final_path = root_directory + path
    val = pc.process_selector(path= final_path, type= 'i')
    return val

@app.route('/processVideo')
def get_video_info():
    
    path = request.args.get('path')
    final_path = root_directory + path
    val = pc.process_selector(path= final_path, type= 'v')
    return val

@app.route('/processFile')
def get_file_info():
    
    path = request.args.get('path')
    type = request.args.get('type')
    final_path = root_directory + path
    val = pc.process_selector(path= final_path, type= type)
    return val

@app.route('/getHeartValues')
def get_heart_values():
    
    path = request.args.get('path')
    type = request.args.get('type')
    final_path = root_directory + path
    val = pc.process_values(path= final_path,type = type)
    return val


if __name__ == '__main__':
    app.run(debug= True, port= 4000)

