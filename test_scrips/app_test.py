from operator import concat
from flask import Flask, jsonify, request
import processor as pc
import boto3
import os

s3_client = boto3.client('s3',
                    aws_access_key_id= 'AKIA4BMMFY4ZGPTTZ556',
                    aws_secret_access_key= '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')
s3_bucket_name = 'cardiov-assets'

app = Flask(__name__)
#root_directory = "C:/Users/matia/cardiov/"
#root_directory = "s3://cardiov-assets/f4d6648d-7cdb-4032-8061-5c8ba99c241f/"

@app.route('/ping')
def ping():
    return 'Pong'

@app.route('/processImage')
def get_img_info():
    
    file = request.args.get('file')
    url_img = s3_client.generate_presigned_url( ClientMethod= 'get_object', Params= { 'Bucket': s3_bucket_name, 'Key': file } )
    val = pc.process_selector(path= url_img, type= 'i')
    return val

@app.route('/processVideo')
def get_video_info():
    
    file = request.args.get('file')
    url_vid = s3_client.generate_presigned_url( ClientMethod= 'get_object', Params= { 'Bucket': s3_bucket_name, 'Key': file } )
    val = pc.process_selector(path= url_vid, type= 'v')
    return val

@app.route('/processFile')
def get_file_info():
    
    file = request.args.get('file')
    print(f"FILE TO PROCESS: {file}")
    url = s3_client.generate_presigned_url( ClientMethod='get_object', Params={ 'Bucket': s3_bucket_name, 'Key': file } )
    type = request.args.get('type')
    val = pc.process_selector(path= url, type= type)
    return val

@app.route('/getHeartValues')
def get_heart_values():
    
    file = request.args.get('file')
    print(f"FILE TO PROCESS: {file}")
    url = s3_client.generate_presigned_url( ClientMethod='get_object', Params={ 'Bucket': s3_bucket_name, 'Key': file } )
    type = request.args.get('type')
    val = pc.process_values(path= url, type = type)
    return val


if __name__ == '__main__':
    app.run(debug= True, port= 4000)

