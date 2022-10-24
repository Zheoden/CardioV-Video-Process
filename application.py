from operator import concat
from flask import Flask, jsonify, request
import processor as pc
import boto3
import os
from process_functions import model_loader as ml


s3_client = boto3.client('s3',
                    aws_access_key_id= os.environ['AWS_KEY_ID'],
                    aws_secret_access_key= os.environ['AWS_ACCESS_KEY'])
s3_bucket_name = os.environ['BUCKET_NAME']

application = Flask(__name__)
loaded_model = ml.cargarModelo()

@application.route('/ping')
def ping():
    return 'Pong'

@application.route('/getHeartValues')
def get_heart_values():
    
    file = request.args.get('file')
    print(f"FILE TO PROCESS: {file}")
    url = s3_client.generate_presigned_url( ClientMethod='get_object', Params={ 'Bucket': s3_bucket_name, 'Key': file } )
    
    type = request.args.get('type')
    
    if type == 'i':
        val = pc.process_image(url= url, model = loaded_model)
    else:
        val = pc.process_video(url= url, model = loaded_model)
    
    return val

if __name__ == '__main__':
    application.run(debug= True, port= 4000)
