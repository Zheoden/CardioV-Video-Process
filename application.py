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
    #url = "https://assets.cardiov.org/4889cc19-a5e5-49a0-a900-22d219c707fc.avi" video
    #url = 'https://cardiov-assets.s3.amazonaws.com/04050916-69a2-4751-9fbe-c164fdf7fad3.jpg' # img
    url = s3_client.generate_presigned_url( ClientMethod='get_object', Params={ 'Bucket': s3_bucket_name, 'Key': file } )
    type = request.args.get('type')
    scale = request.args.get('e')
    
    scale = 1 if scale == None else scale 
    
    if type == 'i':
        val, img_path = pc.process_image(url= url, file = file, original_scale=scale, model = loaded_model)
        img_name = img_path.split("/")[-1]
        s3_client.upload_file(img_path, s3_bucket_name, img_name)
    else:
        val, video_path, dias_path, sys_path = pc.process_video(url= url, file = file, original_scale=scale, model = loaded_model)
        video_name = video_path.split("/")[-1]
        dias_name = dias_path.split("/")[-1]
        sys_name = sys_path.split("/")[-1]
        s3_client.upload_file(video_path, s3_bucket_name, video_name, ExtraArgs={'ACL':'public-read'})
        s3_client.upload_file(dias_path, s3_bucket_name, dias_name, ExtraArgs={'ACL':'public-read'})
        s3_client.upload_file(sys_path, s3_bucket_name, sys_name, ExtraArgs={'ACL':'public-read'})
        
        os.remove(video_path) if os.path.isfile(video_path) else print("No video to be deleted!")
        os.remove(dias_path) if os.path.isfile(dias_path) else print("No dias to be deleted!")
        os.remove(sys_path) if os.path.isfile(sys_path) else print("No sys to be deleted!")

    return val

if __name__ == '__main__':
    application.run(debug= True, port= 4000)
