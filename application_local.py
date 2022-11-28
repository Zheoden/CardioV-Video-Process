from operator import concat
from flask import Flask, jsonify, request
import processor_local as pc
import boto3
import os
from process_functions import model_loader as ml

application = Flask(__name__)
root_directory = "C:\\Users\\matia\\cardiov\\example_video2.mp4"
root_directory2 = "C:\\Users\\matia\\cardiov\\muestra_2.mp4"
root_directory3 = "C:\\Users\\matia\\cardiov\\muestra3.avi"
root_directory4 = "C:\\Users\\matia\\cardiov\\muestra4.avi"
root_directory5 = "C:\\Users\\matia\\cardiov\\test_img2.jpg"
root_directory6 = "C:\\Users\\matia\\cardiov\\bend.jpg" 
root_directory7 = "C:\\Users\\matia\\cardiov\\muestra4.jpg" 


s3_client = boto3.client('s3',
                    aws_access_key_id= 'AKIA4BMMFY4ZGPTTZ556',
                    aws_secret_access_key= '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')
s3_bucket_name = 'cardiov-assets'


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
    
    scale = request.args.get('e')
    
    scale = 1 if scale == None else scale 
    print(f"SCALE : {scale}")
    
    if type == 'i':
        val, img_path = pc.process_image(path= path, file = file, original_scale=scale, model = loaded_model)
        img_name = img_path.split("/")[-1]
        s3_client.upload_file(img_path, s3_bucket_name, img_name)
    else:
        val, video_path, dias_path, sys_path = pc.process_video(path= path, file = file, original_scale=scale, model = loaded_model)
        video_name = video_path.split("/")[-1]
        dias_name = dias_path.split("/")[-1]
        sys_name = sys_path.split("/")[-1]
        
        s3_client.upload_file(video_path, s3_bucket_name, video_name, ExtraArgs={'ACL':'public-read'})
        #s3_client.upload_file(dias_path, s3_bucket_name, dias_name, ExtraArgs={'ACL':'public-read'})
        #s3_client.upload_file(sys_path, s3_bucket_name, sys_name, ExtraArgs={'ACL':'public-read'})
        os.remove(video_path) if os.path.isfile(video_path) else print("No video to be deleted!")
        os.remove(dias_path) if os.path.isfile(dias_path) else print("No dias to be deleted!")
        os.remove(sys_path) if os.path.isfile(sys_path) else print("No sys to be deleted!")

    return val

@application.route('/showGraph')
def show_graph():
    
    pc.make_a_graph()


if __name__ == '__main__':
    application.run(debug= True, port= 4000)
