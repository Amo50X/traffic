import json
import cv2
from flask import Flask, render_template, Response, request,redirect, flash
from modules import VideoSink, generate_frame, RunningCamera, setCamera, PolyDraw
import numpy as np
import os
import threading
import face_recognition
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
from supervision import BoxAnnotator, LabelAnnotator
import pandas as pd
import pickle
import queue

app = Flask(__name__)

app.secret_key = "5a4wd45awd4wa5d4aw5d4a5wd45aw4daw5d"

modelList = os.listdir('static/models')
licenseModel = YOLO('numPlatev8n.pt')

addresses = {}
threads = {}
camDict = {}
queue_cam_dict = {}
cameras = 0

box_annotator = BoxAnnotator()
label_annotator = LabelAnnotator()

isConfig = False
cap_Path = "static/clips/captured"
pdt_Path = "static/clips/predicted"
weight_path = "static/models/"
config_data_path = "static/config.adx"
address_path = "static/address.adx"
record_path = 'static/record.csv'


def gen_camera():
     #=================================================================================================================
    if cameras == 1:
        @app.route(f'/video0')
        def video0():
            return Response(generate_frame(camDict["cap0"],model=model, real=real_time,conf=conf, pdt_Path=pdt_Path, queue_cam_dict=queue_cam_dict, points=points ),
                             mimetype='multipart/x-mixed-replace; boundary=frame')
    #----------------------------------------------------------------------------
    if cameras == 2:
        @app.route(f'/video0')
        def video0():
            return Response(generate_frame(camDict["cap0"],model=model, real=real_time, queue_cam_dict=queue_cam_dict, points=points ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route(f'/video1')
        def video1():
            return Response(generate_frame(camDict["cap1"],model=model, real=real_time, queue_cam_dict=queue_cam_dict, points=points ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    #----------------------------------------------------------------------------   
    if cameras == 3:
        @app.route(f'/video0')
        def video0():
            return Response(generate_frame(camDict["cap0"],model=model, real=real_time, queue_cam_dict=queue_cam_dict, points=points ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route(f'/video1')
        def video1():
            return Response(generate_frame(camDict["cap1"],model=model, real=real_time, queue_cam_dict=queue_cam_dict, points=points ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route(f'/video2')
        def video2():
            return Response(generate_frame(camDict["cap2"],model=model, real=real_time, queue_cam_dict=queue_cam_dict, points=points ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    #----------------------------------------------------------------------------
    if cameras == 4:
        @app.route(f'/video0')
        def video0():
            return Response(generate_frame(camDict["cap0"],model=model, real=real_time, queue_cam_dict=queue_cam_dict, points=points ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route(f'/video1')
        def video1():
            return Response(generate_frame(camDict["cap1"],model=model, real=real_time, queue_cam_dict=queue_cam_dict, points=points ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route(f'/video2')
        def video2():
            return Response(generate_frame(camDict["cap2"],model=model, real=real_time, queue_cam_dict=queue_cam_dict, points=points ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        @app.route(f'/video3')
        def video3():
            return Response(generate_frame(camDict["cap3"],model=model, real=real_time, queue_cam_dict=queue_cam_dict, points=points ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    #================================================================================================

if os.path.isfile(record_path):
    db = pd.read_csv(record_path)
    

if os.path.isfile(config_data_path):
    isConfig = True
    load_config = pickle.load(open(config_data_path, 'rb')) 
    cameras = int(load_config['Cameras'])
    conf = float(load_config["Confidence"])
    model = YOLO(f"{weight_path}{load_config['Model']}" )
    user = load_config["User"]
    password = load_config["Password"]
    real_time = load_config['Real']

    mode = load_config['Mode'] 

else:
    isConfig = False
    cameras = 0
    load_config = "No Configeration" 

if os.path.isfile(address_path):
    cam_config = pickle.load(open(address_path, 'rb')) 
    for cam in range(cameras):
        ip = cam_config[f'IP{cam}']
        port = cam_config[f'PORT{cam}']
        points = tuple(cam_config[f"Point{cam}"])
        camDict = setCamera(cam,ip,port,user,password)
    for cap in camDict:
        queue_cam_dict.update({f'{camDict[cap]}': queue.Queue() })
    gen_camera()


def run_app():
    app.run(host="0.0.0.0", port=5050, threaded=True)#, debug=True,threaded=True

@app.route('/')
def index():
    global isConfig
    return render_template('index.html', isConfigured = isConfig)

@app.route('/dashboard')
def states():
    global isConfig
    if os.path.isfile(record_path):
        db = pd.read_csv(record_path)

    series_vehicle = db["vehicle"].value_counts().to_dict()

    

    dash_data = {
        "Length": len(db["id"]),
        "Cars": db["vehicle"].value_counts()['car'],
        "Trucks": db["vehicle"].value_counts()['truck'],
        "Bus": db["vehicle"].value_counts()['bus'],
        "Motorcycle": db["vehicle"].value_counts()['motorcycle'],
    }
    return render_template('dashboard.html', 
                           isConfigured = isConfig, 
                           cameras=cameras, 
                           tables=[db.to_html()], 
                           titles=[''], dash = dash_data,
                           vehicles_vs_totals =json.dumps(series_vehicle)
                           )

@app.route('/settings', methods=['GET','POST'])
def config():
    global isConfig,load_config, cameras, user, password
    if isConfig:
        load_config = pickle.load(open(config_data_path, 'rb')) 
        cameras = int(load_config['Cameras'])
    else:
        cameras = 0
    if request.method == "POST":
        if request.form.get('submit') == "Submit":
            print("Data Submited")
            cameras = request.form.get('camera')
            user = request.form.get('admin')
            password = request.form.get('password')
            conf = request.form.get('conf')
            model = request.form.get('model')
            mode = request.form.get('mode')
            real = request.form.get('real')

            saved_config = {
                "Cameras" : cameras,
                "User" : user,
                "Password" : password,
                "Confidence": conf,
                "Model": model,
                "Mode": mode,
                "Real": real
            }    
            isConfig = True
            print(saved_config)
            pickle.dump(saved_config, open(config_data_path, 'wb') )
            model = YOLO(f"{weight_path}{model}" )
            return redirect("/settings")

        elif request.form.get('submit') == "Delete":
            print("Data Deleted")
            isConfig = False
            os.remove(config_data_path)
            return redirect("/settings")

        elif request.form.get('submit') == "Config":
            print("Camera Configuration")
            for cam in range(cameras):
                ip = request.form.get(f'ip{cam}')
                port = request.form.get(f'port{cam}')
                addresses.update({f'IP{cam}': ip})
                addresses.update({f'PORT{cam}': port})
                camDict = setCamera(cam,ip,port,user,password)
            

            for index, cam in enumerate(camDict):
                _, frame = camDict[cam].read()
                if not _:
                    flash("Something is wrong with cameras", category='error')
                    return render_template('settings.html', models = modelList, isConfigured = isConfig, config = addresses, cameras=cameras)
                width = int(camDict[cam].get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(camDict[cam].get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(width)
                test_img = cv2.resize(frame, [int(width / 2),int(height / 2)])

                point = cv2.selectROI(img=test_img)
                cv2.destroyAllWindows()
                addresses.update({f"Point{index}": point})       
            pickle.dump(addresses, open(address_path, 'wb') )

            return redirect("/settings")

    return render_template('settings.html', models = modelList, isConfigured = isConfig, config = addresses, cameras=cameras)

@app.route('/cameras')
def cctv():
    return render_template('camera.html')

@app.route('/clips')
def captured():
    clipList = os.listdir(cap_Path)
    predictList = os.listdir(pdt_Path)

    if request.method == "POST":
        if request.form.get('sink') == "Sink":
            VideoSink(model=model,conf=conf,cap_Path=cap_Path, pdt_Path=pdt_Path)

    return render_template('clips.html', clips = clipList, preds = predictList)

# ========================================================================================================================================================================
if __name__ == "__main__":
    # run_app()
    app_thread = threading.Thread(target=run_app)
    if os.path.isfile(address_path):
        for index, cam in enumerate(camDict):
            print('cap')
            _thread = threading.Thread(target=RunningCamera, args=(camDict[cam], cap_Path, queue_cam_dict))
            threads.update({f'thread{index}':_thread})

    app_thread.start()
    if os.path.isfile(address_path):
        for thread in threads:
            print(thread)
            threads[thread].start()