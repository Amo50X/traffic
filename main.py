import json
from flask import Flask, render_template, Response, request,redirect
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
            return Response(generate_frame(camDict["cap0"],model=model, real=real_time, queue_cam_dict=queue_cam_dict ),
                             mimetype='multipart/x-mixed-replace; boundary=frame')
    #----------------------------------------------------------------------------
    if cameras == 2:
        @app.route(f'/video0')
        def video0():
            return Response(generate_frame(camDict["cap0"],model=model, real=real_time, queue_cam_dict=queue_cam_dict ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route(f'/video1')
        def video1():
            return Response(generate_frame(camDict["cap1"],model=model, real=real_time, queue_cam_dict=queue_cam_dict ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    #----------------------------------------------------------------------------   
    if cameras == 3:
        @app.route(f'/video0')
        def video0():
            return Response(generate_frame(camDict["cap0"],model=model, real=real_time, queue_cam_dict=queue_cam_dict ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route(f'/video1')
        def video1():
            return Response(generate_frame(camDict["cap1"],model=model, real=real_time, queue_cam_dict=queue_cam_dict ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route(f'/video2')
        def video2():
            return Response(generate_frame(camDict["cap2"],model=model, real=real_time, queue_cam_dict=queue_cam_dict ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    #----------------------------------------------------------------------------
    if cameras == 4:
        @app.route(f'/video0')
        def video0():
            return Response(generate_frame(camDict["cap0"],model=model, real=real_time, queue_cam_dict=queue_cam_dict ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route(f'/video1')
        def video1():
            return Response(generate_frame(camDict["cap1"],model=model, real=real_time, queue_cam_dict=queue_cam_dict ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route(f'/video2')
        def video2():
            return Response(generate_frame(camDict["cap2"],model=model, real=real_time, queue_cam_dict=queue_cam_dict ), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        @app.route(f'/video3')
        def video3():
            return Response(generate_frame(camDict["cap3"],model=model, real=real_time, queue_cam_dict=queue_cam_dict ), 
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
        # camDict = setCamera(cam,ip,port,user,password)
    for cap in camDict:
        queue_cam_dict.update({f'{camDict[cap]}': queue.Queue() })
    # gen_camera()

def run_app():
    app.run(host="0.0.0.0", port=8080, debug=True)#, debug=True,threaded=True

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
    global isConfig,load_config
    if isConfig:
        load_config = pickle.load(open(config_data_path, 'rb')) 
        cameras = int(load_config['Cameras'])
    else:
        cameras = 0
    if request.method == "POST":
        if request.form.get('submit') == "Submit":
            print("Data Submited")
            CAMERAS = request.form.get('camera')
            USER = request.form.get('admin')
            PASSWORD = request.form.get('password')
            CONF = request.form.get('conf')
            MODEL = request.form.get('model')
            MODE = request.form.get('mode')
            REAL = request.form.get('real')

            saved_config = {
                "Cameras" : CAMERAS,
                "User" : USER,
                "Password" : PASSWORD,
                "Confidence": CONF,
                "Model": MODEL,
                "Mode": MODE,
                "Real": REAL
            }    
            isConfig = True
            print(saved_config)
            pickle.dump(saved_config, open(config_data_path, 'wb') )
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
            pickle.dump(addresses, open(address_path, 'wb') )
            print(addresses)
            # for index,cam in enumerate(camDict):
            #     print(index)
            #     _, frame = camDict[cam].read()
            #     frame, poly_points = PolyDraw().setPolygon(frame)   
            #     cv2.imwrite(f'static/cam{index}.jpg', frame)
            return redirect("/settings")

    return render_template('settings.html', models = modelList, isConfigured = isConfig, config = addresses, cameras=cameras)

@app.route('/clips', methods=["POST","GET"])
def captured():
    clipList = os.listdir(cap_Path)
    predictList = os.listdir(pdt_Path)

    if request.method == "POST":
        if request.form.get('sink') == "Sink":
            VideoSink(model=model,conf=conf,cap_Path=cap_Path, pdt_Path=pdt_Path)

    return render_template('clips.html', clips = clipList, preds = predictList)

# ========================================================================================================================================================================
if __name__ == "__main__":
    run_app()
    # app_thread = threading.Thread(target=run_app)
    # for index, cam in enumerate(camDict):
    #     print('cap')
    #     _thread = threading.Thread(target=RunningCamera, args=(camDict[cam], cap_Path, queue_cam_dict))
    #     threads.update({f'thread{index}':_thread})

    # app_thread.start()
    # for thread in threads:
    #     print(thread)
    #     threads[thread].start()