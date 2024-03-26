import cv2
import numpy as np
import os
import easyocr
import supervision as sv
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import queue
import time

queue_cam_dict = {}

fourcc = cv2.VideoWriter_fourcc(*'h264') #h264
camDict = {}

record_path = 'static/record.csv'
LICENSE_PLATE_MODEL = "numPlatev8n.pt"


licenseModel = YOLO(LICENSE_PLATE_MODEL)
reader = easyocr.Reader(['en'] )

distance = 36

# SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
# SOURCE = np.array([[398, 339], [865, 333], [1561, 719], [-447, 719]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


def setCamera (index, ip, port, user, password):
    addr = f'rtsp://{user}:{password}@{ip}:{port}'
    # addr = f'http://{ip}:{port}/video'
    cap =  cv2.VideoCapture(addr)
    camDict.update({f'cap{index}': cap})
        
    return camDict


class PolyDraw:
    def __init__(self):
        self.tempPoint = []
        self.pointlist = []

    def draw_shape(self,event, x, y, flags, param):
        global ix, iy, mx, my
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pointlist.append((x, y))
            ix, iy = x, y

    def setPolygon(self, frame):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_shape)

        while True:
            img = frame.copy()
            if self.pointlist:
                for point in self.pointlist:
                    cv2.circle(img, point, 5, (255,0,0), -1)

                    if self.tempPoint:
                        cv2.line(img, point,self.tempPoint,(0,0,255),2)
                    self.tempPoint = point

                lastPoint = self.pointlist[len(self.pointlist) - 1]

                # cv2.line(img, lastPoint, (mx,my),(0,0,255),2)   

            cv2.imshow('image', img)

            if cv2.waitKey(20) & 0xFF == 27:
                cv2.destroyAllWindows()
                return img, tuple(self.pointlist)
 
     

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def get_number(frame, box):

    cropped_frame = sv.crop_image(image=frame, xyxy=box)   
    if 0 in cropped_frame.shape:
        return 0, 'Unclear'

    plate_results = licenseModel(cropped_frame, verbose=False)[0]

    detect_plate = sv.Detections.from_ultralytics(plate_results)

    plate_bbox = detect_plate.xyxy
    if len(plate_bbox) > 0:
        for box in plate_bbox:
            plate_img = sv.crop_image(image=cropped_frame, xyxy=box)
            plate_text = reader.readtext(plate_img , detail=0, min_size=30)

            plate_text = ''.join(plate_text)
            plate_text = ''.join(x for x in plate_text if x.isalnum())

            if len(plate_text) < 8:
                plate_text = f'{plate_text} - Unclear'

            return plate_img, plate_text
        
    return 0,'Unclear'

def VideoSink(model, conf: float, cap_Path: str, pdt_Path: str, points):
    
    cap_list = os.listdir(cap_Path)
    pdt_list = os.listdir(pdt_Path)
    CLASS_DICT_NAME = model.model.names
    
    if len(cap_list) == 0:
        return
    
    for video in cap_list:
        vehicle_data = {}
        print("in it")
        if video in pdt_list:
            print("Video Exist")
        else:
            rec_data = []
            videoName = video.split('.')[0]
            source_path = cap_Path + '/' + video
            target_path = pdt_Path + '/' + video
            video_info = sv.VideoInfo.from_video_path(source_path)

            frame_generator = sv.get_video_frames_generator(source_path)
            width, height = video_info.resolution_wh
            
            down = {}
            up = {}
            counter_down = []
            counter_up = []

            offset = 6

            
            red_line_y = points[1] * 2
            blue_line_y = (points[3] + points[1])*2
            
            poly_point = np.array([[points[0]*2, points[1]*2],
                        [points[0]*2, (points[3] + points[1])*2],
                        [(points[2] + points[0])*2,(points[3] + points[1])*2],
                        [(points[2] + points[0])*2, points[1]*2]])

            thickness = sv.calculate_dynamic_line_thickness(
                            resolution_wh=(width,height)
                    )
            text_scale = sv.calculate_dynamic_text_scale(
                resolution_wh=(width,height)
            )
            
            byteTracker = sv.ByteTrack()
            
            poly_annotator = sv.PolygonAnnotator(
                        color=sv.Color.red,
                        thickness=thickness
                    )
            
            polygon_zone = sv.PolygonZone(
                            polygon=poly_point, 
                            frame_resolution_wh=(width,height)
                            , triggering_position=sv.Position.CENTER
                        )

            trace_annotator = sv.TraceAnnotator(
                thickness=thickness,
                position=sv.Position.CENTER
                )

            label_annotator = sv.LabelAnnotator(
                text_scale=text_scale,
                text_thickness=thickness,
                text_position=sv.Position.BOTTOM_CENTER,
            )

            bounding_box_annotator = sv.BoundingBoxAnnotator(
                thickness=thickness
            )
            CLASS_NAMES_DICT = model.model.names

            with sv.VideoSink(target_path=target_path, video_info=video_info, codec="h264") as sink:
                for frame in frame_generator:
                    results = model(frame, verbose=False)[0]
    
                    detections = sv.Detections.from_ultralytics(results)
                    detections = detections[detections.confidence > conf]
                    detections = detections[polygon_zone.trigger(detections)]
                    detections = byteTracker.update_with_detections(detections)
                    
                    labels = []
                    for tracker_id, xyxy, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
                        labels.append(f'{CLASS_NAMES_DICT[class_id]} {tracker_id}')
                        x3, y3, x4, y4 = xyxy
                        id = tracker_id
                        cx = int(x3 + x4) // 2
                        cy = int(y3 + y4) // 2
                        

                        if red_line_y<(cy+offset) and red_line_y > (cy-offset):   
                            down[id]=time.time()   # current time when vehichle touch the first line
                        if id in down:
                            
                            if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
                                elapsed_time=time.time() - down[id]  # current time when vehicle touch the second line. Also we a re minusing the previous time ( current time of line 1)
                                if counter_down.count(id)==0:
                                    counter_down.append(id)
                                    a_speed_ms = distance / elapsed_time
                                    a_speed_kh = int(a_speed_ms * 3.6)
                                    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                                    plate_img, numPlate = get_number(frame, xyxy)
                                    vehicle_data = f"{id},{videoName},{CLASS_NAMES_DICT[class_id]},{a_speed_kh},{numPlate}\n"
                                    if not os.path.isfile(record_path):
                                        with open(record_path,'w') as record:
                                                titles = f'id,Time,vehicle,Speed(km/h),Number-Plate'
                                                record.write(titles+'\n')
                                                
                                                record.write(vehicle_data)
                                    else:
                                        with open(record_path,'a') as record:
                                            record.write(vehicle_data)


                        #####going UP blue line#####     
                        if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
                            up[id]=time.time()
                        if id in up:

                            if red_line_y<(cy+offset) and red_line_y > (cy-offset):
                                elapsed1_time=time.time() - up[id]
                                # formula of speed= distance/time 
                                if counter_up.count(id)==0:
                                    counter_up.append(id)
                                    a_speed_ms1 = distance / elapsed1_time
                                    a_speed_kh1 = int(a_speed_ms1 * 3.6)
                                    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                                    plate_img, numPlate = get_number(frame, xyxy)
                                    vehicle_data = f"{id},{videoName},{CLASS_NAMES_DICT[class_id]},{a_speed_kh1},{numPlate}\n"
                                    if not os.path.isfile(record_path):
                                        with open(record_path,'w') as record:
                                                titles = f'id,Time,vehicle,Speed(km/h),Number-Plate'
                                                record.write(titles+'\n')
                                                
                                                record.write(vehicle_data)
                                    else:
                                        with open(record_path,'a') as record:
                                            record.write(vehicle_data)

                        
                    
                    text_color = (0, 0, 0)  # Black color for text
                    yellow_color = (0, 255, 255)  # Yellow color for background
                    red_color = (0, 0, 255)  # Red color for lines
                    blue_color = (255, 0, 0)  # Blue color for lines

                    cv2.line(frame, (points[0], red_line_y), (width, red_line_y), red_color, 2)
                    cv2.putText(frame, ('Red Line'), (points[0], red_line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

                    cv2.line(frame, (points[0], blue_line_y), (width, blue_line_y), blue_color, 2)
                    cv2.putText(frame, ('Blue Line'), (points[0], blue_line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

                    frame = trace_annotator.annotate(
                        scene=frame, detections=detections
                    )
                    frame = label_annotator.annotate(
                                        scene=frame, detections=detections, labels=labels
                                    )
                    frame = bounding_box_annotator.annotate(
                        scene=frame, detections=detections
                    )
                    frame = poly_annotator.annotate(scene=frame, detections=detections)
                    sink.write_frame(frame)


def generate_frame(cap, queue_cam_dict):
    while True:
        frame = queue_cam_dict[f'{cap}'].get()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpag\r\n\r\n' + frame + b'\r\n') 
        
def RunningCamera(cap,model, conf, cap_Path, queue_cam_dict, real, points):

    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    out = cv2.VideoWriter(f'{cap_Path}/{curr_datetime}.mp4', fourcc, fps, (width, height)) 

    while True:
        success, frame = cap.read()
        if not success:
            print(cap,queue_cam_dict)
            break
        else:
            
            out.write(frame)
            if real == "on":
                frame = RealTimeDetection(frame=frame, cap=cap, model=model,conf=conf,distance=36,points=points)
            queue_cam_dict[f'{cap}'].put(frame)

        if count == 1000:
            print(cap, 'Done')
            out.release()
            count = 0
            curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            out = cv2.VideoWriter(f'{cap_Path}/{curr_datetime}.mp4', fourcc, fps, (width, height)) 
    
        count = count + 1

def RealTimeDetection(frame, cap, model, conf, distance, points):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    down = {}
    up = {}
    counter_down = []
    counter_up = []

    offset = 6

    
    red_line_y = points[1] * 2
    blue_line_y = (points[3] + points[1])*2
    
    poly_point = np.array([[points[0]*2, points[1]*2],
                [points[0]*2, (points[3] + points[1])*2],
                [(points[2] + points[0])*2,(points[3] + points[1])*2],
                [(points[2] + points[0])*2, points[1]*2]])

    thickness = sv.calculate_dynamic_line_thickness(
                    resolution_wh=(width,height)
            )
    text_scale = sv.calculate_dynamic_text_scale(
        resolution_wh=(width,height)
    )
    
    byteTracker = sv.ByteTrack()
    
    poly_annotator = sv.PolygonAnnotator(
                color=sv.Color.red,
                thickness=thickness
            )
    
    polygon_zone = sv.PolygonZone(
                    polygon=poly_point, 
                    frame_resolution_wh=(width,height)
                    , triggering_position=sv.Position.CENTER
                )

    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        position=sv.Position.CENTER
        )

    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )

    bounding_box_annotator = sv.BoundingBoxAnnotator(
        thickness=thickness
    )
    CLASS_NAMES_DICT = model.model.names


    results = model(frame, verbose=False)[0]
    
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.confidence > conf]
    detections = detections[polygon_zone.trigger(detections)]
    detections = byteTracker.update_with_detections(detections)
    
    labels = []
    for tracker_id, xyxy, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
        labels.append(f'{CLASS_NAMES_DICT[class_id]}')
        x3, y3, x4, y4 = xyxy
        id = tracker_id
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        

        if red_line_y<(cy+offset) and red_line_y > (cy-offset):   
            down[id]=time.time()   # current time when vehichle touch the first line
        if id in down:
            
            if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
                elapsed_time=time.time() - down[id]  # current time when vehicle touch the second line. Also we a re minusing the previous time ( current time of line 1)
                print(f'{elapsed_time}-whats' )
                if counter_down.count(id)==0:
                    counter_down.append(id)
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = int(a_speed_ms * 3.6)
                    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                    plate_img, numPlate = get_number(frame, xyxy)
                    vehicle_data = f"{id},{curr_datetime},{CLASS_NAMES_DICT[class_id]},{a_speed_kh},{numPlate}\n"
                    if not os.path.isfile(record_path):
                        with open(record_path,'w') as record:
                                titles = f'id,Time,vehicle,Speed(km/h),Number-Plate'
                                record.write(titles+'\n')
                                
                                record.write(vehicle_data)
                    else:
                        with open(record_path,'a') as record:
                            record.write(vehicle_data)
                    # cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                    # cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    # cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)


        #####going UP blue line#####     
        if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
            up[id]=time.time()
        if id in up:

            if red_line_y<(cy+offset) and red_line_y > (cy-offset):
                elapsed1_time=time.time() - up[id]
                # formula of speed= distance/time 
                if counter_up.count(id)==0:
                    counter_up.append(id)
                    a_speed_ms1 = distance / elapsed1_time
                    a_speed_kh1 = int(a_speed_ms1 * 3.6)
                    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                    plate_img, numPlate = get_number(frame, xyxy)
                    vehicle_data = f"{id},{curr_datetime},{CLASS_NAMES_DICT[class_id]},{a_speed_kh1},{numPlate}\n"
                    if not os.path.isfile(record_path):
                        with open(record_path,'w') as record:
                                titles = f'id,Time,vehicle,Speed(km/h),Number-Plate'
                                record.write(titles+'\n')
                                
                                record.write(vehicle_data)
                    else:
                        with open(record_path,'a') as record:
                            record.write(vehicle_data)
                    # cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                    # cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    # cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        
    
    text_color = (0, 0, 0)  # Black color for text
    yellow_color = (0, 255, 255)  # Yellow color for background
    red_color = (0, 0, 255)  # Red color for lines
    blue_color = (255, 0, 0)  # Blue color for lines

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)

    cv2.line(frame, (points[0], red_line_y), (width, red_line_y), red_color, 2)
    cv2.putText(frame, ('Red Line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.line(frame, (points[0], blue_line_y), (width, blue_line_y), blue_color, 2)
    cv2.putText(frame, ('Blue Line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    # cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    frame = trace_annotator.annotate(
        scene=frame, detections=detections
    )
    frame = label_annotator.annotate(
                        scene=frame, detections=detections, labels=labels
                    )
    frame = bounding_box_annotator.annotate(
        scene=frame, detections=detections
    )
    frame = poly_annotator.annotate(scene=frame, detections=detections)
    return frame