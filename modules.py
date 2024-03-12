import cv2
import numpy as np
import os
import easyocr
import supervision as sv
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import queue

queue_cam_dict = {}

fourcc = cv2.VideoWriter_fourcc(*'h264') #h264
camDict = {}

width, height = 0,0


record_path = 'static/record.csv'
LICENSE_PLATE_MODEL = "numPlatev8n.pt"


licenseModel = YOLO(LICENSE_PLATE_MODEL)
reader = easyocr.Reader(['en'] )

# SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
SOURCE = np.array([[398, 339], [865, 333], [1561, 719], [-447, 719]])

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
    global width,height
    addr = f'rtsp://{user}:{password}@{ip}:{port}'
    print(addr)
    cap =  cv2.VideoCapture(addr)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
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

        # if event == cv2.EVENT_MOUSEMOVE:
        #     mx, my = x, y

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
                return img, self.pointlist
 
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

def VideoSink(model, conf: float, cap_Path: str, pdt_Path: str):
    
    cap_list = os.listdir(cap_Path)
    pdt_list = os.listdir(pdt_Path)
    CLASS_DICT_NAME =model.model.names
    
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

            byte_track = sv.ByteTrack(
                frame_rate=video_info.fps, track_thresh=conf
            )

            thickness = sv.calculate_dynamic_line_thickness(
                resolution_wh=video_info.resolution_wh
        )
            text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)
            bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
            poly_annotator = sv.PolygonAnnotator(
                color=sv.Color.red,
                thickness=thickness
            )
            label_annotator = sv.LabelAnnotator(
                text_scale=text_scale,
                text_thickness=thickness,
                text_position=sv.Position.BOTTOM_CENTER,
            )
            trace_annotator = sv.TraceAnnotator(
                thickness=thickness,
                trace_length=video_info.fps * 2,
                position=sv.Position.BOTTOM_CENTER,
            )

            frame_generator = sv.get_video_frames_generator(source_path=source_path)

            polygon_zone = sv.PolygonZone(
                polygon=SOURCE, frame_resolution_wh=video_info.resolution_wh
            )
            view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

            coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))


            with sv.VideoSink(target_path=target_path, video_info=video_info, codec="h264") as sink:
                for frame in frame_generator:
                    results = model(frame, verbose=False)[0]

                    detections = sv.Detections.from_ultralytics(results)
                    detections = detections[detections.confidence > conf]
                    detections = detections[polygon_zone.trigger(detections)]
                    detections = detections.with_nms(threshold=0.5)
                    detections = byte_track.update_with_detections(detections=detections)

                    points = detections.get_anchors_coordinates(
                        anchor=sv.Position.BOTTOM_CENTER
                    )
                    points = view_transformer.transform_points(points=points).astype(int)

                    for tracker_id, [_, y] in zip(detections.tracker_id, points):
                        coordinates[tracker_id].append(y)

                    labels = []
                    for tracker_id, xyxy, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
                        if len(coordinates[tracker_id]) < video_info.fps / 2:
                            labels.append(f"#{tracker_id}")
                        else:
                            coordinate_start = coordinates[tracker_id][-1]
                            coordinate_end = coordinates[tracker_id][0]
                            distance = abs(coordinate_start - coordinate_end)
                            time = len(coordinates[tracker_id]) / video_info.fps
                            speed = distance / time * 3.6
                            labels.append(f"#{tracker_id} {int(speed)} km/h")
                            plate_img, numPlate = get_number(frame, xyxy)
                            class_name = CLASS_DICT_NAME[class_id]
                            vehicle_data.update({f'{tracker_id}': f"{videoName},{class_name},{int(speed)},{numPlate}"})
                            
                            # if vehicle_data:
                            #     for vehicle in vehicle_data:
                            #         if not f'{vehicle}.jpg' in os.listdir('static/TrufficData'):
                            #             cv2.imwrite(f'static/TrufficData/{class_id}/{videoName}-{vehicle}.jpg', sv.crop_image(image=frame, xyxy=xyxy))

                    annotated_frame = frame.copy()
                    annotated_frame = trace_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                    annotated_frame = bounding_box_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                    annotated_frame = poly_annotator.annotate(scene=annotated_frame,detections=detections)
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels
                    )
                    print(len(vehicle_data))
                    sink.write_frame(annotated_frame)

            if not os.path.isfile(record_path):
                with open(record_path,'w') as record:
                        titles = f'id,Time,vehicle,Speed(km/h),Number-Plate'
                        record.write(titles+'\n')
                        for data in vehicle_data:
                            record.write(f'{data},{vehicle_data[data]}\n')
            else:
                with open(record_path,'a') as record:
                    for data in vehicle_data:
                            record.write(f'{data},{vehicle_data[data]}\n')


def generate_frame(cap, real, model,queue_cam_dict):
    while True:
        frame = queue_cam_dict[f'{cap}'].get()
        box_annotator = sv.BoxAnnotator()


        if real == 'on':
            results = model(frame, verbose=False)[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == 0]


            frame = box_annotator.annotate(scene=frame, detections=detections)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()


        yield (b'--frame\r\n'
                b'Content-Type: image/jpag\r\n\r\n' + frame + b'\r\n') 
        
def RunningCamera(cap, cap_Path, queue_cam_dict):

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
            queue_cam_dict[f'{cap}'].put(frame)

        if count == 1000:
            print(cap, 'Done')
            out.release()
            count = 0
            curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            out = cv2.VideoWriter(f'{cap_Path}/{curr_datetime}.mp4', fourcc, fps, (width, height)) 
    
        count = count + 1

def RealTimeDetection(cap, model, conf, pdt_Path):
    count = 0

    box_annotator = sv.BoxAnnotator()
    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    out = cv2.VideoWriter(f'{pdt_Path}/{curr_datetime}.mp4', fourcc, 20.0, (width, height)) 
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:

            results = model(frame, verbose=False)[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == 0]
            detections = detections[detections.confidence > conf]


            frame = box_annotator.annotate(scene=frame, detections=detections)
            # output the frame 
            out.write(frame)

            # ret, buffer = cv2.imencode('.jpg', frame)
            # frame = buffer.tobytes()

        if count == 1000:
            out.release()
            count = 0
            curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            out = cv2.VideoWriter(f'{pdt_Path}/{curr_datetime}.mp4', fourcc, 20.0, (width, height)) 

        count = count + 1
        if count % 50 == 0:
            print(count)




