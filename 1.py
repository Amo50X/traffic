import cv2 
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time


model = YOLO("yolov8x.pt")

source = "vlc-record-2024-03-06-14h39m58s-Traffic_4.mp4-.mp4"

down = {}
up = {}
counter_down = []
counter_up = []

offset = 6

cap = cv2.VideoCapture(source)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#================================================
byteTracker = sv.ByteTrack()
#========================================1========
distance = int(input("Enter Distance of 2 points"))

_,test_img = cap.read()
test_img = cv2.resize(test_img, [int(width / 2),int(height / 2)])

points = cv2.selectROI(img=test_img)
cv2.destroyAllWindows()
print(points, width, height)


red_line_y = points[1] * 2
blue_line_y = (points[3] + points[1])*2

thickness = sv.calculate_dynamic_line_thickness(
                resolution_wh=(width,height)
        )
text_scale = sv.calculate_dynamic_text_scale(
    resolution_wh=(width,height)
)
poly_point = np.array([[points[0]*2, points[1]*2],
              [points[0]*2, (points[3] + points[1])*2],
              [(points[2] + points[0])*2,(points[3] + points[1])*2],
              [(points[2] + points[0])*2, points[1]*2]])

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

while True:
    suc, frame =  cap.read()   
    results = model(frame, verbose=False)[0]
    
    detections = sv.Detections.from_ultralytics(results)
    detections = byteTracker.update_with_detections(detections)
    detections = detections[polygon_zone.trigger(detections)]
    
    
    labels = []
    for tracker_id, xyxy, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
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
                    a_speed_kh = a_speed_ms * 3.6  # this will give kilometers per hour for each vehicle. This is the condition for going downside
                    # cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                    # cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    # cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    print(f'{a_speed_kh} - UP')

                
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
                a_speed_kh1 = a_speed_ms1 * 3.6
                # cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                # cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                # cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                print(f'{a_speed_kh} - UP')
    
    
    text_color = (0, 0, 0)  # Black color for text
    yellow_color = (0, 255, 255)  # Yellow color for background
    red_color = (0, 0, 255)  # Red color for lines
    blue_color = (255, 0, 0)  # Blue color for lines

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)

    cv2.line(frame, (points[0], red_line_y), (width, red_line_y), red_color, 2)
    cv2.putText(frame, ('Red Line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.line(frame, (points[0], blue_line_y), (width, blue_line_y), blue_color, 2)
    cv2.putText(frame, ('Blue Line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    # annotated_frame = label_annotator.annotate(
    #                     scene=annotated_frame, detections=detections, labels=labels
    #                 )
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = poly_annotator.annotate(scene=annotated_frame, detections=detections)
    # annotated_frame = label_annotator.annotate(
    #     scene=annotated_frame, detections=detections, labels=labels
    # )
    
    
    
    cv2.imshow("Trace", annotated_frame)
    
    key = cv2.waitKey(10)
    
    if key == 27:
        break
    
    
cap.release()
cv2.destroyAllWindows() 