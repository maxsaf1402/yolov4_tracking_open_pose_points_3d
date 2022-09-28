import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from object_tracker import object_tracker_to_csv
from transform import transform
from multi_person_openpose import open_pose

video_path = './data/video/test_1.mp4'
framework = 'tf'
path_to_weights = './checkpoints/yolov4-416'
output_path = None
output_format = 'XVID'
model_yolo = 'yolov4'
typy_model = False
iou = 0.45
score = 0.50
count_obj = False
is_info = False
is_show = False


# # Object Tracker and Save to csv (frame, tracker_id, bbox coordinates)
# object_tracker_to_csv(video_path,framework,path_to_weights, output_path, output_format, model_yolo, typy_model, iou, score, count_obj, is_info, is_show)

# OpenPose parameters
nPoints = 18
POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]
# read bboxes
df = pd.read_csv('boxes_data.csv')
list_frames = df['frame_num'].unique().tolist()

vid = cv2.VideoCapture(video_path)
cmap = plt.get_cmap('tab10')
colors = [cmap.colors[i] for i in range(len(cmap.colors))]

frame_num = 0

# Main
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_render = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        image = Image.fromarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        print('Video has ended or failed, try a different video format!')
        break
    frame_num += 1
    print('Frame{}'.format(frame_num))
    # For each frame from df with bboxes
    if frame_num in (list_frames):

        df_frame = df.query('frame_num==@frame_num')
        # For each tracker_id in frame
        for index, row in df_frame.iterrows():
            # Coordinates of bbox
            xmin = int(row['xmin'])
            xmax = int(row['xmax'])
            ymin = int(row['ymin'])
            ymax = int(row['ymax'])
            center_x = int((xmin+xmax)/2)
            center_y = int((ymin+ymax)/2)
            center_box = (center_x, center_y)
            # Color for each tracker_id
            color = colors[int(row['tracker_id']) % len(colors)]
            color = [i * 255 for i in color]
            # Draw bboxes with tracker_id on main window
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, 'person' + "-" + str(row['tracker_id']), (xmin, int(ymin - 10)), 0, 0.75, (255, 255, 255), 2)

            # Take part of frame with bbox
            temp_frame = cv2.cvtColor(frame[ymin:ymax,xmin:xmax], cv2.COLOR_BGR2RGB)
            # OpenPose (key points and skeleton) on bbox
            frame[ymin:ymax,xmin:xmax], detected_keypoints = open_pose(temp_frame, color)

            # transform matrix parameters
            trans = tuple((0.0, 0.0, 0.0))
            rot = tuple((0.0, 0.0, 7.560000000000002))
            scale = tuple((0.956, 0.552, 0.96))
            shear = tuple((0.0, 0.0, 0.0))

            # draw each point of skeleton after transform
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    temp_point = detected_keypoints[i][j][0:2]
                    temp_point = (temp_point[0]+xmin, temp_point[1]+ymin)
                    new_point = transform(temp_point, trans, rot, scale, shear)
                    cv2.circle(frame_render, new_point, radius=1, color=color, thickness=2)

    else:
        pass

    frame_render = cv2.resize(frame_render, (960, 540))
    frame = cv2.resize(frame, (960, 540))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.waitKey(1)
    cv2.imshow('Rendering', frame_render)
    cv2.imshow('Frame', frame)