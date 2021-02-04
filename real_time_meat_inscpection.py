"""
Meat type tracking and detection model
- video based analysis

"""
from deep_sort.detection import Detection
import numpy as np
from ROI_EXtraction.ROI_EX_from_image_Sort import ROX_EX_image
from deep_sort import preprocessing
from timeit import time
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from collections import *
from tools.generate_detections import create_box_encoder
import cv2
from Data_visual.Live_plot_demo import Text_indicator, PiChart_demonstration, show_multiple_images, \
    BarChart_demonstration

Encoder_dir = '/home/pk/Desktop/Github_project/Object_Tracking/deep_sort_meat_stage1/' \
              'pre_generated_detection/Network/mars-small128.pb'
Video_input_path = '/media/pk/Data/Project/dataset/Meat_image/RMIT/Original_dataset/test_image/test_lab1_original.mp4'
Video_Out_path = '/media/pk/Data/Project/dataset/Meat_image/RMIT/Original_dataset/test_image/'
graph_out_path = '/media/pk/Data/Project/Project/Deep_Sort/Code/deep_sort_1/Graph_output/'

PATH_TO_FROZEN_GRAPH = '/media/pk/Data/Project/Project/Venice/Maet_stage1/efficientdet_d1_coco17_tpu-32/6class/' \
                       'TRT_model/With_calibration_input'
PATH_TO_LABELS = '/media/pk/Data/Project/dataset/Meat_image/RMIT/Original_dataset/' \
                 'Annotation_Pascal_Voc/TFrecord_format/6_class/Meat_label_map_6_classes.pbtxt'




# Init parameters
Target_Class = ['ChunkRoll', 'CubeRoll', 'Rump', 'ShinBeef', 'Striploin', 'Tenderloin']
Fr_th = 15  # threshold for counting classes
startig_point = 27  # Start analyzing Video from 'Startig_point '( second based)
ending_point = 88  # End analyzing Video from 'Startig_point '( second based)
writeVideo_flag = False  # Set Writing Video output
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.3


# tracker and coder init
def tracker_coder_init(Encoder_dir, max_cosine_distance, nn_budget):
    # initialize encoder and tracker
    encoder = create_box_encoder(Encoder_dir, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    return encoder, tracker


# Crop video from top and bottom of the frame by size_reduction = ( 0.5, 0.5)
def crop_frame(frame, size_reduction):
    h, w, _ = frame.shape
    h_red = int((size_reduction[0] * h) / 2)
    w_red = int((size_reduction[1] * w) / 2)
    new_frame = frame[(0 + h_red): (h - h_red + 1), (0 + w_red): (w - w_red + 1), :]
    return new_frame


# Return class majority of the detected object
def Class_majority_find(container, ID, class_total):
    IDs_classes = [i for i, j in enumerate(container) if j == ID]
    Class_accum = []
    for C2 in IDs_classes:
        Class_accum.append(class_total[C2])
    out_CMF = Counter(Class_accum)
    maxval = (max(out_CMF.values()))
    Class_major = [key for key, value in out_CMF.items() if value == maxval][0]
    return Class_major


# Main video analysis
def video_analysis_main(Encoder_dir, Video_input_path, Video_Out_path, Target_Class, Fr_th,
                        startig_point, ending_point, writeVideo_flag, max_cosine_distance,
                        nn_budget, nms_max_overlap):
    # tracker and coder init
    encoder, tracker = tracker_coder_init(Encoder_dir, max_cosine_distance, nn_budget)

    # read video frames
    video_capture = cv2.VideoCapture(Video_input_path)
    Vid_fr = int(video_capture.get(cv2.CAP_PROP_FPS))
    file_name = Video_input_path.split('/')[-1]
    if writeVideo_flag:
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(Video_Out_path + file_name.split('.')[0] + '_output_' + str(Fr_th) + '.mp4', fourcc, 20,
                              (1346, 433))
        # out_IP = cv2.VideoWriter(Video_Out_path + file_name.split('.')[0] + '_Indicator_output.mp4', fourcc, 30,
        #                          (I_P.shape[1], I_P.shape[0]))
        # list_file = open(Video_Out_path + 'detection.txt', 'w')

    ##inter initialization
    # starting point of the algorithm
    start = time.time()
    # Counter of Each object
    counter_total = []
    class_total = []
    # Final Meat type
    Meat_Category = [0] * len(Target_Class)
    # Create an Indicator Page
    I_P = np.zeros([700, 1000, 3], dtype=np.uint8)
    I_P.fill(255)

    # fps and frame_index init
    fps_accum = []
    fps = 0.0
    frame_index = 0
    while True:
        ret, frame = video_capture.read()
        if ret:
            frame = crop_frame(frame, size_reduction=(0.6, 0.65))
            cv2.putText(frame, 'Meat Labelling Automation (PK)', (10, 20), 0, 5e-3 * 150, (150, 150, 150), 1)
            if (frame_index >= (startig_point * Vid_fr)) and (frame_index <= (ending_point * Vid_fr)):
                t1 = time.time()
                # if frame.shape[0] < frame.shape[1]: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # frame = cv2.rotate(frame, cv2.ROTATE_180)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxs, class_names = ROX_EX_image(image)
                # print(boxs, class_names)
                features = encoder(frame, boxs)
                # score to 1.0 here).
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                i = int(0)
                indexIDs = []

                for det in detections:
                    bbox = det.to_tlbr()
                    # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255,
                    # 255), 2)

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    # boxes.append([track[0], track[1], track[2], track[3]])
                    indexIDs.append(int(track.track_id))
                    bbox = track.to_tlbr()

                    if len(class_names) > 0:
                        class_name = class_names[0]
                        counter_total.append(int(track.track_id))
                        class_total.append(class_names[0])
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (250, 203, 150), 3)
                        # cv2.putText(frame, str('Analysing'), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150,
                        # color, 2)
                    i += 1

                # Filtering repeattion of face w/n occurance in video
                Rep_IDs = Counter(counter_total)
                # Rep_IDs = OrderedCounter(Rep_IDs)
                for Key, value in Rep_IDs.items():
                    if value == Fr_th:
                        # C_Id_total = class_total[counter_total.index(Key)]
                        C_Id_total = Class_majority_find(counter_total, Key, class_total)
                        category_index = [i for i, s in enumerate(Target_Class) if C_Id_total in s]
                        Meat_Category[category_index[0]] += 1
                        counter_total.append(int(Key))
                        class_total.append(C_Id_total)

                ## Data Visualization

                fps = (fps + (1. / (time.time() - t1))) / 2
                fps_accum.append(fps)
                cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 100, (0, 0, 0), 2)

                # Text_indicator(Meat_Category, Target_Class, I_P)
                I_P = Text_indicator(Meat_Category, Target_Class, I_P)
                multiple_image = show_multiple_images(frame, I_P)
                # print(multiple_image.shape)
                # cv2.imshow('SSD', frame)
                # cv2.imshow('Indicator', I_P)

                # # bar Chart
                # BarChart_demonstration(Meat_Category, Target_Class)

                if writeVideo_flag:
                    # save a frame
                    out.write(multiple_image)
                    # out_IP.write(I_P)

                I_P.fill(255)
                print(Rep_IDs)
                print(Target_Class)
                print(Meat_Category)
                print('==================================')

                # Press Q to stop!
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        else:
            break
        frame_index = frame_index + 1
        print('Time---->', round((frame_index / Vid_fr), 1), 'Second')
        print('FPS average--->', np.average(fps_accum))

    print(" ")
    print("[Finish]")
    end = time.time()

    video_capture.release()

    if writeVideo_flag:
        out.release()
    cv2.destroyAllWindows()


video_analysis_main(Encoder_dir, Video_input_path, Video_Out_path, Target_Class, Fr_th,
                    startig_point, ending_point, writeVideo_flag, max_cosine_distance,
                    nn_budget, nms_max_overlap)

# creat_video_of_graph_images
