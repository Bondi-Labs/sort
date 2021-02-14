from sort import *
import cv2
from collections import *
from utilities import crop_frame, Text_indicator, show_multiple_images, Class_majority_find
from ROI_EX_from_image_Sort import ROX_EX_image


Video_input_path = '/media/pk/Data/Project/dataset/Meat_image/RMIT/Original_dataset/test_image/test_lab1_original.mp4'
Video_Out_path = '/media/pk/Data/Project/dataset/Meat_image/RMIT/Original_dataset/test_image/'

# Init parameters
Target_Class = ['ChunkRoll', 'CubeRoll', 'Rump', 'ShinBeef', 'Striploin', 'Tenderloin']
Fr_th = 15  # threshold for counting classes
startig_point = 27  # Start analyzing Video from 'Startig_point '( second based)
ending_point = 88  # End analyzing Video from 'Startig_point '( second based)
writeVideo_flag = False  # Set Writing Video output


# create instance of SORT
mot_tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)


# read video frames
video_capture = cv2.VideoCapture(Video_input_path)
Vid_fr = int(video_capture.get(cv2.CAP_PROP_FPS))
file_name = Video_input_path.split('/')[-1]



# read video frames
video_capture = cv2.VideoCapture(Video_input_path)
Vid_fr = int(video_capture.get(cv2.CAP_PROP_FPS))
file_name = Video_input_path.split('/')[-1]

if writeVideo_flag:
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(Video_Out_path + file_name.split('.')[0] + '_SORT_output_' + str(Fr_th) + '.mp4', fourcc, 20,
                          (1346, 433))

##inter initialization
# starting point of the algorithm
start = time.time()
# Counter of Each object
counter_total = []
class_total = []

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
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, class_names = ROX_EX_image(image)

            # Call the tracker
            # track_bbs_ids format is [bbox,Id] = [ x1,y1,x2,y2, Id]
            track_bbs_ids = mot_tracker.update(detections)

            i = int(0)
            indexIDs = []

            for track in track_bbs_ids:
                indexIDs.append(track[4].astype(int))
                bbox = track[0:4].astype(int)

                if len(class_names) > 0:
                    class_name = class_names[0]
                    counter_total.append(track[4].astype(int))
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
print(Target_Class)
print(Meat_Category)
print(" ")
print("[Finish]")
end = time.time()
video_capture.release()

if writeVideo_flag:
    out.release()
cv2.destroyAllWindows()
