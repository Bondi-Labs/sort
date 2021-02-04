from sort import *
import cv2
from collections import *
from utilities import *
from ROI_EXtraction.ROI_EX_from_image_Sort import ROX_EX_image


Video_input_path = '/media/pk/Data/Project/dataset/Meat_image/RMIT/Original_dataset/test_image/test_lab1_original.mp4'
Video_Out_path = '/media/pk/Data/Project/dataset/Meat_image/RMIT/Original_dataset/test_image/'

# Init parameters
Target_Class = ['ChunkRoll', 'CubeRoll', 'Rump', 'ShinBeef', 'Striploin', 'Tenderloin']
Fr_th = 5  # threshold for counting classes
startig_point = 27  # Start analyzing Video from 'Startig_point '( second based)
ending_point = 88  # End analyzing Video from 'Startig_point '( second based)

# read video frames
video_capture = cv2.VideoCapture(Video_input_path)
Vid_fr = int(video_capture.get(cv2.CAP_PROP_FPS))
file_name = Video_input_path.split('/')[-1]

# create instance of SORT
mot_tracker = Sort()

# read video frames
video_capture = cv2.VideoCapture(Video_input_path)
Vid_fr = int(video_capture.get(cv2.CAP_PROP_FPS))
file_name = Video_input_path.split('/')[-1]

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



def Text_indicator(Meat_Category, Target_Class, Indicator_page):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 0)
    thickness = 3
    cv2.putText(Indicator_page, "STATISTICAL INDICATOR", (int(120), int(60)), font, 1.5, color, 5, cv2.LINE_AA)
    cv2.putText(Indicator_page, Target_Class[0] + ' : ' + str(Meat_Category[0]), (int(120), int(150)), font, fontScale,
                color,
                thickness, cv2.LINE_AA)
    cv2.putText(Indicator_page, Target_Class[1] + ' : ' + str(Meat_Category[1]), (int(120), int(250)), font, fontScale,
                color,
                thickness, cv2.LINE_AA)
    cv2.putText(Indicator_page, Target_Class[2] + ' : ' + str(Meat_Category[2]), (int(120), int(350)), font, fontScale,
                color,
                thickness, cv2.LINE_AA)
    cv2.putText(Indicator_page, Target_Class[3] + ' : ' + str(Meat_Category[3]), (int(120), int(450)), font, fontScale,
                color,
                thickness, cv2.LINE_AA)
    cv2.putText(Indicator_page, Target_Class[4] + ' : ' + str(Meat_Category[4]), (int(120), int(550)), font, fontScale,
                color,
                thickness, cv2.LINE_AA)
    cv2.putText(Indicator_page, Target_Class[5] + ' : ' + str(Meat_Category[5]), (int(120), int(650)), font, fontScale,
                color,
                thickness, cv2.LINE_AA)
    return Indicator_page

def show_multiple_images(image1, image2):
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    numpy_horizontal_concat = np.concatenate((image1, image2), axis=1)
    cv2.imshow('Meat Labelling Automation (PK) ', numpy_horizontal_concat)
    return numpy_horizontal_concat

while True:
    ret, frame = video_capture.read()
    if ret:
        frame = crop_frame(frame, size_reduction=(0.6, 0.65))
        cv2.putText(frame, 'Meat Labelling Automation (PK)', (10, 20), 0, 5e-3 * 150, (150, 150, 150), 1)
        if (frame_index >= (startig_point * Vid_fr)) and (frame_index <= (ending_point * Vid_fr)):
            t1 = time.time()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, class_names = ROX_EX_image(image)
            print(detections)

            # Call the tracker
            # track_bbs_ids format is [bbox,Id] = [ x1,y1,x2,y2, Id]


            track_bbs_ids = mot_tracker.update(detections)

            i = int(0)
            indexIDs = []

            if track_bbs_ids.any():
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

                I_P.fill(255)
                print(Rep_IDs)
                print(Target_Class)
                print(Meat_Category)
                print('==================================')
    else:
        break
    frame_index = frame_index + 1
    print('Time---->', round((frame_index / Vid_fr), 1), 'Second')
    print('FPS average--->', np.average(fps_accum))


print(" ")
print("[Finish]")
end = time.time()
