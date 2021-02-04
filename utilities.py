from collections import *
import cv2
import numpy as np


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



#
# def Text_indicator(Meat_Category, Target_Class, Indicator_page):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 1
#     color = (0, 0, 0)
#     thickness = 3
#     cv2.putText(Indicator_page, "STATISTICAL INDICATOR", (int(120), int(60)), font, 1.5, color, 5, cv2.LINE_AA)
#     cv2.putText(Indicator_page, Target_Class[0] + ' : ' + str(Meat_Category[0]), (int(120), int(150)), font, fontScale,
#                 color,
#                 thickness, cv2.LINE_AA)
#     cv2.putText(Indicator_page, Target_Class[1] + ' : ' + str(Meat_Category[1]), (int(120), int(250)), font, fontScale,
#                 color,
#                 thickness, cv2.LINE_AA)
#     cv2.putText(Indicator_page, Target_Class[2] + ' : ' + str(Meat_Category[2]), (int(120), int(350)), font, fontScale,
#                 color,
#                 thickness, cv2.LINE_AA)
#     cv2.putText(Indicator_page, Target_Class[3] + ' : ' + str(Meat_Category[3]), (int(120), int(450)), font, fontScale,
#                 color,
#                 thickness, cv2.LINE_AA)
#     cv2.putText(Indicator_page, Target_Class[4] + ' : ' + str(Meat_Category[4]), (int(120), int(550)), font, fontScale,
#                 color,
#                 thickness, cv2.LINE_AA)
#     cv2.putText(Indicator_page, Target_Class[5] + ' : ' + str(Meat_Category[5]), (int(120), int(650)), font, fontScale,
#                 color,
#                 thickness, cv2.LINE_AA)
#     return Indicator_page


# def show_multiple_images(image1, image2):
#     image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
#     numpy_horizontal_concat = np.concatenate((image1, image2), axis=1)
#     cv2.imshow('Meat Labelling Automation (PK) ', numpy_horizontal_concat)
#     return numpy_horizontal_concat