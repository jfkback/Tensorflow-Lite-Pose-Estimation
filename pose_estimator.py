import numpy as np
import cv2
import tensorflow as tf
from skimage import io
from PIL import Image
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.misc import imread, imresize

file_name = 'man3.jpg'
cap = cv2.VideoCapture('tiktok.mp4')
# fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000)

input_mean = 127.5
input_std = 127.5

interpreter = tf.lite.Interpreter(model_path="multi_person_mobilenet_v1_075_float.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get_points(img):
    print(img.shape)
    input_data = np.expand_dims(img, axis=0)
    input_data = (np.float32(input_data) - input_mean) / input_std
    print(input_data.dtype, input_details)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    heatmap = interpreter.get_tensor(output_details[0]['index'])
    float_short_offset = interpreter.get_tensor(output_details[1]['index'])
    scores = expit(heatmap)
    keypoint = []
    for j in range(0, 17):
        heatmap_pos = np.unravel_index(np.argmax(scores[0, :, :, j]), scores[0, :, :, j].shape)
        offset_vector = (float_short_offset[0, heatmap_pos[0], heatmap_pos[1], j], float_short_offset[0, heatmap_pos[0], heatmap_pos[1], 17 + j])
        keypoint.append(np.array(heatmap_pos) * 16 + np.array(offset_vector))
    return keypoint


def resize_img(img):
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = img.resize((width, height))
    return img


def resize_img_np(img):
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    # img = np.resize(img, (height, width, 3))
    img_height, img_width, _ = img.shape
    horz_crop = 400
    vert_crop = 0
    img = img[vert_crop:img_height - vert_crop, horz_crop:img_width - horz_crop]
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    return img

# img = Image.open(file_name)
# img = resize_img(img)
#
# plt.imshow(img)
# keypoint = get_points(img)
# for j in range(0, 17):
#     plt.scatter(keypoint[j][1], keypoint[j][0], s=5, c='r')
# plt.show()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
while(cap.isOpened()):
    ret, frame = cap.read()

    img_frame = resize_img_np(frame)
    fgmask = fgbg.apply(img_frame)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(img_frame, img_frame, mask= fgmask)

    keypoints = get_points(img_frame)
    for keypoint in keypoints:
        img_frame = cv2.circle(img_frame, (int(keypoint[1]), int(keypoint[0])), 3, (255,0,0), -1)
    # cv2.imshow('frame', res)
    cv2.imshow('frame', img_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()

# Closes all the frames
cv2.destroyAllWindows()
