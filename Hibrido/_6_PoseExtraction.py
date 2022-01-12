from skimage.transform import resize
from os.path import basename, join, exists
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd 
import cv2
import math
import pickle


from object_detection.utils import config_util
from object_detection.builders import model_builder
import utils

#policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
#tf.keras.mixed_precision.experimental.set_policy(policy)
tf.constant(1) # força os logs do tensorflow logo, ao invés de no começo do treino
print('\nTensorFlow 2 is ready!\n\n\n\n') # algumas quebras de linha pra melhorar a visibilidade

dataset_root_folder = 'D:/Datasets/Multiple Cameras Fall Dataset/videos/'
#coco_root = 'D:/Projeto/Unifall/Datasets/'
model_path = './Modelos_Ready/human_baselines_treinado_4.h5'
input_shape = (224, 224)

with open('labels.pickle', 'rb') as file:
    moments = pickle.load(file)

df = pd.DataFrame(moments, columns=["Scenario","Camera","Start","End","Position Code","Event Name","URL"])
df['Start'] = df['Start'].astype(int)
df['End'] = df['End'].astype(int)
df['Camera'] = df['Camera'].astype(int)
df = df.sort_values(['Scenario', 'Camera','URL'], ascending=True)
labels_pickle = np.array(df)


model = tf.keras.models.load_model(model_path, compile=False)
#model = load_from_tensorflow_1_checkpoint(model, include_top=True)
model.trainable = False
heatmap_shape = tuple(model.output.shape[1::])

center_net_path = './Modelos_Ready/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8/'
configs = config_util.get_configs_from_pipeline_file(center_net_path + 'pipeline.config')
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(center_net_path + 'checkpoint/ckpt-0').expect_partial()

def get_model_detection_function(model):
    @tf.function
    def detect_fn(image):
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])
    return detect_fn
detect_fn = get_model_detection_function(detection_model)

def detect_people(frame, cutoff=0.3):
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    bboxes = detections['detection_boxes'][0].numpy()
    classes = (detections['detection_classes'][0].numpy() + 1).astype(int)
    scores = detections['detection_scores'][0].numpy()
    persons = bboxes[classes==1]
    scores = scores[classes==1]

    h, w, _ = frame.shape
    persons = persons * [h, w, h, w]

    persons = persons[scores > cutoff]
    scores = scores[scores > cutoff]
    return persons, scores

def track_people(persons, scores, _persons, _scores):
    if len(_persons) > 0:
        y11, x11, y12, x12 = np.split(persons, 4, axis=1)
        y21, x21, y22, x22 = np.split(_persons, 4, axis=1)
        yA = np.maximum(y11, np.transpose(y21))
        xA = np.maximum(x11, np.transpose(x21))
        yB = np.minimum(y12, np.transpose(y22))
        xB = np.minimum(x12, np.transpose(x22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        iou_copy = np.copy(iou)

        ps, qs = [], []
        for i in range(0, len(persons)):
            p, q = np.unravel_index(np.argmax(iou), iou.shape)
            if iou[p, q] <= 0.1:
                break

            iou[:, q] = 0
            iou[p, :] = 0
            ps.append(p)
            qs.append(q)

        out_persons = np.copy(_persons)# + _speed
        out_persons[qs] = persons[ps]
        out_scores = np.copy(_scores) - 0.05
        out_scores[qs] = np.clip((1 + scores[ps]) / 2, out_scores[qs], 1) 


        new_persons = np.array(list(set(range(0, len(persons))) - set(ps)))
        if len(new_persons) > 0:
            new_persons_originality = 1 - np.sum(iou_copy, axis=1)[new_persons]
            new_persons = new_persons[new_persons_originality > 0.9]
            for n in new_persons:
                nid = np.argmin(out_scores) # se tiver um id livre a pelo menos 1 frame, recicla
                if out_scores[nid] < 0.1:
                    out_persons[nid] = persons[n]
                    out_scores[nid] = scores[n]
                else:
                    out_persons = np.vstack([out_persons, persons[n]])
                    out_scores = np.hstack([out_scores, scores[n]])

        out_scores[out_scores < 0.1] = 0
        out_persons[out_scores <= 0] = [-999, -999, -999, -999]
        out_scores = np.trim_zeros(out_scores, trim='b')
        out_persons = out_persons[0:len(out_scores)]

        return out_persons, out_scores
    else:
        return persons, scores


cap = cv2.VideoCapture('D:/PythonProjects/MultipleCamerasFallDataset/')

connectivity = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [ 5, 11], [ 6, 12], [ 5,  6], [ 5,  7], [ 6,  8], [ 7,  9], [ 8, 10], [ 1,  2], [ 0,  1], [ 0,  2], [ 1,  3], [ 2,  4], [ 3,  5], [ 4,  6]]
_persons, _scores = [], []
alpha = 0.1
beta = 0.3

#Funções criadas pelo Pedro começam aqui :D
def get_video(i, j):
    return dataset_root_folder + 'chute' + str(i + 1).zfill(2) + '/cam' + str(j + 1) + '.avi'

def extract_features(rectangle, ellipsis):
    aspect_ratio = (rectangle[3] - rectangle[1]) / (rectangle[2] - rectangle[0])
    angle = ellipsis[2]
    area = math.pi * ellipsis[1][0] * ellipsis[1][1] / 4
    return aspect_ratio, angle, area

def findEllipse(x0, y0, x1, y1):
    center = int((y0 + y1)/2), int((x0 + x1)/2)
    radius = int((y1 - y0)/2), int((x1 - x0)/2)
    ellipse_angle = math.degrees(np.arctan2(radius[1], radius[0]))
    ellipse = center, radius, ellipse_angle
    return ellipse

def is_falling(aspect_ratio, angle, area):
    if (aspect_ratio > 1):
        if (4000 <= area <= 12000):
            if (5 <= angle <= 40 or 70 <= angle <= 100):
                return 1

    return -1    

def print_result(tp, fp, fn, tn, ndt, ndf):
    total_empe = fp + tn + ndf
    total_queda = tp + fn + ndt
    total_validos = tp + fp + fn + tn
    if total_empe + total_queda == 0:
        return

    print('Resultado (Detecção):')
    print(f' -Nem Detectou (Em Pé): {ndf} ({100 * ndf / total_empe:.2f}% dos frames em pé)')
    print(f' -Nem Detectou (Queda): {ndt} ({100 * ndt / total_queda:.2f}% dos frames deitado)')
    print()
    print('Resultados (Dos frames detectados)')
    print(f' -TPs = {tp} ({100 * tp / total_validos:.2f}%')
    print(f' -FPs = {fp} ({100 * fp / total_validos:.2f}%')
    print(f' -FNs = {fn} ({100 * fn / total_validos:.2f}%')
    print(f' -TNs = {tn} ({100 * tn / total_validos:.2f}%')
    print(f' -Sensitividade = {tp / (tp + fn + 1e-7)}')
    print(f' -Especificidade = {tn / (tn + fp + 1e-7)}')  
    print()

current_vid = -1 
current_frame = 0
tp, fp, fn, tn, ndt, ndf, marker = 0, 0 , 0, 0, 0, 0, 0
cam_min, cam_max = 0, 8
videos = [get_video(i, j) for i in range(0, 24) for j in range(cam_min, cam_max)]
cap = cv2.VideoCapture('D:/PythonProjects/MultipleCamerasFallDataset/')

while True:
    ret, frame = cap.read()
    if not ret:
        current_vid = current_vid + 1 
        if current_vid < len(videos):
            print_result(tp, fp, fn, tn, ndt, ndf)

            print('current_vid Video: ' + videos[current_vid]) 
            cap = cv2.VideoCapture(videos[current_vid])
            current_frame = 0    
            continue
        else:
            break

    current_frame = current_frame + 1
    start_label = int(labels_pickle[marker, 2])
    end_label = int(labels_pickle[marker, 3])
    label = labels_pickle[marker, 5] in ['Lying on the ground' , 'Falling']
    if current_frame < start_label or (current_vid//8) != labels_pickle[marker, 0] or (current_vid%8) != labels_pickle[marker, 1]:
        continue
    elif current_frame > end_label:
        marker = marker + 1
        _persons, _scores = [], [] # reseta o tracking
        continue
    #else:
    #    print(str(start_label) + '---' + str(end_label) + '--- : ' + str(current_frame))

    fall = None
    frame = (resize(frame, (480, 853)) * 255).astype(np.uint8)
    persons, scores = detect_people(frame)
    persons, scores = track_people(persons, scores, _persons, _scores) 
    for id, ((y0, x0, y1, x1), s) in enumerate(zip(persons.astype(np.int), scores)):
        if s < 0.1:
            continue

        ellipse = findEllipse(y0,x0,y1,x1)
        rect = [y0, x0, y1, x1]
        aspect_ratio, angle, area = extract_features(rect, ellipse)
        cv2.ellipse(frame, ellipse[0], ellipse[1], 0, 0, 360, (0, 255, 255), 3)
        if is_falling(aspect_ratio, angle, area) == True:
            fall = 1
            continue
        else:
            fall = -1
    
    if start_label <= current_frame <= end_label:
        if fall == None and label:
            ndt = ndt + 1
        elif fall == None and (not label):
            ndf = ndf + 1
        elif label and fall == 1:
            tp = tp + 1
        elif (not label) and fall == 1:
            fp = fp + 1
        elif label and fall == -1:
            fn = fn + 1
        elif (not label) and fall == -1:
            tn = tn + 1
        else:
            raise Exception()       
    
    _persons, _scores = persons, scores 
    cv2.imshow('frame', frame)
    cv2.waitKey(1)  

print_result(tp, fp, fn, tn, ndt, ndf)
cv2.destroyAllWindows()
cap.release()