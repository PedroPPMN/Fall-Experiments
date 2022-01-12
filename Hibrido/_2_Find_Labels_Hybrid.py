

from joblib import Parallel, delayed
from glob import glob
import numpy as np
import pickle
import cv2


#dataset_root = 'D:/Datasets/Multiple Cameras Fall Dataset/'
dataset_root = './Datasets/MultiCamera/'
#keypoints_roots = list(glob(dataset_root + 'keypoints_processed_*/'))
#print(keypoints_roots)
names = [
    'Walking, standing up',
    'Falling',
    'Lying on the ground',
    'Crounching',
    'Moving down',
    'Moving up',
    'Sitting',
    'Lying on a sofa',
    'Moving horizontal',
]

# Pegando o delay de cada câmera para compensação
#with open('./Datasets/MultiCamera/Data/MultiCameraDataset_Delays.csv', mode='r') as f:
    #lines = f.readlines()

#content = [x.strip().split(' ') for x in lines]
#content = [[int(delay) for delay in line] for line in content]
#delays = content

# Pegando os frames iniciais e finais de cada evento de cada vídeo
with open('./Datasets/MultiCamera/Data/MultiCameraDataset_Events.csv', mode='r') as f:
    lines = f.readlines()

content = [x.strip().split(',') for x in lines] 
content = [l for l in content if len(l[0]) == 0 or l[0][0] != 's'] 

references = []
for i, row in enumerate(content):
    if row[0] == '':
        row[0] = content[i - 1][0]

    action_id = int(row[4]) - 1
    references.append((int(row[0]) - 1, int(row[2]), int(row[3]), action_id, names[action_id]))

def get_video_name(i, j):
    return dataset_root + 'chute' + str(i + 1).zfill(2) + '/cam' + str(j + 1) + '.avi'

moments = []
for ref in references:
    i = ref[0]
    for j in range(0, 8):
        path = get_video_name(i, j)
        #delay = delays[i][j]
        #moments.append((i, j, ref[1] + delay, ref[2] + delay, ref[3], ref[4], path))
        moments.append((i, j, ref[1], ref[2], ref[3], ref[4], path))

print(moments)
with open('labels.pickle', 'wb') as handle:
    pickle.dump(moments, handle, protocol=pickle.HIGHEST_PROTOCOL)

#for keypoints_root in keypoints_roots:
    #print('Saving labels to ' + keypoints_root)
    #with open(keypoints_root + 'labels.pickle', 'wb') as file:
        #pickle.dump(moments, file)
        #pickle.dump(delays, file)

