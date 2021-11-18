import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.categorical as C
import torch.distributions.bernoulli as bn
from torch.utils.tensorboard import SummaryWriter


# In[4]:


use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')


# In[5]:


mtcnn = MTCNN(image_size=200, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=False, device=device)
path = os.path.join(os.getcwd(), 'trainset/')


max_batch_size = 1

for root, subdirs, filename in os.walk(path, topdown=True):
    if len(filename):
        batch_images = []
        file_list = []
        skip = False
        
        for f in filename:
            if 'bbox' in f:
                skip = True
                break
        
        if skip:
            print('skipping..', root)
            continue
        
        for i, file in enumerate(filename):
            file_path = os.path.join(root, file)
            arr = cv2.imread(file_path)
            img = arr[:,:,::-1]
            batch_images.append(torch.FloatTensor(img.copy()))
            file_list.append(file_path)
            
            if len(batch_images) >= max_batch_size or i == len(filename) - 1:
                images = torch.stack(batch_images).to(device)
                batch_images = []
                out, prob, landmark = mtcnn.detect(images, landmarks=True)                
                
                for j in range(len(out)):
                    if out[j] is None:
                        continue
                    
                    save_map = {}
                    save_map['box'] = list(out[j][0])
                    save_map['left_eye'] = list(landmark[j][0][0])
                    save_map['right_eye'] = list(landmark[j][0][1])
                    save_map['nose'] = list(landmark[j][0][2])
                    save_map['left_lips'] = list(landmark[j][0][3])
                    save_map['right_lips'] = list(landmark[j][0][4])
                    save_map['confidence'] = prob[j][0]
                    
                    json = open(file_list[j].split('.')[0] + '_bbox' , 'wb')
                    pickle.dump(save_map, json)
                    json.close()
                
                file_list = []
        
        print('done path', root)

