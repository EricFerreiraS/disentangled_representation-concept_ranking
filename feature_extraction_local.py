#!/usr/bin/env python
# coding: utf-8

# ## Experiment to extract segments from images using NetDissection

# In[3]:


import torch
from torchsummary import summary
from loader.model_loader import loadmodel
from PIL import Image
from torchvision import transforms as T
import os
import csv
import pandas as pd
import settings
from torch.utils.data import DataLoader
import torchvision
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import cv2 as cv
import pickle
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import operator
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from collections import Counter


# ### Load model defined in the settings.py

# In[2]:


features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


# In[3]:


model = loadmodel(hook_feature)


# In[ ]:


model.eval()


# ### Function to extract values from the model for each image

# In[5]:


activations_avg=[]
def hook_fn(m, i, o):
    unit_sum={}
    unit_max={}
    unit_avg={}
    for grad in o:
        try:
            for i,j in enumerate(grad):
                ##avg
                unit_avg[i+1]=j.mean().item()
            activations_avg.append(unit_avg)
        except AttributeError: 
            print ("None found for Gradient")


# Layer that I'm going to use. After all, try to get all layers

# In[6]:


layer = model.layer4


# In[7]:


layer.register_forward_hook(hook_fn)


# Extracting unique features for each image

# In[17]:


def extract_features_folder(path):

    transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.4715, 0.4413, 0.4020],
                                     std=[0.2732, 0.2650, 0.2742])])
    imgs = os.listdir(path)
    not_used=[]
    data_class=[]
    image_name=[]
    for i,img in enumerate(imgs):
        try:
            image = Image.open(path+img)#.convert("RGB")
            x = transform(image).unsqueeze(dim=0).to(device)
        except:
            image = Image.open(path+img).convert("RGB")
            x = transform(image).unsqueeze(dim=0).to(device)
        try:
            model(x)
            name = str(img.split('.')[0])[:-4]
            data_class.append(name)
            image_name.append(img)
        except:
            not_used.append(i+1)
    return data_class, image_name


# In[ ]:


data_class, image_name = extract_features_folder('dataset/Stanford40/JPEGImages/')


# In[ ]:


d_avg = pd.DataFrame(activations_avg)


# In[10]:


n = pd.DataFrame(image_name, columns={'name'})
dt_avg = d_avg.merge(n,how='inner',left_index=True, right_index=True)


# Ranking the positive features by highest activation map mean

# In[12]:


pos_df=[]
for i in dt_avg.values:
    cl = i[-1:]
    rank_p = (i[:-1].ravel().argsort()[-20:]+1)
    pos_df.append(list(np.append(rank_p,cl)))


# In[14]:


positive_df = pd.DataFrame(pos_df).rename(
    columns={0:'19',1:'18',2:'17',3:'16',4:'15',5:'14',6:'13',7:'12',8:'11',9:'10',
             10:'9',11:'8',12:'7',13:'6',14:'5',15:'4',16:'3',17:'2',18:'1',19:'0',20:'name'}).set_index(['name']).stack()
positive_df = pd.DataFrame(positive_df).rename(columns={0:'unit'}).reset_index().rename(
    columns={'level_0':'class','level_1':'unit_rank'})


# Join with the result of netdissection

# In[15]:


net_result = pd.read_csv(f'result/pytorch_{settings.MODEL}_{settings.DATASET}/tally.csv')


# In[16]:


positive_net=positive_df.merge(net_result,on='unit',how='inner')


# Selecting unique features

# In[31]:


positive_net.unit_rank = positive_net.unit_rank.astype(np.int16)


# In[32]:


pos_unique = positive_net.sort_values(['name','unit_rank']).drop_duplicates(['name','label']).groupby(['name']).head(10)


# In[34]:


pos_unique['class']= pos_unique['name'].str[:-8]


# In[36]:


pos_unique.to_csv('positive_unique_feature.csv',sep=',',encoding='utf-8', index=False)

