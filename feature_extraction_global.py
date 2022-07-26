#!/usr/bin/env python
# coding: utf-8

# ## Experiment to extract segments from images using NetDissection

# In[8]:


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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ### Load model defined in the settings.py

# In[9]:


features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


# In[10]:


model = loadmodel(hook_feature)


# In[ ]:


model.eval()


# ### Function to extract values from the model for each image

# In[12]:


activations_sum=[]
activations_max=[]
activations_avg=[]
def hook_fn(m, i, o):
    unit_sum={}
    unit_max={}
    unit_avg={}
    for grad in o:  
        try:
            for i,j in enumerate(grad):
                ##sum of all values in the activation map
                unit_sum[i+1]=j.sum().item()
                ##max
                unit_max[i+1]=j.max().item()
                ##avg
                unit_avg[i+1]=j.mean().item()
            activations_sum.append(unit_sum)
            activations_max.append(unit_max)
            activations_avg.append(unit_avg)
        except AttributeError: 
            print ("None found for Gradient")


# Layer that I'm going to use. After all, try to get all layers

# In[16]:


layer = model._modules['layer4']


# In[17]:


layer.register_forward_hook(hook_fn)


# Generate activation map for each image and write in a csv file

# In[7]:


#Calculate mean and std tensor from the data
path='dataset/action40/train'#path to the dataset

transform_img = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
])

image_data = torchvision.datasets.ImageFolder(
  root=path, transform=transform_img
)

image_data_loader = DataLoader(
  image_data, 
  # batch size is whole datset
  batch_size=len(image_data), 
  shuffle=False, 
  num_workers=0)

def mean_std(loader):
    images, lebels = next(iter(loader))
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0,2,3]), images.std([0,2,3])
    return mean, std

mean, std = mean_std(image_data_loader)
print("mean and std: \n", mean, std)


# In[18]:


#Feature Extraction process
path='dataset/Stanford40/JPEGImages/' #path to the images
transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

imgs = os.listdir(path)
not_used=[]
data_class=[]
image_name=[]
for i,img in enumerate(imgs):
    try:
        image = Image.open(path+img)
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


# In[19]:


d_sum = pd.DataFrame(activations_sum)
d_max = pd.DataFrame(activations_max)
d_avg = pd.DataFrame(activations_avg)


# In[24]:


t = pd.DataFrame(data_class, columns={'target'})


# In[27]:


#Merging dataframe with the target
dt_sum = d_sum.merge(t,how='inner',left_index=True, right_index=True)
dt_max = d_max.merge(t,how='inner',left_index=True, right_index=True)
dt_avg = d_avg.merge(t,how='inner',left_index=True, right_index=True)


# In[29]:


#adding the image name
n = pd.DataFrame(image_name, columns={'name'})
dt_sum = dt_sum.merge(n,how='inner',left_index=True, right_index=True)
dt_max = dt_max.merge(n,how='inner',left_index=True, right_index=True)
dt_avg = dt_avg.merge(n,how='inner',left_index=True, right_index=True)


# In[30]:


#saving into csv files
dt_sum.to_csv(f'result/experiment/dataset_stanford_{settings.MODEL}_{settings.DATASET}_{settings.FEATURE_NAMES}_sum.csv')
dt_max.to_csv(f'result/experiment/dataset_stanford_{settings.MODEL}_{settings.DATASET}_{settings.FEATURE_NAMES}_max.csv')
dt_avg.to_csv(f'result/experiment/dataset_stanford_{settings.MODEL}_{settings.DATASET}_{settings.FEATURE_NAMES}_avg.csv')

