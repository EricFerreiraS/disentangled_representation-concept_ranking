#!/usr/bin/env python
# coding: utf-8

# ## SVM execution on feature extraction from the dataset

# In[1]:


from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd
import settings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,scale
from sklearn.model_selection import GridSearchCV, cross_validate,train_test_split,KFold
import json
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix,ConfusionMatrixDisplay
import pickle
import copy
pd.set_option("display.max_columns", None)


# In[2]:


df_sum = pd.read_csv(f'result/experiment/dataset_stanford_{settings.MODEL}_{settings.DATASET}_{settings.FEATURE_NAMES}_sum.csv',index_col=0)
df_max = pd.read_csv(f'result/experiment/dataset_stanford_{settings.MODEL}_{settings.DATASET}_{settings.FEATURE_NAMES}_max.csv',index_col=0)
df_avg = pd.read_csv(f'result/experiment/dataset_stanford_{settings.MODEL}_{settings.DATASET}_{settings.FEATURE_NAMES}_avg.csv',index_col=0)


# Defining X and y

# In[8]:


X_sum = df_sum.drop(['target','name'],axis=1)
y_sum = df_sum['target']


# In[9]:


X_max = df_max.drop(['target','name'],axis=1)
y_max = df_max['target']


# In[10]:


X_avg = df_avg.drop(['target','name'],axis=1)
y_avg = df_avg['target']


# Getting train and test

# In[7]:


X_train_sum, X_test_sum, y_train_sum, y_test_sum = train_test_split(X_sum, y_sum, test_size=0.3,random_state=7) 


# In[8]:


X_train_max, X_test_max, y_train_max, y_test_max = train_test_split(X_max, y_max, test_size=0.3,random_state=7) 


# In[9]:


X_train_avg, X_test_avg, y_train_avg, y_test_avg = train_test_split(X_avg, y_avg, test_size=0.3,random_state=7) 


# Get the best hyperparameters

# In[ ]:


'''
param_grid = {
    'C': [0.001,0.01, 0.1],  
    'penalty': [1, 0.1, 0.01], 
    'loss': ['hinge', 'squared_hinge'],
    'dual': [True, False],
    'multi_class': ['crammer_singer','ovr'],
    'class_weight': ['balanced', None],
    'max_iter': [1000, 100000, 10000000]
    }

svm=LinearSVC()
svm_cv=GridSearchCV(svm,param_grid,cv=5)
svm_cv.fit(X_train_sum, y_train_sum)

print("best parameters",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)
'''


# SUM:
# best parameters {'C': 0.001, 'class_weight': 'balanced', 'dual': True, 'loss': 'hinge', 'max_iter': 1000, 'multi_class': 'crammer_singer', 'penalty': 1}
# accuracy : 0.7801271274475121

# best parameters {'C': 0.01, 'class_weight': 'balanced', 'dual': True, 'loss': 'hinge', 'max_iter': 1000, 'multi_class': 'crammer_singer', 'penalty': 1}
# accuracy : 0.5751774674460524

# Training the classifier

# In[11]:


clf_sum = LinearSVC(multi_class='crammer_singer',loss='hinge',class_weight='balanced',C=0.001,
               dual=True) 
clf_max = LinearSVC(multi_class='crammer_singer',loss='hinge',class_weight='balanced',C=0.001,
               dual=True) 
clf_avg = LinearSVC(multi_class='crammer_singer',loss='hinge',class_weight='balanced',C=0.001,
               dual=True) 


# Running a cross-validation

# In[12]:


def training_evaluation(X,y,pipe,k=5):
    acc=[]
    pre=[]
    rec=[]
    f1=[]
    
    better_model=0
    better_predi=0
    better_metric=0
    better_test=0
    qtd_class=0
    classes=[]
    
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = copy.deepcopy(pipe)
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)

        #calculating the metrics by each class
        acc.append(metrics.accuracy_score(y_test,predictions))
        pre.append(metrics.precision_score(y_test,predictions,average='macro'))
        rec.append(metrics.recall_score(y_test,predictions,average='macro'))
        f1.append(metrics.f1_score(y_test,predictions,average='macro'))
        classes = model.classes_

        if f1[-1] > better_metric:
            better_metric = f1[-1]
            better_model = copy.deepcopy(model)#model
            better_pred = model.predict(X_test)
            better_test = y_test.copy()
            better_x_test = X_test.copy()

    #plotting the metrics for the kfold
    fig, ax = plt.subplots(figsize=(10,10)) 
    width = 0.2
    r = 4
    n=np.arange(r)

    plt.bar(n+width, height=[np.mean(pre),np.mean(rec),np.mean(f1),np.mean(acc)],
           yerr=[np.std(pre),np.std(rec),np.std(f1),np.std(acc)])

    plt.title('Model Evaluation')
    plt.xticks(np.arange(r),['Precision','Recall','F1','Accuracy'],rotation=90)
    plt.legend(bbox_to_anchor=(1.20,1))
    plt.grid()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()

    return better_model, better_pred, better_test, better_x_test


# In[28]:


model_sum,pred_sum,y_test_sum,X_test_sum = training_evaluation(X_sum,y_sum,clf_sum)


# In[31]:


model_max,pred_max,y_test_max,X_test_max = training_evaluation(X_max,y_max,clf_max)


# In[13]:


model_avg,pred_avg,y_test_avg,X_test_avg = training_evaluation(X_avg,y_avg,clf_avg)


# save the model

# In[15]:


clf = copy.deepcopy(model_avg)
y_test = y_test_avg
y_pred = pred_avg
X_test = X_test_avg


# In[17]:


pickle.dump(clf, open(f'svm_{settings.MODEL}_{settings.DATASET}_{settings.FEATURE_NAMES}_avg.pkl', 'wb'))
pickle.dump(y_test, open(f'svm_y_test_{settings.MODEL}_{settings.DATASET}_{settings.FEATURE_NAMES}_avg.pkl', 'wb'))
pickle.dump(y_pred, open(f'svm_y_pred_{settings.MODEL}_{settings.DATASET}_{settings.FEATURE_NAMES}_avg.pkl', 'wb'))
pickle.dump(X_test, open(f'svm_X_test_{settings.MODEL}_{settings.DATASET}_{settings.FEATURE_NAMES}_avg.pkl', 'wb'))


# In[ ]:


print(classification_report(y_test, y_pred, target_names=clf.classes_))


# confusion matrix

# In[ ]:


fig, ax = plt.subplots(figsize=(20, 20))
plot_confusion_matrix(clf, X_test, y_test, ax=ax,normalize='true',values_format='.0%')  
plt.xticks(rotation=90,fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('True Label',fontsize=20)
plt.xlabel('Predicted Label',fontsize=20)
plt.show()  


# In[21]:


feature_importance_dict={}

for clas,coef in zip(clf.classes_,clf.coef_):
    feature_importance_dict[clas]=coef


# #### Export files with ranked features per class

# In[26]:


positive={}
negative={}
for i,j in enumerate(clf.classes_):
    coef = clf.coef_[i].ravel()
    p=[]
    for k in np.argsort(coef)[-10:]:
        p.append(k+1)
    positive[j]= p


# In[27]:


positive_df = pd.DataFrame.from_dict(positive,orient='index').rename(
    columns={0:'9',1:'8',2:'7',3:'6',4:'5',5:'4',6:'3',7:'2',8:'1',9:'0'}).stack()


# In[28]:


positive_df = pd.DataFrame(positive_df).rename(columns={0:'unit'}).reset_index().rename(
    columns={'level_0':'class','level_1':'unit_rank'})


# Join with the result of netdissection

# In[30]:


net_result = pd.read_csv(f'result/pytorch_{settings.MODEL}_{settings.DATASET}/tally.csv')


# In[32]:


positive_net=positive_df.merge(net_result,on='unit',how='inner')


# Selecting unique features

# In[37]:


positive_net.unit_rank = positive_net.unit_rank.astype(np.int16)


# In[38]:


pos_unique = positive_net.sort_values(['class','unit_rank']).drop_duplicates(['class','label']).groupby(['class']).head(10)


# In[41]:


p_unique = {}
for p in pos_unique.values:
    if p[0] in p_unique:
        if p[3] in p_unique[p[0]]:
            p_unique[p[0]][p[3]].append(p[4])
        else:
            p_unique[p[0]][p[3]]= [p[4]]
    else:
        p_unique[p[0]]= {p[3]:[p[4]]}
        
n_unique = {}
for n in neg_unique.values:
    if n[0] in n_unique:
        if n[3] in n_unique[n[0]]:
            n_unique[n[0]][n[3]].append(n[4])
        else:
            n_unique[n[0]][n[3]]= [n[4]]
    else:
        n_unique[n[0]]= {n[3]:[n[4]]}


# In[42]:


pickle.dump(p_unique,open('global_positive_features_svm.pkl','wb'))

