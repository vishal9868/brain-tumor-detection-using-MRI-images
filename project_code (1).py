#!/usr/bin/env python
# coding: utf-8

# In[2]:


from PIL import Image as im
import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # image preprocessing

# # Topics:
# 
# 1) Reading an image file and converting it to a numpy array
# 
# 2) Resizing an image
# 
# 2) RGB to Grayscale conversion
# 

# In[4]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


image = plt.imshow(Te-no_0039.jpg)
image.show()


# In[ ]:





# In[ ]:





# In[ ]:


import numpy as np 
from tqdm import tqdm
import cv2
import os
import imutils



def crop_img(img):
	"""
	Finds the extreme points on the image and crops the rectangular out of them
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# find contours in thresholded image, then grab the largest one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# find the extreme points
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
	ADD_PIXELS = 0
	new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
	
	return new_img
	
if __name__ == "__main__":
	train = "/content/drive/MyDrive/proect/Train"
	train_dir = os.listdir(train)
	IMG_SIZE = 260

	for dir in train_dir:
		save_path = '/content/drive/MyDrive/proect/Train'+ dir
		path = os.path.join(train,dir)
		image_dir = os.listdir(path)
		for img in image_dir:
			image = cv2.imread(os.path.join(path,img))
			new_img = crop_img(image)
			new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			cv2.imwrite(save_path+'/'+img, new_img)
	


# ##conferming the image size 

# In[2]:


im_size=cv2.imread('/content/drive/MyDrive/proect/Trainglioma/Te-glTr_0000.jpg',0)
im_size.shape


# In[5]:


class image_To_Matrix_Class:
  def __init__(self , image_path , image_width , image_height):
    self.image_path = image_path
    self.image_width = image_width
    self.image_height = image_height
    self.image_size = image_width * image_height

  def get_matrix(self):
    col = len(os.listdir(self.image_path))
    img_mat = np.zeros((col , self.image_size ))
    dir = self.image_path
    i = 0
    for file_name in os.listdir(dir):
      image = os.path.join(dir , file_name)
      gray = cv2.imread(image,0)
      mat_gray = np.asmatrix(gray) 
      vec = mat_gray.ravel()

      img_mat[i , :] = vec
      i += 1 
    return img_mat


# # converting second data set into matrix

# ## first two file glioma & meningioma ( as mat1 ) 

# In[6]:


my_mat = image_To_Matrix_Class('/content/drive/MyDrive/proect/Trainglioma',260,260)


# In[7]:


mat_glioma = my_mat.get_matrix()
mat_glioma.shape


# In[8]:


my_mat = image_To_Matrix_Class('/content/drive/MyDrive/proect/Trainmeningioma',260,260)


# In[9]:


mat_mening = my_mat.get_matrix()
mat_mening.shape


# In[10]:


mat_final = np.concatenate((mat_glioma , mat_mening))
mat_final.shape


# In[13]:


mat_final = pd.DataFrame(mat_final)


#  ## first two file notumor & pituitary ( as mat2 ) 

# In[14]:


mat_final.to_csv('/content/drive/MyDrive/proect/Csvfile/mat1')


# In[15]:


my_mat = image_To_Matrix_Class('/content/drive/MyDrive/proect/Trainnotumor',260,260)


# In[16]:


mat_notu = my_mat.get_matrix()
mat_notu.shape


# In[17]:


my_mat = image_To_Matrix_Class('/content/drive/MyDrive/proect/Trainpituitary',260,260)


# In[18]:


mat_pit = my_mat.get_matrix()
mat_pit.shape


# In[19]:


mat_final2 = np.concatenate((mat_glioma , mat_mening))
mat_final2.shape


# In[20]:


mat_final2 = pd.DataFrame(mat_final2)


# In[21]:


mat_final2.to_csv('/content/drive/MyDrive/proect/Csvfile/mat2')


# In[ ]:





# # concatenate two csv files (mat1 & mat2 )

# In[26]:


mat1=pd.read_csv('mat1')
mat2=pd.read_csv('mat2')


# In[27]:


mat1.shape


# In[30]:


mat1 = mat1.iloc[:,1:]
mat1


# In[31]:


mat1.shape


# In[28]:


mat2.shape


# In[32]:


mat2 = mat2.iloc[:,1:]
mat2


# In[33]:


mat2.shape


# In[35]:


import pandas as pd 
#merging two csv file
finalcsv=pd.concat(map(pd.read_csv,['mat1','mat2']),ignore_index=True)
print(finalcsv)


# In[36]:


finalcsv.shape


# In[37]:


finalcsv = finalcsv.iloc[:,1:]
finalcsv


# In[39]:


finalcsv.shape


# In[8]:


finalcsv.to_csv('finaldata_csv')


# ## Adding target column
# 

# In[ ]:


mridata.info()


# In[ ]:


mridata.describe()


# In[13]:


mridata=pd.read_csv('finaldata_csv')


# In[18]:


tumor=pd.read_csv('tumor.csv')


# In[19]:


mridata['target'] = tumor['target']
mridata


# In[20]:


mridata.to_csv('mridata_csv')


# # Data read for testing

# In[3]:


finalcsv=pd.read_csv('mridata_csv')


# In[4]:


finalcsv


# In[5]:


finalcsv = finalcsv.iloc[:,2:]
finalcsv


# In[6]:


X = finalcsv.iloc[ : , :-1]


# In[7]:


X


# In[8]:


Y = finalcsv.target
Y


# # SVD Singular value decomposition

# In[9]:


u, s, v = np.linalg.svd(X, full_matrices=False)


# In[10]:


u.shape


# In[11]:


s.shape


# In[12]:


v.shape


# In[13]:


# Variance explaind by 200 


# In[14]:


var_explained = np.round(s**2/np.sum(s**2), decimals=6)
var_explained[:200].sum()


# test, train split

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report


# In[16]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[17]:


X_train, X_test,y_train, y_test = train_test_split(X,Y ,
                                   random_state=42, 
                                   test_size=0.20)


# In[18]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[19]:


X= v[: , :200]


# In[20]:


X.shape


# In[21]:


## Randomforest


# In[22]:


clf = RandomForestClassifier(n_estimators = 200)


# In[23]:


clf.fit(X_train , y_train)


# In[24]:


y_pred = clf.predict(X_test)


# In[25]:


accuracy_score(y_pred ,y_test)


# In[40]:


import seaborn as sns
cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm,xticklabels=['glioma','meningioma','notumor','pituitary'],
            yticklabels=['glioma','meningioma','notumor','pituitary'],annot=True)
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()


# In[35]:


# AdaBoost model


# In[36]:


adb = AdaBoostClassifier(clf , n_estimators = 10 , learning_rate = 0.05)


# In[37]:


adb.fit(X_train , y_train)


# In[38]:


adb_pred = adb.predict(X_test)


# In[39]:


accuracy_score(adb_pred , y_test)


# In[41]:


learn_rate=np.arange(0.05, 1.1, 0.01) 
test_accuracy = np.empty (len(learn_rate))

for i, k in enumerate (learn_rate):
    adab = AdaBoostClassifier(clf ,n_estimators=10, learning_rate=k)
    adab.fit(X_train, y_train)
    
    test_accuracy[i] = adab.score (X_test, y_test)

plt.plot(learn_rate, test_accuracy, label = 'Testing dataset Accuracy')

plt.legend()
plt.xlabel('learn_rate')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


learn_rate=np.arange(0.05, 0.1, 0.01) 
test_accuracy = np.empty (len(learn_rate))

for i, k in enumerate (learn_rate):
    adab = AdaBoostClassifier(clf ,n_estimators=20, learning_rate=k)
    adab.fit(X_train, y_train)
    
    test_accuracy[i] = adab.score (X_test, y_test)

plt.plot(learn_rate, test_accuracy, label = 'Testing dataset Accuracy')

plt.legend()
plt.xlabel('learn_rate')
plt.ylabel('Accuracy')
plt.show()


# In[42]:


learn_rate=np.arange(0.05, 0.1, 0.01) 
test_accuracy = np.empty (len(learn_rate))

for i, k in enumerate (learn_rate):
    adab = AdaBoostClassifier(clf ,n_estimators=10, learning_rate=k)
    adab.fit(X_train, y_train)
    
    test_accuracy[i] = adab.score (X_test, y_test)

plt.plot(learn_rate, test_accuracy, label = 'Testing dataset Accuracy')

plt.legend()
plt.xlabel('learn_rate')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


##xgboost


# 
# 

# In[43]:


get_ipython().system('pip3 install xgboost')


# In[96]:


get_ipython().system('pip install graphviz')


# In[44]:


from xgboost import XGBClassifier
from xgboost import plot_tree
from graphviz import Source
from sklearn.model_selection import GridSearchCV


# In[45]:


xg_clf = XGBClassifier(learning_rate= 0.05, max_depth = 8, n_estimators = 500, subsample=0.5, eval_metric = 'auc',
                       objective = 'multi:softprob', verbosity = 1)


# In[ ]:


xg_clf.fit(X_train, y_train)


# In[105]:


xg_pred = xg_clf.predict(X_test)
accuracy_score (y_test, xg_pred)


# In[ ]:





# In[144]:


learning_rate_list = [0.02, 0.05, 0.1]
max_depth_list = [8, 10]
n_estimator_list = [100, 200, 500]
params_dict = { 'learning_rate': learning_rate_list, 'max_depth': max_depth_list, 'n_estimators': n_estimator_list}


# In[ ]:





# In[145]:


xg_hp = GridSearchCV(estimator = xg_clf, param_grid = params_dict, cv = 2, verbose = 4)


# In[146]:


xg_hp.fit(X_train, y_train)


# In[147]:


hp_pred = xg_hp.predict(X_test)
accuracy_score(y_test, hp_pred)


# #####SVM

# In[164]:


from sklearn.svm import SVC
svc_clf = SVC()


# In[165]:


svc_clf.fit(X_train, y_train)


# In[166]:


svm_pred = svc_clf.predict(X_test)


# In[167]:


print(accuracy_score(svm_pred , y_test))


# In[ ]:




