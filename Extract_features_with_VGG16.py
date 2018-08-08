
# coding: utf-8

# # Extract features from images with VGG16

# In[3]:


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image


# In[6]:


images = np.load('images.npy')
labels = np.load('labels.npy')

model = VGG16(weights='imagenet', include_top=False, pooling='avg')

features = []
for img in images:
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    feature = np.squeeze(model.predict(x))
    features.append(feature)
features=np.array(features)
features.shape


# We can extract features from different layers, but here we only extract from the last pooling layer of VGG16, before the fully connected layers.

# In[7]:


np.save('features',features) #on enregistre les features extraites

