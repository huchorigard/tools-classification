
# coding: utf-8

# # Create an Array containing every images 
# ## Create images.npy and labels.npy containing the images and their labels

# We import numpy and keras.preprocessing to do so

# In[1]:


import numpy as np
from keras.preprocessing import image


# In[2]:


get_ipython().system('ls tools')


# The folder tools is the database. It contains all the images, classified in different folders : folder for conditions and sub-folders for tools class.
# 
# It needs to be copied in the same folder as this piece of code.

# In[11]:


cond = ['conditions1','conditions2','conditions3','conditions4','conditions5']
np_images = []
labels = []
for i in cond:
    l = get_ipython().getoutput('ls tools/$i* #tools is the joris database')
    for j in l:
        ll = get_ipython().getoutput('ls tools/$i*/$j*')
        for k in ll:
            np_images.append(image.img_to_array(image.load_img('tools/'+str(i)+'/'+str(j)+'/'+str(k))))
            labels.append([str(i),str(j)])
np_images = np.array(np_images)
labels = np.array(labels)


# In[17]:


print('images shape', np_images.shape, '\nlabels shape', labels.shape)


# In[18]:


np.save('images',np.array(np_images))
np.save('labels',np.array(labels))


# Now we can work easily with clean arrays containing all our data
