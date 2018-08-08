
# coding: utf-8

# # Classify the features with a Neural Network

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
import seaborn
from keras.utils import np_utils
import keras


# In[5]:


data = np.array(np.load('features.npy'))
labels = np.load('labels.npy')
data.shape, labels.shape


# In[7]:


import keras
from keras import Sequential, Model
from keras.layers import Dense


# In[16]:


y = np.array(np.transpose(np.transpose(labels)[1]))
from sklearn import preprocessing

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_Y)


# In[15]:


model = Sequential()
model.add(Dense(50, input_shape=(512,),activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
#model.add(keras.layers.Dropout(rate=0.5, noise_shape=None, seed=2))
model.add(Dense(7,activation="softmax"))
print('model.output_shape',model.output_shape)
print(model.summary())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train,y_train,verbose=0, epochs=10,validation_split=0.2)
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# # We add Dropout to reduce the overfitting

# In[22]:


model = Sequential()
model.add(Dense(50, input_shape=(512,),activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(keras.layers.Dropout(rate=0.3, noise_shape=None, seed=2))
model.add(Dense(50,activation='relu'))
model.add(keras.layers.Dropout(rate=0.2, noise_shape=None, seed=2))
model.add(Dense(7,activation="softmax"))
print('model.output_shape',model.output_shape)
print(model.summary())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train,y_train,verbose=0, epochs=10,validation_split=0.2)
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# # Now we test the robustness with training on condition 1,2,3,5 and test on conditon 4

# We prepare the test and train set

# In[37]:


tool = ['allen','clamp','driver','flat','pen','screw','usb']
cond=['cond1','cond2','cond3','cond4','cond5']
#print(labels[1][0])
#print(len([i for i in labels[1] if i=='cond5']))
y_test=[]
X_test=[]
y_train=[]
X_train=[]

y = np.array(np.transpose(np.transpose(labels)[1]))
conditions = np.array(np.transpose(np.transpose(labels)[0]))
from sklearn import preprocessing

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

for i in range(len(y)):
  if conditions[i]=='conditions4':
    y_test.append(dummy_y[i])
    X_test.append(data[i])
  else:
    y_train.append(dummy_y[i])
    X_train.append(data[i])
y_test = np.array(y_test)
X_test = np.array(X_test)#les valeurs de la cond4
y_train = np.array(y_train)
X_train = np.array(X_train)#les valeurs de cond1,2,3,5
print('Xtest',X_test.shape)
print('Xtrain',X_train.shape)


# We use the same network as previously, with the dropout

# In[40]:


model = Sequential()
model.add(Dense(50, input_shape=(512,),activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(keras.layers.Dropout(rate=0.3, noise_shape=None, seed=2))
model.add(Dense(50,activation='relu'))
model.add(keras.layers.Dropout(rate=0.2, noise_shape=None, seed=2))
model.add(Dense(7,activation="softmax"))
print('model.output_shape',model.output_shape)
print(model.summary())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train,y_train,verbose=0, epochs=20, validation_data=(X_test,y_test))
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

