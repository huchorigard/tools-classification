# Python script for image classification

Note: To use the scripts, you must add the tool dataset in th same folder. You can find it at : tools.tar.gz .It contains everything related to this dataset.

### 1. TITLE: 

      Image classificaion algorithm for a tool dataset

### 2. CONTACT: 

      Hugo RICHARD
      Ecole Nationale supérieure des Arts et Métiers
      8 Boulevard Louis XIV
      59800 LILLE
      FRANCE
      Tel. (+33) 760964267
      email: hugo.richard@ensam.eu

### 3. RELEVANT INFORMATION:
      
      This repository contains :
            + Jupyter Notebooks, that are useful to first understand the code : 
      - Extract features wit VGG16.ipynb
      - Feature Visualization (PCA and t-SNE).ipynb
      - NN Classification.ipynb
      - Stock images into numpy array.ipynb
      - SVM Classification.ipynb

            + saved numpy arrays (only useful inside th code) :
      - features.npy
      - other numpy arrays in the features_fc1 and labels folder

            + Python scripts : 
      - Extract features wit VGG16.py
      - Feature Visualization (PCA and t-SNE).py
      - NN Classification.py
      - Stock images into numpy array.py
      - SVM Classification.py



### 4. HOW TO USE THE SCRIPTS

      First, with the 'Stock images into numpy array.ipynb' we transform the database from images in subfolders into a usefull numpy array.
      Then with the 'Extract features wit VGG16.ipynb', we use a pretrained CNN to extract features from the images.
      Now w want to visualize the features with PCA and t-SNE in 'Feature Visualization (PCA and t-SNE).ipynb'
      The next step is to manage to classify these features. First we try with a SVM in 'SVM Classification.ipynb', then with a Neural Network in 'NN Classification.ipynb'.

### 4. UPCOMING WORK
      
      I am currently working on classifying the features with siamese networks, but before posting anything I still have to work on it.    
