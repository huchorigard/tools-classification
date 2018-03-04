# tools-classification

This repository aims to develop a robust algorithm to classifiy a dataset of tools available here : 
http://www.usine-agile.fr/datas/22-cnn-datas-1.html) 

The first solution consists in using transfer learning. We use Keras and VGG16 CNN to extract features from the ilmages, and then,
due to the little amount of images, we use a SVM to classify the images.

This first approach is a Supervised learning problem. But the final aim is to have a solution that is able to detect novelties 
and to continuously learn. In fact the algorithm is aimed to be deployed for collaborative robots (colrobots).

The insight is to give a robot the abilty to classify the tools in a workspace, in order to help the human in their every day tasks.

Example :
Human : "give me the screwdriver!"
Robot has to recognize the screwdriver and to give it to the human.
