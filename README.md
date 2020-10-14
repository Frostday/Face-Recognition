# Face-Recognition
Implementation of Multi Face Recognition using deep learning and OpenCV.

## Installation and Setup
* Fork the repo and clone it.
```
git clone https://github.com/Frostday/Face-Recognition.git
```
* To download the required packages run the commands below 
```
pip install -r requirements.txt
```
* Next you need to make a folder inside the images directory for every person who you want the model to recognize.
* Next you place images of each person in their respective folders. It is recommended to use atleast 5 images per person.
* To train the model and finish the setup run the following command
```
python faces_train.py
```
* Finally to see the results, run the following command
```
python faces.py
```

## Results
![](assets/video.gif)