# People counting system by Ido Galil & Or Farfara
![](https://github.com/IdoGalil/People-counting-system/blob/master/Central%20Library%20entrance.gif)
![](https://github.com/IdoGalil/People-counting-system/blob/master/Medical%20Library%20Entrance.gif)

## Short Overview
The system counts the people entering and leaving an entrance, using a Deep Neural Network as a detector (YOLOv3) and a tracking algorithm to track and count (DCF-CSR \ CSRT). It was developed by myself and Or Farfara as a project in Machine Learning & Computer Vision at the Technion, and intended for use by the Technion's libraries (though it could be optimized for any entrance).

## Demo
Short demo of the system with some of its features turned on:
https://www.youtube.com/watch?v=XJ_s2oy9_hc&t=4s

## Requirements
- Python 3.6.10
- GPU and CUDA 9.0 installed

## Setup
- Clone repo
- Install dependencies in requirements.txt file (pip install requirements.txt)
- Download the detector YOLOv3-416's h5 file from here: https://pjreddie.com/darknet/yolo/
And insert it into the model_data folder.

## Important notes and how to use
- You *should* read the user's manual attached https://github.com/IdoGalil/People-counting-system/blob/master/Counting%20System%20User's%20manual.pdf
- The system has some parameters that should be optimized to the specific entrance it's used on. Most importantly, DI and MCDF.
- For more information about the components and ideas of the system and how they were developed, read the project's report.
- The code is relatively modular, in such a way it would be easy to modify the detector and the tracker components as better updated ones are made.
- The system is intended for real-time performance (more info in the report), and as such requires GPU. note that it could recieve its input from an IP camera, but to work in near real-time the video must be obtained at high speeds.
