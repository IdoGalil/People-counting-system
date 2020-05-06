# People-counting-system
![Alt Text](https://github.com/IdoGalil/People-counting-system/blob/master/Central%20Library%20entrance.gif)
![Alt Text](https://github.com/IdoGalil/People-counting-system/blob/master/Medical%20Library%20Entrance.gif)
Short demo of the system with some of its features turned on:
https://www.youtube.com/watch?v=XJ_s2oy9_hc&t=4s

System to count the people entering and leaving an entrance, using a DNN as a detector (YOLOv3) and a tracking algorithm to count and track (CSRT). It was made by myself and Or Farfara as a final project on our B.Sc in Computer's Science at the Technion, and intended for use by the Technion's libraries (though it could be optimized for any entrance).

The system has some parameters that should be optimized to the specific entrance it's used on. You should read the user's manual attached.
For more information, read the project's report.

The code is relatively modular, in such a way it would be easy to modify the detector and the tracker components as better updated ones are made.
