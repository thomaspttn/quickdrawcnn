# ECSE 484 FINAL PROJECT - README.md
##### Thomas Patton (tjp94)

### Data 
Data was obtained from https://github.com/googlecreativelab/quickdraw-dataset as NumPy bitmaps

### ``cnn_main.py``
The primary program for this project is ``cnn_main.py``. This code does not require any arguments to run and ``python cnn_main.py`` will run the program as described in the paper. Within the program there is the option to set a random seed for reproducibility. A random seed value of 999 will give values which match that of the corresponding paper for this project.

### ``utils.py``
This file contians the function I used to generate the 28x28 images out of the NumPy bitmaps. This program will be useful if the project is to be modified to add more classes in the future. Note that the NumPy bitmaps are not included in this repository due to size limitations. 

### --
Author: Thomas Patton (<tjp94@case.edu>)
