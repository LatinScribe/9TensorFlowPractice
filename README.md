# TensorFlowPractice
### About

This project is for trying out TensorFlow

Practicing tensor flow in preparation to develop my own neural network to score Go games using image recognition

Functions include:

1) Basic image classification neural network capable of identifying 
clothing from Fashion MNIST with ~88% accuracy
2) Basic text classification neural network capable of analyzing movie reviews to identify
binary sentiment of positive versus negative with ~ 87% accuracy on validation set
3) StackOverflow question classifier. Capable of categorising questions based on their raw text
as either Python, Java, CSharp, or JavaScript. Acheives ~ 76% accuracy on validation set

Practice problems were from tensorflow.org

---

### How to try the code for yourself
To set up your own tensorflow environment, see https://www.tensorflow.org/install/pip

Note: Setting up GPU support with tensorflow can be finnicky espcially on Windows

Personally, I am using WSL2 to run a Linux environment, in which Anaconda is used to run Tensorflow and relevant files
with GPU support.

#### Instructions for Windows WSL2

Assuming you followed the instructions from the guide and have Tensorflow, Anaconda, and the nesscary Nvidia toolkits installed.
This also applies only to using Windows WSL2. 

** This also assumes you have created a conda enviroment named "tf"

** Double check that all things are properly configured with the test specified in the installtion guide

Getting started
- Run Windows PowerShell (perferably in administrator privledges)
- Type "wsl" to start WSL2 virtualisation of Linus (default distribution is Ubuntu)
- Then use "conda activate tf" to initialise the conda environment

Basics of navigating file system in Linux (skip this if you already know how)
- Next, use "cd" to move to the home directory
- Then, use "cd some_directory_name" to navigate to a lower directory (i.e 'cd /mnt/d')
- You can also use "ls" to show the contents of the current directory
- For complete commands, refer to https://www.linode.com/docs/guides/linux-navigation-commands/ 

Running a python program/file
- Navigate to the directory containing the .py file you want to run
- Type "python filename.py" - it should just run!
- Note, if you receive the error regarding libdevice not found (I certainly did), see this thread for a solution
(scroll towards the bottom): https://github.com/tensorflow/tensorflow/issues/58681


To edit Python files yourself, I recommend using your own IDE of your choice (I'm using Pycharm)

---
