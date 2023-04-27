# RoboticsCV-PointCloudAnalysis

HW Guide: https://github.com/notsky23/RoboticsCV-PointCloudAnalysis/blob/master/hw4-v2.pdf.<br><br>

## What is this practice about?<br>

This module is a cross between computer vision and robotics. We will be using point clouds to help a robot analyze and make sense of it's surroundings.<br><br>

In this module, we will be applying 2 algorithms to point clouds:<br>
1. Random sample consensus (RANSAC)<br>
2. Iterative closest point (ICP)<br><br>

## Installation Instructions
Make sure you are using Python>=3.6.  We provide instructions that use conda as
a virtual environment.  You are free to use another virtual environment (such as
`virtualenv`) if you wish.

0. Install conda.  Follow this [guide](https://docs.anaconda.com/anaconda/install/).
1. Create conda environment using your terminal
```shell
conda create -n rss23 python=3.8 numpy scipy matplotlib
```
2. Activate enviroment (you will have to do this every time you open terminal)
```shell
conda activate rss23
```

## Results:<br>

Here are the results I got.<br>

The code is included in this repo.<br><br>

### Q1 - Plane fitting:<br>

a. Implement the function q1 a in questions.py: fit a plane by calculating the sample mean and covariance matrix of the points. You will need to obtain the Eigen values and vectors of the covariance matrix in order to complete this question. You can test your implementation by running the command $python q1 a in your terminal.

![image](https://user-images.githubusercontent.com/98131995/234903586-5c188424-4ae5-4bc7-a8a6-e6485ce84c8e.png)


<img src="https://user-images.githubusercontent.com/98131995/234774183-aa43c871-c027-4e08-88fc-be1bba319672.png" width=50% height=50%><br><br>
![image](https://user-images.githubusercontent.com/98131995/234774248-c49252f4-8ae4-4d06-8d06-47ec421bfb46.png)<br><br>
