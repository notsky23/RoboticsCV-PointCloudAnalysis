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
