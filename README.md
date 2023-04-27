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

a. Implement the function q1 a in questions.py: fit a plane by calculating the sample mean and covariance matrix of the points. You will need to obtain the Eigen values and vectors of the covariance matrix in order to complete this question. You can test your implementation by running the command $python q1 a in your terminal.<br><br>

![image](https://user-images.githubusercontent.com/98131995/234903586-5c188424-4ae5-4bc7-a8a6-e6485ce84c8e.png)<br><br>

b. Test your plane fitting on an example with outliers by running the command $python q1 b. How is this different from the result in part (a) and why?<br>

-	The result of the plane fitting in part (b) with outliers is likely to be different from the result in part (a) without outliers. This is because in part (b), the input point cloud contains outliers that deviate significantly from the underlying plane. As a result, the sample covariance matrix computed in part (b) will be biased by the presence of outliers, and the eigenvectors and eigenvalues obtained from this matrix may not accurately represent the underlying plane.<br>
-	In contrast, in part (a), the input point cloud is assumed to be noise-free, and the sample covariance matrix computed from this cloud accurately represents the underlying plane. The eigenvectors and eigenvalues obtained from this matrix accurately represent the orientation and scale of the plane, respectively.<br>
-	Therefore, in the presence of outliers, the result of plane fitting obtained from the sample covariance matrix may not accurately represent the underlying plane, and alternative methods such as RANSAC may be more appropriate to robustly estimate the plane parameters.<br><br>

![image](https://user-images.githubusercontent.com/98131995/234905039-d4958b2d-b470-4779-b91f-6c19f085ba3c.png)<br><br>

c.	Implement the function q1  c in questions.py: fit a plane using a ransac based method. You can test your implementation by running $python hw4.py q1 c in your terminal. What are the strengths and weaknesses of each approach?<br>

-	Strengths:<br>
  o	RANSAC is more robust to outliers compared to the other methods because it only considers a subset of points to fit the model and ignores the rest.<br>
  o	RANSAC can handle non-linear models and does not require the assumption of linearity.<br>
-	Weaknesses:<br>
  o	RANSAC requires more computation time because it samples points and fits the model multiple times to obtain the best result.<br>
  o	The parameters used in RANSAC, such as the number of iterations and the inlier threshold, need to be carefully tuned to obtain a good result.<br>

![image](https://user-images.githubusercontent.com/98131995/234906032-b4deda77-ea10-4117-b1e9-2ef3b7155b7a.png)<br><br>


<img src="https://user-images.githubusercontent.com/98131995/234774183-aa43c871-c027-4e08-88fc-be1bba319672.png" width=50% height=50%><br><br>
![image](https://user-images.githubusercontent.com/98131995/234774248-c49252f4-8ae4-4d06-8d06-47ec421bfb46.png)<br><br>
