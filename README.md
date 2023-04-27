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

a. Implement the function q1 a in questions.py: fit a plane by calculating the sample mean and covariance matrix of the points. You will need to obtain the Eigen values and vectors of the covariance matrix in order to complete this question. You can test your implementation by running the command $python q1 a in your terminal.<br>

![image](https://user-images.githubusercontent.com/98131995/234914301-e5f33005-4f74-41b6-b111-958280192180.png)<br>
<img src="https://user-images.githubusercontent.com/98131995/234914301-e5f33005-4f74-41b6-b111-958280192180.png" width=50% height=50%><br><br>

![image](https://user-images.githubusercontent.com/98131995/234903586-5c188424-4ae5-4bc7-a8a6-e6485ce84c8e.png)<br><br>

b. Test your plane fitting on an example with outliers by running the command $python q1 b. How is this different from the result in part (a) and why?<br>

-	The result of the plane fitting in part (b) with outliers is likely to be different from the result in part (a) without outliers. This is because in part (b), the input point cloud contains outliers that deviate significantly from the underlying plane. As a result, the sample covariance matrix computed in part (b) will be biased by the presence of outliers, and the eigenvectors and eigenvalues obtained from this matrix may not accurately represent the underlying plane.<br>
-	In contrast, in part (a), the input point cloud is assumed to be noise-free, and the sample covariance matrix computed from this cloud accurately represents the underlying plane. The eigenvectors and eigenvalues obtained from this matrix accurately represent the orientation and scale of the plane, respectively.<br>
-	Therefore, in the presence of outliers, the result of plane fitting obtained from the sample covariance matrix may not accurately represent the underlying plane, and alternative methods such as RANSAC may be more appropriate to robustly estimate the plane parameters.<br>

![image](https://user-images.githubusercontent.com/98131995/234905039-d4958b2d-b470-4779-b91f-6c19f085ba3c.png)<br><br>

c.	Implement the function q1  c in questions.py: fit a plane using a ransac based method. You can test your implementation by running $python hw4.py q1 c in your terminal. What are the strengths and weaknesses of each approach?<br>

<img src="https://user-images.githubusercontent.com/98131995/234912850-c1756140-af14-4998-939f-0d999e44cc95.png" width=50% height=50%><br><br>

-	Strengths:<br>
  o	RANSAC is more robust to outliers compared to the other methods because it only considers a subset of points to fit the model and ignores the rest.<br>
  o	RANSAC can handle non-linear models and does not require the assumption of linearity.<br>
-	Weaknesses:<br>
  o	RANSAC requires more computation time because it samples points and fits the model multiple times to obtain the best result.<br>
  o	The parameters used in RANSAC, such as the number of iterations and the inlier threshold, need to be carefully tuned to obtain a good result.<br>

![image](https://user-images.githubusercontent.com/98131995/234906032-b4deda77-ea10-4117-b1e9-2ef3b7155b7a.png)<br><br>

### Q2 - Sphere fitting:<br>

![image](https://user-images.githubusercontent.com/98131995/234908515-704bd325-ad01-4381-8cb5-9b2b0278b5d2.png)<br><br>

### Q3 - Cylinder fitting:<br>

![image](https://user-images.githubusercontent.com/98131995/234908962-85062eb3-6c32-4836-9a0c-a81fc237f615.png)<br><br>

### Q4 - ICP:<br>

a. Implement the function q4 a to find the transformation matrix that aligns two point clouds given full correspondences between points in the two clouds. In other words, D and M are the same point cloud but in different poses. You can test your implementation by running: $python hw4.py q4_a<br>

![image](https://user-images.githubusercontent.com/98131995/234910694-32c3ce34-8357-4fb1-9a2f-fee9ca6ee068.png)<br><br>

b. Run $python hw4.py q4  b to test your implementation from part (a) on noisy data. Explain why the algorithm still works when gaussian noise is added to one of the point clouds, but does not work when the order of the points is shuffled.<br>

![image](https://user-images.githubusercontent.com/98131995/234911071-39dcef61-e78a-4560-888e-d49382e20b46.png)<br><br>

c. Implement the function q4 c to perform iterative closest point (ICP). Your implementation should get reasonably alignment on shuffled and noisy data: run $python hw4.py q4 c to test this.<br>

![image](https://user-images.githubusercontent.com/98131995/234911356-7850710b-0b52-432c-9fa9-380bae1755cb.png)<br><br>
