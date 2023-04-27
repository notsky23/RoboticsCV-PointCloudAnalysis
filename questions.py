from typing import Tuple
import numpy as np

import utils
import random


def q1_a(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit a least squares plane by taking the Eigen values and vectors
    of the sample covariance matrix

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space

    Returns
    -------
    normal : np.ndarray
        array of shape (3,) denoting surface normal of the fitting plane
    center : np.ndarray
        array of shape (3,) denoting center of the points
    '''
    # compute the center of the points
    center = P.mean(axis=0)

    # subtract center to get the points centered at origin
    P_centered = P - center

    # compute the sample covariance matrix
    cov = np.cov(P_centered, rowvar=False)

    # compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov)

    # select the eigenvector corresponding to the smallest eigenvalue as the normal vector
    normal = eigenvecs[:, 0]

    return normal, center


def q1_c(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit a plane using RANSAC

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space

    Returns
    -------
    normal : np.ndarray
        array of shape (3,) denoting surface normal of the fitting plane
    center : np.ndarray
        array of shape (3,) denoting center of the points
    '''
    num_iterations = 1000
    num_points_to_sample = 3
    inlier_threshold = 0.01
    best_score = 0

    for i in range(num_iterations):
        # Randomly select 3 points
        samples = P[np.random.choice(P.shape[0], num_points_to_sample, replace=False)]

        # Calculate the plane from the samples
        center = samples.mean(axis=0)
        P_centered = samples - center
        cov = np.cov(P_centered, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        normal = eigenvecs[:, 0]

        # Score the plane based on the number of inliers within the threshold
        distances = np.abs((P - center).dot(normal))
        inliers = distances < inlier_threshold
        score = np.sum(inliers)

        # Update the best model if the current score is better
        if score > best_score:
            best_score = score
            best_normal = normal
            best_center = center

    return best_normal, best_center


def q2(P: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    Localize a sphere in the point cloud. Given a point cloud as
    input, this function should locate the position and radius
    of a sphere

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space
    N : np.ndarray
        Nx3 matrix denoting normals of pointcloud

    Returns
    -------
    center : np.ndarray
        array of shape (3,) denoting sphere center
    radius : float
        scalar radius of sphere
    '''
    # Set up RANSAC parameters
    num_iterations = 2000
    num_points_to_sample = 1
    inlier_threshold = 0.1  #epsilon
    best_score = 0

    # Find sphere by RANSAC
    for i in range(num_iterations):
        # Randomly sample a point
        ind = np.random.randint(P.shape[0])
        point = P[ind]
        # Surface normals of point cloud
        normal = N[ind]

        # Sample a radius
        r = np.random.uniform(0.05, 0.11)

        # Compute the center of the sphere
        center = point + r * normal

        # Compute the distances of all points to the center
        distances = np.linalg.norm(P - center, axis=1)

        # Compute the inliers as the points that lie within the sphere
        inliers = distances <= r

        # Score the sphere based on the number of points within epsilon of the candidate sphere surface
        score = np.sum(distances[inliers] <= inlier_threshold)

        # Update the best model if the current score is better
        if score > best_score:
            best_score = score
            best_center = center
            best_radius = r

    return best_center, best_radius


def q3(P: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Localize a cylinder in the point cloud. Given a point cloud as
    input, this function should locate the position, orientation,
    and radius of the cylinder

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting 100 points in 3D space
    N : np.ndarray
        Nx3 matrix denoting normals of pointcloud

    Returns
    -------
    center : np.ndarray
        array of shape (3,) denoting cylinder center
    axis : np.ndarray
        array of shape (3,) pointing along cylinder axis
    radius : float
        scalar radius of cylinder
    '''
    # Set up RANSAC parameters
    num_iterations = 5000
    num_points_to_sample = 2
    inlier_threshold = 0.01  # epsilon
    best_score = 0

    # Find cylinder by RANSAC
    for i in range(num_iterations):
        # Sample a radius between 0.05 and 0.1m
        r = np.random.uniform(0.05, 0.10)

        # Randomly sample two points
        ind = np.random.choice(P.shape[0], size=num_points_to_sample, replace=False)
        pt1 = P[ind[0]]
        pt2 = P[ind[1]]
        # Surface normals of point cloud at sampled points
        normal1 = N[ind[0]]
        normal2 = N[ind[1]]

        # Set the cylinder axis direction equal to the direction of the cross product between the surface normal associated with the two sampled points.
        # Get axis which will be the base of the cylinder
        axis = np.cross(normal1, normal2)
        if np.linalg.norm(axis) < 1e-6:
            continue
        axis /= np.linalg.norm(axis)

        # Pick one of the sampled points from the cloud and use it to estimate a candidate center, just as you did in Q2
        center = pt1 + r * normal1

        # Project the points in the cloud onto the plane orthogonal to the axis you just calculated.
        floor_normal = np.array([0, 1, 0])  # assuming the floor is aligned with the XY plane
        proj = np.eye(3) - np.outer(axis, axis)
        Q = np.dot(proj, P.T)
        center_proj = np.dot(proj, (P[ind[0]] - np.dot(P[ind[0]], axis) * axis).T).T
        center_proj = np.tile(center_proj, (Q.shape[1], 1)).T

        # Evaluate number of inliers (i.e. ∥P¯ −c¯∥2 < ϵ where P¯ is the projected point cloud, ¯c is the projected candidate center, and ϵ is the noise threshold)
        distances = np.linalg.norm(Q - center_proj, axis=1)
        inliers = abs(distances - r) <= inlier_threshold

        # Check if the cylinder is roughly perpendicular to the z-axis (within 3 degrees)
        angle_to_z = np.abs(np.arccos(np.abs(np.dot(axis, np.array([0, 1, 0])))) - np.pi / 2)
        if angle_to_z > np.deg2rad(3):
            continue

        # Score the cylinder based on the number of inliers within the threshold
        score = np.sum(inliers)

        # Update the best model if the current score is better
        if score > best_score:
            best_score = score
            best_center = center - np.array([0, 0, r])
            best_axis = axis
            best_radius = r

    return best_center, best_axis, best_radius


def q4_a(M: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Find transformation T such that D = T @ M. This assumes that M and D are
    corresponding (i.e. M[i] and D[i] correspond to same point)

    Attributes
    ----------
    M : np.ndarray
        Nx3 matrix of points
    D : np.ndarray
        Nx3 matrix of points

    Returns
    -------
    T : np.ndarray
        4x4 homogenous transformation matrix

    Hint
    ----
    use `np.linalg.svd` to perform singular value decomposition
    '''
    # Compute the centroids of M and D
    cent_M = np.mean(M, axis=0)
    cent_D = np.mean(D, axis=0)

    # Center the point clouds
    M_centered = M - cent_M
    D_centered = D - cent_D

    # Compute the covariance matrix
    H = M_centered.T @ D_centered

    # Perform SVD on the covariance matrix
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Compute translation vector
    t = cent_D - R @ cent_M

    # Construct homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def q4_c(M: np.ndarray, D: np.ndarray) -> np.ndarray:
    '''
    Solves iterative closest point (ICP) to generate transformation T to best
    align the points clouds: D = T @ M

    Attributes
    ----------
    M : np.ndarray
        Nx3 matrix of points
    D : np.ndarray
        Nx3 matrix of points

    Returns
    -------
    T : np.ndarray
        4x4 homogenous transformation matrix

    Hint
    ----
    you should make use of the function `q4_a`
    '''

    # Initialize T as an identity matrix
    T = np.eye(4)

    # Set a threshold for the error between M and D
    threshold = 1e-6

    # Set a maximum number of iterations
    max_iter = 100

    # Loop until the error is below the threshold or the iteration limit is reached
    for i in range(max_iter):
        # Find the closest points between M and D using a distance metric (e.g. Euclidean)
        dists = np.sqrt(np.sum((M[:, None, :] - D[None, :, :])**2, axis=-1))
        indices = np.argmin(dists, axis=1)
        closest_points = D[indices]

        # Call q4_a with M and closest_points as inputs to get T
        T_new = q4_a(M, closest_points)

        # Get R and t from T_new
        R = T_new[:3, :3]
        t = T_new[:3, 3]

        # Update T by multiplying it with T_new
        T = T_new @ T

        # Apply t and R to M to align it with D
        M = (R @ M.T + t[:, None]).T

        # Compute the error between M and closest_points (e.g. mean squared error)
        error = np.mean(np.sum((M - closest_points) ** 2, axis=1))

        # Break if error is below threshold
        if error < threshold:
            break

    return T