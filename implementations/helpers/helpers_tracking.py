import numpy as np
import cv2

"""Decompose camera projection matrix and return decomposition as vector.

Parameters
----------
proj_cam : camera projection matrix
dists : distortion coefficients of camera

Returns
-------
decomposed : decomposed proj_cam concatenated with distortion coeffs
"""
def projection_decomposed_format(proj, dist):
    cam, rot, translation, x, y, z, euler = cv2.decomposeProjectionMatrix(proj)
    translation_euc = translation[:-1] / translation[-1]
    trans = -rot @ translation_euc
    p = np.zeros((3, 4))
    p[:3, :3] = rot
    p[:, 3] = np.squeeze(trans)

    rot = cv2.Rodrigues(rot)[0]

    focal_length = cam[0][0]
    cx = cam[0, 2]
    cy = cam[1, 2]

    decomposed = np.concatenate((np.squeeze(rot),
                                 np.squeeze(trans),
                                 np.asarray([focal_length, cx, cy]),
                                 dist)).reshape((14,))
    return decomposed

"""Get the 2D locations of vertitces from mesh estimated by the eos library, using the estimated affine camera matrix.
Implementation from 4DFace python demo.
"""
def get_vertices2d(vertices, modelview_matrix, projection_matrix, image_width, image_height):
    # We follow the standard OpenGL projection model to project the 3D vertices to the screen:
    # (See e.g. glm::project, https://glm.g-truc.net/0.9.9/api/a00666.html#gaf36e96033f456659e6705472a06b6e11)
    vertices2d = []
    for vertex in vertices:
        vertex_h = np.append(vertex, 1)  # homogeneous coordinates
        projected_vertex = projection_matrix @ modelview_matrix @ vertex_h
        projected_vertex = projected_vertex[0:3]
        # Viewport transformation:
        # Create an OpenGL compatible viewport vector that flips y and has the origin on the top-left, like in OpenCV:
        viewport = np.array([0, image_height, image_width, -image_height])
        projected_vertex = projected_vertex * 0.5 + 0.5
        projected_vertex[0] = projected_vertex[0] * viewport[2] + viewport[0]
        projected_vertex[1] = projected_vertex[1] * viewport[3] + viewport[1]
        vertices2d.append(np.asarray([projected_vertex[0], projected_vertex[1], 1.0]))
    return vertices2d


"""
 RALIGN - Rigid alignment of two sets of points in k-dimensional
          Euclidean space.  Given two sets of points in
          correspondence, this function computes the scaling,
          rotation, and translation that define the transform TR
          that minimizes the sum of squared errors between TR(X)
          and its corresponding points in Y.  This routine takes
          O(n k^3)-time.

 Inputs:
   X - a k x n matrix whose columns are points 
   Y - a k x n matrix whose columns are points that correspond to
       the points in X
 Outputs: 
   c, R, t - the scaling, rotation matrix, and translation vector
             defining the linear map TR as 

                       TR(x) = c * R * x + t

             such that the average norm of TR(X(:, i) - Y(:, i))
             is minimized.
"""

"""
Copyright: Carlo Nicolini, 2013
Code adapted from the Mark Paskin Matlab version
from http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m 
"""


def ralign(X, Y):
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc * Xc, 0))
    sy = np.mean(np.sum(Yc * Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U, D, V = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = V.T.copy()
    # print U,"\n\n",D,"\n\n",V
    r = Sxy.ndim
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if (np.det(Sxy) < 0):
            S[m, m] = -1;
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R, c, t

    R = np.dot(np.dot(U, S), V.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    c = 1
    t = my - c * np.dot(R, mx)

    transfromation_mat = np.zeros((3, 4))
    transfromation_mat[:, :3] = c * R
    transfromation_mat[:, 3] = t

    return transfromation_mat
