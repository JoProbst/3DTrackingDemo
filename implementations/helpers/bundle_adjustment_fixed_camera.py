import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2


def project_points(points, cameraArray):
    pnts2d_all = []
    f = cameraArray[:, 6]
    cx = cameraArray[:, 7]
    cy = cameraArray[:, 8]
    for idx, point in enumerate(points):
        cam = np.zeros((3, 3))
        cam[0, 0] = f[idx]
        cam[1, 1] = f[idx]
        cam[0, 2] = cx[idx]
        cam[1, 2] = cy[idx]
        pnts2d, jac = cv2.projectPoints(np.asarray(point), cameraArray[idx][:3], cameraArray[idx][3:6], cam,
                                        np.zeros(5))
        pnts2d_all.append(pnts2d[0][0])
    pnts2d_all = np.asarray(pnts2d_all)
    return pnts2d_all


def fun(params, n_points, cameraArray, camera_indices, point_indices, points_2d):
    camera_params = cameraArray
    points_3d = params.reshape((n_points, 3))
    points_proj = project_points(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


"""Optimize 3D point locations using 2D detections from 3 or more cameras

Parameters
----------
cameraArray : decomposed camera matrices of used cameras
points3D : initial estimates for 3D points
points2D : 2D locations of points in all cameras
cameraIndices : camera index for each detected point
point2DIndices : point indices for 2D point corresponding to 3D points

Returns
-------
points3D_optimized : optimized 3D points
"""
def ba_fixed_cam(cameraArray, points3D, points2D, cameraIndices, point2DIndices):
    numPoints = points3D.shape[0]
    points2D = np.asarray(points2D).reshape(len(point2DIndices), 2)
    x0 = points3D.ravel()

    m = cameraIndices.size * 2
    n = numPoints * 3

    # Construct jacobian sparsity matrix
    sparse_jac = lil_matrix((m, n), dtype=int)
    i = np.arange(cameraIndices.size)
    for s in range(3):
        sparse_jac[2 * i, point2DIndices * 3 + s] = 1
        sparse_jac[2 * i + 1, point2DIndices * 3 + s] = 1

    res = least_squares(fun, x0, jac_sparsity=sparse_jac, verbose=1, x_scale='jac', ftol=1e-6, method='trf',
                        args=(numPoints, cameraArray, cameraIndices, point2DIndices, points2D))
    points3D_optimized = res.x.reshape((numPoints, 3))
    return points3D_optimized
