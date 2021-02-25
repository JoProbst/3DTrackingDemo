import numpy as np
import eos

import multiprocessing
import open3d as o3d
import os
from scipy.sparse import lil_matrix
from scipy.linalg import lstsq

from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from scipy.optimize import least_squares
from scipy.optimize import nnls
from scipy.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import curve_fit
import transforms3d as t3d
from sklearn.neighbors import LocalOutlierFactor

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import cv2

import transforms3d as t3d
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

    # c = np.trace(np.dot(np.diag(D), S)) / sx
    c = 1
    t = my - c * np.dot(R, mx)

    transformation_mat = np.zeros((3, 4))
    transformation_mat[:, :3] = c * R
    transformation_mat[:, 3] = t

    return transformation_mat


def find_closest_vertex3D(vertex, candidate_vertices):
    dists = np.linalg.norm(candidate_vertices - vertex, axis=1)
    min_dist = np.argmin(dists)
    return candidate_vertices[min_dist]


def find_closest_vertex_index(vertex, candidate_vertices):
    dists = np.linalg.norm(candidate_vertices - vertex, axis=1)
    min_dist = np.argmin(dists)
    return min_dist


def get_contour_correspondence_3d(vertices3d,
                                  mean_mesh,
                                  shape_coeffs,
                                  blendshape_coeffs,
                                  trans,
                                  zoom,
                                  shape_model,
                                  blendshapes_model,
                                  num_shape_coeffs,
                                  used_vertices,
                                  contours_lms,
                                  contours_vertices):
    used_vertices_3 = []
    for vert in contours_vertices[0] + contours_vertices[1]:
        for i in range(3):
            used_vertices_3.append(3 * vert + i)
    blendshapes_mat = np.asarray([bs.deformation for bs in blendshapes_model]).transpose()[used_vertices_3]
    shapes_mat = shape_model[used_vertices_3, :len(shape_coeffs)]
    shape_blendshape = np.concatenate((shapes_mat, blendshapes_mat), axis=1)

    mean_mesh_contours = mean_mesh[contours_vertices[0] + contours_vertices[1]]

    T = trans[:3]
    R = t3d.quaternions.quat2mat(trans[3:7])
    Z = zoom

    total_coeffs = np.concatenate([shape_coeffs, blendshape_coeffs])
    mesh_transformed = np.expand_dims(
        mean_mesh_contours + (shape_blendshape @ total_coeffs).reshape(mean_mesh_contours.shape), axis=2)
    contours_trans = np.squeeze(np.matmul(R, mesh_transformed)) * zoom + T

    contour_verts_selected = np.zeros((2, len(contours_lms[0])), dtype=int)

    for i, side in enumerate(contours_lms):
        for j, lm in enumerate(side):
            closest = find_closest_vertex_index(vertices3d[lm], contours_trans[(i * 17):((i * 17) + 17)])
            contour_verts_selected[i, j] = closest + 50 + i * 17
    return contour_verts_selected


def ralign_model(vertices3d,
                 morphablemodel_with_expressions,
                 shape_coeffs,
                 blendshape_coeffs,
                 used_vertices,
                 contours_lms,
                 contours_vertices, has_contours):
    mesh = np.asarray(
        morphablemodel_with_expressions.draw_sample(shape_coeffs.tolist(), blendshape_coeffs.tolist(), []).vertices)[
        used_vertices]
    if has_contours:
        vertices_ordered = np.concatenate([contours_vertices[0], [0], contours_vertices[1], range(1, 50)])
    else:
        vertices_ordered = list(range(0, 50))
        vertices3d = vertices3d[[8] + list(range(17, 66))]
    vertices_selected = mesh[vertices_ordered]
    trans = ralign(vertices_selected.T, vertices3d.T)

    trans_mat = np.vstack((trans, np.array([0.0, 0.0, 0.0, 1.0])))
    T, R, Z, S = t3d.affines.decompose(trans_mat)
    R = t3d.quaternions.mat2quat(R)
    # trz = np.concatenate((T, R.flatten(), Z))
    tr = np.concatenate((T, R.flatten()))

    return tr


'''
Linear fitting of shape coefficients to multiple point clouds,
given fixed blendshape parameters and transformation matrices.
'''


def get_shape_multiple3d_svd(all_vertices_3d,
                             mean_mesh,
                             blendshape_coeffs,
                             trans_BA,
                             zoom,
                             shape,
                             model_blendshapes,
                             num_shape_coeffs,
                             used_vertices,
                             contours_lms,
                             model_contour_lists,
                             contour_correspondece,
                             has_contours,
                             lam):
    used_vertices_3 = []
    if has_contours:
        used_vertices = np.concatenate([used_vertices, model_contour_lists[0], model_contour_lists[1]])
        num_landmarks = 66
    else:
        used_vertices = used_vertices[:50]
        num_landmarks = 50
    for vert in used_vertices:
        for i in range(3):
            used_vertices_3.append(3 * vert + i)

    num_observations = blendshape_coeffs.shape[0]

    if not has_contours:
        all_vertices_inner = np.zeros((num_observations * 50, 3))
        for obs_num in range(num_observations):
            inner_inidces = [7] + list(range(16, 65))
            all_vertices_inner[obs_num * 50:(obs_num + 1) * 50] = all_vertices_3d[obs_num * 66:(obs_num + 1) * 66][
                inner_inidces]
        all_vertices_3d = all_vertices_inner

    num_coeffs_opt = num_shape_coeffs

    blendshapes_mat = np.asarray([bs.deformation for bs in model_blendshapes]).transpose()[used_vertices_3]
    shapes_mat = shape[used_vertices_3, :num_shape_coeffs]

    S = np.zeros((num_observations, num_landmarks * 3, num_coeffs_opt))
    for i in range(num_observations):
        if has_contours:
            inner_and_contours = np.concatenate(
                [contour_correspondece[i][0], [0], contour_correspondece[i][1], range(1, 50)])
            inner_and_contours_3 = []
            for vert in inner_and_contours:
                for j in range(3):
                    inner_and_contours_3.append(3 * vert + j)
            S[i] = shapes_mat[inner_and_contours_3]
        else:
            S[i] = shapes_mat

    all_rots = []
    all_trans = []

    for i, trans in enumerate(trans_BA):
        R = t3d.quaternions.quat2mat(trans[3:7])
        t = trans[:3]
        all_rots.append(R)
        all_trans.append(t)
    Li = np.zeros((num_observations, num_landmarks * 3, num_landmarks * 3))
    for i in range(num_observations):
        for j in range(num_landmarks):
            for x in range(3):
                for y in range(3):
                    Li[i][j * 3 + x, j * 3 + y] = all_rots[i][x, y]

    Qi = Li @ S * zoom
    Qt = np.zeros((num_observations, num_coeffs_opt, num_coeffs_opt))
    for i in range(num_observations):
        Qt[i] = Qi[i].T @ Qi[i]
    Q = np.sum(Qt, axis=0)

    lam_diag = np.ones(num_coeffs_opt)
    I = num_observations * np.diag(lam_diag) * lam
    l = Q + I

    m = num_observations * num_landmarks * 3
    mean_model_stacked = np.zeros((m, 1), dtype=float)
    for i in range(num_observations):
        if has_contours:
            inner_and_contours = np.concatenate(
                [contour_correspondece[i][0], [0], contour_correspondece[i][1], range(1, 50)])
            inner_and_contours_3 = []
            for vert in inner_and_contours:
                for j in range(3):
                    inner_and_contours_3.append(3 * vert + j)
            blendshapes_mat_i = blendshapes_mat[inner_and_contours_3]
            mean_mesh_i = mean_mesh[used_vertices][inner_and_contours]
        else:
            blendshapes_mat_i = blendshapes_mat
            mean_mesh_i = mean_mesh[used_vertices]

        blends = blendshapes_mat_i @ blendshape_coeffs[i]
        mean_model_stacked[i * num_landmarks * 3:(i + 1) * num_landmarks * 3] = \
            (mean_mesh_i).reshape(num_landmarks * 3, 1) + blends.reshape((num_landmarks * 3, 1))

    m = num_observations * num_landmarks * 3
    trans_stacked = np.zeros((m, 1), dtype=float)
    for i in range(num_observations):
        for j in range(num_landmarks):
            trans_stacked[i * num_landmarks * 3 + j * 3:i * num_landmarks * 3 + (j + 1) * 3] = np.expand_dims(
                all_trans[i], axis=1)

    m = num_observations * num_landmarks * 3
    n = num_observations * num_landmarks * 3
    rot_sparse = lil_matrix((m, n), dtype=float)
    for i in range(num_observations):
        for j in range(num_landmarks):
            for x in range(3):
                for y in range(3):
                    rot_sparse[i * num_landmarks * 3 + j * 3 + x, i * num_landmarks * 3 + j * 3 + y] = \
                        all_rots[i][x, y]

    model_transformed = (rot_sparse @ mean_model_stacked) * zoom + trans_stacked

    vertices_3d = all_vertices_3d.reshape((num_observations * num_landmarks * 3, 1))
    b = (vertices_3d - model_transformed)
    b_summed = b.reshape(num_observations, num_landmarks * 3, 1)
    Qty = np.zeros((num_observations, num_coeffs_opt, 1))
    for i in range(num_observations):
        Qty[i] = Qi[i].T @ b_summed[i]
    rhs = np.sum(Qty, axis=0)
    alpha = np.linalg.pinv(l) @ rhs

    return np.squeeze(alpha)


def get_blendshape_3d_linear(vertices_3d,
                             mean_mesh,
                             shape_coeffs,
                             trans_BA,
                             zoom,
                             shape,
                             model_blendshapes,
                             num_shape_coeffs,
                             used_vertices,
                             contours_lms,
                             model_contour_lists,
                             contour_correspondece,
                             has_contours
                             ):
    used_vertices_3 = []
    if has_contours:
        used_vertices = np.concatenate([used_vertices, model_contour_lists[0], model_contour_lists[1]])
        num_landmarks = 66
    else:
        used_vertices = used_vertices[:50]
        num_landmarks = 50
        vertices_3d = vertices_3d[[7] + list(range(16, 65))]

    for vert in used_vertices:
        for i in range(3):
            used_vertices_3.append(3 * vert + i)

    blendshapes_mat = np.asarray([bs.deformation for bs in model_blendshapes]).transpose()[used_vertices_3]
    shapes_mat = shape[used_vertices_3, :num_shape_coeffs]

    if has_contours:
        inner_and_contours = np.concatenate(
            [contour_correspondece[0], [0], contour_correspondece[1], range(1, 50)])
        inner_and_contours_3 = []
        for vert in inner_and_contours:
            for i in range(3):
                inner_and_contours_3.append(3 * vert + i)
        B = blendshapes_mat[inner_and_contours_3]
    else:
        B = blendshapes_mat

    R = t3d.quaternions.quat2mat(trans_BA[3:7])
    t = trans_BA[:3]
    m = num_landmarks * 3
    n = num_landmarks * 3
    rot_sparse = lil_matrix((m, n), dtype=float)
    for j in range(num_landmarks):
        for x in range(3):
            for y in range(3):
                rot_sparse[j * 3 + x, j * 3 + y] = R[x, y]

    Q = (rot_sparse @ B) * zoom

    if has_contours:
        inner_and_contours = np.concatenate(
            [contour_correspondece[0], [0], contour_correspondece[1], range(1, 50)])
        inner_and_contours_3 = []
        for vert in inner_and_contours:
            for i in range(3):
                inner_and_contours_3.append(3 * vert + i)
        shapes_mat = shapes_mat[inner_and_contours_3]
        mean_mesh = mean_mesh[used_vertices][inner_and_contours]
    else:
        mean_mesh = mean_mesh[used_vertices]

    shape_deform = shapes_mat @ shape_coeffs
    mean_model = (mean_mesh).reshape(num_landmarks * 3, 1) + shape_deform.reshape((num_landmarks * 3, 1))

    m = num_landmarks * 3
    trans_stacked = np.zeros((m, 1), dtype=float)
    for j in range(num_landmarks):
        trans_stacked[j * 3:(j + 1) * 3] = np.expand_dims(t, axis=1)

    model_transformed = (rot_sparse @ mean_model) * zoom + trans_stacked
    vertices_3d = vertices_3d.reshape((num_landmarks * 3, 1))
    b = (vertices_3d - model_transformed)

    beta = nnls(Q, b.squeeze())
    return np.squeeze(beta[0]), beta[1]


def min_coeffs(x, trz, zoom, shape_coeffs, model, mean_mesh, vertices, num_shape_coeffs, num_blendshapes):
    num_coeffs = num_shape_coeffs + num_blendshapes
    T = trz[:3]
    R = t3d.quaternions.quat2mat(trz[3:7])
    Z = np.ones(3) * zoom
    Rt = R.T
    t = Rt @ T

    total_coeffs = np.concatenate([shape_coeffs, x])
    mesh_transformed = np.expand_dims((mean_mesh.flatten() + (model @ total_coeffs)).reshape(mean_mesh.shape) * Z + t,
                                      axis=2)
    mesh_transformed = np.squeeze(np.matmul(R, mesh_transformed))

    vertices_triangulated = np.array(vertices)

    error = (mesh_transformed - vertices_triangulated).flatten()
    return error


def get_coeffs_from_mesh(vertices3d,
                         mean_mesh,
                         shape_coeffs,
                         blendshape_prev,
                         trans_BA,
                         zoom,
                         shape,
                         blendshapes,
                         num_shape_coeffs,
                         used_vertices,
                         contours_lms,
                         contours_vertices,
                         contour_correspondence,
                         has_contour):
    if has_contour:
        vertices_ordered = np.concatenate(
            [contour_correspondence[0], [0], contour_correspondence[1], range(1, 50)])
        used_vertices = used_vertices[vertices_ordered]
    else:
        vertices3d = vertices3d[[7] + list(range(16, 65))]
        used_vertices = [7] + list(range(16, 65))
    used_vertices_3 = []
    for vert in used_vertices:
        for i in range(3):
            used_vertices_3.append(3 * vert + i)
    blendshapes_mat = np.asarray([bs.deformation for bs in blendshapes]).transpose()[used_vertices_3]
    shapes_mat = shape[used_vertices_3, :num_shape_coeffs]
    shape_blendshape = np.concatenate((shapes_mat, blendshapes_mat), axis=1)

    bounds = [np.zeros(len(blendshapes)), np.ones(len(blendshapes)) * 1.5]

    res = least_squares(min_coeffs, blendshape_prev, args=(
        trans_BA, zoom, shape_coeffs, shape_blendshape, mean_mesh[used_vertices], vertices3d, num_shape_coeffs,
        len(blendshape_prev)), loss="linear", bounds=bounds,
                        method='trf')
    coeffs = res.x
    return coeffs, res.cost
