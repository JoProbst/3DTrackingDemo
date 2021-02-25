import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2
import transforms3d as t3d
from scipy.optimize import nnls

def find_closest_landmark(lm, candidate_lms):
    dists = np.linalg.norm(candidate_lms - lm, axis=1)
    min_dist = np.argmin(dists)
    return candidate_lms[min_dist]


def find_closest_landmark_index(lm, candidate_lms):
    dists = np.linalg.norm(candidate_lms - lm, axis=1)
    min_dist = np.argmin(dists)
    return min_dist

'''
Calculate the contour correspondences for one detection of a face.
'''
def get_contour_correspondence_2d(camsBA, cam_indices, trans, zoom, shape_coeffs, blendshape_coeffs, shape_model,
                                  blendshapes_model,
                                  mean_mesh, landmarks_all,
                                  used_vertices, contours_lms, contours_vertices):
    n_observations = np.unique(cam_indices).size
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
    total_coeffs = np.concatenate([shape_coeffs, blendshape_coeffs])
    mesh_transformed = np.expand_dims(
        mean_mesh_contours + ((shape_blendshape @ total_coeffs).reshape(mean_mesh_contours.shape) ), axis=2)
    contours_trans = np.squeeze(np.matmul(R, mesh_transformed)) * zoom + T

    contour_verts_selected = np.zeros((n_observations, 2, len(contours_lms[0])), dtype=int)
    for idx, cam_num in enumerate(np.unique(cam_indices)):
        cam = camsBA[cam_num]
        landmarks = landmarks_all[idx * 66:(idx + 1) * 66]
        dist = cam[-5:]
        trans = cam[3:6]
        rot = cam[:3]
        internal = np.zeros((3, 3))
        internal[0, 0] = cam[6]
        internal[1, 1] = cam[6]
        internal[0, 2] = cam[7]
        internal[1, 2] = cam[8]
        pnts2d, jac = cv2.projectPoints(contours_trans, rot, trans, internal, dist)
        pnts2d = np.squeeze(pnts2d)
        for i, side in enumerate(contours_lms):
            for j, lm in enumerate(side):
                closest = find_closest_landmark_index(landmarks[lm], pnts2d[(i * 17):((i * 17) + 17)])
                contour_verts_selected[idx, i, j] = closest + 50 + i * 17

    return contour_verts_selected


def min_coeffs2d(x, camsBA, cam_indices, trz, zoom, shape_coeffs, model, mean_mesh, landmarks_all, num_shape_coeffs,
                 num_blendshapes, used_vertices, contours_lms, contours_vertices, num_landmarks, has_contours, contour_correspondence):
    T = trz[:3]
    R = t3d.quaternions.quat2mat(trz[3:7])
    blendshape_coeffs = x
    total_coeffs = np.concatenate([shape_coeffs, blendshape_coeffs])

    mesh_transformed = np.expand_dims((mean_mesh.flatten() + (model @ total_coeffs)).reshape(mean_mesh.shape),
                                      axis=2)
    mesh_transformed = np.squeeze(np.matmul(R,
                                            mesh_transformed)) * zoom + T
    # mean_mesh_transformed = np.asarray(mean_mesh_transformed).reshape(mean_mesh.shape)[used_vertices].flatten()
    error = []
    for idx, cam_num in enumerate(np.unique(cam_indices)):
        cam = camsBA[cam_num]
        landmarks = landmarks_all[idx * num_landmarks:(idx + 1) * num_landmarks]
        dist = cam[-5:]
        trans = cam[3:6]
        rot = cam[:3]
        internal = np.zeros((3, 3))
        internal[0, 0] = cam[6]
        internal[1, 1] = cam[6]
        internal[0, 2] = cam[7]
        internal[1, 2] = cam[8]

        pnts2d, jac = cv2.projectPoints(mesh_transformed, rot, trans, internal, dist)
        pnts2d = np.squeeze(pnts2d)
        landmarks_proj = np.zeros_like(landmarks)
        if has_contours:
            landmarks_proj[8] = pnts2d[0]
            landmarks_proj[17:] = pnts2d[1:50]
            landmarks_proj[:8] = pnts2d[contour_correspondence[idx][0]]
            landmarks_proj[9:17] = pnts2d[contour_correspondence[idx][1]]
        else:
            landmarks_proj = pnts2d
        error.append((landmarks_proj - landmarks).flatten())

    return np.concatenate(error)

'''
Non linear blendshape fitting to multiple sets of 2D landmarks from different cameras.
'''
def get_coeffs_from_mesh2d(camsBA,
                           cam_indices,
                           landmarks,
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
                           contours_vertices, has_contours, contour_correspondence):
    if has_contours:
        used_vertices = np.concatenate([used_vertices, contours_vertices[0], contours_vertices[1]])
        num_landmarks = 66
    else:
        num_landmarks = 50
    used_vertices_3 = []
    for vert in used_vertices:
        for i in range(3):
            used_vertices_3.append(3 * vert + i)
    blendshapes_mat = np.asarray([bs.deformation for bs in blendshapes]).transpose()[used_vertices_3]
    shapes_mat = shape[used_vertices_3, :num_shape_coeffs]
    shape_blendshape = np.concatenate((shapes_mat, blendshapes_mat), axis=1)
    total_coeffs = blendshape_prev
    num_all_coeffs = len(total_coeffs)
    bounds = [np.zeros(len(blendshapes)), np.ones(len(blendshapes)) * 2]

    res = least_squares(min_coeffs2d, total_coeffs, args=(camsBA,
                                                          cam_indices,
                                                          trans_BA, zoom, shape_coeffs, shape_blendshape,
                                                          mean_mesh[used_vertices], landmarks, num_shape_coeffs,
                                                          len(blendshape_prev), used_vertices, contours_lms,
                                                          contours_vertices, num_landmarks, has_contours, contour_correspondence),
                        loss="linear", method='trf',
                        bounds=bounds)
    coeffs = res.x
    return coeffs, res.cost


def min_trans2d(x, camsBA, cam_indices, total_coeffs, model, mean_mesh, landmarks_all,
                 used_vertices, contours_lms, contours_vertices, num_landmarks, has_contours):
    T = x[:3]
    R = t3d.quaternions.quat2mat(x[3:-1])
    zoom = x[-1]
    mesh_transformed = np.expand_dims(mean_mesh + (model @ total_coeffs).reshape(mean_mesh.shape), axis=2)
    mesh_transformed = np.squeeze(np.matmul(R, mesh_transformed)) * zoom + T
    error = []
    for idx, cam_num in enumerate(np.unique(cam_indices)):
        cam = camsBA[cam_num]
        landmarks = landmarks_all[idx * num_landmarks:(idx + 1) * num_landmarks]
        dist = cam[-5:]
        trans = cam[3:6]
        rot = cam[:3]
        internal = np.zeros((3, 3))
        internal[0, 0] = cam[6]
        internal[1, 1] = cam[6]
        internal[0, 2] = cam[7]
        internal[1, 2] = cam[8]
        pnts2d, jac = cv2.projectPoints(mesh_transformed, rot, trans, internal, dist)
        pnts2d = np.squeeze(pnts2d)
        landmarks_proj = np.zeros_like(landmarks)
        if has_contours:
            landmarks_proj[8] = pnts2d[0]
            landmarks_proj[17:] = pnts2d[1:50]
            landmarks_proj[:8] = pnts2d[contours_vertices[idx][0]]
            landmarks_proj[9:17] = pnts2d[contours_vertices[idx][1]]
        else:
            landmarks_proj = pnts2d

        error.append((landmarks_proj - landmarks).flatten())

    return np.concatenate(error)

'''
Non linear transformation matrix optimization for multiple sets of 2D landmarks for one face at the same frame.
'''
def get_trans_from_mesh2d(camsBA,
                          cam_indices,
                          landmarks,
                          mean_mesh,
                          shape_prev,
                          blendshape_prev,
                          trans_BA,
                          zoom,
                          shape,
                          model_blendshapes,
                          num_shape_coeffs,
                          used_vertices,
                          contours_lms,
                          contours_vertices,
                          model_contour_lists,
                          has_contours
                          ):
    if has_contours:
        used_vertices = np.concatenate([used_vertices, model_contour_lists[0], model_contour_lists[1]])
        num_landmarks = 66
    else:
        num_landmarks = 50
    used_vertices_3 = []
    for vert in used_vertices:
        for i in range(3):
            used_vertices_3.append(3 * vert + i)
    blendshapes_mat = np.asarray([bs.deformation for bs in model_blendshapes]).transpose()[used_vertices_3]
    shapes_mat = shape[used_vertices_3, :num_shape_coeffs]
    shape_blendshape = np.concatenate((shapes_mat, blendshapes_mat), axis=1)
    total_coeffs = np.append(shape_prev, blendshape_prev)

    bounds = [np.concatenate([-np.ones(3) * np.inf, np.ones(4) * -1, [0]]),
              np.concatenate([np.ones(3) * np.inf, np.ones(4) * 1, [np.inf]])]



    res = least_squares(min_trans2d, np.concatenate([trans_BA, [zoom]]),
                        args=(camsBA,
                              cam_indices, total_coeffs, shape_blendshape, mean_mesh[used_vertices], landmarks,
                              used_vertices, contours_lms,
                              contours_vertices, num_landmarks, has_contours), bounds=bounds, loss="linear",
                        method='trf')

    return res.x[:-1], res.cost / num_landmarks

def min_shape2d(x, camsBA, cam_indices_all, n_cams_used, landmarks2d, blendshapes, trans_BA, zoom, model, mean_mesh,
                num_shape_coeffs, num_blendshapes, contours_lms, contour_correspondence):
    error = []
    dets_total = landmarks2d.shape[0] / 66
    for i, blend in enumerate(blendshapes):
        cams_prev = np.sum(n_cams_used[:i])
        landmarks_all = landmarks2d[cams_prev * 66:(cams_prev + n_cams_used[i]) * 66]
        cam_indices = cam_indices_all[cams_prev * 66:(cams_prev + n_cams_used[i]) * 66]
        contours = contour_correspondence[cams_prev:(cams_prev + n_cams_used[i])]
        trans = trans_BA[i]
        T = trans[:3]
        R = t3d.quaternions.quat2mat(trans[3:7])

        shape_coeffs = x
        blendshape_coeffs = blend

        total_coeffs = np.concatenate([shape_coeffs, blendshape_coeffs])
        mesh_transformed = np.expand_dims(
            (mean_mesh.flatten() + (model @ total_coeffs)).reshape(mean_mesh.shape), axis=2)
        mesh_transformed = np.squeeze(np.matmul(R, mesh_transformed)) * zoom + T
        # mean_mesh_transformed = np.asarray(mean_mesh_transformed).reshape(mean_mesh.shape)[used_vertices].flatten()
        unq = sorted(np.unique(cam_indices))
        for idx, cam_num in enumerate(unq):
            cam = camsBA[cam_num]
            landmarks = landmarks_all[idx * 66:(idx + 1) * 66]
            dist = cam[-5:]
            trans = cam[3:6]
            rot = cam[:3]
            internal = np.zeros((3, 3))
            internal[0, 0] = cam[6]
            internal[1, 1] = cam[6]
            internal[0, 2] = cam[7]
            internal[1, 2] = cam[8]
            pnts2d, jac = cv2.projectPoints(mesh_transformed, rot, trans, internal, dist)
            pnts2d = np.squeeze(pnts2d)
            landmarks_proj = np.zeros_like(landmarks)
            landmarks_proj[8] = pnts2d[0]
            landmarks_proj[17:] = pnts2d[1:50]
            landmarks_proj[:8] = pnts2d[contours[idx][0]]
            landmarks_proj[9:17] = pnts2d[contours[idx][1]]

            error.append((landmarks_proj - landmarks).flatten())

    return np.concatenate(error)

'''
Non-linear shape coefficient fitting to multiple sets of 2D landmarks of the same face.
'''
def get_shape_multiple2d(camsBA, cam_indices, n_cams_used,
                         landmarks2d,
                         mean_mesh,
                         shape_prev,
                         blendshape_prev,
                         trans_BA,
                         zoom,
                         shape,
                         model_blendshapes,
                         num_shape_coeffs,
                         used_vertices,
                         contours_lms,
                         contours_vertices,
                         has_contours,
                        contour_correspondence
                         ):
    if has_contours:
        used_vertices = np.concatenate([used_vertices, contours_vertices[0], contours_vertices[1]])
        num_landmarks = 66
    else:
        num_landmarks = 50
    used_vertices_3 = []
    for vert in used_vertices:
        for i in range(3):
            used_vertices_3.append(3 * vert + i)
    used_vertices_3 = []
    for vert in used_vertices:
        for i in range(3):
            used_vertices_3.append(3 * vert + i)
    blendshapes_mat = np.asarray([bs.deformation for bs in model_blendshapes]).transpose()[used_vertices_3]
    shapes_mat = shape[used_vertices_3, :num_shape_coeffs]
    shape_blendshape = np.concatenate((shapes_mat, blendshapes_mat), axis=1)

    bounds = [np.ones(num_shape_coeffs) * (-3), np.ones(num_shape_coeffs) * 3]


    res = least_squares(min_shape2d, shape_prev,
                       args=(camsBA, cam_indices, n_cams_used, landmarks2d, blendshape_prev, trans_BA, zoom,
                             shape_blendshape, mean_mesh[used_vertices], num_shape_coeffs,
                             len(blendshape_prev), contours_lms, contour_correspondence), verbose=1,
                       method='trf', bounds=bounds)

    return res.x

'''
Linear blendshape fitting to multiple images of the same person at the same time. Shape and transformation stays fixed.
'''
def get_blendshape_2d_linear_proj(camsBA, cam_indices,
                             landmarks2d,
                             mean_mesh,
                             shape_prev,
                             trans_BA,
                             zoom,
                             shape,
                             model_blendshapes,
                             num_shape_coeffs,
                             used_vertices,
                             contours_lms,
                             model_contour_lists,
                             contour_correspondece,
                             has_contours):
    used_vertices_3 = []
    if has_contours:
        used_vertices = np.concatenate([used_vertices, model_contour_lists[0], model_contour_lists[1]])
        num_landmarks = 66
    else:
        num_landmarks = 50
    for vert in used_vertices:
        for i in range(3):
            used_vertices_3.append(3 * vert + i)

    blendshapes_mat = np.asarray([bs.deformation for bs in model_blendshapes]).transpose()[used_vertices_3]
    shapes_mat = shape[used_vertices_3, :num_shape_coeffs]

    num_blendshape_coeffs = len(model_blendshapes)

    num_observations = int(len(landmarks2d) / num_landmarks)
    cam_single = cam_indices[::num_landmarks]
    projections_affine = []
    for cam in camsBA:
        trans = cam[3:6]
        rot = cam[:3]
        rot = cv2.Rodrigues(rot)[0]
        internal = np.zeros((3, 3))
        internal[0, 0] = cam[6] / trans[2]
        internal[1, 1] = cam[6] / trans[2]
        internal[0, 2] = cam[7]
        internal[1, 2] = cam[8]
        internal[2, 2] = 1

        p = np.zeros((3, 4))
        p[:3, :3] = rot
        p[:, 3] = np.squeeze(trans)
        p[2, :] = [0, 0, 0, 1]
        p = internal @ p
        projections_affine.append(p)

    projections_persp = []
    for cam in camsBA:
        trans = cam[3:6]
        rot = cam[:3]
        rot = cv2.Rodrigues(rot)[0]
        internal = np.zeros((3, 3))
        internal[0, 0] = cam[6]
        internal[1, 1] = cam[6]
        internal[0, 2] = cam[7]
        internal[1, 2] = cam[8]
        internal[2, 2] = 1

        p = np.zeros((3, 4))
        p[:3, :3] = rot
        p[:, 3] = np.squeeze(trans)
        p = internal @ p
        projections_persp.append(p)

    m = num_observations * num_landmarks * 2
    n = num_observations * num_landmarks * 3
    rotcam_sparse = np.zeros((m, n), dtype=float)
    projections_sparse = lil_matrix((m, n), dtype=float)
    t_stacked = np.zeros((m, 1))

    Li = np.zeros((num_observations, num_landmarks * 2, num_landmarks * 3))
    for i in range(num_observations):
        for j in range(num_landmarks):
            for x in range(2):
                for y in range(3):
                    rotcam_sparse[i * num_landmarks * 2 + j * 2 + x, i * num_landmarks * 3 + j * 3 + y] = \
                        projections_affine[cam_single[i]][x, y]
                    projections_sparse[i * num_landmarks * 2 + j * 2 + x, i * num_landmarks * 3 + j * 3 + y] = \
                        projections_affine[cam_single[i]][x, y]
                    Li[i][j * 2 + x, j * 3 + y] = projections_affine[cam_single[i]][x, y]
                t_stacked[i * num_landmarks * 2 + j * 2 + x] = projections_affine[cam_single[i]][x, 3]


    m = num_observations * num_landmarks * 3
    n = num_observations * num_landmarks * 3
    persp_rotcam_sparse = np.zeros((m, n), dtype=float)
    persp_t_stacked = np.zeros((m, 1))
    for i in range(num_observations):
        for j in range(num_landmarks):
            for x in range(3):
                for y in range(3):
                    persp_rotcam_sparse[i * num_landmarks * 3 + j * 3 + x, i * num_landmarks * 3 + j * 3 + y] = \
                        projections_persp[cam_single[i]][x, y]
                persp_t_stacked[i * num_landmarks * 3 + j * 3 + x] = projections_persp[cam_single[i]][x, 3]

    R = t3d.quaternions.quat2mat(trans_BA[3:7])
    t = trans_BA[:3]

    m = num_observations * num_landmarks * 3
    n = num_observations * num_landmarks * 3
    rot_sparse = lil_matrix((m, n), dtype=float)

    for i in range(num_observations):
        for j in range(num_landmarks):
            for x in range(3):
                for y in range(3):
                    rot_sparse[i * num_landmarks * 3 + j * 3 + x, i * num_landmarks * 3 + j * 3 + y] = \
                        R[x, y]

    S = np.zeros((num_observations, num_landmarks * 3, num_blendshape_coeffs))
    for i in range(num_observations):
        # shape_model_stacked[i * num_landmarks * 3:(i + 1) * num_landmarks * 3, :] = shapes_mat
        if has_contours:
            inner_and_contours = np.concatenate(
                [contour_correspondece[i][0], [0], contour_correspondece[i][1], range(1, 50)])
            inner_and_contours_3 = []
            for vert in inner_and_contours:
                for j in range(3):
                    inner_and_contours_3.append(3 * vert + j)

            S[i] = rot_sparse[:num_landmarks * 3,:num_landmarks * 3] @ blendshapes_mat[inner_and_contours_3]
        else:
            S[i] = rot_sparse[:num_landmarks * 3,:num_landmarks * 3] @ blendshapes_mat

    Qi = (Li @ S) * zoom
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
            shapes_mat_i = shapes_mat[inner_and_contours_3]
            mean_mesh_i = mean_mesh[used_vertices][inner_and_contours]
        else:
            shapes_mat_i = shapes_mat
            mean_mesh_i = mean_mesh[used_vertices]

        shape_deform = shapes_mat_i @ shape_prev
        mean_model_stacked[i * num_landmarks * 3:(i + 1) * num_landmarks * 3] = \
            (mean_mesh_i).reshape(num_landmarks * 3, 1) + shape_deform.reshape((num_landmarks * 3, 1))

    m = num_observations * num_landmarks * 3
    trans_stacked = np.zeros((m, 1), dtype=float)
    for i in range(num_observations):
        for j in range(num_landmarks):
            trans_stacked[i * num_landmarks * 3 + j * 3:i * num_landmarks * 3 + (j + 1) * 3] = np.expand_dims(
                t, axis=1)

    model_transformed = (rot_sparse @ mean_model_stacked) * zoom + trans_stacked

    y = persp_rotcam_sparse @ model_transformed + persp_t_stacked

    y = y.reshape(num_landmarks * num_observations, 3)
    y[:, 0] = y[:, 0] / y[:, 2]
    y[:, 1] = y[:, 1] / y[:, 2]
    y = y[:, :2]
    b = (np.array(landmarks2d).ravel() - y.ravel())

    qi_stack = Qi.reshape((num_observations * num_landmarks * 2, num_blendshape_coeffs))
    beta = nnls(qi_stack, b.squeeze())

    return np.squeeze(beta[0]), beta[1]

'''
Linear shape fitting to 2D landmarks from multiple cameras. Blendshapes and transformations stay fixed.
'''
def get_shape_multiple2d_svd_projective(camsBA, cam_indices, n_cams_used,
                             landmarks2d,
                             mean_mesh,
                             shape_prev,
                             blendshape_prev,
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
        num_landmarks = 50
    for vert in used_vertices:
        for i in range(3):
            used_vertices_3.append(3 * vert + i)

    blendshapes_mat = np.asarray([bs.deformation for bs in model_blendshapes]).transpose()[used_vertices_3]
    shapes_mat = shape[used_vertices_3, :num_shape_coeffs]

    blendshape_coeffs_obs = []
    for i in n_cams_used:
        for _ in range(i):
            blendshape_coeffs_obs.append(blendshape_prev[i])
    num_observations = np.sum(n_cams_used)
    cam_single = cam_indices[::num_landmarks]
    projections_affine = []
    for cam in camsBA:
        trans = cam[3:6]
        rot = cam[:3]
        rot = cv2.Rodrigues(rot)[0]
        internal = np.zeros((3, 3))
        internal[0, 0] = cam[6] / trans[2]
        internal[1, 1] = cam[6] / trans[2]
        internal[0, 2] = cam[7]
        internal[1, 2] = cam[8]
        internal[2, 2] = 1


        p = np.zeros((3, 4))
        p[:3, :3] = rot
        p[:, 3] = np.squeeze(trans)
        p[2, :] = [0, 0, 0, 1]
        p = internal @ p

        projections_affine.append(p)
    projections_persp = []
    for cam in camsBA:
        trans = cam[3:6]
        rot = cam[:3]
        rot = cv2.Rodrigues(rot)[0]
        internal = np.zeros((3, 3))
        internal[0, 0] = cam[6]
        internal[1, 1] = cam[6]
        internal[0, 2] = cam[7]
        internal[1, 2] = cam[8]
        internal[2, 2] = 1

        p = np.zeros((3, 4))
        p[:3, :3] = rot
        p[:, 3] = np.squeeze(trans)
        p = internal @ p

        projections_persp.append(p)

    m = num_observations * num_landmarks * 3
    n = num_observations * num_landmarks * 3
    persp_rotcam_sparse = lil_matrix((m, n), dtype=float)
    persp_t_stacked = np.zeros((m, 1))
    for i in range(num_observations):
        for j in range(num_landmarks):
            for x in range(3):
                for y in range(3):
                    persp_rotcam_sparse[i * num_landmarks * 3 + j * 3 + x, i * num_landmarks * 3 + j * 3 + y] = \
                        projections_persp[cam_single[i]][x, y]
                persp_t_stacked[i * num_landmarks * 3 + j * 3 + x] = projections_persp[cam_single[i]][x, 3]


    m = num_observations * num_landmarks * 2
    n = num_observations * num_landmarks * 3
    rotcam_sparse = lil_matrix((m, n), dtype=float)
    projections_sparse = lil_matrix((m, n), dtype=float)
    t_stacked = np.zeros((m, 1))

    Li = np.zeros((num_observations, num_landmarks * 2, num_landmarks * 3))
    for i in range(num_observations):
        for j in range(num_landmarks):
            for x in range(2):
                for y in range(3):
                    rotcam_sparse[i * num_landmarks * 2 + j * 2 + x, i * num_landmarks * 3 + j * 3 + y] = \
                        projections_affine[cam_single[i]][x, y]
                    projections_sparse[i * num_landmarks * 2 + j * 2 + x, i * num_landmarks * 3 + j * 3 + y] = \
                        projections_affine[cam_single[i]][x, y]
                    Li[i][j * 2 + x, j * 3 + y] = projections_affine[cam_single[i]][x, y]
                t_stacked[i * num_landmarks * 2 + j * 2 + x] = projections_affine[cam_single[i]][x, 3]

    all_rots = []
    all_trans = []
    trans_centers = []
    for i, trans in enumerate(trans_BA):
        R = t3d.quaternions.quat2mat(trans[3:7])
        t = trans[:3]
        trans_center = projections_affine[cam_single[i]] @ np.hstack((t, 1))


        for _ in range(n_cams_used[i]):
            for _ in range(num_landmarks):
                trans_centers.append(trans_center.T)
            all_rots.append(R)
            all_trans.append(t)

    m = num_observations * num_landmarks * 3
    n = num_observations * num_landmarks * 3
    rot_sparse = lil_matrix((m, n), dtype=float)

    for i in range(num_observations):
        for j in range(num_landmarks):
            for x in range(3):
                for y in range(3):
                    rot_sparse[i * num_landmarks * 3 + j * 3 + x, i * num_landmarks * 3 + j * 3 + y] = \
                        all_rots[i][x, y]

    S = np.zeros((num_observations, num_landmarks * 3, num_shape_coeffs))
    for i in range(num_observations):
        # shape_model_stacked[i * num_landmarks * 3:(i + 1) * num_landmarks * 3, :] = shapes_mat
        if has_contours:
            inner_and_contours = np.concatenate(
                [contour_correspondece[i][0], [0], contour_correspondece[i][1], range(1, 50)])
            inner_and_contours_3 = []
            for vert in inner_and_contours:
                for j in range(3):
                    inner_and_contours_3.append(3 * vert + j)
            S[i] = rot_sparse[i * num_landmarks * 3:(i + 1) * num_landmarks * 3,
                   i * num_landmarks * 3:(i + 1) * num_landmarks * 3] @ shapes_mat[inner_and_contours_3]
        else:
            S[i] = rot_sparse[i * num_landmarks * 3:(i + 1) * num_landmarks * 3,
                   i * num_landmarks * 3:(i + 1) * num_landmarks * 3] @ shapes_mat

    Qi = (Li @ S) * zoom
    Qt = np.zeros((num_observations, num_shape_coeffs, num_shape_coeffs))
    #omega = np.diagflat(np.ones(num_landmarks * 2) * np.sqrt(3))
    for i in range(num_observations):
        Qt[i] = Qi[i].T @ Qi[i]
    Q = np.sum(Qt, axis=0)

    I = num_observations * np.diag(np.ones(num_shape_coeffs)) * lam

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

        blends = blendshapes_mat_i @ blendshape_coeffs_obs[i]
        mean_model_stacked[i * num_landmarks * 3:(i + 1) * num_landmarks * 3] = \
            (mean_mesh_i).reshape(num_landmarks * 3, 1) + blends.reshape((num_landmarks * 3, 1))

    m = num_observations * num_landmarks * 3
    trans_stacked = np.zeros((m, 1), dtype=float)
    for i in range(num_observations):
        for j in range(num_landmarks):
            trans_stacked[i * num_landmarks * 3 + j * 3:i * num_landmarks * 3 + (j + 1) * 3] = np.expand_dims(
                all_trans[i], axis=1)

    model_transformed = (rot_sparse @ mean_model_stacked) * zoom + trans_stacked

    y = persp_rotcam_sparse @ model_transformed + persp_t_stacked
    y = y.reshape(num_landmarks * num_observations, 3)
    y[:,0] = y[:,0]/y[:,2]
    y[:,1] = y[:,1]/y[:,2]
    y = y[:,:2]

    b = (np.array(landmarks2d).ravel() - y.ravel())
    b_summed = b.reshape(num_observations, num_landmarks * 2, 1)
    Qty = np.zeros((num_observations, num_shape_coeffs, 1))
    for i in range(num_observations):
        Qty[i] = Qi[i].T @ b_summed[i]
    rhs = np.sum(Qty, axis=0)
    alpha = np.linalg.pinv(l) @ rhs
    #qi_stack = Qi.reshape((num_observations * num_landmarks * 2, num_shape_coeffs))
    #alpha = lstsq(qi_stack, b.squeeze())[0]

    return np.squeeze(alpha)