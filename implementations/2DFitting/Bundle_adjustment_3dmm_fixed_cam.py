
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2
import transforms3d as t3d

class optimize_2D_non_linear:
    """
    Performs non linear lest squares optimization to fit a Morphable Model to multiple sets of 2D landmarks.
    Optimizes for shape coefficients, blendshape coefficients and transformation matrices
    """

    def __init__(self, cameraArray, shape_params, mean_zoom, blendshape_params, transformation_mats, points2D,
           cameraIndices, morphablemodel_with_expressions, used_lms_numbers, contour_vertices_list,
                 contour_lms_list, n_shape_params, n_blendshape_params, n_lms_used, n_cams_used):


        self.cameraArray = cameraArray
        self.num_camera_params = cameraArray.shape[1]
        self.points2D = points2D
        self.shape_params = shape_params
        self.cameraIndices = cameraIndices
        self.numObservations = blendshape_params.shape[0]
        self.blendshape_params = blendshape_params
        self.transformation_mats = transformation_mats
        self.mean_zoom = mean_zoom

        self.num_shape_params = n_shape_params
        self.num_blendshape_params = n_blendshape_params
        self.n_persons = shape_params.shape[0]
        self.contour_vertices_list = contour_vertices_list
        self.contour_lms_list = contour_lms_list
        self.n_lms_used = n_lms_used
        self.n_cams_used = n_cams_used

        self.cameraIndices_per_observation = cameraIndices[::66]
        self.morphable_model = morphablemodel_with_expressions
        self.used_lms_numbers = used_lms_numbers

    '''
    For each detection, move the generated face to the correct position and project the landmarks to all cameras 
    on which the face was detected.
    '''
    def project_full_cam_contours(self,mean_zooms, shape_coeffs, blendshape_coeffs, trans_mats, cameraArray):
        pnts2d_inner = []
        pnts2d_right = []
        pnts2d_left = []
        pnts2d_all = [pnts2d_inner, pnts2d_right, pnts2d_left]

        f = cameraArray[:, 6]
        cx = cameraArray[:, 7]
        cy = cameraArray[:, 8]
        vertices3D_inner = []
        vertices3D_right = []
        vertices3D_left = []
        for i in range(blendshape_coeffs.shape[0]):
            mesh_from_transformation = \
               self.morphable_model.draw_sample(shape_coeffs, blendshape_coeffs[i], ())
            used_lms = self.used_lms_numbers[:self.n_lms_used].squeeze()
            trans_mat = t3d.affines.compose(trans_mats[i][:3], t3d.quaternions.quat2mat(trans_mats[i][3:7]),
                                            np.ones(3) * mean_zooms)

            vertices3D = []
            used_lms = used_lms.tolist() + self.contour_vertices_list[0] + self.contour_vertices_list[1]
            for idx, vertex in enumerate(np.array(mesh_from_transformation.vertices)[used_lms]):
                vertex_hom = np.ones(4)
                vertex_hom[:3] = vertex
                vertex = trans_mat @ vertex_hom
                # vertex = vertex[:3]/vertex[3]
                vertices3D.append(vertex.astype('float32'))

            for _ in range(self.n_cams_used[i]):
                vertices3D_inner.append(vertices3D[:self.n_lms_used])
                vertices3D_right.append(vertices3D[self.n_lms_used:(self.n_lms_used+len(self.contour_vertices_list[0]))])
                vertices3D_left.append(vertices3D[(self.n_lms_used+len(self.contour_vertices_list[0])):])


        for j, vertices3D in enumerate([vertices3D_inner, vertices3D_right, vertices3D_left]):
            for idx, points in enumerate(vertices3D):
                cam = np.zeros((3, 3))
                cam_num = self.cameraIndices_per_observation[idx]
                cam[0, 0] = f[cam_num]
                cam[1, 1] = f[cam_num]
                cam[0, 2] = cx[cam_num]
                cam[1, 2] = cy[cam_num]

                pnts2d, jac = cv2.projectPoints(np.asarray(points).T[:3,:], cameraArray[cam_num][:3], cameraArray[cam_num][3:6], cam,
                                                cameraArray[cam_num][-5:])
                pnts2d_all[j].append(np.squeeze(pnts2d))

        pnts2d_inner = pnts2d_all[0]
        pnts2d_right = pnts2d_all[1]
        pnts2d_left = pnts2d_all[2]

        return pnts2d_inner, pnts2d_right, pnts2d_left

    def find_closest_vertex(self, landmark, candidate_points):
        dists = np.linalg.norm(candidate_points - landmark, axis=1)
        min_dist = np.argmin(dists)
        return candidate_points[min_dist]

    '''
    Compute the residuals based on the curretn parameters. 2D reprojection error for each landmark.
    '''
    def fun_contours(self, params, numCameras, numObservations, points2D):
        camera_params = self.cameraArray
        mean_zooms = self.mean_zoom

        shape_coeffs = params[:self.num_shape_params].flatten()
        blendshape_end = self.num_shape_params + numObservations * self.num_blendshape_params
        blendshape_coeffs = params[self.num_shape_params:blendshape_end].reshape((numObservations, self.num_blendshape_params))
        trans_mats = params[blendshape_end:].reshape((numObservations, 7))

        points_proj = np.zeros_like(points2D)
        points_inner, points_right, points_left = self.project_full_cam_contours(mean_zooms, shape_coeffs, blendshape_coeffs, trans_mats, camera_params)

        inner_idx = 0
        for idx in range(points2D.shape[0]):
            lm_idx = idx % 66
            obs_num = int(np.floor(idx/66))

            if lm_idx in self.contour_lms_list[0]:
                points_proj[idx] = self.find_closest_vertex(points2D[idx],
                                    points_right[obs_num])
            elif lm_idx in self.contour_lms_list[1]:
                points_proj[idx] = self.find_closest_vertex(points2D[idx],
                                    points_left[obs_num])
            else:
                points_proj[idx] = points_inner[obs_num][inner_idx]
                inner_idx += 1
                if inner_idx == 50:
                    inner_idx = 0

        return (points_proj - points2D).ravel()

    '''
    Generate the sparse jacobian matrix for non linear least squares optimization.
    '''
    def bundle_adjustment_sparsity_3dmm(self, numCameras, numTransformationParams, cameraIndices):
        numShapes = self.num_shape_params
        numBlendshapes = self.num_blendshape_params
        m = self.points2D.shape[0] * 2
        num_lms = int(self.points2D.shape[0] / np.sum(self.n_cams_used))
        n = numShapes +\
           numBlendshapes * self.numObservations + numTransformationParams * self.numObservations
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(int(m/2))

        for s in range(numShapes):
            A[2 * i, s] = 1
            A[2 * i + 1, s] = 1

        numObservations_repeat = []
        for label in range(self.numObservations):
            for _ in range(num_lms*self.n_cams_used[label]):
                numObservations_repeat.append(label)
        numObservations_repeat = np.array(numObservations_repeat)
        for s in range(numBlendshapes):
            A[2 * i, self.num_shape_params + numObservations_repeat * numBlendshapes + s] = 1
            A[2 * i + 1,self.num_shape_params + numObservations_repeat * numBlendshapes + s] = 1

        for s in range(numTransformationParams):
            A[2 * i,self.num_shape_params + self.numObservations * numBlendshapes +
              numObservations_repeat * numTransformationParams + s] = 1
            A[2 * i + 1, self.num_shape_params + self.numObservations * numBlendshapes +
              numObservations_repeat * numTransformationParams + s] = 1

        return A

    '''
    return the optimized parameters in corect shape
    '''
    def optimizedParams(self, params, numCameras, numObservations):
        camera_params = self.cameraArray
        mean_zooms = self.mean_zoom

        shape_coeffs = params[:self.num_shape_params].flatten()
        blendshape_end = self.num_shape_params + numObservations * self.num_blendshape_params
        blendshape_coeffs = params[self.num_shape_params:blendshape_end].reshape((numObservations, self.num_blendshape_params))
        trans_mats = params[blendshape_end:].reshape((numObservations, 7))

        return camera_params, shape_coeffs, mean_zooms, blendshape_coeffs, trans_mats

    '''
    Returns the optimized rotation and translation vectors as well as the
    optimized shape and blendshape coefficients.
    '''
    def optimize(self):
        numCameras = self.cameraArray.shape[0]
        x0 = np.hstack((self.shape_params.ravel(),
                        self.blendshape_params.ravel(), self.transformation_mats.ravel()))

        A = self.bundle_adjustment_sparsity_3dmm(numCameras, self.transformation_mats.shape[1], self.cameraIndices)

        trans_mins = []
        trans_maxs = []
        for idx in range(len(self.transformation_mats)):
            bounds_trans_min = [-np.inf, -np.inf, -np.inf, -1, -1, -1, -1]
            bounds_trans_max = [np.inf, np.inf, np.inf, 1, 1, 1, 1]
            trans_mins.append(bounds_trans_min)
            trans_maxs.append(bounds_trans_max)

        shapes_min = np.ones_like(self.shape_params).flatten() * -3
        shapes_max = np.ones_like(self.shape_params).flatten() * 3
        blends_min = np.zeros_like(self.blendshape_params).flatten()
        blends_max = np.ones_like(self.blendshape_params).flatten() * 2
        trans_mins = np.concatenate(trans_mins)
        trans_maxs = np.concatenate(trans_maxs)

        bounds_max = np.concatenate([shapes_max, blends_max, trans_maxs])
        bounds_min = np.concatenate([shapes_min, blends_min, trans_mins])

        bounds = (bounds_min, bounds_max)
        res = least_squares(self.fun_contours, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf',
                            args=(numCameras, self.numObservations, self.points2D), bounds=bounds, diff_step=0.1)

        params = self.optimizedParams(res.x, numCameras, self.numObservations)

        return params
