import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import transforms3d as t3d

class optimize_3D_non_linear:
    """
    Performs non linear lest squares optimization to fit a Morphable Model to multiple sets of 3D landmarks.
    Optimizes for shape coefficients, blendshape coefficients and transformation matrices
    """

    def __init__(self, vertices3d, shape_params, blendshape_params, transformation_mats,
                 morphablemodel_with_expressions, used_lms_numbers, contour_vertices_list,
                 contour_lms_list, n_shape_params, n_blendshape_params, n_lms_used):
        self.vertices3d = vertices3d
        self.shape_params = shape_params
        self.numObservations = blendshape_params.shape[0]
        self.blendshape_params = blendshape_params
        self.transformation_mats = transformation_mats

        self.num_shape_params = n_shape_params
        self.num_blendshape_params = n_blendshape_params
        self.contour_vertices_list = contour_vertices_list
        self.contour_lms_list = contour_lms_list
        self.n_lms_used = n_lms_used

        self.morphable_model = morphablemodel_with_expressions
        self.used_lms_numbers = used_lms_numbers

    def transform_meshes(self, shape_coeffs, blendshape_coeffs, trans_mats):
        vertices3D_inner = []
        vertices3D_right = []
        vertices3D_left = []
        for i in range(blendshape_coeffs.shape[0]):
            mesh_from_transformation = \
               self.morphable_model.draw_sample(shape_coeffs, blendshape_coeffs[i], ())
            used_lms = self.used_lms_numbers[:self.n_lms_used].squeeze()
            trans_mat = t3d.affines.compose(trans_mats[i][:3], t3d.quaternions.quat2mat(trans_mats[i][3:7]),
                                            np.ones(3))

            vertices3D = []
            used_lms = used_lms.tolist() + self.contour_vertices_list[0] + self.contour_vertices_list[1]
            for idx, vertex in enumerate(np.array(mesh_from_transformation.vertices)[used_lms]):
                vertex_hom = np.ones(4)
                vertex_hom[:3] = vertex
                vertex = trans_mat @ vertex_hom
                vertex = vertex[:3]/vertex[3]
                vertices3D.append(vertex.astype('float32'))

            for _ in range(1):
                vertices3D_inner.append(vertices3D[:self.n_lms_used])
                vertices3D_right.append(vertices3D[self.n_lms_used:(self.n_lms_used+len(self.contour_vertices_list[0]))])
                vertices3D_left.append(vertices3D[(self.n_lms_used+len(self.contour_vertices_list[0])):])

        return vertices3D_inner, vertices3D_right, vertices3D_left

    def find_closest_vertex3D(self, vertex, candidate_vertices):
        dists = np.linalg.norm(candidate_vertices - vertex, axis=1)
        min_dist = np.argmin(dists)
        return candidate_vertices[min_dist]

    def fun_contours(self, params):
        """
        Calculate the residuals between meshes generated by current estimates and triangulated 3D landmarks.
        """
        shape_coeffs = params[:self.num_shape_params]
        blendshape_end = self.num_shape_params + self.numObservations * self.num_blendshape_params
        blendshape_coeffs = params[self.num_shape_params:blendshape_end].reshape((self.numObservations, self.num_blendshape_params))
        trans_mats = params[blendshape_end:].reshape((self.numObservations, 7))

        vertices3d = self.vertices3d
        vertices3d_from_mesh = np.zeros_like(vertices3d)
        vertices3d_inner, vertices3d_right, vertices3d_left = self.transform_meshes(shape_coeffs, blendshape_coeffs, trans_mats)

        inner_idx = 0
        for idx in range(vertices3d.shape[0]):
            lm_idx = idx % 66
            obs_num = int(np.floor(idx/66))

            if lm_idx in self.contour_lms_list[0]:
                vertices3d_from_mesh[idx] = self.find_closest_vertex3D(vertices3d[idx],
                                    vertices3d_right[obs_num])
            elif lm_idx in self.contour_lms_list[1]:
                vertices3d_from_mesh[idx] = self.find_closest_vertex3D(vertices3d[idx],
                                    vertices3d_left[obs_num])
            else:
                vertices3d_from_mesh[idx] = vertices3d_inner[obs_num][inner_idx]
                inner_idx += 1
                if inner_idx == 50:
                    inner_idx = 0

        return (vertices3d_from_mesh - vertices3d).ravel()


    def bundle_adjustment_sparsity_vertices(self, numTransformationParams):
        """
        create sparse jacobian for non linear least squares optimization
        """
        numShapes = self.num_shape_params
        numBlendshapes = self.num_blendshape_params
        m = self.vertices3d.shape[0] * 3
        n = self.num_shape_params +\
           numBlendshapes * self.numObservations + numTransformationParams * self.numObservations
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(self.vertices3d.shape[0])


        for s in range(numShapes):
            A[3 * i, s] = 1
            A[3 * i + 1, s] = 1
            A[3 * i + 2, s] = 1


        numObservations_repeat = []
        for label in range(self.numObservations):
            for _ in range(int(self.vertices3d.shape[0]/self.numObservations)):
                numObservations_repeat.append(label)
        numObservations_repeat = np.array(numObservations_repeat)
        for s in range(numBlendshapes):
            A[3 * i, numShapes + numObservations_repeat * numBlendshapes + s] = 1
            A[3 * i + 1, numShapes + numObservations_repeat * numBlendshapes + s] = 1
            A[3 * i + 2, numShapes + numObservations_repeat * numBlendshapes + s] = 1


        for s in range(numTransformationParams):
            A[3 * i, numShapes + self.numObservations * numBlendshapes +
              numObservations_repeat * numTransformationParams + s] = 1
            A[3 * i + 1, numShapes + self.numObservations * numBlendshapes +
              numObservations_repeat * numTransformationParams + s] = 1
            A[3 * i + 2, numShapes + self.numObservations * numBlendshapes +
              numObservations_repeat * numTransformationParams + s] = 1

        return A

    def optimizedParams(self, params):
        shape_coeffs = params[:self.num_shape_params]
        blendshape_end = self.num_shape_params + self.numObservations * self.num_blendshape_params
        blendshape_coeffs = params[self.num_shape_params:blendshape_end].reshape((self.numObservations, self.num_blendshape_params))
        trans_mats = params[blendshape_end:].reshape((self.numObservations, 7))

        return shape_coeffs, blendshape_coeffs, trans_mats
    '''
    Returns the optimized rotation and translation vectors as well as the
    optimized shape and blendshape coefficients.
    '''
    def optimize(self):
        x0 = np.hstack((self.shape_params.ravel(),
                                       self.blendshape_params.ravel(), self.transformation_mats.ravel()))
        A = self.bundle_adjustment_sparsity_vertices(self.transformation_mats.shape[1])

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
                            diff_step=0.1,
                            bounds=bounds)

        params = self.optimizedParams(res.x)

        return params
