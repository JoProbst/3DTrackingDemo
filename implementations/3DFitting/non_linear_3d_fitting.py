import glob
import shutil
import time
import numpy as np
import os
import eos
import cv2
import transforms3d as t3d
from colour import Color
import json


from optimize_3D_non_linear import optimize_3D_non_linear

n_shape_params = 5

folder = "../outBaseline/blender/"
out_folder = "out_non_linear_3D/"

if os.path.isdir(out_folder):
    shutil.rmtree(out_folder)
os.makedirs(out_folder)

# Load Morphable Model with mappings
model = eos.morphablemodel.load_model("../share/sfm_shape_3448.bin")
blendshapes = eos.morphablemodel.load_blendshapes("../share/expression_blendshapes_3448.bin")
# Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                    color_model=eos.morphablemodel.PcaModel(),
                                                                    vertex_definitions=None,
                                                                    texture_coordinates=model.get_texture_coordinates())
landmark_mapper = eos.core.LandmarkMapper('../share/ibug_to_sfm.txt')
edge_topology = eos.morphablemodel.load_edge_topology('../share/sfm_3448_edge_topology.json')
contour_landmarks = eos.fitting.ContourLandmarks.load('../share/ibug_to_sfm.txt')
model_contour = eos.fitting.ModelContour.load('../share/sfm_model_contours.json')
model_contour_lists = [[], []]
with open('../share/sfm_model_contours.json') as f:
    model_contour_dict = json.load(f)['model_contour']
model_contour_lists[0] = model_contour_dict['right_contour']
model_contour_lists[1] = model_contour_dict['left_contour']
contour_lms = [range(0, 8), range(9, 17)]  # right,# left
no_mapping = [61, 65]
ignored_landmarks = list(range(1, 9)) + list(range(10, 18)) + no_mapping
used_lms = []
used_lms_int = []
for lm in range(1, 69):
    if lm not in ignored_landmarks:
        used_lms_int.append(lm - 1)
        used_lms.append(str(lm))
used_vertex_indices = [int((landmark_mapper.convert(lm))) for lm in used_lms]
n_blendshape_params = 6

# Load necessary data for fitting
camera_indices = np.int32(
    np.load(folder + 'fitting_data/camera_indices.npy', allow_pickle=True))
cams_BA = np.load(folder + 'fitting_data/camsBA.npy', allow_pickle=True)
vertices2d_coords = np.load(folder + 'fitting_data/vertices2d_coords.npy', allow_pickle=True)
frame_numbers = np.int32(np.load(folder + 'fitting_data/frame_numbers.npy', allow_pickle=True))
used_lms_numbers = np.int32(np.load(folder + 'fitting_data/used_lms_numbers.npy', allow_pickle=True))
observations_used = np.int32(
    np.load(folder + 'fitting_data/observations_used.npy', allow_pickle=True))
person_label = np.int32(np.load(folder + 'fitting_data/person_label.npy', allow_pickle=True))
trans_mats = np.load(folder + 'fitting_data/trans_mats.npy', allow_pickle=True)
shape_params = np.load(folder + 'fitting_data/shapes.npy', allow_pickle=True)
blendshape_params = np.load(folder + 'fitting_data/blendshapes.npy', allow_pickle=True)
vertices3d_all = np.load(folder + 'fitting_data/vertices3d.npy', allow_pickle=True)
vertices3d_indices = np.int32(np.load(folder + 'fitting_data/vertices3d_indices.npy', allow_pickle=True))
n_cams_used = np.int32(np.load(folder + 'fitting_data/n_cams_used.npy', allow_pickle=True))

num_people = shape_params.shape[0]

n_observations = person_label.size

blendshape_params = blendshape_params.reshape(n_observations, int(blendshape_params.size / n_observations))
trans_mats = trans_mats.reshape((n_observations, 3, 4))

shape_coeffs = np.copy(shape_params)
if shape_coeffs.shape[1] < n_shape_params:
    expand = np.zeros((num_people, n_shape_params - shape_coeffs.shape[1]))
    shape_coeffs = np.hstack((shape_coeffs, expand))


trans_matsBA = np.zeros((n_observations, 7))
blendshape_coeffs = np.zeros_like(blendshape_params)
shape_coeffs = np.zeros_like(shape_params)

t1 = time.time()

for person_idx in range(num_people):
    indices = np.squeeze(np.argwhere(person_label == person_idx))

    trans_mats_transform = []
    # initialize transformation matrices with values from preprocessing
    for idx, mat in enumerate(trans_mats[indices]):
        trans_mat = np.vstack((mat, np.array([0.0, 0.0, 0.0, 1.0])))
        T, R, Z, S = t3d.affines.decompose(trans_mat)
        R = t3d.quaternions.mat2quat(R)
        # trz = np.concatenate((T, R.flatten(), Z))
        tr = np.concatenate((T, R.flatten()))
        trans_matsBA[indices[idx]] = tr

    indices_for_init = np.round(np.linspace(0, len(indices) - 1, 10)).astype(int)
    indices_for_init = indices[indices_for_init]
    indices_for_init = indices
    lmarks_indices = []
    cam_indices = []
    for idx in indices_for_init:
        cams_prev = np.sum(n_cams_used[:idx])
        cams = camera_indices[cams_prev * 66:(cams_prev + n_cams_used[idx]) * 66]
        lms = vertices3d_all[idx * 66:(1 + idx) * 66]
        for lm in lms:
            lmarks_indices.append(lm)
        for cam in cams:
            cam_indices.append(cam)


    ba = optimize_3D_non_linear(np.asarray(lmarks_indices), shape_coeffs[person_idx], blendshape_coeffs[indices],
                                trans_matsBA[indices], morphablemodel_with_expressions, used_lms_numbers,
                                model_contour_lists, contour_lms, n_shape_params, n_blendshape_params, 66)
    shape_coeffs[person_idx], blendshape_coeffs[indices], trans_matsBA[indices] = ba.optimize()


t2 = time.time()
print('Time passed :', t2-t1)
np.savetxt(out_folder + 'shapes.txt', shape_coeffs, fmt='%1.3f')
trans_mats_composed = []
for idx, trans_mat in enumerate(trans_matsBA):
    trans_mat = t3d.affines.compose(trans_mat[:3], t3d.quaternions.quat2mat(trans_mat[3:7]),
                                    np.ones(3))
    trans_mats_composed.append(trans_mat[:3,:])
observations_before = 0

all_images = glob.glob(folder + '*.png')
for image in all_images:
    path, name = os.path.split(image)
    try:
        img = cv2.imread(image)
        cv2.imwrite(out_folder + name[:-3] + "jpg", img)
    except:
        print('Unable to copy image')

red = Color("red")
blue = Color("blue")
colors = [tuple(int(255 * v) for v in c.rgb) for c in list(red.range_to(blue, num_people))]
for idx, frame_number in enumerate(frame_numbers):
    print(frame_number)
    label = person_label[idx]
    observations = observations_before + 66

    transformation_mat_model_obj = trans_mats_composed[idx]

    mesh = morphablemodel_with_expressions.draw_sample(shape_coeffs[label].tolist(), blendshape_coeffs[idx].tolist(), [])
    vertices_transformed = []
    for vertex in mesh.vertices:
        vertex_hom = np.ones(4)
        vertex_hom[:3] = vertex
        vertex = transformation_mat_model_obj @ vertex_hom
        # vertex = vertex[:3]/vertex[3]
        vertices_transformed.append(vertex.astype('float32'))
    mesh.vertices = vertices_transformed

    for k, cam in enumerate(cams_BA):
        image = cv2.imread(out_folder + "cam" + str(k) + "frame" + str(frame_number) + ".jpg")
        if image is None:
            image = cv2.imread(folder + "cam" + str(k) + "frame" + str(frame_number) + ".jpg")
        dist = cam[-5:]
        trans = cam[3:6]
        rot = cam[:3]
        internal = np.zeros((3, 3))
        internal[0, 0] = cam[6]
        internal[1, 1] = cam[6]
        internal[0, 2] = cam[7]
        internal[1, 2] = cam[8]
        pnts2d, jac = cv2.projectPoints(np.asarray(vertices_transformed), rot, trans, internal, dist)
        pnts2d = (np.int32(np.reshape(pnts2d, (int(pnts2d.size / 2), 2))))
        for vertex in pnts2d:
            # vertex = np.round(vertex).astype(int)
            cv2.circle(image, tuple(vertex), 1, colors[label])
        cv2.imwrite(out_folder + "cam" + str(k) + "frame" + str(frame_number) + ".jpg", image)

    eos.core.write_obj(mesh,
                       out_folder + 'mesh_face' + str(label) + 'frame' + str(frame_number) + '.obj')
    np.savetxt(out_folder + 'transformation_mat_face' + str(label) + 'frame' + str(frame_number) + '.txt',
               transformation_mat_model_obj, fmt='%1.3f')
    np.savetxt(out_folder + 'blendshapes_face' + str(label) + 'frame' + str(frame_number) + '.txt',
               blendshape_coeffs[idx], fmt='%1.3f')
    observations_before = observations
