import numpy as np
from helpers_evaluation import cosineSimilarity
from glob import glob
import eos

model = eos.morphablemodel.load_model("../implementations/share/sfm_shape_3448.bin")
blendshapes = eos.morphablemodel.load_blendshapes("../implementations/share/expression_blendshapes_3448.bin")
# Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                    color_model=eos.morphablemodel.PcaModel(),
                                                                    vertex_definitions=None,
                                                                    texture_coordinates=model.get_texture_coordinates())
shape1 = [2, -1, 2, 0, 1]  # gab
shape0 = [-1, 2, 1, -1, 0.5] #jonas

gt_dir0 = '../ground_truth/blender_videos/face_jonas_sfm'
gt_dir1 = '../ground_truth/blender_videos/face_gab_sfm'

shapes_gt = [shape0, shape1]
bs_gt_0 = glob(gt_dir0 +'/ble*')
bs_gt_1 = glob(gt_dir1 +'/ble*')
trans_gt0 = glob(gt_dir0 +'/tr*')
trans_gt1 = glob(gt_dir1 +'/tr*')

dirName = "../implementations/outBaseline/blender/"
method = ""
folder_tracking0 = dirName + method
folder_tracking1 = dirName + method

bs_prop_0 = glob(folder_tracking0 + 'ble*face0*')
bs_prop_1 = glob(folder_tracking0 + 'ble*face1*')
shape_prop0 = np.loadtxt(folder_tracking0 + 'shapes.txt')[0]
shape_prop1 = np.loadtxt(folder_tracking0 + 'shapes.txt')[1]
shape_prop = [shape_prop0, shape_prop1]

trans_prop_0 = glob(folder_tracking0 + 'tr*face0*')
trans_prop_1 = glob(folder_tracking0 + 'tr*face1*')

blend_gt = [bs_gt_0, bs_gt_1]
trans_gt = [trans_gt0, trans_gt1]
blend_prop = [bs_prop_0, bs_prop_1]
trans_prop = [trans_prop_0, trans_prop_1]
total_l2 = np.ones((2,120)) * np.nan
total_cosine = np.ones((2,120)) * np.nan
bs_all = np.zeros((2,120,2, 6)) * np.nan
euc_dist = np.zeros((2,120)) * np.nan
euc_dist_xy = np.zeros((2,120)) * np.nan
bs_norm = np.zeros((2,120)) * np.nan

dist_verts = np.zeros((2, 120, 3448)) * np.nan
for person in range(2):
    for i in range(120):
        bs_gt = [s for s in blend_gt[person] if "l" + str(i) + "." in s]
        bs_gt = np.loadtxt(bs_gt[0])
        t_gt = [s for s in trans_gt[person] if "l" + str(i) + "." in s]
        t_gt = np.loadtxt(t_gt[0])
        bs_all[person,i, 0] = bs_gt
        bs_prop = [s for s in blend_prop[person] if "e" + str(i) + "." in s]
        t_prop = [s for s in trans_prop[person] if "e" + str(i) + "." in s]
        if len(bs_prop) == 0:
            continue
        bs_prop = np.loadtxt(bs_prop[0])
        t_prop = np.loadtxt(t_prop[0])
        bs_all[person, i, 1] = bs_prop
        bs_norm[person, i] = np.linalg.norm(bs_prop, ord=1)

        cos = cosineSimilarity(bs_prop, bs_gt)
        euc = np.linalg.norm((bs_gt-bs_prop))
        total_l2[person, i] = euc
        total_cosine[person, i] = cos

        mesh_g = morphablemodel_with_expressions.draw_sample(shapes_gt[person], bs_gt, [])
        vertices_transformed = []
        t_gt[:, 3] = t_gt[:, 3]
        for vertex in mesh_g.vertices:
            vertex_hom = np.ones(4)
            vertex_hom[:3] = vertex
            vertex = t_gt @ vertex_hom
            # vertex = vertex[:3]/vertex[3]
            vertices_transformed.append(vertex.astype('float32'))
        vert_g = np.array(vertices_transformed)
        mesh_b = morphablemodel_with_expressions.draw_sample(shape_prop[person], bs_prop, [])
        vertices_transformed = []
        for vertex in mesh_b.vertices:
            vertex_hom = np.ones(4)
            vertex_hom[:3] = vertex
            vertex = t_prop @ vertex_hom
            #vertex = vertex[:3]/vertex[3]
            vertices_transformed.append(vertex.astype('float32') )
        vert_b = np.array(vertices_transformed)
        euc_dist[person, i] = np.mean(np.linalg.norm((vert_g-vert_b), axis=1))
        dist_verts[person, i] = np.linalg.norm((vert_g-vert_b), axis=1)

print('blendshape cos', np.nanmean(total_cosine))

print('blendshape norm ', np.nanmean(bs_norm))
print('euc', np.nanmean(euc_dist))
dists_mean = np.nanmean(dist_verts, axis=1)
np.save('mesh_euclidean_error/' + 'real_baseline.npy', np.mean(dists_mean, axis=0))

euc_shape = []
cos_shape = []
norm = []
for i in range(2):
    mesh_g = morphablemodel_with_expressions.draw_sample(shapes_gt[i], [], [])
    vert_g = np.array(mesh_g.vertices)
    mesh_b = morphablemodel_with_expressions.draw_sample(shape_prop[i], [], [])
    vert_b = np.array(mesh_b.vertices)
    euc_dist = np.mean(np.linalg.norm((vert_g - vert_b), axis=1))
    norm.append(np.linalg.norm(shape_prop[i]))
    cos = cosineSimilarity(shapes_gt[i], shape_prop[i])
    euc_shape.append(euc_dist)
    cos_shape.append(cos)
print('shape norm ', np.mean(norm))
print('shape cos ', np.mean(cos_shape))
