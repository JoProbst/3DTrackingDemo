import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from helpers_real import cosineSimilarity
from glob import glob
from scipy import spatial
import os
import shutil
import toml
import eos
import pandas as pd

def read_ground_truth(filename):
    df = pd.read_csv(filename, skiprows=5, header=None).fillna(0.0)
    tracks = df.iloc[:, 2:].to_numpy().reshape((750, 6, 3))
    return tracks[1::2,:,:]

model = eos.morphablemodel.load_model("share/sfm_shape_3448.bin")
blendshapes = eos.morphablemodel.load_blendshapes("share/expression_blendshapes_3448.bin")
# Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                    color_model=eos.morphablemodel.PcaModel(),
                                                                    vertex_definitions=None,
                                                                    texture_coordinates=model.get_texture_coordinates())

dirName = "/media/jonas/Elements/ViconDataSplit/ViconCombined"

# dirName = "D:/BirdTrackingProject/20180906_1BirdBackpack"
# sessionName = "TT_JoKö_2020_10_20_14_32_09"
# sessionName = "20180906_1BirdBackpack_Bird201"

dirName = "/media/jonas/Elements/ViconDataSplit/ViconCombined"
method = ""
folder_tracking = dirName + "/out_sfm_vicon_lam30/" + method

#shapes_baseline = [ [0.625, -1.6, 0.24, -0.15, 0.79], [-0.25, -0.59, -0.037, -0.23, 0.31]]
bs_prop_all = glob(folder_tracking + 'ble*face*')
shape_prop = np.loadtxt(folder_tracking + 'shapes.txt')
trans_prop_0 = glob(folder_tracking + 'tr*face*')

sfm_vertices_gt = np.array([658, 225, 114, 812, 398, 33], dtype=np.int)

blend_prop = []
trans_prop = []
for person in range(3):
    bs_prop = glob(folder_tracking + 'ble*face' + str(person) + "*")
    t_prop = glob(folder_tracking + 'tr*face' + str(person) + "*")
    blend_prop.append(bs_prop)
    trans_prop.append(t_prop)

person_gt = ["TT_JoKö_2020_10_20_14_32_04_Jonas Chair C.csv",
             "TT_JoKö_2020_10_20_14_32_06_Johanna Chair B.csv",
             "TT_JoKö_2020_10_20_14_32_08_Urs Chair A.csv"]

tracks_prop = np.ones((3, 375, 6, 3)) * np.nan

for person in range(3):
    for i in range(375):
        bs_prop = [s for s in blend_prop[person] if "e" + str(i) + "." in s]
        t_prop = [s for s in trans_prop[person] if "e" + str(i) + "." in s]
        if len(bs_prop) == 0:
            continue
        bs_prop = np.loadtxt(bs_prop[0])
        t_prop = np.loadtxt(t_prop[0])

        mesh_b = morphablemodel_with_expressions.draw_sample(shape_prop[person], bs_prop, [])
        vertices_transformed = []
        for vertex in np.array(mesh_b.vertices)[sfm_vertices_gt]:
            vertex_hom = np.ones(4)
            vertex_hom[:3] = vertex
            vertex = t_prop @ vertex_hom
            #vertex = vertex[:3]/vertex[3]
            vertices_transformed.append(vertex.astype('float32') )
        vert_b = np.array(vertices_transformed)
        tracks_prop[person,i] = vert_b

gt_all = np.ones((3, 375, 6, 3)) * np.nan

for person in range(3):
    ground_truth = read_ground_truth(
        filename="/media/jonas/Elements/ViconDataSplit/Facetracking VICON Mock up Daten/" + person_gt[person])
    #ground_truth = ground_truth[0:75]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ground_truth[0])
    o3d.io.write_point_cloud(folder_tracking + 'gt_vertices_face' + str(person) + 'frame' + str(0) + '.ply',
                             pcd)
    ground_truth[ground_truth == 0] = np.nan
    gt_all[person] = ground_truth

diff = gt_all-tracks_prop
diff_norm = np.linalg.norm(diff, axis=3)
diff_norm_person = np.nanmean(diff_norm, axis=1)
diff_norm_total = np.nanmean(diff_norm_person)
print(diff_norm_total)

diff_norm_over_time = np.nanmean(diff_norm, axis=2)

fig, ax = plt.subplots()
plt.xlim([0, 375])
ax.plot(np.arange(0,375), diff_norm_over_time[0], label='Person 0')
ax.plot(np.arange(0,375), diff_norm_over_time[1], label='Person 1')
#ax.plot(np.arange(0,375), diff_norm_over_time[2], label='Person 2')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.xlabel('Frame numbers', fontsize=18)
plt.ylabel('Average distance in mm', fontsize=16)
plt.title('Average error over time', fontsize=20)
ax.grid()
fig.savefig("avg_dist_person_base_lam30.png")
plt.show()



