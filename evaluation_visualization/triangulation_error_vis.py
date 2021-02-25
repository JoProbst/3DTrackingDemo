import numpy as np
import eos
import transforms3d as t3d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import os
import open3d as o3d

from colour import Color
import pptk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx

from mpl_toolkits.mplot3d import Axes3D
from transforms3d.euler import euler2mat, mat2euler
import toml
import json

def visualize_vertices_with_mapping():
    model = eos.morphablemodel.load_model("../implementations/share/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("../implementations/share/expression_blendshapes_3448.bin")
    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())
    lms = toml.load('../implementations/share/ibug_to_sfm.txt')['landmark_mappings']
    with open('../implementations/share/sfm_model_contours.json') as file:
        contours = json.load(file)['model_contour']

    mean_mesh = morphablemodel_with_expressions.draw_sample([], [], [])
    green = Color("lime")
    colors = list(green.range_to(Color("red"), 100))
    colors = [col.rgb for col in colors]

    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud(folder + "mean_trian_error.ply", pcd)

    x_angle = -np.pi/2
    y_angle = -np.pi/2
    z_angle = 0
    R = euler2mat(x_angle, y_angle, z_angle, 'sxyz')
    vertics = np.asarray(mean_mesh.vertices) @ R

    contours_combined = contours['left_contour'] + contours['right_contour']
    colors = np.ones(vertics.shape[0])
    colors[list(lms.values())] = 0
    colors[contours_combined] = 2

    cols = ['tab:green', 'tab:gray', 'tab:red']
    colors_simple = []
    for col in colors:
        colors_simple.append(cols[int(col)])
    size = np.ones(vertics.shape[0])
    size[list(lms.values())] = 30
    size[contours_combined] = 30

    sort_zip = sorted(zip(size, colors_simple, vertics),
                      key=lambda pair: pair[0], reverse=False)
    size, colors, vertics = zip(*sort_zip)
    vertics = np.array(vertics)
    size = np.array(size)
    colors = np.array(colors)
    scatter3d_mappings(vertics[:,0], vertics[:,1], vertics[:,2], colors, size, proj='persp', colorBar=False)

def scatter3d_mappings(x, y, z, cs, size, proj='ortho', colorBar=True, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    #cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig, azim=0, elev=0, proj_type=proj)
    scatter = ax.scatter(x, y, z, s=size, c=cs, depthshade=False)

    plt.axis('off')
    colors = ['tab:green', 'tab:red', 'tab:gray']
    texts = ["Inner Landmarks", "Contour Landmarks", "No Mapping"]
    patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=colors[i],
                        label="{:s}".format(texts[i]))[0] for i in range(len(texts))]
    plt.legend(handles=patches, ncol=1, numpoints=1, loc='upper center')
    plt.gca().set_axis_off()
    plt.savefig('visualizations/lms_to_vertices_correspondence.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_mean_dists():
    model = eos.morphablemodel.load_model("../implementations/share/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("../implementations/share/expression_blendshapes_3448.bin")
    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())
    model_contour = eos.fitting.ModelContour.load('../implementations/share/sfm_model_contours.json')
    lms = toml.load('../implementations/share/ibug_to_sfm.txt')['landmark_mappings']


    folder = "mesh_euclidean_error/"
    type = 'real'
    d1 = np.load(folder + type +"_2D_iterative.npy")
    d2 = np.load(folder + type +"_3D_iterative.npy")
    d3 = np.load(folder + type +"_2D_nonlin.npy")
    d4 = np.load(folder + type +"_3D_nonlin.npy")
    d5 = np.load(folder + type +"_baseline.npy")


    all_d = [d1, d2, d3, d4, d5]
    all_means = np.asarray(all_d).flatten()

    #all_means = np.mean(means, axis=0)
    titles = ["2D linear", "3D linear", "2D non-linear", "3D non-linear", "Baseline"]
    for idx, d in enumerate(all_d):
        total_mean = d

        mean_mesh = morphablemodel_with_expressions.draw_sample([], [], [])
        green = Color("lime")
        colors = list(green.range_to(Color("red"), 100))
        colors = [col.rgb for col in colors]

        max = np.max(all_means)
        all_dist = np.around((all_means / max), decimals=2) * 10
        dist_frame = np.around((total_mean / max), decimals=2) * 10

        x_angle = -np.pi/2
        y_angle = -np.pi/2
        z_angle = 0
        R = euler2mat(x_angle, y_angle, z_angle, 'sxyz')
        vertics = np.asarray(mean_mesh.vertices) @ R

        scatter3d(vertics[:,0], vertics[:,1], vertics[:,2], total_mean, all_means, folder, titles[idx])

def scatter3d(x, y, z, cs, all_cs, folder, title, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(all_cs), vmax=max(all_cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig, azim=0, elev=0, proj_type='ortho')
    ax.scatter(x, y, z, s=2, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)

    cb = fig.colorbar(scalarMap, shrink=0.7)
    cb.set_label('Mean distance to ground truth in mm', fontsize=16)
    plt.axis('off')
    plt.title(title, fontsize=24)
    plt.margins(0, 0, 0)
    plt.savefig('visualizations/mean_mesh_diff_real' + title + '.png')
    plt.show()




if __name__ == "__main__":
    visualize_mean_dists()
    visualize_vertices_with_mapping()