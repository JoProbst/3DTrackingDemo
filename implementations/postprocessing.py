import numpy as np
import eos
import transforms3d as t3d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import os

from colour import Color
import cv2
import glob

def main():
    model = eos.morphablemodel.load_model("share/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("share/expression_blendshapes_3448.bin")
    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())


    nframes = 120
    nfaces = 2
    n_blendshapes = 6

    red = Color("red")
    blue = Color("blue")
    colors = [tuple(int(255 * v) for v in c.rgb) for c in list(red.range_to(blue, 3))]

    folder_tracking ="outBaseline/blender/"
    cams_BA = np.load(folder_tracking + "fitting_data/camsBA.npy", allow_pickle=True)
    write_images = True
    video_dir = "../ground_truth/blender_videos"
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.avi")))


    caps = []
    for idx, camera in enumerate(video_files):
        cap = cv2.VideoCapture(camera)
        caps.append(cap)

    folder_out = folder_tracking + "postprocessed/"
    if not os.path.isdir(folder_out):
        os.makedirs(folder_out)
    transformation_mats_detected = []
    blendshape_coeffs_mesh = []
    shape_coeffs_from_mesh = []

    missing_detection = []
    missing_start = []
    shape_coeffs_from_mesh = np.loadtxt(folder_tracking + 'shapes.txt')
    #print("Load parameters for detected faces...")
    for face in range(nfaces):
        missing_start.append(False)
        n_frames_detected = 0

        transformation_mats_detected.append([])
        blendshape_coeffs_mesh.append([])
        missing_detection.append([])
        for i in range(nframes):
            frame_num = i
            #fs = cv2.FileStorage(
            #    "mean_mesh_transformed/" + "transformation_matrix_face0" + str(face) + "_frame0" + str(i) + ".xml",
            #    cv2.FILE_STORAGE_READ)
            #fn = fs.getNode("transformation_matrix_face0" + str(face) + "_frame0" + str(i)).mat()
            path = folder_tracking + 'transformation_mat_face' + str(face) + 'frame' + str(frame_num) + '.txt'
            try:
                trans = np.loadtxt(path)
                if trans.shape == (3,4):
                    trans = np.vstack((trans, np.array([0.0, 0.0, 0.0, 1.0])))
                transformation_mats_detected[face].append(trans)
                missing_detection[face].append(False)
                n_frames_detected += 1
            except IOError:
                if len(transformation_mats_detected[face]) == 0:
                    missing_start[face] = True
                    missing_detection[face].append(True)
                    transformation_mats_detected[face].append(np.zeros((3,4)))
                    blendshape_coeffs_mesh[face].append(np.zeros(n_blendshapes))
                    continue
                transformation_mats_detected[face].append(transformation_mats_detected[face][i - 1])
                blendshape_coeffs_mesh[face].append(np.zeros(n_blendshapes))

                missing_detection[face].append(True)
                continue


            bs = getCoeffsFromFile_np(
                folder_tracking + "blendshape_coeffs_eos_face" + str(face) + "frame" + str(frame_num) + ".txt")
            #bs = getCoeffsFromFile_np(
            #        folder_tracking + "blendshapes_face" + str(face) + "frame" + str(frame_num) + ".txt")
            blendshape_coeffs_mesh[face].append(bs)

    for face in range(nfaces):
        #print("Fill missing values...")
        if missing_start[face]:
            for i in range(nframes):
                if missing_detection[face][i]:
                    continue
                else: #first detection
                    first_trans = transformation_mats_detected[face][i]
                    first_blend = blendshape_coeffs_mesh[face][i]
                    break
            for i in range(nframes):
                if missing_detection[face][i]:
                    transformation_mats_detected[face][i] = first_trans
                    blendshape_coeffs_mesh[face][i] = first_blend
                    continue
                else:
                    break
    # smooth the transformation matrices
    zoom = []
    for face in range(nfaces):
        zoom.append([])
        for i in range(nframes):
            T, R, Z, S = t3d.affines.decompose(transformation_mats_detected[face][i])
            zoom[face].append(Z)
        zoom[face] = np.median(np.asarray(zoom[face]), axis=0)
        for i in range(nframes):
            T, R, Z, S = t3d.affines.decompose(transformation_mats_detected[face][i])
            Z = zoom[face]
            transformation_mats_detected[face][i] = t3d.affines.compose(T, R, Z, S)

    for face in range(nfaces):
        #print("Fill missing values...")
        i = 0
        while i < nframes:
            if missing_detection[face][i]:
                j = i+1
                while j < nframes:
                    if missing_detection[face][j]:
                        j = j+1
                    else:
                        break
                if j == nframes:
                    break

                n_missing_frames = (j - (i-1))

                blendshapes_start = blendshape_coeffs_mesh[face][i-1]
                blendshapes_end = blendshape_coeffs_mesh[face][j]
                blendshapes_step_mesh = (blendshapes_end - blendshapes_start) / n_missing_frames

                #trans_mat_start = np.vstack((transformation_mats_detected[face][i-1], np.array([0.0, 0.0, 0.0, 1.0])))
                trans_mat_start = transformation_mats_detected[face][i-1]
                T_start, R_start, Z_start, S_start = t3d.affines.decompose(trans_mat_start)

                #trans_mat_end = np.vstack((transformation_mats_detected[face][j], np.array([0.0, 0.0, 0.0, 1.0])))
                trans_mat_end = transformation_mats_detected[face][j]
                T_end, R_end, Z_end, S_end = t3d.affines.decompose(trans_mat_end)

                translation_diff = T_end - T_start
                translation_diff_step = translation_diff / n_missing_frames

                # Rotation
                key_rots = Rot.from_matrix([R_start, R_end])
                key_times = [i-1, j]
                slerp = Slerp(key_times, key_rots)

                print(i)
                print(j)
                for k in range(i,j):
                    #trans_mat = np.vstack((transformation_mats_detected[face][k], np.array([0.0, 0.0, 0.0, 1.0])))
                    T, R, Z, S = t3d.affines.decompose(transformation_mats_detected[face][k])
                    T = T + (translation_diff_step * (k-i+1))
                    R = slerp(k).as_matrix()
                    Z = zoom[face]
                    transformation_mats_detected[face][k] = t3d.affines.compose(T, R, Z, S)
                    blendshape_coeffs_mesh[face][k] = blendshape_coeffs_mesh[face][k-1] + blendshapes_step_mesh

                i = j
            else:
                i = i+1

    #print("generate detected meshes")
    for face in range(nfaces):
        for frame in range(nframes):
            mesh = morphablemodel_with_expressions.draw_sample(shape_coeffs_from_mesh[face],
                                                               blendshape_coeffs_mesh[face][frame], ())
            vertices_transformed = []
            for vertex in mesh.vertices:
                vertex_hom = np.ones(4)
                vertex_hom[:3] = vertex
                vertex = transformation_mats_detected[face][frame] @ vertex_hom
                vertex = vertex[:3] / vertex[3]
                vertices_transformed.append(vertex.astype('float32'))
            mesh.vertices = vertices_transformed
            for cam_num, cam in enumerate(cams_BA):
                if write_images:
                    if face == 0:
                        ret, image = caps[cam_num].read()
                    else:
                        image = cv2.imread(folder_out + "cam" + str(cam_num) + "frame" + str(frame) + ".jpg")
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
                        cv2.circle(image, tuple(vertex), 1, colors[face])
                    cv2.imwrite(folder_out + "cam" + str(cam_num) + "frame" + str(frame) + ".jpg", image)
            eos.core.write_obj(mesh, folder_out + "face" + str(face) + "frame" + str(frame) + ".obj")



def getCoeffsFromFile(fileName):
    try:
        with open(fileName) as f:
            bs = f.read().splitlines()
            return list(map(float, bs))
    except:
        return []

def getCoeffsFromFile_np(fileName):
    try:
        coeffs = np.loadtxt(fileName)
        return coeffs
    except:
        return []
main()
