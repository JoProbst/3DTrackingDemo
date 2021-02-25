import os
import cv2
import eos
import face4d
import numpy as np
from colour import Color

from helpers.bundle_adjustment_fixed_camera import ba_fixed_cam
import helpers.helpers_tracking as helpers
import helpers.mergings as mergings


def main():
    # Load Morphable Model and corresponding mappings
    model = eos.morphablemodel.load_model("share/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("share/expression_blendshapes_3448.bin")
    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())
    landmark_mapper = eos.core.LandmarkMapper('share/ibug_to_sfm.txt')
    edge_topology = eos.morphablemodel.load_edge_topology('share/sfm_3448_edge_topology.json')
    contour_landmarks = eos.fitting.ContourLandmarks.load('share/ibug_to_sfm.txt')
    model_contour = eos.fitting.ModelContour.load('share/sfm_model_contours.json')
    n_blendshape_coeffs = 6
    no_mapping = [61, 65]
    lms_with_mapping = [l for l in (range(68)) if l not in no_mapping]
    ignored_landmarks = list(range(1, 9)) + list(range(10, 18)) + no_mapping
    used_lms = []
    used_lms_int = []
    for lm in range(1, 69):
        if lm not in ignored_landmarks:
            used_lms_int.append(lm - 1)
            used_lms.append(str(lm))
    used_vertex_indices = [int((landmark_mapper.convert(lm))) for lm in used_lms]

    # Define fonts for writing on images
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    fontColor = (0, 0, 0)
    lineType = 1
    thickness = 2

    # Parameters for the face and landmark detection
    n_people = 2
    n_shape_coeffs = 5
    shape_lambda_eos = 30
    visualise_fitted_mesh = False
    skip_frames = 1
    confidence_threshold = 0.9
    size_threshold = 3000
    project_triangulated_face = False

    data_type = "blender"
    out_folder = "outBaseline/" + data_type + "/"

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # load video files and camera calibrations
    caps = []
    dists_coeffs = []
    projection = []
    if data_type == "vicon":
        for i in range(4):
            cap = cv2.VideoCapture("../ground_truth/vicon_videos/cam0" + str(i) + ".avi")
            caps.append(cap)
            P = np.loadtxt('../ground_truth/vicon_videos/proj_mat' + str(i) + '.txt')
            projection.append(P)
            dist = np.loadtxt('../ground_truth/vicon_videos/distortion_coeffs' + str(i) + '.txt')
            dists_coeffs.append(dist)
    elif data_type == 'blender':
        for i in range(4):
            cap = cv2.VideoCapture("../ground_truth/blender_videos/cam0" + str(i) + "sfm.avi")
            caps.append(cap)
            P = np.loadtxt('../ground_truth/blender_videos/P3x4' + str(i) + '.txt')
            P[:,3] = P[:,3] * 1000
            projection.append(P)
            dists_coeffs.append(np.zeros(5))

    # load previously defined track centers, or manually select the faces on each camera
    track_centers = np.zeros((len(caps), n_people, 2))
    try:
        track_centers = np.load(out_folder + "track_centers.npy")
    except:
        for l in range(n_people):
            for j, cap in enumerate(caps):
                ret, frame = cap.read()
                cv2.namedWindow("Select Person " + str(l + 1) + " on camera " + str(j), flags=cv2.WINDOW_NORMAL)
                r = cv2.selectROI("Select Person " + str(l + 1) + " on camera " + str(j), frame,
                                  showCrosshair=True)
                track_centers[j, l][0] = r[0] + r[2] / 2
                track_centers[j, l][1] = r[1] + r[3] / 2
                cv2.destroyAllWindows()
        np.save(out_folder + "track_centers.npy", np.asarray(track_centers))


    # decompose projection cameras, needed for opencv project points
    cam_for_BA = []
    for idx, proj in enumerate(projection):
        cam_ba = helpers.projection_decomposed_format(proj, dists_coeffs[idx])
        cam_for_BA.append(cam_ba)
    cam_for_BA = np.asarray(cam_for_BA)

    landmark_detector = face4d.detection.CNNLandmarkDetector("share/detectors/landmark_detector_large.json")
    face_detector = face4d.detection.CNNFaceDetector(pnet_filename="share/detectors/facedetector_pnet.json",
                                                     rnet_filename="share/detectors/facedetector_rnet.json",
                                                     onet_filename="share/detectors/facedetector_onet.json",
                                                     min_face_size=50)

    pcaShapeCoefficientMergings_4dface = []
    pcaShapeCoefficientMergings_Mesh = []
    transformation_matrices = []
    blendshapes_prev = []
    red = Color("red")
    blue = Color("blue")
    colors = [tuple(int(255 * v) for v in c.rgb) for c in list(red.range_to(blue, n_people))]
    for i in range(n_people):
        pcaShapeCoefficientMergings_4dface.append(mergings.PcaCoefficientMerging(ncoeffs=n_shape_coeffs))
        pcaShapeCoefficientMergings_Mesh.append(mergings.PcaCoefficientMerging(ncoeffs=n_shape_coeffs))
        transformation_matrices.append(np.eye(N=3, M=4))
        blendshapes_prev.append(np.zeros(n_blendshape_coeffs))

    # Lists used to save parameters needed for the 2D and 3D optimization methods
    vertices3d = []
    vertices3d_indices = []
    camera_indices = []
    vertices2d_coords = []
    frame_numbers = []
    used_lms_numbers = []
    observations_used = []
    person_label = []
    all_trans_mats = []
    all_blendshapes = []
    n_cams_used = []
    frame_number = 0
    while True:
        print("Frame number " + str(frame_number))

        pcaBlendshapeMergings = []
        all_vertices2d = []
        all_vertices2d_names = []
        all_poses = []
        detected = []
        all_shape_coeffs = []
        all_blendshape_coeffs = []
        all_confidence = []

        for i in range(n_people):
            pcaBlendshapeMergings.append(mergings.PcaCoefficientMerging(ncoeffs=n_blendshape_coeffs))

        for cam_num, cap in enumerate(caps):
            all_vertices2d.append([])
            all_vertices2d_names.append([])
            detected.append([])
            all_poses.append([])
            all_shape_coeffs.append([])
            all_blendshape_coeffs.append([])
            all_confidence.append([])
            for i in range(n_people):
                detected[cam_num].append(False)
                all_poses[cam_num].append([])
                all_vertices2d_names[cam_num].append([])
                all_vertices2d[cam_num].append([])
                all_shape_coeffs[cam_num].append([])
                all_blendshape_coeffs[cam_num].append([])
                all_confidence[cam_num].append([])

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            returned, frame = cap.read()
            if not returned or frame is None:
                SystemExit(0)

            frame_out = frame.copy()  # keep the output image in BGR order for displaying
            frame = frame[..., [2, 1, 0]]  # convert BGR to RGB for the 4dface SDK

            dets = face_detector.detect(frame)
            cv2.imwrite(out_folder + "cam" + str(cam_num) + "frame" + str(frame_number) + ".jpg", frame_out)
            for i, det in enumerate(dets):
                det = eos.core.Rect(x=int(det.x), y=int(det.y),
                                    width=det.width, height=det.height)
                confidence = face_detector.estimate_confidence(det, frame)
                center = [det.x + 0.5 * det.width, det.y + 0.5*det.height]
                dists = np.linalg.norm(track_centers[cam_num] - center, axis=1)
                if np.min(dists) > 250:
                    label = -1
                else:
                    label = np.argmin(dists)
                    # Check if face was already detected with higher confidence in that image
                    if len(all_confidence[cam_num][label]) > 0:
                        if confidence < all_confidence[cam_num][label][0]:
                            label = -1
                # center_face = np.asarray([rect.x + rect.width / 2, rect.y + rect.height / 2])
                if label == -1:
                    print("Outlier detected")
                    frame_out = cv2.circle(frame_out, (int(center[0]), int(center[1])), 5, (0,0,255))
                    cv2.putText(frame_out, 'Dist' + str(np.min(dists)),
                               ((int(center[0]+10), int(center[1]))),
                               font,
                               fontScale,
                               fontColor,
                               thickness,
                               lineType)
                    cv2.imwrite(out_folder + "cam" + str(cam_num) + "frame" + str(frame_number) + ".jpg", frame_out)
                    continue
                rescaled_facebox = face4d.detection.rescale_bbox(face4d.detection.make_bbox_square(det),
                                                                 landmark_detector.get_facebox_rescale_factor())
                lms = landmark_detector.detect(frame, rescaled_facebox)
                rect = face4d.detection.get_enclosing_bbox(lms)
                size = rect.width * rect.height
                if confidence < confidence_threshold or size < size_threshold:
                    continue
                if lms == []:
                    continue
                lm_coords = []
                lm_names = []
                for lm in lms:
                    lm_names.append(lm.name)
                    lm_coords.append(lm.coordinates)

                (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(
                    morphablemodel_with_expressions,
                    lms,
                    landmark_mapper,
                    frame.shape[1],
                    frame.shape[0],
                    edge_topology,
                    contour_landmarks,
                    model_contour,
                    num_iterations=5,
                    num_shape_coefficients_to_fit=n_shape_coeffs,
                    lambda_identity=shape_lambda_eos,
                    num_expression_coefficients_to_fit=n_blendshape_coeffs,
                    lambda_expressions=30.0)
                pcaShapeCoefficientMergings_4dface[label].add_and_merge(shape_coeffs)
                pcaBlendshapeMergings[label].add_and_merge(blendshape_coeffs)

                if visualise_fitted_mesh:
                    # Visualise the projected vertices (coarse 3D face fitting):
                    vertices2d = helpers.get_vertices2d(np.asarray(mesh.vertices), pose.get_modelview(),
                                                        pose.get_projection(),
                                                        image_width=frame_out.shape[1],
                                                        image_height=frame_out.shape[0])
                    for vertex in vertices2d:
                        cv2.circle(frame_out, tuple(np.round(vertex[0:2]).astype(int)), 1, colors[label])
                for lm in lms:
                    cv2.circle(frame_out, tuple(np.round(lm.coordinates).astype(int)), 2, colors[label])
                cv2.putText(frame_out, 'Confidence, Size:',
                            (int(lm_coords[7][0] - 70), int(lm_coords[7][1] + 40)),
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                cv2.putText(frame_out, str(confidence)[:5] + ', ' + str(size),
                            (int(lm_coords[7][0] - 70), int(lm_coords[7][1] + 70)),
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                # Mark face as detected for this camera, add 2d landmarks and other values to list for triangulation
                detected[cam_num][label] = True
                all_vertices2d_names[cam_num][label].append(lm_names)
                all_vertices2d[cam_num][label].append(lm_coords)
                all_confidence[cam_num][label] = confidence
                cv2.imwrite(out_folder + "cam" + str(cam_num) + "frame" + str(frame_number) + ".jpg", frame_out)

        # All faces and landmarks for each camera detected, start triangulation
        for label in range(n_people):
            cams_for_triangulation = []
            vertices2d = []
            vertices2d_names = []
            for cam in range(len(caps)):
                if detected[cam][label]:
                    cams_for_triangulation.append(cam)
                    vertices2d.append(np.asarray(all_vertices2d[cam][label][0]))
                    vertices2d_names.append(all_vertices2d_names[cam][label])

            # if the face was detected on less than 2 cameras, continue with next face
            num_candidates = len(cams_for_triangulation)
            if num_candidates == 0:
                print("Face " + str(label) + " not detected.")
                continue
            elif num_candidates == 1:
                print("Face " + str(label) + " detected on one camera.")
                # skip
                continue
            elif num_candidates >= 2:
                print("Face " + str(label) + " detected on " + str(num_candidates) + " cameras.")

            cam_num1 = cams_for_triangulation[0]
            cam_num2 = cams_for_triangulation[1]

            p3d = cv2.triangulatePoints(projection[cam_num1], projection[cam_num2],
                                        vertices2d[0].T, vertices2d[1].T)
            p3d_euclid = np.squeeze(cv2.convertPointsFromHomogeneous(p3d.T))

            if len(cams_for_triangulation) > 2:
                # Face was detected on more than two cameras, optimize 3D landmark positions using bundle adjustment
                # with fixed cameras
                cams_indices = []
                point_indices = []
                landmarks_2d = []
                for cam in cams_for_triangulation:
                    landmarks_2d.append(all_vertices2d[cam][label][0])
                    for i in range(len(vertices2d[0])):
                        cams_indices.append(cam)
                        point_indices.append(i)

                p3d_euclid =\
                    ba_fixed_cam(cam_for_BA, p3d_euclid, landmarks_2d, np.array(cams_indices), np.array(point_indices))
            n_cams_used.append(len(cams_for_triangulation))

            # Project centroid of all vertices to all cameras, update track center for label
            for cam_num, cam_BA in enumerate(cam_for_BA):
                centroid = np.mean(p3d_euclid, axis=0)
                dist = cam_BA[-5:]
                trans = cam_BA[3:6]
                rot = cam_BA[:3]
                internal = np.zeros((3, 3))
                internal[0, 0] = cam_BA[6]
                internal[1, 1] = cam_BA[6]
                internal[0, 2] = cam_BA[7]
                internal[1, 2] = cam_BA[8]
                pnts2d, jac = cv2.projectPoints(centroid, rot, trans, internal, dist)
                track_centers[cam_num][label] = np.squeeze(pnts2d)


            if len(vertices3d_indices) > 0:
                point_ids = np.asarray(range(len(lms_with_mapping))) + np.amax(vertices3d_indices[-1]) + 1
            else:
                point_ids = np.asarray(range(len(lms_with_mapping)))

            for j, cam_num in enumerate(cams_for_triangulation):
                camera_indices.append(np.ones(len(lms_with_mapping), dtype='int') * cam_num)
                vertices2d_coords.append(vertices2d[j][lms_with_mapping])
                vertices3d_indices.append(point_ids)

            mesh = morphablemodel_with_expressions.draw_sample(pcaShapeCoefficientMergings_Mesh[label].merged_coeffs,
                                                               pcaBlendshapeMergings[label].merged_coeffs, [])
            vertices_obj_space = np.asarray(mesh.vertices)
            vertices_obj_space_use = vertices_obj_space[used_vertex_indices]

            p3d_with_mapping = p3d_euclid[used_lms_int]
            transformation_mat_model_obj = helpers.ralign((vertices_obj_space_use.T), (p3d_with_mapping.T))

            vertices_transformed = []
            for idx, vertex in enumerate(mesh.vertices):
                vertex_hom = np.ones(4)
                vertex_hom[:3] = vertex
                vertex = transformation_mat_model_obj @ vertex_hom
                vertices_transformed.append(vertex.astype('float32'))
            mesh.vertices = vertices_transformed

            # add detection parameters to optimization problem
            all_trans_mats.append(transformation_mat_model_obj)
            frame_numbers.append(frame_number)
            observations_used.append(len(used_lms))
            person_label.append(label)
            vertices3d.append(p3d_euclid[[lms_with_mapping]])

            all_blendshapes.append(pcaBlendshapeMergings[label].merged_coeffs)
            used_lms_numbers.append(used_vertex_indices)

            if project_triangulated_face:
                for camera_number, cam_BA in enumerate(cam_for_BA):
                    try:
                        image = cv2.imread(
                            out_folder + "cam" + str(camera_number) + "frame" + str(frame_number) + ".jpg")
                        dist = cam_BA[-5:]
                        trans = cam_BA[3:6]
                        rot = cam_BA[:3]
                        internal = np.zeros((3, 3))
                        internal[0, 0] = cam_BA[6]
                        internal[1, 1] = cam_BA[6]
                        internal[0, 2] = cam_BA[7]
                        internal[1, 2] = cam_BA[8]
                        pnts2d, jac = cv2.projectPoints(np.asarray(vertices_transformed), rot, trans, internal, dist)
                        pnts2d = (np.int32(np.reshape(pnts2d, (int(pnts2d.size / 2), 2))))
                        for vertex in pnts2d:
                            # vertex = np.round(vertex).astype(int)
                            cv2.circle(image, tuple(vertex), 1, colors[label])
                        cv2.imwrite(out_folder + "cam" + str(camera_number) + "frame" + str(frame_number) + ".jpg",
                                    image)
                    except Exception as e:
                        print(e)
                        print('Error in projecting mesh to image')

            # Save the constructed and transformed mesh as obj file.
            # Save the corresponding shape and blendshape coefficients and the transformation matrix
            eos.core.write_obj(mesh,
                               out_folder + 'mesh_face' + str(label) + 'frame' + str(frame_number) + '.obj')
            np.savetxt(out_folder + 'transformation_mat_face' + str(label) + 'frame' + str(frame_number) + '.txt',
                       transformation_mat_model_obj, fmt='%1.3f')
            np.savetxt(out_folder + 'shape_coeffs_eos_face' + str(label) + 'frame' + str(frame_number) + '.txt',
                       pcaShapeCoefficientMergings_4dface[label].merged_coeffs)
            np.savetxt(out_folder + 'blendshape_coeffs_eos_face' + str(label) + 'frame' + str(frame_number) + '.txt',
                       pcaBlendshapeMergings[label].merged_coeffs, fmt='%1.3f')

        if not os.path.isdir(out_folder + 'fitting_data/'):
            os.makedirs(out_folder + 'fitting_data/')

        # save all data needed for the 2D and 3D fitting methods
        try:
            file_names = ['camsBA', 'vertices3d', 'vertices2d_coords',
                          'camera_indices', 'vertices3d_indices', 'used_lms_numbers', 'frame_numbers',
                          'observations_used', 'person_label', 'trans_mats', 'shapes', 'blendshapes',
                          'n_cams_used']
            for idx, data in enumerate([cam_for_BA, np.concatenate(vertices3d), np.concatenate(vertices2d_coords),
                                        np.concatenate(camera_indices), np.concatenate(vertices3d_indices),
                                        np.concatenate(used_lms_numbers), np.asarray(frame_numbers),
                                        np.asarray(observations_used), np.asarray(person_label),
                                        np.concatenate(all_trans_mats, axis=0),
                                        np.array([pca.merged_coeffs for pca in pcaShapeCoefficientMergings_4dface]),
                                        np.concatenate(all_blendshapes), np.array(n_cams_used, dtype=int)]):
                np.save(out_folder + 'fitting_data/' + file_names[idx] + '.npy', data)
        except Exception as e:
            print(e)
            print("Error in saving fitting data")
        frame_number += skip_frames


if __name__ == "__main__":
    main()
