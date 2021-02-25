import eos
import numpy as np
import os

import scipy.interpolate
import transforms3d as t3d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

def main():
	model = eos.morphablemodel.load_model("share/sfm_shape_3448.bin")
	blendshapes = eos.morphablemodel.load_blendshapes("share/expression_blendshapes_3448.bin")
	# Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
	morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
																		color_model=eos.morphablemodel.PcaModel(),
																		vertex_definitions=None,
																		texture_coordinates=model.get_texture_coordinates())

	#shape = (0.0943917, 0.47008, -1.34564, -0.564144, 0.464408)  # jonas
	#shape = (0.0883723, 0.359962, -0.394978, -1.08614, 1.04865)  # gab
	shape0 = (2, -1, 2, 0, 1)  # gab
	shape1 = (-1, 2, 1, -1, 0.5)  # jonas
	file_name = "face_jonas_sfm"
	folder = "face_jonas_sfm_real_size/"


	if not os.path.isdir(folder):
		os.makedirs(folder)
	file_name = folder + file_name
	nframes = 120
	steps = range(0,nframes+1, int(nframes / 6))
	steps = np.asarray(steps)

	shape = shape1

	# 0 minus:mund auf plus:mund zu
	# 1-4 mundwinkel
	# 5 kinn verschieben
	# 6 lippen
	# 7 egal
	#8 mund auf
	# 9 augenbrauen
	bs = np.zeros((7, 6))
	bs[0] = np.zeros(6)
	bs[1, :] = [2, 0, 0, -0, 0, 0]
	bs[2, :] = [1, 0, 0, -0, 0, 0]
	bs[3, :] = [0, 0, 2, -0, 0, 0]
	bs[4, :] = [0, 0, 1, -0, 0, 0]
	bs[5, :] = [0, 0, 0, -0, 2, 0]
	bs[6, :] = [0, 0, 0, -0, 1, 0]
	bs = scipy.interpolate.interp1d(steps, bs, axis=0)
#
	trans = np.zeros(((7, 3)))
	trans[0] = [0.5, -0.5, 1]
	trans[1] = [0.5, -0.7, 1.2]
	trans[2] = [0.4, -0.4, 0.8]
	trans[3] = [0.4, -0.6, 0.8]
	trans[4] = [0.6, -0.5, 1]
	trans[5] = [0.7, -0.7, 0.7]
	trans[6] = [0.5, -0.5, 1]
	trans[:,1] = trans[:,1] + 0.25
	trans = trans * 1000
	trans = scipy.interpolate.interp1d(steps, trans, axis=0)

	key_rots = Rot.from_euler("xyz", [(75, 0, 210), (80, 0, 220), (70, 10, 250), (75, 0, 230), (80, 0, 220), (90, -20, 215), (90, 0, 250)], degrees=True)
	slerp = Slerp(steps, key_rots)

	#bs = np.zeros((7, 6))
	#bs[0] = np.zeros(6)
	#bs[1, :] = [0, 1.5, 0, -0, 0, 0]
	#bs[2, :] = [1, 0, 0, -0, 0, 0]
	#bs[3, :] = [0, 0, 0, -0, 0, 1.5]
	#bs[4, :] = [0, 0, 0, -0, 0, 1]
	#bs[5, :] = [0, 0, 0, 1, 0, 0]
	#bs[6, :] = [0, 0, 0, 0, 1, 0]
	#bs = scipy.interpolate.interp1d(steps, bs, axis=0)
	#trans = np.zeros(((7, 3)))
	#trans[0] = [0.5, 0.7, 1]
	#trans[1] = [0.5, 0.6, 1.05]
	#trans[2] = [0.4, 0.5, 0.8]
	#trans[3] = [0.4, 0.8, 0.8]
	#trans[4] = [0.6, 0.7, 1]
	#trans[5] = [0.7, 0.7, 0.7]
	#trans[6] = [0.5, 0.8, 1]
	#trans[:,0] = trans[:,0] - 0.7
	#trans = scipy.interpolate.interp1d(steps, trans, axis=0)
	## Rotation
	#key_rots = Rot.from_euler("xyz", [(75, 0, 15), (80, 0, 25), (70, 10, 25), (75, 0, 25), (80, 0, 20), (90, -20, 25), (90, 0, 30)], degrees=True)
	#slerp = Slerp(steps, key_rots)

	np.savetxt(folder + 'shape_original.txt', shape, fmt='%1.3f')
	for j in range(nframes):
		mesh = morphablemodel_with_expressions.draw_sample(shape, bs(j), ())
		verts = np.array(mesh.vertices)# * 0.001
		rot = slerp(j).as_matrix()
		verts_trans = (rot @ verts.T).T + trans(j)
		mesh.vertices = verts_trans.tolist()
		eos.core.write_textured_obj(mesh, file_name + str(j) + ".obj")
		f = open(file_name + str(j) + ".mtl", "w")
		f.write("newmtl FaceTexture\nmap_Kd gab.isomap.png")
		f.close()
		np.savetxt(folder + 'blendshapes_original' + str(j) + '.txt', bs(j), fmt='%1.3f')
		np.savetxt(folder + 'transmat_original' + str(j) + '.txt', np.hstack([rot, trans(j)[:, np.newaxis]]), fmt='%1.3f')



if __name__ == "__main__":
	main()
