import numpy as np


class WeightedIsomapAveraging:
    def __init__(self, merge_threshold=60.0):
        """

        :type merge_threshold: View angle threshold
        """
        self.merged_isomap = np.zeros((512, 512, 3))
        self.visibility_counter = np.zeros((512, 512))
        # scale merge threshold from 0-90 to 0-255
        self.merge_threshold = (-255.0 / 90.0) * merge_threshold + 255


    def add_and_merge(self, isomap):
        mask = np.where(isomap[:, :, 3] > self.merge_threshold, 1.0, 0.0)
        masks = np.stack((mask, mask, mask), axis=-1)
        isomap = np.array(isomap[:, :, :3], dtype=float) * masks
        vis_counter_stacked = np.stack((self.visibility_counter, self.visibility_counter, self.visibility_counter), axis=-1)
        current_isomaps = self.merged_isomap * vis_counter_stacked
        self.merged_isomap = (current_isomaps + isomap) / (vis_counter_stacked + 1)
        visible = np.where(np.any(isomap > 0, axis=-1), 1.0, 0.0)
        self.visibility_counter = self.visibility_counter + visible
        return np.array(self.merged_isomap, dtype=np.uint8)

    def get_merged_isomap(self):
        return np.array(self.merged_isomap, dtype=np.uint8)



class PcaCoefficientMerging:
    def __init__(self, ncoeffs):
        self.merged_coeffs = np.zeros(ncoeffs)
        self.num_processed_frames = 0.0

    def add_and_merge(self, coeffs):
        coeffs = np.asarray(coeffs)
        assert coeffs.shape == self.merged_coeffs.shape, "Shape of added coefficients is different than merged coefficients"
        self.merged_coeffs = (self.merged_coeffs * self.num_processed_frames + coeffs) / (self.num_processed_frames + 1.0)
        self.num_processed_frames += 1


