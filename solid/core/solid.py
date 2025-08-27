import numpy as np
from pydantic_settings import BaseSettings

class SOLiDModule:
    def __init__(self, config: BaseSettings):
        self.fov_u = config.fov_u
        self.fov_d = config.fov_d
        self.num_angle = config.num_angle
        self.num_range = config.num_range
        self.num_elevation = config.num_elevation
        self.max_length = config.max_distance

        self.gap_ring = self.max_length / self.num_range
        self.gap_sector = 360 / self.num_angle
        self.gap_height = (self.fov_u - self.fov_d) / self.num_elevation   

    def xy2theta(self, x, y):
        theta = np.empty_like(x, dtype=float)

        mask1 = (x >= 0) & (y >= 0)
        mask2 = (x < 0) & (y >= 0)
        mask3 = (x < 0) & (y < 0)
        mask4 = (x >= 0) & (y < 0)

        theta[mask1] = 180 / np.pi * np.arctan(y[mask1] / x[mask1])
        theta[mask2] = 180 - (180 / np.pi) * np.arctan(y[mask2] / (-x[mask2]))
        theta[mask3] = 180 + (180 / np.pi) * np.arctan(y[mask3] / x[mask3])
        theta[mask4] = 360 - (180 / np.pi) * np.arctan((-y[mask4]) / x[mask4])

        return theta

    def pt2rah(self, points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Avoid division by zero
        x = np.where(x == 0.0, 0.001, x)
        y = np.where(y == 0.0, 0.001, y)

        faraway = np.hypot(x, y)
        phi = np.degrees(np.arctan2(z, faraway)) - self.fov_d

        idx_ring = np.floor_divide(faraway, self.gap_ring).astype(np.int32)
        idx_height = np.floor_divide(phi, self.gap_height).astype(np.int32)

        np.clip(idx_ring, 0, self.num_range - 1, out=idx_ring)
        np.clip(idx_height, 0, self.num_elevation - 1, out=idx_height)

        return idx_ring, idx_height

    def get_descriptor(self, scan):
        rh_counter = np.zeros([self.num_range, self.num_elevation])             
        idx_rings, idx_heights = self.pt2rah(scan)
        rh_counter[idx_rings, idx_heights] = rh_counter[idx_rings, idx_heights] + 1
     
        number_vector = np.sum(rh_counter, axis=0)
        min_val = number_vector.min()
        max_val = number_vector.max()
        number_vector = (number_vector - min_val) / (max_val - min_val)
            
        r_solid = rh_counter.dot(number_vector)
        return r_solid

    def loop_detection(self, query, candidates):
        cosine_similarities = (query @ candidates.T) / (np.linalg.norm(query) * np.linalg.norm(candidates, axis=1))
        return cosine_similarities

    def pose_estimation(self, query, candidate):
        # Use broadcasting for efficiency
        rolled_queries = np.array([np.roll(query, i) for i in range(len(query))])
        initial_cosdist = np.sum(np.abs(candidate - rolled_queries), axis=1)
        angle_difference = np.argmin(initial_cosdist) * (360 / self.num_angle)
        return angle_difference
