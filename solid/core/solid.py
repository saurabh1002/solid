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
        self.gap_height = ((self.fov_u - self.fov_d)) / self.num_elevation   

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
        
        x = np.where(x == 0.0, 0.001, x)
        y = np.where(y == 0.0, 0.001, y)

        theta   = self.xy2theta(x, y) 
        faraway = np.sqrt(np.square(x) + np.square(y))
        phi     = np.rad2deg(np.arctan2(z, faraway)) - self.fov_d

        idx_ring   = np.divmod(faraway, self.gap_ring)[0]      
        idx_sector = np.divmod(theta, self.gap_sector)[0]   
        idx_height = np.divmod(phi, self.gap_height)[0]
        
        idx_ring = np.where(idx_ring >= self.num_range, self.num_range - 1, idx_ring)
        idx_height = np.where(idx_height >= self.num_elevation, self.num_elevation - 1, idx_height)

        return idx_ring.astype(int), idx_sector.astype(int), idx_height.astype(int)

    def get_descriptor(self, scan):
        rh_counter = np.zeros([self.num_range, self.num_elevation])             
        sh_counter = np.zeros([self.num_angle, self.num_elevation])   
        idx_rings, idx_sectors, idx_heights = self.pt2rah(scan)
        rh_counter[idx_rings, idx_heights] = rh_counter[idx_rings, idx_heights] + 1
        sh_counter[idx_sectors, idx_heights] = sh_counter[idx_sectors, idx_heights] + 1
     
        ring_matrix = rh_counter    
        sector_matrix = sh_counter
        number_vector = np.sum(ring_matrix, axis=0)
        min_val = number_vector.min()
        max_val = number_vector.max()
        number_vector = (number_vector - min_val) / (max_val - min_val)
            
        r_solid = ring_matrix.dot(number_vector)
        a_solid = sector_matrix.dot(number_vector)
        return r_solid, a_solid

    def loop_detection(self, query, candidate):
        cosine_similarity = np.dot(query, candidate) / (np.linalg.norm(query) * np.linalg.norm(candidate))
        return cosine_similarity

    def pose_estimation(self, query, candidate):
        initial_cosdist = np.zeros(len(query))
        for shift_index in range(len(query)):
            initial_cosdist[shift_index] = np.sum(np.abs(candidate - np.roll(query, shift_index)))
        angle_difference = np.argmin(initial_cosdist) * (360 / self.num_angle)
        return angle_difference