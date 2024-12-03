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

    def xy2theta(self, x, y):
        if (x >= 0 and y >= 0): 
            theta = 180/np.pi * np.arctan(y/x)
        if (x < 0 and y >= 0): 
            theta = 180 - ((180/np.pi) * np.arctan(y/(-x)))
        if (x < 0 and y < 0): 
            theta = 180 + ((180/np.pi) * np.arctan(y/x))
        if ( x >= 0 and y < 0):
            theta = 360 - ((180/np.pi) * np.arctan((-y)/x))
        return theta

    def pt2rah(self, point, gap_ring, gap_sector, gap_height):
        x = point[0]
        y = point[1]
        z = point[2]
        
        if(x == 0.0):
            x = 0.001  
        if(y == 0.0):
            y = 0.001 

        theta   = self.xy2theta(x, y) 
        faraway = np.sqrt(x*x + y*y) 
        phi     = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2))) - self.fov_d

        idx_ring   = np.divmod(faraway, gap_ring)[0]      
        idx_sector = np.divmod(theta, gap_sector)[0]   
        idx_height = np.divmod(phi, gap_height)[0]
        
        if(idx_ring >= self.num_range):
            idx_ring = self.num_range-1

        if(idx_height >= self.num_elevation):
            idx_height = self.num_elevation-1

        return int(idx_ring), int(idx_sector), int(idx_height)

    def ptcloud2solid(self, ptcloud):
        num_points = ptcloud.shape[0]               
        
        gap_ring = self.max_length/self.num_range            
        gap_sector = 360/self.num_angle              
        gap_height = ((self.fov_u-self.fov_d))/self.num_elevation              

        rh_counter = np.zeros([self.num_range, self.num_elevation])             
        sh_counter = np.zeros([self.num_angle, self.num_elevation])   
        for pt_idx in range(num_points): 
            point = ptcloud[pt_idx, :]
            idx_ring, idx_sector, idx_height = self.pt2rah(point, gap_ring, gap_sector, gap_height) 
            rh_counter[idx_ring, idx_height] = rh_counter[idx_ring, idx_height] + 1     
            sh_counter[idx_sector, idx_height] = sh_counter[idx_sector, idx_height] + 1  
     
        ring_matrix = rh_counter    
        sector_matrix = sh_counter
        number_vector = np.sum(ring_matrix, axis=0)
        min_val = number_vector.min()
        max_val = number_vector.max()
        number_vector = (number_vector - min_val) / (max_val - min_val)
            
        r_solid = ring_matrix.dot(number_vector)
        a_solid = sector_matrix.dot(number_vector)
                
        return r_solid, a_solid

    def get_descriptor(self, scan):
        r_solid, a_solid = self.ptcloud2solid(scan)
        return r_solid, a_solid

    def loop_detection(self, query, candidate):
            cosine_similarity = np.dot(query, candidate) / (np.linalg.norm(query) * np.linalg.norm(candidate))
            return cosine_similarity

    def pose_estimation(self, query, candidate):
        initial_cosdist = []
        for shift_index in range(len(query)):
            initial_cosine_similarity = np.sum(np.abs(candidate - np.roll(query, shift_index)))
            initial_cosdist.append(initial_cosine_similarity)
        angle_difference = (np.argmin(initial_cosdist))*(360/self.num_angle)
        return angle_difference