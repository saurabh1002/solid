import numpy as np
import open3d as o3d
from pydantic_settings import BaseSettings

class PointModule:
    def __init__(self, config: BaseSettings):
        self.min_distance = config.min_distance
        self.max_distance = config.max_distance
        self.voxel_size   = config.voxel_size

    def remove_closest_points(self, points):
        dists = np.sum(np.square(points[:, :3]), axis=1)
        cloud_out = points[dists > self.min_distance*self.min_distance]
        return cloud_out

    def remove_far_points(self, points):
        dists = np.sum(np.square(points[:, :3]), axis=1)
        cloud_out = points[dists < self.max_distance*self.max_distance]
        return cloud_out

    def down_sampling(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        down_points_np = np.asarray(down_pcd.points)
        return down_points_np

