import numpy as np
import open3d as o3d
from pydantic_settings import BaseSettings

class PointModule:
    def __init__(self, config: BaseSettings):
        self.min_distance = config.min_distance
        self.max_distance = config.max_distance
        self.voxel_size   = config.voxel_size

    def preprocess(self, points):
        dists = np.sum(np.square(points[:, :3]), axis=1)
        cloud_filtered = points[(dists > self.min_distance * self.min_distance) & (dists < self.max_distance * self.max_distance), :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_filtered))
        down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return np.asarray(down_pcd.points)

