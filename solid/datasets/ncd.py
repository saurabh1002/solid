# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import importlib
import os
import re
from pathlib import Path

import numpy as np


class NewerCollegeDataset:
    def __init__(self, data_dir: Path, *_, **__):
        try:
            self.PyntCloud = importlib.import_module("pyntcloud").PyntCloud
        except ModuleNotFoundError:
            print(f'Newer College requires pnytccloud: "pip install pyntcloud"')

        self.data_dir = os.path.join(data_dir, "")
        self.scan_folder = os.path.join(self.data_dir, "raw_format/ouster_scan")
        self.sequence_id = os.path.basename(data_dir)

        # Load scan files and poses
        self.scan_files = self.get_pcd_filenames(self.scan_folder)
        try:
            self.gt_closure_indices = np.loadtxt(
                os.path.join(self.data_dir, "loop_closure", "gt_closures.txt")
            )
        except FileNotFoundError:
            self.gt_closure_indices = None

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.scan_folder, self.scan_files[idx])
        return self.getitem(file_path)

    def getitem(self, scan_file: str):
        points = self.PyntCloud.from_file(scan_file).points[["x", "y", "z"]].to_numpy()
        return points.astype(np.float64)

    @staticmethod
    def get_pcd_filenames(scans_folder):
        # cloud_1583836591_182590976.pcd
        regex = re.compile("^cloud_(\d*_\d*)")

        def get_cloud_timestamp(pcd_filename):
            m = regex.search(pcd_filename)
            secs, nsecs = m.groups()[0].split("_")
            return int(secs) * int(1e9) + int(nsecs)

        return sorted(os.listdir(scans_folder), key=get_cloud_timestamp)
