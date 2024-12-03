# MIT License
#
# Copyright (c) 2023 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch,
# Cyrill Stachniss.
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
import datetime
import os
from pathlib import Path
from typing import Optional

import numpy as np

from solid.config import load_config
from solid.core.solid import SOLiDModule
from solid.core.point_module import PointModule
from solid.tools.pipeline_results import PipelineResults
from solid.tools.progress_bar import get_progress_bar


class SolidPipeline:
    def __init__(
        self,
        dataset,
        results_dir: Path,
        config: Optional[Path] = None,
    ):
        self._dataset = dataset
        self._first = 0
        self._last = len(self._dataset)

        self.results_dir = results_dir

        self.config = load_config(config)
        self.solid = SOLiDModule(self.config)
        self.preprocess = PointModule(self.config)
        self.rsolid_database = []
        self.asolid_database = []
        self.dataset_name = self._dataset.sequence_id

        self.closures = []
        self.gt_closure_indices = self._dataset.gt_closure_indices

        solid_thresholds = np.arange(self.config.loop_threshold, 0.04, 0.004)
        self.results = PipelineResults(
            self.gt_closure_indices, self.dataset_name, solid_thresholds
        )

    def run(self):
        self._run_pipeline()
        if self.gt_closure_indices is not None:
            self._run_evaluation()
        self._log_to_file()

        return self.results

    def _run_pipeline(self):
        for query_idx in get_progress_bar(self._first, self._last):
            scan = self._dataset[query_idx]
            scan = self.preprocess.remove_closest_points(scan)
            scan = self.preprocess.remove_far_points(scan)
            scan_downsampled = self.preprocess.down_sampling(scan)
            r_solid_desc, a_solid_desc = self.solid.get_descriptor(scan_downsampled)
            self.rsolid_database.append(r_solid_desc)
            self.asolid_database.append(a_solid_desc)
            
            if query_idx > 100:
                cosdist = []
                for candidate_idx in range(query_idx - 100):
                    query_R_solid     = self.rsolid_database[query_idx]
                    candidate_R_solid = self.rsolid_database[candidate_idx]
                    cosine_similarity = self.solid.loop_detection(query_R_solid, candidate_R_solid)
                    cosdist = 1-cosine_similarity
                    if cosdist < self.config.loop_threshold:
                        query_A_solid     = self.asolid_database[query_idx]
                        candidate_A_solid = self.asolid_database[candidate_idx]
                        angle_difference  = self.solid.pose_estimation(query_A_solid, candidate_A_solid)
                        self.closures.append(np.r_[candidate_idx, query_idx, angle_difference])
                    self.results.append(query_idx, candidate_idx, cosdist)


    def _run_evaluation(self) -> None:
        self.results.compute_metrics()

    def _log_to_file(self) -> None:
        self.results_dir = self._create_results_dir()
        if self.gt_closure_indices is not None:
            self.results.log_to_file_pr(os.path.join(self.results_dir, "metrics.txt"))
        self.results.log_to_file_closures(self.results_dir)
        np.savetxt(os.path.join(self.results_dir, "closures.txt"), np.asarray(self.closures))

    def _create_results_dir(self) -> Path:
        def get_timestamp() -> str:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results_dir = os.path.join(
            self.results_dir,  f"{self.dataset_name}_results", get_timestamp()
        )
        latest_dir = os.path.join(
            self.results_dir, f"{self.dataset_name}_results", "latest"
        )
        os.makedirs(results_dir, exist_ok=True)
        os.unlink(latest_dir) if os.path.exists(latest_dir) or os.path.islink(latest_dir) else None
        os.symlink(results_dir, latest_dir)

        return results_dir
