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

import numpy as np

def scan_indices_to_map_indices(dataset_size, local_maps_scan_range):
    start = local_maps_scan_range[:, 0][:, None]
    end = local_maps_scan_range[:, 1][:, None]

    scan_indices = np.arange(dataset_size)
    ref_mask = (scan_indices >= start) & (scan_indices < end)
    map_indices = np.argmax(ref_mask, axis=0).astype(np.uint16)
    return map_indices

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
        self.preprocessor = PointModule(self.config)
        self.rsolid_database = np.zeros((self._last, self.config.num_range))
        self.dataset_name = self._dataset.sequence_id

        self.gt_closure_indices = self._dataset.gt_closure_indices
        self.local_maps_scan_range = self._dataset.local_maps_scan_range
        self.map_indices = scan_indices_to_map_indices(self._last, self.local_maps_scan_range)

        solid_thresholds = np.arange(0.001, 0.1, 0.001)
        self.results = PipelineResults(
            self.gt_closure_indices, solid_thresholds
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
            scan_downsampled = self.preprocessor.preprocess(scan)
            query_R_solid, _ = self.solid.get_descriptor(scan_downsampled)
            self.rsolid_database[query_idx] = query_R_solid
            
            if query_idx > 100:
                candidate_indices = np.arange(query_idx - 100)
                candidates_R_solid = self.rsolid_database[candidate_indices]
                cosdistances = 1 - self.solid.loop_detection(query_R_solid, candidates_R_solid)
                map_query = self.map_indices[query_idx]
                map_refs = self.map_indices[candidate_indices]
                keep_indices = np.where((map_query - map_refs > 3) & (cosdistances <= 0.1))[0]
                self.results.append(map_refs[keep_indices], map_query, cosdistances[keep_indices])

    def _run_evaluation(self) -> None:
        self.results.compute_metrics()

    def _log_to_file(self) -> None:
        self.results_dir = self._create_results_dir()
        if self.gt_closure_indices is not None:
            self.results.log_to_file_pr(os.path.join(self.results_dir, "metrics.txt"))

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
