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
        dataset_query,
        dataset_ref,
        results_dir: Path,
        config_ref: Optional[Path] = None,
        config_query: Optional[Path] = None,
    ):
        self._query_dataset = dataset_query
        self._query_dataset_name = (
            self._query_dataset.sequence_id
            if hasattr(self._query_dataset, "sequence_id")
            else os.path.basename(self._query_dataset.data_dir)
        )
        self._ref_dataset = dataset_ref
        self._ref_dataset_name = (
            self._ref_dataset.sequence_id
            if hasattr(self._ref_dataset, "sequence_id")
            else os.path.basename(self._ref_dataset.data_dir)
        )

        self.results_dir = results_dir

        self.config_ref = load_config(config_ref)
        self.config_query = load_config(config_query)
        self.solid_ref = SOLiDModule(self.config_ref)
        self.preprocessor_ref = PointModule(self.config_ref)
        self.rsolid_database = np.zeros((len(self._ref_dataset), self.config_ref.num_range))

        self.solid_query = SOLiDModule(self.config_query)
        self.preprocessor_query = PointModule(self.config_query)

        self.closures = []
        base_dir_query = self._query_dataset.sequence_dir
        file_path_closures_query = os.path.join(
            base_dir_query,
            "loop_closure",
            f"{self._ref_dataset.sequence_id}_local_map_gt_closures.txt",
        )
        if os.path.exists(file_path_closures_query) and os.path.exists(file_path_closures_query):
            self.gt_closures = np.loadtxt(file_path_closures_query, dtype=int)
            print(f"[INFO] Found closure ground truth at {file_path_closures_query}")
        else:
            self.gt_closures = None
            print(f"[INFO] No closure ground truth found at {file_path_closures_query}")

        self.ref_local_maps_scan_range = self._ref_dataset.local_maps_scan_range
        self.query_local_maps_scan_range = self._query_dataset.local_maps_scan_range

        self.ref_map_indices = scan_indices_to_map_indices(
            len(self._ref_dataset), self.ref_local_maps_scan_range
        )
        self.query_map_indices = scan_indices_to_map_indices(
            len(self._query_dataset), self.query_local_maps_scan_range
        )

        solid_thresholds = np.arange(0.001, 0.1, 0.001)
        self.results = PipelineResults(self.gt_closures, solid_thresholds)

    def run(self):
        self._run_pipeline()
        if self.gt_closures is not None:
            self._run_evaluation()
        self._log_to_file()

        return self.results

    def _run_pipeline(self):
        for ref_idx in get_progress_bar(0, len(self._ref_dataset)):
            scan = self._ref_dataset[ref_idx]
            scan_downsampled = self.preprocessor_ref.preprocess(scan)
            self.rsolid_database[ref_idx] = self.solid_ref.get_descriptor(scan_downsampled)

        for query_idx in get_progress_bar(0, len(self._query_dataset)):
            scan = self._query_dataset[query_idx]
            scan_downsampled = self.preprocessor_query.preprocess(scan)
            query_R_solid = self.solid_query.get_descriptor(scan_downsampled)

            cosdistances = 1 - self.solid_query.loop_detection(query_R_solid, self.rsolid_database)
            keep_indices = np.where(cosdistances <= 0.1)[0]

            self.results.append(
                self.ref_map_indices[keep_indices],
                self.query_map_indices[query_idx],
                cosdistances[keep_indices],
            )

    def _run_evaluation(self) -> None:
        self.results.compute_metrics()

    def _log_to_file(self) -> None:
        self.results_dir = self._create_results_dir()
        if self.gt_closures is not None:
            self.results.log_to_file_pr(os.path.join(self.results_dir, "metrics.txt"))

    def _create_results_dir(self) -> Path:
        results_dir = os.path.join(
            self.results_dir, f"{self._query_dataset_name}", f"{self._ref_dataset_name}"
        )
        os.makedirs(results_dir, exist_ok=True)

        return results_dir
