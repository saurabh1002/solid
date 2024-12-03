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
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class SolidConfig(BaseModel):
    min_distance: float = 3.0
    max_distance: float = 80.0
    fov_u: float = 24.8
    fov_d: float = -2.0
    num_angle: int = 60
    num_elevation: int = 64
    num_range: int = 40
    voxel_size: float = 0.5
    loop_threshold: float = 0.004


def load_config(config_file: Optional[Path]) -> SolidConfig:
    """Load configuration from an Optional yaml file. Additionally, deskew and max_range can be
    also specified from the CLI interface"""

    config = None
    if config_file is not None:
        try:
            yaml = importlib.import_module("yaml")
        except ModuleNotFoundError:
            print(
                "[ERROR] Custom configuration file specified but PyYAML is not installed on your system,"
                " run `pip install pyyaml`"
            )
            sys.exit(1)
        with open(config_file) as cfg_file:
            config = yaml.safe_load(cfg_file)
        return SolidConfig(**config)
    else:
        return SolidConfig()


def write_config(config: SolidConfig, filename: str):
    with open(filename, "w") as outfile:
        try:
            yaml = importlib.import_module("yaml")
            yaml.dump(config.model_dump(), outfile, default_flow_style=False)
        except ModuleNotFoundError:
            outfile.write(str(config.model_dump()))
