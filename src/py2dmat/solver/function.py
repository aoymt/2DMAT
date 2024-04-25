# 2DMAT -- Data-analysis software of quantum beam diffraction experiments for 2D material structure
# Copyright (C) 2020- The University of Tokyo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import os
import numpy as np
import py2dmat

# type hints
from pathlib import Path
from typing import Callable, Optional, Union, Dict


class Solver:
    root_dir: Path
    output_dir: Path
    proc_dir: Path
    work_dir: Path
    dimension: Optional[int]
    _name: str
    _func: Optional[Callable[[np.ndarray], float]]
    timer: Dict[str, Dict]
    x: np.ndarray
    fx: float

    def __init__(self, info: Optional[py2dmat.Info] = None,
                 *,
                 root_dir: Union[Path,str] = ".",
                 output_dir: Union[Path,str] = ".",
                 dimension: Optional[int] = None,
                 function: Optional[Callable[[np.ndarray],float]] = None,
                 function_name: str = "function",
                 **kwargs) -> None:
        """
        Initialize the solver.

        Parameters
        ----------
        info: Info
        """
        if info is not None:
            self.root_dir = info.base["root_dir"]
            self.output_dir = info.base["output_dir"]
            self.dimension = info.solver.get("dimension") or info.base.get("dimension")
        else:
            self.root_dir = Path(root_dir).expanduser().absolute()
            self.output_dir = (self.root_dir / output_dir).absolute()
            self.dimension = dimension
        self.proc_dir = self.output_dir / str(py2dmat.mpi.rank())
        self.work_dir = self.proc_dir

        self._func = function
        self._name = function_name

        self.timer = {"prepare": {}, "run": {}, "post": {}}

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, x: np.ndarray, args = (), nprocs: int = 1, nthreads: int = 1) -> float:
        self.prepare(x, args)
        cwd = os.getcwd()
        os.chdir(self.work_dir)
        self.run(nprocs, nthreads)
        os.chdir(cwd)
        result = self.get_results()
        return result

    def prepare(self, x: np.ndarray, args = ()) -> None:
        self.x = x

    def run(self, nprocs: int = 1, nthreads: int = 1) -> None:
        if self._func is None:
            raise RuntimeError(
                "ERROR: function is not set. Make sure that `set_function` is called."
            )
        self.fx = self._func(self.x)

    def get_results(self) -> float:
        return self.fx

    def set_function(self, f: Callable[[np.ndarray], float]) -> None:
        self._func = f
