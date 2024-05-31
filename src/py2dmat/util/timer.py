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

import sys
import time
from statistics import mean

import py2dmat.mpi

class Timer:
    def __init__(self, initial_keys=None):
        if initial_keys:
            self._record = { k: [] for k in initial_keys }
        else:
            self._record = {}
        self._start = {}

    def start(self, key):
        current = time.perf_counter()
        if key in self._start.keys():
            raise KeyError("timer {} is started more than once".format(key))
        self._start[key] = current

    def stop(self, key):
        current = time.perf_counter()
        if not key in self._start.keys():
            raise KeyError("timer {} has not yet started".format(key))
        start = self._start.pop(key)
        if not key in self._record.keys():
            self._record[key] = []
        self._record[key].append(current-start)

    def report(self, fp=sys.stdout, detail=True):
        mpisize = py2dmat.mpi.size()
        mpirank = py2dmat.mpi.rank()

        if mpisize > 1:
            grecord = py2dmat.mpi.comm().allgather(self._record)
        else:
            grecord = [self._record]

        if mpirank == 0:
            for key in grecord[0].keys():
                vss = [g.get(key, []) for g in grecord]
                vv = [v for vs in vss for v in vs]
                count = len(vv)
                if count > 0:
                    total, average = sum(vv), mean(vv)
                else:
                    total, average = 0.0, 0.0
                fp.write("{} {:.6f} {:.6f} ({})\n".format(key, total, average, count))
                if detail and mpisize > 1:
                    for r in range(mpisize):
                        vs = grecord[r].get(key, [])
                        count = len(vs)
                        if count > 0:
                            total, average = sum(vs), mean(vs)
                        else:
                            total, average = 0.0, 0.0
                        fp.write("\trank={} {:.6f} {:.6f} ({})\n".format(r, total, average, count))
