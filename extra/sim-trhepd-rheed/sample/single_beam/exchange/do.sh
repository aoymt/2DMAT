#!/bin/sh

MPIEXEC=mpiexec --oversubscribe
#MPIEXEC=mpirun

sh prepare.sh

./bulk.exe

time $MPIEXEC -np 4 sim-trhepd-rheed input.toml

echo diff best_result.txt ref.txt
res=0
diff best_result.txt ref.txt || res=$?
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: best_result.txt and ref.txt differ
  false
fi

