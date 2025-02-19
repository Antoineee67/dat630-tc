#!/bin/sh


echo Running treecode
rm -r out.txt
mpirun --allow-run-as-root -n 8 ./treecode out=out.txt nbody=16384 tstop=1 seed=123
echo Comparing output to standard nbody=16384 tstop=1 seed=123 output.
echo
python3 ./tools/verifyOutput.py out.txt ./tools/verificationData.txt