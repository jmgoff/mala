# Walks through training ACE mala model and calculating forces

Following similar procedure as with SNAP:

## first calculate ACE grid descriptors:

<i>to be ran with MALA_DATA_REPO='./'</i>

python ex07_preprocess_ace.py

## second, train ACE model:

python ex08_train_ace.py


## third, run prediction OR calculate force coefficients:

to run prediction of band energy


python ex09_run_prediction_ace.py


to calculate force coefficients using modified version of RandomDefaultUser's script:

python ex20_ace_atomic_forces.py

note that you may need to update the base calculation date (QE output file) to get 
ex20 running.

## Numerical differentiation of forces

To run the example used for numerical differentiation, see the lammps input file:

`in.numdiff`

It uses the python interface to mala set up in the `mala_betas.py` file. Note that
the settings in this file must be consistent with the model you fit.

Run with:
/path/to/build/lmp -in in.numdiff &> out.txt

It will take a minute. After it is finished, it will dump files that contain the
atomic forces but you will need to parse the 'energies'. For now, I do that with:

grep 'mala "energy"' ./out.txt  | awk '{print $(NF)}' > en_file.txt

Follow this with 

python e_vs_f.py to run numerical differentiation. It will find the 'energies' in
the 'en_file.txt' file. If a lot of 'nan' show up for forces from lammps, you will
need to rerun. This is the result of a memory error that only shows up when the
mala interface is on (it does not show up in valgrind for dummy betas).

##NOTES

<b>NOTE that the preprocessing and training have already been performed in this example</b>

This 'backup' has these steps completed so that errors with the callback function 
can be reproduced rapidly.

You may want to retrain the model now that Matthew's normalization fix is in the code.
