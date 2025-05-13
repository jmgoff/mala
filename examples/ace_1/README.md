# Walks through training ACE mala model (very simple one) and performing infer or calculating force coefficients

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

## failing example required for lammps force callback

The new lammps fix capable of running MD with mala requires that a python function
returns the force coefficients (dE_I/dB_k) for all descriptors k per grid point I. 
While `ex20_ace_atomic_forces.py` calculates these correctly, there are errors when
trying to access these values within the callback function that lammps needs. 

To try the lammps/mala callback for forces, try:

python test.py

##NOTES

