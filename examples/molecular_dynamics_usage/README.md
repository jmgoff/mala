# How to run MD when MALA is being called correctly

Using serial lammps:
<path/to>/lmp -in in.force_interface 

# NOTE

currently this will throw a seg fault. (Lammps throws a segfault if there are errors in the python function)
See the error that the python function throws by running the 'test.py' script
