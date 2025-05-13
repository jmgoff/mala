# New example for ace forces

Attempting to do numerical differentiation outside of lammps numdiff. Instead
of using energy model in lammps, and forces in lammps to do numerical differentiation in the
lammps c++ code, the forces are dumped from lammps in dump.x.mala for displacements
and (band) energies are printed to stdout.

to run lammps and get forces as a function of displacements, run

`/path/to/lmp -in in.continuous.numdiff > outnumdiff.txt`

process your results with `proc_out.py`. this script is new and may need editing.

As it is set up, running this example will calculate forces using dE/dB from
`mala_betas.py` and use that to evaluate atomic forces in lammps.

When using `dummy_betas.py` and some flags in lammps to use a linear energy model,
the forces are consistent with those from numerical differentiation.


