import os

from ase.io import read
from ase import Atoms,Atom
import mala
from mala import printout
import numpy as np
from lammps import lammps
import ctypes
from ctypes import *

from mala.datahandling.data_repo import data_repo_path
data_path = './' #os.path.join(data_repo_path, "Be2")

assert os.path.exists("be_model.zip"), "Be model missing, run ex01 first."

def run_prediction(backprop=False,atoms=None):
    """
    This just runs a regular MALA prediction for a two-atom Beryllium model.
    """
    parameters, network, data_handler, predictor = mala.Predictor. \
        load_run("be_model")

    parameters.targets.target_type = "LDOS"
    parameters.targets.ldos_gridsize = 11
    parameters.targets.ldos_gridspacing_ev = 2.5
    parameters.targets.ldos_gridoffset_ev = -5

    parameters.descriptors.descriptor_type = "ACE"
    parameters.descriptors.bispectrum_cutoff = 5.8
    parameters.ace_types_like_snap = False

    assert atoms != None,"need to supply ase atoms object or file"
    if type(atoms) == str:
        atoms = read(atoms)
    else:
        atoms = atoms
    ldos = predictor.predict_for_atoms(atoms, save_grads=backprop)
    ldos_calculator: mala.LDOS = predictor.target_calculator
    ldos_calculator.read_from_array(ldos)
    return ldos, ldos_calculator, parameters, predictor


#boxlo, boxhi, xy, yz, xz, periodicity, box_change
def lammps_box_2_ASE_cell(lmpbox):
    Lx = lmpbox[1][0] - lmpbox[0][0]
    Ly = lmpbox[1][1] - lmpbox[0][1]
    Lz = lmpbox[1][2] - lmpbox[0][2]
    xy = lmpbox[2]
    yz = lmpbox[3]
    xz = lmpbox[4]
    a = [Lx,0,0]
    b = [xy,Ly,0]
    c = [xz,yz,Lz]
    cel = [a,b,c]
    return cel
#print('box',lmp.extract_box())
#print('cell', lammps_box_2_ASE_cell(lmp.extract_box()))

def lammps_2_ase_atoms(lmp,typ_map):
    cell = lammps_box_2_ASE_cell(lmp.extract_box())
    x= lmp.extract_atom("x")
    natoms = lmp.get_natoms()
    pos = np.array([[x[i][0], x[i][1], x[i][2]] for i in range(natoms)])
    # Extract atom types
    atom_types = lmp.extract_atom("type")
    # Convert atom types to NumPy array
    atom_types_lst = [atom_types[i] for i in range(natoms)]
    atom_syms = [typ_map[typi] for typi in atom_types_lst]
    atoms = Atoms(atom_syms)
    atoms.positions = pos
    atoms.set_cell(cell)
    atoms.set_pbc(True)# assume pbc
    return atoms
    

#def backpropagation():
def pre_force_callback(lmp):
    #L = lammps(ptr=lmp)
    """
    Test whether backpropagation works. To this end, the entire forces are
    computed, and then backpropagated through the network.
    """
    # Only compute a specific part of the forces.
    atoms = lammps_2_ase_atoms(lmp,typ_map={1:'Be'})
    ldos, ldos_calculator, parameters, predictor = run_prediction(backprop=True,atoms=atoms)
    ldos_calculator.debug_forces_flag = "band_energy"
    ldos_calculator.input_data_derivative = predictor.input_data
    ldos_calculator.output_data_torch = predictor.output_data
    mala_forces = ldos_calculator.atomic_forces
    print("FORCE TEST: Backpropagation machinery.")
    print(mala_forces.size())
    return mala_forces

#backpropagation()
