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

    # try to manually set grid?
    grd_loc = [18,18,27]
    predictor.data.grid_dimension = grd_loc
    print('predictor vars',vars(predictor))
    print('predictor vars',vars(predictor.data))
    print('predictor vars',vars(predictor.parameters))
    if type(atoms) == str:
        atoms = read(atoms)
    else:
        atoms = atoms
    predictor.parameters.inference_data_grid = grd_loc
    print('param grid',predictor.parameters.inference_data_grid)
    print('grid size',predictor.data.grid_size)
    #ldos = predictor.predict_for_atoms(atoms, save_grads=backprop,**{"grid_dimensions":[11,11,11]})
    ldos = predictor.predict_for_atoms(atoms, save_grads=backprop)
    ldos_calculator: mala.LDOS = predictor.target_calculator
    #ldos_calculator.read_from_array(ldos)
    return ldos, ldos_calculator, parameters, predictor


#function to convert lammps triclinic cell into ASE format cell
def lammps_box_2_ASE_cell(lmpbox):
    #boxlo, boxhi, xy, yz, xz, periodicity, box_change
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

# function to take lammps object (lammps run) and extract ase atoms object
def lammps_2_ase_atoms(lmp,typ_map):
    """
    lmp : lammps.lammps python object (after run 0 command at least)
    typ_map : dictionary to map lammps types to atomic symbols: e.g. {1:'H',2:'Be'}
    """
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
    
# backprop function to be used in LAMMPS pre_force callback 
def pre_force_callback(lmp):
    """
    lmp : lammps.lammps python object
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


lmp = lammps()
me = lmp.extract_setting("world_rank")
nprocs = lmp.extract_setting("world_size")

cmds = ["-screen", "none", "-log", "none"]
lmp = lammps(cmdargs = cmds)

def run_lammps():

    # simulation settings
    lmp.command("clear")
    lmp.command("units metal")
    lmp.command("boundary       p p p")
    lmp.command("atom_modify    map hash")
    lmp.command("variable    lat equal 3.16")
    lmp.command("variable    nrep index 1")
    lmp.command("variable    ngridxy index 18")
    lmp.command("variable    ngridz index 27")
    lmp.command("variable        nx equal ${nrep}")
    lmp.command("variable    ny equal ${nrep}")
    lmp.command("variable    nz equal ${nrep}")
    lmp.command("lattice         bcc ${lat}")
    lmp.command("region         box block 0 ${nx} 0 ${ny} 0 ${nz}")
    lmp.command(f"create_box    1 box")
    lmp.command(f"create_atoms  1 box")
    lmp.command("mass           * 9.012182")
    lmp.command("displace_atoms         all random 0.01 0.01 0.01 123456")

    # potential settings

    lmp.command("variable       ace_options string coupling_coefficients.yace")
    lmp.command(f"pair_style    zero {rcutfac}")
    lmp.command(f"pair_coeff    * *")
    lmp.command("compute       mygridlocal all pace/grid/local  grid ${ngridxy} ${ngridxy} ${ngridz} ${ace_options} ugridtype 1")
    lmp.command(f"thermo                100")
    lmp.command(f"run {nsteps}")

nsteps = 0
nrep = 1
ntypes = 1
nx = nrep
ny = nrep
nz = nrep

# declare compute snap variables

rcutfac = 5.8

run_lammps()


pre_force_callback(lmp)
