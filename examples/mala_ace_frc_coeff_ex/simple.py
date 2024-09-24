import os

from ase.io import read
import mala
from mala import printout
import numpy as np

from mala.datahandling.data_repo import data_repo_path
#data_path = os.path.join(data_repo_path, "Be2")
data_path = './'
"""
ex19_at.py: Show how a prediction can be made using MALA.
Using nothing more then the trained network and atomic configurations, 
predictions can be made. 
"""


# Uses a network to make a prediction.
assert os.path.exists("be_model.zip"), "Be model missing, run ex01 first."


def run_prediction(backprop=False):
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
    parameters.descriptors.ace_ranks = [1,2,3]
    parameters.descriptors.ace_lmax = [0,2,2]
    parameters.descriptors.ace_nmax = [12,2,2]
    parameters.descriptors.ace_lmin = [0,0,0]
    #parameters.ace_types_like_snap = False
    parameters.descriptors.ace_types_like_snap = False
    parameters.lammps_compute_file = "/home/jmgoff/Software/mala_frc_combine/mala/descriptors/in.acegrid.python"

    parameters.targets.pseudopotential_path = data_path

    #atoms = read(os.path.join(data_path, "Be_snapshot3.out"))
    atoms = read(os.path.join(data_path, "Be_snapshot0.out"))
    ldos = predictor.predict_for_atoms(atoms, save_grads=backprop)
    ldos_calculator: mala.LDOS = predictor.target_calculator
    ldos_calculator.read_from_array(ldos)
    return ldos, ldos_calculator, parameters, predictor


def backpropagation():
    """
    Test whether backpropagation works. To this end, the entire forces are
    computed, and then backpropagated through the network.
    """
    # Only compute a specific part of the forces.
    ldos, ldos_calculator, parameters, predictor = run_prediction(backprop=True)
    ldos_calculator.debug_forces_flag = 'band_energy'
    ldos_calculator.input_data_derivative = predictor.input_data
    ldos_calculator.output_data_torch = predictor.output_data
    mala_forces = ldos_calculator.atomic_forces
    # Should be 8748, 91
    print("FORCE TEST: Backpropagation machinery.")
    print(mala_forces.size())


#band_energy_contribution()
#entropy_contribution()
#hartree_contribution()
backpropagation()
