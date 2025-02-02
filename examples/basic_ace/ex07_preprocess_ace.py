import os

import mala

from mala.datahandling.data_repo import data_repo_path

data_path = './'

"""
Shows how this framework can be used to preprocess
data. Preprocessing here means converting raw DFT calculation output into 
numpy arrays of the correct size. For the input data, this means descriptor
calculation.

REQUIRES LAMMPS.
"""


####################
# 1. PARAMETERS
# Select the correct parameters with which to preprocess the electronic
# structure simulation data. For the LDOS, these parameters have to be chosen
# consistent with the DFT calculation (i.e., if you specified an energy grid
# with 250 values lying between -10 and 15 eV, and therefore 0.1 eV between
# grid points, you will have to set ldos_gridsize to 250, gridspacing_ev to 0.1
# and grid_offset_ev to -10). Of course, the values are dependent on the
# physical system simulated.
####################

"""
        ranks = self.parameters.ace_ranks
        lmax = self.parameters.ace_lmax
        nmax = self.parameters.ace_nmax
        lmin = self.parameters.ace_lmin
"""

parameters = mala.Parameters()
# ACE parameters.
parameters.descriptors.descriptor_type = "ACE"
parameters.descriptors.bispectrum_cutoff = 5.8
parameters.descriptors.ace_ranks = [1,2,3]
parameters.descriptors.ace_lmax = [0,1,1]
parameters.descriptors.ace_nmax = [1,1,1]
parameters.descriptors.ace_lmin = [0,0,0]
#NOTE that the following setting should rarely be 'True'
parameters.descriptors.ace_types_like_snap = False
#NOTE that with ACE, you always need to specify an extra "element" for the grid
#  points. The "element" assigned to gridpoints should be labeled 'G'
parameters.descriptors.ace_elements = ['Be','G']
# LDOS parameters.
parameters.targets.target_type = "LDOS"
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5

####################
# 2. ADDING DATA FOR DATA CONVERSION
# Data conversion itself is simple. We select input and output data
# to be converted, add this data snapshot-wise and tell MALA to
# convert snapshots. Inputs and outputs can be processed individually.
# Further, via the additional_info_input_* keywords, calculation output
# can be processed from the original simulation *.out output files into
# more convenient *.json files that can be used in their stead. This saves
# on disk space.
# To only process parts of the data, omit/add descriptor_input*, target_input_*
# and additional_info_input_* at your leisure.
# Make sure to set the correct units - for QE, this should always be
# 1/(Ry*Bohr^3).
####################

data_converter = mala.DataConverter(parameters)
outfile = os.path.join(data_path, "Be_snapshot0.out")
ldosfile = os.path.join(data_path, "cubes/tmp.pp*Be_ldos.cube")

data_converter.add_snapshot(
    descriptor_input_type="espresso-out",
    descriptor_input_path=outfile,
    target_input_type=".cube",
    target_input_path=ldosfile,
    additional_info_input_type="espresso-out",
    additional_info_input_path=outfile,
    target_units="1/(Ry*Bohr^3)",
)
####################
# 3. Converting the data
# To convert the data we now simply have to call the convert_snapshot function.
# Input (descriptor) and output (target) data can be saved in individual
# locations, as can the additional output data files. Fine-granular access
# to the calculators is enabled via the descriptor_calculation_kwargs and
# target_calculation_kwargs arguments, but usually not needed.
# If all data files should be stored in the same location, the
# complete_save_path keyword may be used.
####################

data_converter.convert_snapshots(
    descriptor_save_path="./",
    target_save_path="./",
    additional_info_save_path="./",
    naming_scheme="Be_snapshot*.npy",
    descriptor_calculation_kwargs={"working_directory": data_path},
)
