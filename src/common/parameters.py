'''
parameters.py: Includes several classes that structure the parameters used within this framework.
An object of the central parameters() class is the only thing you need to get start training (except from the data of course).
'''


# Subclasses that make up the final parameters class.

class parameters_base():
    def __init__(self):
        raise exception("Error: No constructor implemented.")

    def show(self, indent=""):
        for v in vars(self):
            print(indent + '%-15s: %s' % (v, getattr(self, v)))


class parameters_network(parameters_base):
    """Neural Network parameter subclass."""

    def __init__(self):
        self.nn_type = "feed-forward"
        self.layer_sizes = [10,10,10]
        """Includes the input layer, although no activation is applied to it."""
        self.layer_activations = ["Sigmoid"]
        """If the dimension of layer_activations is smaller than the dimension of
        layer_sizes-1, than the first entry is used for all layers."""
        self.loss_function_type = "mse"


class parameters_descriptors(parameters_base):
    """Fingerprint descriptor parameter subclass."""

    def __init__(self):
        self.descriptor_type = "SNAP"


class parameters_data(parameters_base):
    """Dataset interface parameter subclass."""

    def __init__(self):
        self.datatype = "QE+LDOS"
        """
        Specifies the kind of input data we are working with.
            Implemented so far:
                - MNIST for the MNIST data set (for testing purposes).
                - QE+LDOS for a dataset consisting of a Quantum espresso outfile and an LDOS file.
        """
        self.directory = "~/data"
        self.qe_calc_list = []
        """
        A list detailing all subdirectories, each containing a QE out file and
        the calculated LDOS (using pp.x).
        """
        self.twojmax = 10
        """
        SNAP creation: 2*jmax-parameter used for calculation of SNAP descriptors. Standard value
        for jmax is 5, so standard value for twojmax is 10
        """
        self.rcutfac = 4.67637
        """
        SNAP creation: radius cutoff factor for the fingerprint sphere in Angstroms.
        """
        self.lammps_compute_file = ""
        """
        LAMMPS input file that is used to calculate the SNAP descriptors.
        If this string is empty, the standard LAMMPS input file found in this
        repository will be used (recommended).
        """
        self.dbg_grid_dimensions = ""
        """
        For debugging purposes smaller grid sizes than the ones specified in the
        QE output can be used.
        """

class parameters_training(parameters_base):
    """Network training parameter subclass."""

    def __init__(self):
        self.trainingtype = "SGD"
        self.learning_rate = 0.5
        self.max_number_epochs = 100
        # TODO: Find a better system for verbosity. Maybe a number.
        self.verbosity = True
        self.mini_batch_size = 10


# TODO: Add keyword arguments that allow for a passing of the arguments in the constructor.

class parameters():
    """Parameter class for ML-DFT@CASUS."""

    def __init__(self):
        self.comment = 0
        '''Comment a used parameter dataset.'''
        self.network = parameters_network()
        self.descriptors = parameters_descriptors()
        self.data = parameters_data()
        self.training = parameters_training()

    def show(self):
        """Prints all the parameters bundled in this class."""
        print("--- " + self.__doc__ + " ---")
        for v in vars(self):
            if (isinstance(getattr(self, v), parameters_base)):
                parobject = getattr(self, v)
                print("--- " + parobject.__doc__ + " ---")
                parobject.show("\t")
            else:
                print('%-15s: %s' % (v, getattr(self, v)))


if __name__ == "__main__":
    p = parameters()
    p.show()
    print("parameters.py - basic functions are working.")
