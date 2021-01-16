import os
from cuMPPM import cu_mppm
from ctypes import *
from typing import Sequence
from extrap.entities.measurement import Measurement
from extrap.modelers import single_parameter
from extrap.modelers.abstract_modeler import MultiParameterModeler as AbstractMultiParameterModeler
from extrap.modelers.abstract_modeler import SingularModeler
from extrap.entities.hypotheses import MultiParameterHypothesis
from extrap.modelers.modeler_options import modeler_options

@modeler_options
class GPUDirectMultiParameterModeler(AbstractMultiParameterModeler, SingularModeler):
    """
    This class implements direct multi parameter modelling using cuda. At least 2 parameters are needed,
    a maximum of 500 measurements is supported. Works up to 5D, however 5D takes > 30 minutes.

    """

    NAME = 'GPU-Direct-Multi-Parameter'
    single_parameter_modeler: 'SingleParameterModeler'
    use_crossvalidation = modeler_options.add(True, bool, 'Enables cross-validation', name='Cross-validation')
    allow_combinations_of_sums_and_products = modeler_options.add(True, bool,
                                                                  description="Allows models that consist of "
                                                                              "combinations of sums and products.")
    compare_with_RSS = modeler_options.add(False, bool,
                                           'If enabled the models are compared using their residual sum of squares '
                                           '(RSS) instead of their symmetric mean absolute percentage error (SMAPE)')

    def __init__(self):
        """
        Initialize SingleParameterModeler object.
        """
        super().__init__(use_median=False, single_parameter_modeler=single_parameter.Default())
        # value for the minimum number of measurement points required for modeling
        self.min_measurement_points = 5
        self.epsilon = 0.0005  # value for the minimum term contribution

    def create_model(self, measurements: Sequence[Measurement]):
        lib = cu_mppm.CuMPPM()
        hypothesis = lib.find_hypothesis(measurements, self.use_median)
        hypothesis.compute_cost(measurements)
        return hypothesis
