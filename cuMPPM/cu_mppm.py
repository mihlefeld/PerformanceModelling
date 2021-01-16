import GPUtil
import os
from extrap.entities.terms import MultiParameterTerm, CompoundTerm
from extrap.entities.functions import MultiParameterFunction
from extrap.entities.hypotheses import MultiParameterHypothesis
from typing import Sequence
from extrap.entities.measurement import Measurement
from ctypes import *


class CPUMatrix(Structure):
    _fields_ = [('width', c_int), ('height', c_int), ('elements', POINTER(c_float))]

    def __repr__(self):
        repstring = "CPUMatrix\n width: "+str(self.width)+", height: "+str(self.height)
        return repstring


class CPUHypothesis(Structure):
    _fields_ = [('d', c_int),
                ('coefficients', c_float * 5),
                ('exponents', c_float * 10),
                ('smape', c_float),
                ('rss', c_float),
                ('combination', c_uint8 * 25)]

    def __repr__(self):
        rep_string = "-----------------------------------------------------------------\n"
        rep_string += f"Hypothesis (SMAPE = {self.smape:.4f}, RSS = {self.rss:.4f}\n"
        rep_string += " ".join(["Coefficients:", *[f"{c:.4f}" for c in self.coefficients[:self.d + 1]]]) + "\n"
        exp = zip(self.exponents[:self.d*2:2], self.exponents[1:self.d*2:2])
        rep_string += " ".join(["Exponents:", *[f"({i:.2f},{j:.2f})" for i, j in exp]]) + "\n"
        rep_string += "Combination:" + "\n"
        for i in range(self.d):
            rep_string += " ".join([str(c) for c in self.combination[i*self.d:(i + 1) * self.d]]) + "\n"
        rep_string += "-----------------------------------------------------------------\n"
        return rep_string

    def as_multi_parameter_hypothesis(self, use_median: bool) -> MultiParameterHypothesis:
        compound_terms = []
        for i in range(self.d):
            iex = self.exponents[i*2]
            jex = self.exponents[i*2 + 1]
            frac = float(iex).as_integer_ratio()
            term = CompoundTerm.create(iex, int(jex))
            compound_terms.append((i, term))

        mp_terms = []
        for i in range(self.d):
            line = self.combination[i*self.d:(i+1)*self.d]
            terms = []
            if not any(line):
                continue
            for j in range(self.d):
                if line[j]:
                    terms.append(compound_terms[j])
            mp_terms.append(MultiParameterTerm(*terms))

        func = MultiParameterFunction(*mp_terms)
        func.constant_coefficient = self.coefficients[0]
        for i, term in enumerate(func.compound_terms):
            term.coefficient = self.coefficients[i + 1]
        hypothesis = MultiParameterHypothesis(func, use_median)
        return hypothesis


def get_best_gpu():
    gpus = GPUtil.getGPUs()
    best_gpu = gpus[0]
    for gpu in gpus:
        if gpu.memoryFree > best_gpu.memoryFree:
            best_gpu = gpu
    return best_gpu


def get_cpu_matrix(measurements: Sequence[Measurement], use_median: bool) -> CPUMatrix:
    w = measurements[0].coordinate.dimensions + 1
    h = len(measurements)
    c_measurements = []
    for measurement in measurements:
        row = list(measurement.coordinate)
        row.append(measurement.mean if not use_median else measurement.median)
        c_measurements += row
    float_arr = c_float * (h * w)
    elements = float_arr(*c_measurements)
    elements_ptr = cast(elements, POINTER(c_float))
    return CPUMatrix(w, h, elements_ptr)


class CuMPPM:
    def __init__(self):
        self.lib = None
        # TODO: decide where .dll/.so should reside
        if os.name == 'nt':
            self.lib = WinDLL("cmake-build-debug/cuMPPM.dll")
        else:
            self.lib = CDLL("../build/libcuMPPM.so")
        self.lib.find_hypothesis.restype = CPUHypothesis

    def find_hypothesis(self, measurements: Sequence[Measurement], use_median: bool) -> MultiParameterHypothesis:
        gpu = get_best_gpu()
        ratio = 0.9 * (gpu.memoryFree / gpu.memoryTotal)
        id = gpu.id
        cpu_measurements = get_cpu_matrix(measurements, use_median)
        hypothesis: CPUHypothesis = self.lib.find_hypothesis(byref(cpu_measurements), c_float(ratio))
        return hypothesis.as_multi_parameter_hypothesis(use_median)
