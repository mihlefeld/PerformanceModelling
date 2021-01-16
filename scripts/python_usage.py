from extrap.entities.callpath import Callpath
from extrap.entities.metric import Metric
from extrap.fileio import text_file_reader
from cuMPPM.gpu_direct_multiparameter_modeler import GPUDirectMultiParameterModeler


def main():
    experiment = text_file_reader.read_text_file("testdata/two_parameter_1_extrap.txt")
    modeller = GPUDirectMultiParameterModeler()
    model = modeller.create_model(experiment.measurements[(Callpath('reg'), Metric('metr'))])
    print(model)


if __name__ == '__main__':
    main()
