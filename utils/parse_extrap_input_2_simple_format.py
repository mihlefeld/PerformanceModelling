import sys

from extrap.entities.callpath import Callpath
from extrap.entities.metric import Metric
from extrap.modelers.multi_parameter.direct_multi_parameter_modeler import DirectMultiParameterModeler
from extrap.fileio import text_file_reader


def main():
    argc = len(sys.argv)
    if argc < 2:
        print("Usage:", sys.argv[0], "<input.txt> [<output.txt>]")
        return

    if argc >= 3:
        sys.stdout = open(sys.argv[2], 'w')

    experiment = text_file_reader.read_text_file(sys.argv[1])  # "tests/data/text/three_parameter_1.txt"
    experiment.measurements  # Dict[Tuple[Callpath, Metric], List[Measurement]]
    # print(len(experiment.callpaths))
    # print(len(experiment.metrics))
    length = len(experiment.callpaths)

    for i in range(length):
        measurements = experiment.measurements[(experiment.callpaths[i], experiment.metrics[i])]

        dimensions = len(measurements[0].coordinate)
        count = len(measurements)
        print("extrap measurements", dimensions, count)


        for measurement in measurements:
            for x in measurement.coordinate:
                print(x, end=" ")
            print(measurement.mean)  # measurement.median, measurement.minimum, measurement.maximum, measurement.std)

    # modeller = DirectMultiParameterModeler()
    # model = modeller.create_model(experiment.measurements[(Callpath('reg'), Metric('metr'))])


if __name__ == '__main__':
    main()
