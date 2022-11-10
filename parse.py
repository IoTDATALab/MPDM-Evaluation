from math import floor
from random import shuffle

from numpy import array, divide, finfo, max, maximum, sum
from yaml import SafeLoader, load

from core import *
from policy import (AugmeconAlgorithm, EvolutionaryAlgorithm,
                    HeuristicAlgorithm, HeuristicTAlgorithm,
                    WeightedSumAlgorithm)


def load_data(path):
    with open(path, 'rb') as file:
        data = file.read()
    return load(data, Loader=SafeLoader)


def parse_system(data):
    # seed(0)

    alpha = data['params']['alpha']
    paradigms = data['params']['paradigms']
    number_each_edge_type = data['params']['number_each_edge_type']
    edge_order = data['params']["edge_order"]

    cloud_resources = data['cloud']['resources']
    cloud_model = data['cloud']['model']
    cloud: Cloud = Cloud(resources=CloudResource(**cloud_resources),
                         model=CloudModel(**cloud_model))
    edge_nodes = []
    i = 0
    for node_type in data['edge']['nodes']:
        edge_resources = node_type['resources']
        for _ in range(number_each_edge_type):
            edge_nodes.append(
                EdgeNode(id=i, resources=EdgeResource(**edge_resources)))
            i += 1
    if edge_order == "OutOfOrder":
        shuffle(edge_nodes)

    edge_models = []

    for model in data['edge']['models']:
        demands = model['demands']
        edge_models.append(
            EdgeModel(id=model['id'],
                      demands=EdgeModelDemands(**demands),
                      accuracy=model['accuracy']))

    edge: Edge = Edge(nodes=edge_nodes, models=edge_models)

    stream_types = data['end']['streams']
    streams = []

    total_cloud_bandwidth = cloud.resources.bandwidth

    resource_pool = sum([
        array(list(asdict(node.resources).values()), dtype=float)
        for node in edge.nodes
    ],
                        axis=0)
    compute_resource_pool = resource_pool[:-1]
    eps = finfo(compute_resource_pool.dtype).eps
    adjusted_compute_resource_pool = maximum(compute_resource_pool, eps)

    def normalized_and_scalar_demands(vector):
        return max(
            divide(array(vector, dtype=float), adjusted_compute_resource_pool))

    minimum_scalar_demands = min([
        normalized_and_scalar_demands(list(asdict(model.demands).values()))
        for model in edge.models
    ])

    maximum_scalar_demands = max([
        normalized_and_scalar_demands(list(asdict(model.demands).values()))
        for model in edge.models
    ])

    demands = stream_types[0]['demands']
    if "PureCloud" in paradigms:

        min_stream_number = min(
            1 / maximum_scalar_demands,
            floor(total_cloud_bandwidth * 1.0 / demands['bandwidth']))
        max_stream_number = floor(1 / minimum_scalar_demands) + floor(
            total_cloud_bandwidth * 1.0 / demands['bandwidth'])

    else:
        min_stream_number = floor(1 / maximum_scalar_demands)
        max_stream_number = floor(1 / minimum_scalar_demands)

    min_number = min_stream_number
    max_number = max_stream_number

    min_accuracy = min([model.accuracy for model in edge.models])

    if "PureCloud" in paradigms:
        max_accuracy = cloud_model['accuracy']

    elif "ECS-CLB" in paradigms:
        max_accuracy = round(
            alpha * cloud_model['accuracy'] +
            (1 - alpha) * max([model.accuracy for model in edge.models]), 3)

    else:
        max_accuracy = max([model.accuracy for model in edge.models])

    min_bandwidth = 0
    max_bandwidth = total_cloud_bandwidth

    number = Range(min_number, max_number)
    accuracy = Range(float(min_accuracy), float(max_accuracy))
    bandwidth = Range(float(min_bandwidth),float(max_bandwidth))

    for i in range(max_stream_number):
        streams.append(
            Stream(id=i, demands=StreamDemands(**demands),
                   state=StreamState()))

    end: End = End(streams=streams)

    system: System = System(alpha, cloud, edge, end, number, accuracy,
                            bandwidth, paradigms)

    return system


def parse_policy(data):
    system = parse_system(data)
    if data['params']['algorithm'] == 'Heuristic':
        s = data['params']['s']
        parts=data['params']['parts']
        parallelization=data['params']['parallelization']
        policy = HeuristicAlgorithm(system, s, parts,parallelization)
    elif data['params']['algorithm'] == 'HeuristicT':
        s = data['params']['s']
        parts=data['params']['parts']
        parallelization=data['params']['parallelization']
        policy = HeuristicTAlgorithm(system, s, parts, parallelization)
    elif data['params']['algorithm'] == 'Augmecon':
        s = data['params']['s']
        parts=data['params']['parts']
        parallelization=data['params']['parallelization']
        policy = AugmeconAlgorithm(system, s, parts, parallelization)
    elif data['params']['algorithm'] == 'WeightedSum':
        s = data['params']['s']
        parts = data['params']['parts']
        parallelization = data['params']['parallelization']
        policy = WeightedSumAlgorithm(system, s, parts, parallelization)
    elif data['params']['algorithm'] in [
            "NSGAII", "SPEA2", "EpsMOEA", "MOEAD", "EpsNSGAII", "PESA2"
    ]:
        nfe = data['params']['nfe']
        population_size = data['params']['population_size']
        algorithm = data['params']['algorithm']
        parallelization = data['params']['parallelization']
        policy = EvolutionaryAlgorithm(system, nfe, population_size,
                                           algorithm,
                                           parallelization)

    return policy
