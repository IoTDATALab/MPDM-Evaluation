from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from itertools import compress
from random import choice, shuffle, uniform

from attr import attrs, attrib
from pandas import DataFrame
from paretoset import paretoset
from platypus.evaluator import _chunks
from psutil import cpu_count
from scipy.spatial.distance import _validate_vector, _validate_weights
from tqdm import tqdm

import numpy as np

from .base import BasePolicy


def wchebyshev(u, v, w=None):
    u = _validate_vector(u)
    v = _validate_vector(v)
    u_v = u - v
    if w is not None:
        w = _validate_weights(w)
        root_w = w
        u_v = root_w * u_v
    return max(abs(u_v))


def c_random_weights(nobjs, size):
    weights = []

    if nobjs == 2:
        weights = [[1, 0], [0, 1]]
        weights.extend([(i / (size - 1.0), 1.0 - i / (size - 1.0))
                        for i in range(1, size - 1)])
    else:
        for i in range(size):
            random_values = [uniform(0.0, 1.0) for _ in range(nobjs)]
            weights.append([x / sum(random_values) for x in random_values])
        for i in range(nobjs):
            weights.append([0] * i + [1] + [0] * (nobjs - i - 1))
    return weights


@attrs
class HeuristicTAlgorithm(BasePolicy):
    s = attrib(type=int)
    parts = attrib(type=int, default=10)
    parallelization = attrib(type=bool, default=False)

    def __attrs_post_init__(self):
        super(HeuristicTAlgorithm, self).__attrs_post_init__()
        self.weights = c_random_weights(3, self.s)
        min_number = self.system.number.start
        max_number = self.system.number.end
        ranges = np.array_split(np.arange(min_number, max_number + 1),
                                min(self.parts, self.s) - 1)
        ranges = [
            tuple([min(array), max(array)]) for array in ranges
            if array.size > 0
        ]
        ranges.append(tuple([min_number, min_number]))
        self.bounds = ranges
        min_bandwidth = self.system.bandwidth.start
        max_bandwidth = self.system.bandwidth.end
        bw_ranges=[]
        step=(max_bandwidth-min_bandwidth)*1.0/(min(self.parts,self.s)-1)
        p=min_bandwidth
        q=min_bandwidth
        bw_ranges.append(tuple([p, q]))
        for _ in range(min(self.parts,self.s)-1):
            p=q
            q=q+step
            if max_bandwidth - q < step:
                q = max_bandwidth
            bw_ranges.append(tuple([p,q]))
        self.bw_ranges = bw_ranges

    def evaluate_objective(
        self,
        weight,
        bound,
        bw_range,
        system=None,
        stream=None,
        decision=None,
    ):
        temp_system = self.system if system is None else system
        min_accuracy = temp_system.accuracy.start
        max_accuracy = temp_system.accuracy.end

        min_number = temp_system.number.start
        max_number = temp_system.number.end

        min_bandwidth = temp_system.bandwidth.start
        max_bandwidth = temp_system.bandwidth.end

        if stream is None:
            if len(self.stream_decisions) == 1:
                return abs(
                    self.normalize(len(temp_system.served_streams()),
                                   min_number, max_number) - 1)

            return wchebyshev([1, 1, 1], [
                round(
                    self.normalize(temp_system.total_accuracy(), min_accuracy *
                                   min_number, max_accuracy * max_number), 3),
                round(
                    self.normalize(len(temp_system.served_streams()), bound[0],
                                   bound[1]), 3),
                round(
                    self.normalize(temp_system.bandwidth_reserved(),
                                   bw_range[0], bw_range[1]) if
                    self.system.paradigms != ['PureEdge'] else self.normalize(
                        temp_system.bandwidth_reserved(), min_bandwidth,
                        max_bandwidth), 3)
            ], weight)
        else:
            temp_total_accuracy = temp_system.total_accuracy(
            ) + decision.accuracy
            temp_total_number = len(temp_system.served_streams()) + 1
            temp_bandwidth_reserved = temp_system.cloud.resources.bandwidth - decision.demands(
                stream)[-1]

            if len(self.stream_decisions) == 1:
                return abs(
                    self.normalize(temp_total_number, min_number, max_number) -
                    1)

            return wchebyshev([1, 1, 1], [
                round(
                    self.normalize(temp_total_accuracy, min_accuracy *
                                   min_number, max_accuracy * max_number), 3),
                round(self.normalize(temp_total_number, bound[0], bound[1]),
                      3),
                round(
                    self.normalize(temp_bandwidth_reserved, bw_range[0],
                                   bw_range[1]) if
                    self.system.paradigms != ['PureEdge'] else self.normalize(
                        temp_bandwidth_reserved, min_bandwidth, max_bandwidth),
                    3)
            ], weight)

    def solve_once(self, weight):
        system = deepcopy(self.system)

        local_stream_decisions = deepcopy(self.stream_decisions)
        shuffle(local_stream_decisions)

        nodes = system.edge.nodes

        bound = choice(self.bounds)
        bw_range = choice(self.bw_ranges)

        for stream in system.end.streams:
            if len(local_stream_decisions) == 0:
                break

            partial_find_by_node = partial(system.find_feasible_decisions,
                                           decisions=local_stream_decisions,
                                           stream=stream)

            nodes = list(filter(partial_find_by_node, nodes))

            partial_find = partial(system.find_feasible_placement,
                                   stream=stream,
                                   nodes=nodes)

            feasible_decisions = list(
                filter(None, map(partial_find, local_stream_decisions)))

            if len(feasible_decisions) == 0:
                break

            ids = set(map(lambda _: _.id, feasible_decisions))
            local_stream_decisions = list(
                filter(lambda _: _.id in ids, local_stream_decisions))

            partial_evaluate_objective = partial(self.evaluate_objective,
                                                 weight=weight,
                                                 bound=bound,
                                                 bw_range=bw_range,
                                                 system=system)

            with ThreadPoolExecutor() as executor:
                evaluations = [
                    executor.submit(partial_evaluate_objective,
                                    stream=stream,
                                    decision=decision)
                    for decision in feasible_decisions
                ]
                min_index = -1
                min_value = float('inf')
                for index, future in enumerate(evaluations):
                    evaluation = future.result()
                    if evaluation <= min_value:
                        min_value = evaluation
                        min_index = index
                delta = min_value - partial_evaluate_objective()
                if delta <= 0:
                    result_stream_decision = feasible_decisions[min_index]
                    stream.decision = result_stream_decision
                    system.add_placement(stream)
                else:
                    break

        return system

    def solve_details(self):
        if self.parallelization:
            with ProcessPoolExecutor() as executor:
                with tqdm(total=len(self.weights),
                          desc="HeuristicT",
                          position=1,
                          leave=False) as pbar:
                    futures = [
                        executor.submit(self.solve_once, weight)
                        for weight in self.weights
                    ]
                    chunksize = cpu_count(True)

                    results = []
                    for chunk in _chunks(futures, chunksize):
                        delta_results = [f.result() for f in chunk]
                        results.extend(delta_results)
                        pbar.update(len(delta_results))
        else:
            results = []
            for weight in tqdm(self.weights, desc="HeuristicT"):
                solution = self.solve_once(weight)
                results.append(solution)

        systems = results

        points = list(
            map(
                lambda system: {
                    "average_accuracy": system.average_accuracy(),
                    "total_accuracy": system.total_accuracy(),
                    "served_ratio": system.served_ratio(),
                    "served": len(system.served_streams()),
                    "bandwidth_usage": system.bandwidth_usage(),
                    "bandwidth_consumed": system.bandwidth_consumed(),
                    "bandwidth_reserved_ratio": system.
                    bandwidth_reserved_ratio(),
                    "bandwidth_reserved": system.bandwidth_reserved()
                }, systems))

        indexes = list(
            map(
                lambda _: all([
                    _['average_accuracy'] in self.system.accuracy, _['served']
                    in self.system.number, _['bandwidth_reserved'
                                             ] in self.system.bandwidth
                ]), points))

        points = list(compress(points, indexes))
        systems = list(compress(systems, indexes))

        if len(points) > 0:
            dataset = DataFrame(points)

            mask = paretoset(dataset[[
                'average_accuracy', 'served_ratio', 'bandwidth_reserved_ratio'
            ]],
                             sense=["max", "max", "max"])
            masked_points = dataset[mask]
            pareto_systems = list(compress(systems, mask))
            pareto_points = masked_points.to_dict(orient='records')

            return pareto_points, pareto_systems
        else:
            return [], []

    def solve(self):
        return self.solve_details()
