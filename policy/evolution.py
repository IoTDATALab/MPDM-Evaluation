import copy
from concurrent.futures import ThreadPoolExecutor
from itertools import compress
from math import floor
from random import shuffle,choice

import numpy as np
from attr import attrib, attrs
from platypus import (HUX, MOEAD, NSGAII, PESA2, SPEA2, BitFlip, Constraint,
                      EpsMOEA, EpsNSGAII, GAOperator,Binary,
                      Problem, ProcessPoolEvaluator, MapEvaluator,
                      TournamentSelector, nondominated,
                      unique)
from switchlang import switch
from tqdm import tqdm
from .base import BasePolicy

class ShuffledHUX(HUX):
    def evolve(self, parents):
        temp_parents = copy.deepcopy(parents)
        shuffle(temp_parents[0].variables)
        return super(ShuffledHUX, self).evolve(temp_parents)


class CustomBinary(Binary):
    def __init__(self, nbits):
        super(CustomBinary, self).__init__(nbits)
        self.slices = np.eye(self.nbits + 1, self.nbits, dtype=bool)

    def rand(self):
        return choice(self.slices).tolist()


@attrs
class EvolutionaryAlgorithm(BasePolicy):
    nfe = attrib(type=int, default=10000)
    population_size = attrib(type=int, default=100)
    algorithm = attrib(type=str, default="nsgaii")
    parallelization = attrib(type=bool, default=False)

    def __attrs_post_init__(self):
        super(EvolutionaryAlgorithm, self).__attrs_post_init__()
        streams = len(self.system.end.streams)
        decisions = len(self.stream_decisions)
        nodes = len(self.system.edge.nodes)

        items = decisions * nodes

        constraint_list = []

        for node in range(nodes):
            resources = self.system.edge.nodes[node].resources
            constraint_list.append(Constraint("<=", resources.cpu))
            constraint_list.append(Constraint("<=", resources.ram))
            constraint_list.append(Constraint("<=", resources.gpu))
            constraint_list.append(Constraint("<=", resources.bandwidth))

        constraint_list.append(
            Constraint("<=", self.system.cloud.resources.bandwidth))

        problem = Problem(streams, 3, len(constraint_list))

        problem.types[:] = [CustomBinary(items) for _ in range(streams)]

        problem.constraints[:] = constraint_list
        problem.function = self.evaluation
        self.problem = problem

    # transform solution into system
    def solution_to_system(self, solution):
        system = copy.deepcopy(self.system)
        if solution.constraint_violation == 0:
            for index, variable in enumerate(solution.variables):
                if any(variable):
                    node_index = variable.index(True) % len(system.edge.nodes)
                    decision_index = floor(variable.index(True) / len(system.edge.nodes))
                    stream_decisions = copy.deepcopy(self.stream_decisions)
                    decision = stream_decisions[decision_index]
                    if decision.type != "PureCloud":
                        decision.edge_node = system.edge.nodes[node_index]
                    system.end.streams[index].decision = decision
                    system.add_placement(system.end.streams[index])
            point = {
                "average_accuracy": system.average_accuracy(),
                "total_accuracy": system.total_accuracy(),
                "served_ratio": system.served_ratio(),
                "served": len(system.served_streams()),
                "bandwidth_usage": system.bandwidth_usage(),
                "bandwidth_consumed": system.bandwidth_consumed(),
                "bandwidth_reserved_ratio": system.bandwidth_reserved_ratio(),
                "bandwidth_reserved": system.bandwidth_reserved()
            }
            return point, system

    def evaluation(self, stream_set):
        origin_streams = copy.copy(self.system.end.streams)
        origin_nodes = copy.copy(self.system.edge.nodes)
        stream_decisions = copy.copy(self.stream_decisions)

        streams = len(origin_streams)
        decisions = len(stream_decisions)
        nodes = len(origin_nodes)

        constraint_list = []

        accuracy_list = []
        cloud_bandwidth_list = []

        def compute(node):
            demands_list = []
            for stream in range(streams):
                for decision in range(decisions):
                    if stream_set[stream][decision * nodes + node]:
                        demands_list.append(
                            self.stream_decisions[decision].demands(
                                self.system.end.streams[stream]))
                        accuracy_list.append(
                            round(self.stream_decisions[decision].accuracy, 3))
            if len(demands_list) == 0:
                return [0, 0, 0, 0, 0]
            else:
                return np.sum(demands_list, axis=0)

        results = list(map(compute, range(nodes)))

        for result in results:
            constraint_list.extend(result[:-1])
            cloud_bandwidth_list.append(result[-1])

        constraint_list.append(sum(cloud_bandwidth_list))

        bandwidth_reserved = self.system.cloud.resources.bandwidth - sum(
            cloud_bandwidth_list)

        served = len(accuracy_list)
        average_accuracy = 0 if len(accuracy_list) == 0 else np.mean(
            accuracy_list)

        min_accuracy = self.system.accuracy.start
        max_accuracy = self.system.accuracy.end

        min_number = self.system.number.start
        max_number = self.system.number.end

        min_bandwidth = self.system.bandwidth.start
        max_bandwidth = self.system.bandwidth.end

        objectives = [
            round(-self.normalize(served, min_number, max_number), 3),
            round(
                -self.normalize(average_accuracy, min_accuracy, max_accuracy),
                3),
            round(
                -self.normalize(bandwidth_reserved, min_bandwidth,
                                max_bandwidth), 3)
        ]

        return objectives, constraint_list

    def solve_details(self):
        if self.parallelization:
            evaluator_class=ProcessPoolEvaluator
        else:
            evaluator_class=MapEvaluator
        with evaluator_class() as evaluator:

            def post_filtering(algorithm):
                solutions = unique(nondominated(algorithm.result))

                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(self.solution_to_system, solution)
                        for solution in solutions
                    ]
                    systems = [future.result() for future in futures]
                paretos = list(filter(None, systems))

                pareto_points = [point for point, _ in paretos]
                pareto_systems = [system for _, system in paretos]

                indexes = list(
                    map(
                        lambda _: all([
                            _['average_accuracy'] in self.system.accuracy, _['served']
                            in self.system.number, _['bandwidth_reserved'
                                                    ] in self.system.bandwidth
                        ]), pareto_points))

                pareto_points = list(compress(pareto_points, indexes))
                pareto_systems = list(compress(pareto_systems, indexes))

                return pareto_points, pareto_systems

            variator = GAOperator(
                ShuffledHUX(0.9),
                BitFlip(1))

            with switch(self.algorithm) as s:
                s.case(
                    "NSGAII",
                    lambda: NSGAII(self.problem,
                                   population_size=self.population_size,
                                   selector=TournamentSelector(2),
                                   variator=variator,
                                   evaluator=evaluator))
                s.case(
                    "EpsNSGAII",
                    lambda: EpsNSGAII(self.problem,
                                      population_size=self.population_size,
                                      selector=TournamentSelector(2),
                                      variator=variator,
                                      evaluator=evaluator,
                                      epsilons=0.05))
                s.case(
                    "MOEAD",
                    lambda: MOEAD(self.problem,
                                  population_size=self.population_size,
                                  variator=variator,
                                  evaluator=evaluator))
                s.case(
                    "SPEA2",
                    lambda: SPEA2(self.problem,
                                  population_size=self.population_size,
                                  variator=variator,
                                  evaluator=evaluator))
                s.case(
                    "EpsMOEA",
                    lambda: EpsMOEA(self.problem,
                                    population_size=self.population_size,
                                    variator=variator,
                                    evaluator=evaluator,
                                    epsilons=0.05))
                s.case(
                    "PESA2",
                    lambda: PESA2(self.problem,
                                  population_size=self.population_size,
                                  variator=variator,
                                  evaluator=evaluator))
            with tqdm(total=self.nfe,
                      desc="Evolution",
                      position=1,
                      leave=False) as pbar:
                algorithm = s.result

                def callback(alg):
                    last_print_n = pbar.last_print_n
                    pbar.update(alg.nfe - last_print_n)

                algorithm.run(self.nfe, callback)

                pareto_points, pareto_systems = post_filtering(algorithm)

        return pareto_points, pareto_systems

    def solve(self):
        return self.solve_details()
