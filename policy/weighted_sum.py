from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from itertools import compress
from random import choice

import numpy as np
from attr import attrib, attrs
from pandas import DataFrame
from paretoset import paretoset
from platypus import random_weights
from platypus.evaluator import _chunks
from psutil import cpu_count
from pyomo.environ import (Binary, ConcreteModel, Constraint, Objective,
                           SolverFactory, SolverStatus, TerminationCondition,
                           Var, maximize)
from tqdm import tqdm

from .base import BasePolicy


@attrs
class WeightedSumAlgorithm(BasePolicy):
    s = attrib(type=int, default=100)
    parts = attrib(type=int, default=10)
    parallelization = attrib(type=bool, default=False)

    def __attrs_post_init__(self):
        super(WeightedSumAlgorithm, self).__attrs_post_init__()
        self.weights = random_weights(2, self.s)
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
            if max_bandwidth-q<step:
                q=max_bandwidth
            bw_ranges.append(tuple([p,q]))
        self.bw_ranges = bw_ranges
    def solve_once(self, weight):
        origin_streams = deepcopy(self.system.end.streams)
        origin_nodes = deepcopy(self.system.edge.nodes)
        stream_decisions = deepcopy(self.stream_decisions)

        streams = len(origin_streams)
        decisions = len(stream_decisions)
        nodes = len(origin_nodes)

        bound = choice(self.bounds)
        bw_range = choice(self.bw_ranges)

        computed_demands = [
            service.demands(origin_streams[0]) for service in stream_decisions
        ]

        services = decisions

        ecs = [
            index for index, decision in enumerate(stream_decisions)
            if decision.type == "ECS-CLB"
        ]

        model = ConcreteModel()
        model.Streams = range(streams)
        model.Nodes = range(nodes)
        model.Services = range(services)
        model.SynergyServices = set(ecs)
        model.x = Var(model.Streams,
                      model.Nodes,
                      model.Services,
                      within=Binary,
                      domain=Binary,
                      initialize=0)

        def single_stream_rule(model, stream):
            return sum(model.x[stream, :, :]) <= 1.0

        model.single_stream = Constraint(model.Streams,
                                         rule=single_stream_rule)

        if self.system.paradigms != ['PureCloud']:
            # "cpu"
            def single_node_cpu_rule(model, node):
                resources = self.system.edge.nodes[node].resources
                capacities = [
                    resources.cpu, resources.ram, resources.gpu,
                    resources.bandwidth
                ]
                return sum(model.x[stream, node, service] *
                           computed_demands[service][0]
                           for service in model.Services
                           for stream in model.Streams) <= capacities[0]

            model.single_node_cpu = Constraint(model.Nodes,
                                               rule=single_node_cpu_rule)

            # "ram"
            def single_node_ram_rule(model, node):
                resources = self.system.edge.nodes[node].resources
                capacities = [
                    resources.cpu, resources.ram, resources.gpu,
                    resources.bandwidth
                ]
                return sum(model.x[stream, node, service] *
                           computed_demands[service][1]
                           for service in model.Services
                           for stream in model.Streams) <= capacities[1]

            model.single_node_ram = Constraint(model.Nodes,
                                               rule=single_node_ram_rule)

            # "gpu"
            def single_node_gpu_rule(model, node):
                resources = self.system.edge.nodes[node].resources
                capacities = [
                    resources.cpu, resources.ram, resources.gpu,
                    resources.bandwidth
                ]
                return sum(model.x[stream, node, service] *
                           computed_demands[service][2]
                           for service in model.Services
                           for stream in model.Streams) <= capacities[2]

            model.single_node_gpu = Constraint(model.Nodes,
                                               rule=single_node_gpu_rule)

            # "bandwidth"
            def single_node_bandwidth_rule(model, node):
                resources = self.system.edge.nodes[node].resources
                capacities = [
                    resources.cpu, resources.ram, resources.gpu,
                    resources.bandwidth
                ]
                return sum(model.x[stream, node, service] *
                           computed_demands[service][3]
                           for service in model.Services
                           for stream in model.Streams) <= capacities[3]

            model.single_node_bandwidth = Constraint(
                model.Nodes, rule=single_node_bandwidth_rule)

        if self.system.paradigms != ['PureEdge']:

            def cloud_rule(model):
                return sum(model.x[stream, node, service] *
                           computed_demands[service][-1]
                           for service in model.Services
                           for node in model.Nodes for stream in model.Streams
                           ) <= self.system.cloud.resources.bandwidth

            model.cloud = Constraint(rule=cloud_rule)

        def served_range_rule(model):
            served = sum(model.x[stream, node, service]
                         for service in model.Services for node in model.Nodes
                         for stream in model.Streams)
            return served <= bound[1]

        model.served_range = Constraint(rule=served_range_rule)
        if self.system.paradigms != ['PureEdge']:

            def bandwidth_range_rule(model):
                bandwidth_reserved = self.system.cloud.resources.bandwidth - sum(
                    model.x[stream, node, service] *
                    computed_demands[service][-1] for service in model.Services
                    for node in model.Nodes for stream in model.Streams)
                return (bw_range[0], bandwidth_reserved, bw_range[1])

            model.bandwidth_range = Constraint(rule=bandwidth_range_rule)

        def objective(model):
            sum_accuracy = sum(model.x[stream, node, service] *
                               stream_decisions[service].accuracy
                               for service in model.Services
                               for node in model.Nodes
                               for stream in model.Streams)
            served = sum(model.x[stream, node, service]
                         for service in model.Services for node in model.Nodes
                         for stream in model.Streams)
            bandwidth_reserved = self.system.cloud.resources.bandwidth - sum(
                model.x[stream, node, service] * computed_demands[service][-1]
                for service in model.Services for node in model.Nodes
                for stream in model.Streams)

            min_accuracy = self.system.accuracy.start
            max_accuracy = self.system.accuracy.end

            min_number = self.system.number.start
            max_number = self.system.number.end

            min_bandwidth = self.system.bandwidth.start
            max_bandwidth = self.system.bandwidth.end

            objectives = [
                self.normalize(sum_accuracy, min_accuracy * min_number,
                               max_accuracy * max_number),
                self.normalize(bandwidth_reserved, min_bandwidth,
                               max_bandwidth)
            ]

            return weight[0] * objectives[0] + weight[1] * objectives[1]

        model.value = Objective(
            rule=objective,
            sense=maximize,
        )

        optimizer = SolverFactory("gurobi", solver_io="python")
        optimizer.options['MIPGap']=1e-4
        results = optimizer.solve(model, tee=False)

        if (results.solver.status
                == SolverStatus.ok) and (results.solver.termination_condition
                                         == TerminationCondition.optimal):
            solution = model.x.extract_values()
            return solution
        else:
            return None

    def solve_details(self, filtered=True):
        if self.parallelization:
            with ProcessPoolExecutor() as executor:
                with tqdm(total=len(self.weights),
                          desc="WeightedSum",
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
            for weight in tqdm(
                    self.weights,
                    desc="WeightedSum",
            ):
                solution = self.solve_once(weight)
                results.append(solution)

        solutions = list(filter(None, results))

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.solution_to_system, solution)
                for solution in solutions
            ]
            systems = [future.result() for future in futures]

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

    # transform solution into system
    def solution_to_system(self, solution):
        system = deepcopy(self.system)
        solution = dict(solution)

        for index, variable in solution.items():
            if variable == 1.0:
                stream_index = index[0]
                node_index = index[1]
                decision_index = index = index[2]
                stream_decisions = deepcopy(self.stream_decisions)
                decision = stream_decisions[decision_index]
                if decision.type != "PureCloud":
                    decision.edge_node = system.edge.nodes[node_index]
                system.end.streams[stream_index].decision = decision
                system.add_placement(system.end.streams[stream_index])
        return system

    def solve(self):
        return self.solve_details()
