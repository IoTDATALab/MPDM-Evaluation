import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from os import system
from itertools import compress
from threading import Thread

import numpy as np
from attr import attrib, attrs
from paretoset import paretoset
from pyaugmecon import PyAugmecon
from pyaugmecon.flag import Flag
from pyaugmecon.model import Model
from pyaugmecon.options import Options
from pyaugmecon.process_handler import ProcessHandler
from pyaugmecon.queue_handler import QueueHandler
from pyomo.environ import (Binary, ConcreteModel, Constraint, ObjectiveList,
                           Var, maximize)
from tqdm import tqdm
from platypus.evaluator import _chunks
from psutil import cpu_count
from .base import BasePolicy


class CustomProcessHandler(ProcessHandler):
    def __init__(self, opts: Options, model: Model, queues: QueueHandler):
        super(CustomProcessHandler, self).__init__(opts, model, queues)
        self.procs = [
            CustomSolverProcess(p_num, self.opts, self.model, self.queues,
                                self.flag)
            for p_num in range(self.queues.proc_count)
        ]

class CustomSolverProcess(Thread):
    def __init__(self, p_num, opts: Options, model: Model,
                 queues: QueueHandler, flag: Flag):
        Thread.__init__(self)
        self.p_num = p_num
        self.opts = opts
        self.model = model
        self.queues = queues
        self.flag = flag

    def run(self):
        jump = 0
        if self.opts.process_logging:
            logger = logging.getLogger(self.opts.log_name)
            logger.setLevel(logging.INFO)

        self.model.unpickle()

        while True:
            work = self.queues.get_work(self.p_num)

            if work:
                for c in work:
                    log = f"Process: {self.p_num}, index: {c}, "

                    cp_end = self.opts.gp - 1

                    self.model.progress.increment()

                    def do_jump(i, jump):
                        return min(jump, abs(cp_end - i))

                    def bypass_range(i):
                        if i == 0:
                            return range(c[i], c[i] + 1)
                        else:
                            return range(c[i], c[i] + b[i] + 1)

                    def early_exit_range(i):
                        if i == 0:
                            return range(c[i], c[i] + 1)
                        else:
                            return range(c[i], cp_end)

                    if self.opts.flag and self.flag.get(c) != 0 and jump == 0:
                        jump = do_jump(c[0] - 1, self.flag.get(c))

                    if jump > 0:
                        jump = jump - 1
                        continue

                    for o in self.model.iter_obj2:
                        log += f"e{o + 1}: {self.model.e[o, c[o]]}, "
                        self.model.model.e[o + 2] = self.model.e[o, c[o]]

                    self.model.obj_activate(0)
                    self.model.solve()
                    self.model.models_solved.increment()

                    if self.opts.early_exit and self.model.is_infeasible():
                        self.model.infeasibilities.increment()
                        if self.opts.flag:
                            self.flag.set(early_exit_range, self.opts.gp,
                                          self.model.iter_obj2)
                        jump = do_jump(c[0], self.opts.gp)

                        log += "infeasible"
                        if self.opts.process_logging:
                            logger.info(log)
                        continue
                    elif self.opts.bypass and self.model.is_optimal():
                        b = []

                        for i in self.model.iter_obj2:
                            step = self.model.obj_range[i] / (self.opts.gp - 1)
                            slack = round(self.model.slack_val(i + 1))
                            b.append(int(slack / step))

                        log += f"jump: {b[0]}, "

                        if self.opts.flag:
                            self.flag.set(bypass_range, b[0] + 1,
                                          self.model.iter_obj2)
                        jump = do_jump(c[0], b[0])

                    sols = []

                    if self.model.is_optimal():
                        sols.append(
                            self.model.obj_val(0) - self.opts.eps *
                            sum(10**(-1 * (o)) * self.model.slack_val(o + 1) /
                                self.model.obj_range[o]
                                for o in self.model.iter_obj2))

                        for o in self.model.iter_obj2:
                            sols.append(self.model.obj_val(o + 1))

                        values_x = self.model.model.x.extract_values()

                        result = dict(sols=tuple(sols), system=values_x)

                        self.queues.put_result(result)

                        log += f"solutions: {sols}"
                        if self.opts.process_logging:
                            logger.info(log)
            else:
                break

class CustomPyAugmecon(PyAugmecon):
    def find_solutions(self):
        self.model.progress.set_message("finding solutions")

        grid_range = range(self.opts.gp)
        indices = [
            tuple([n for n in grid_range]) for _ in self.model.iter_obj2
        ]
        self.cp = list(itertools.product(*indices))
        self.cp = [i[::-1] for i in self.cp]

        self.model.pickle()
        self.queues = QueueHandler(self.cp, self.opts)
        self.queues.split_work()
        self.procs = CustomProcessHandler(self.opts, self.model, self.queues)

        self.procs.start()
        self.unprocesssed_sols = self.queues.get_result()
        self.procs.join()
        self.model.clean()

    def process_solutions(self):
        def convert_obj_goal(sols):
            return np.array(sols) * self.model.obj_goal

        def keep_undominated(pts):
            pts = np.array(pts)
            undominated = np.ones(pts.shape[0], dtype=bool)
            for i, c in enumerate(pts):
                if undominated[i]:
                    undominated[undominated] = np.any(pts[undominated] > c,
                                                      axis=1)
                    undominated[i] = True

            return pts[undominated, :]

        original_unprocesssed_sols = [
            item['sols'] for item in self.unprocesssed_sols
        ]
        # Remove duplicate solutions
        self.sols = list(set(tuple(original_unprocesssed_sols)))
        self.sols = [list(i) for i in self.sols]
        self.num_sols = len(self.sols)

        # Remove duplicate solutions due to numerical issues by rounding
        self.unique_sols = [
            tuple(round(sol, self.opts.round) for sol in item)
            for item in self.sols
        ]
        self.unique_sols = list(set(tuple(self.unique_sols)))
        self.unique_sols = [list(i) for i in self.unique_sols]
        self.num_unique_sols = len(self.unique_sols)

        # Remove dominated solutions
        self.unique_pareto_sols = keep_undominated(self.unique_sols)
        self.num_unique_pareto_sols = len(self.unique_pareto_sols)

        # Multiply by -1 if original objective was minimization
        self.model.payoff = convert_obj_goal(self.model.payoff)
        self.sols = convert_obj_goal(self.sols)
        self.unique_sols = convert_obj_goal(self.unique_sols)
        self.unique_pareto_sols = convert_obj_goal(self.unique_pareto_sols)

        self.systems = [item['system'] for item in self.unprocesssed_sols]


@attrs
class AugmeconAlgorithm(BasePolicy):
    s = attrib(type=int, default=100)
    parts = attrib(type=int, default=10)
    parallelization = attrib(type=bool, default=False)

    def __attrs_post_init__(self):
        super(AugmeconAlgorithm, self).__attrs_post_init__()
        min_number = self.system.number.start
        max_number = self.system.number.end
        # 对["PureEdge"]部署方式支持不好
        if self.parts > 1 and self.system.paradigms != ["PureEdge"]:
            ranges = np.array_split(np.arange(min_number, max_number + 1),
                                    min(self.parts, self.s) - 1)
            ranges = [
                tuple([min(array), max(array)]) for array in ranges
                if array.size > 0
            ]
            ranges.append(tuple([min_number, min_number]))
            bounds = [int(item[1]) for item in ranges]
            allocations = np.array_split(np.arange(1, self.s + 1), len(bounds))
            counts = [
                max(array.size, 2) for array in allocations if array.size > 0
            ]
            self.pairs = list(zip(bounds, counts))
        else:
            self.pairs = [tuple([max_number, self.s])]

    def solve_range(self, pair=None):
        if pair is None:
            bound = None
            count = 100
        else:
            bound = pair[0]
            count = pair[1]

        origin_streams = deepcopy(self.system.end.streams)
        origin_nodes = deepcopy(self.system.edge.nodes)
        stream_decisions = deepcopy(self.stream_decisions)

        streams = len(origin_streams)
        decisions = len(stream_decisions)
        nodes = len(origin_nodes)

        computed_demands = [
            service.demands(origin_streams[0]) for service in stream_decisions
        ]

        services = decisions

        model = ConcreteModel()
        model.Streams = range(streams)
        model.Nodes = range(nodes)
        model.Services = range(services)
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
            return served <= bound

        model.served_range = Constraint(rule=served_range_rule)

        # 不用归一化
        def objective1(model):
            sum_accuracy = sum(model.x[stream, node, service] *
                               stream_decisions[service].accuracy
                               for service in model.Services
                               for node in model.Nodes
                               for stream in model.Streams)
            return sum_accuracy

        # 不用归一化
        def objective2(model):
            bandwidth_reserved = self.system.cloud.resources.bandwidth - sum(
                model.x[stream, node, service] * computed_demands[service][-1]
                for service in model.Services for node in model.Nodes
                for stream in model.Streams)
            return bandwidth_reserved

        model.obj_list = ObjectiveList()
        model.obj_list.add(expr=objective1(model), sense=maximize)
        if self.system.paradigms != ['PureEdge']:
            model.obj_list.add(expr=objective2(model), sense=maximize)
        else:
            model.obj_list.add(expr=objective1(model), sense=maximize)

        for o in range(len(model.obj_list)):
            model.obj_list[o + 1].deactivate()

        opts = {
            'grid_points': count,
            'round_decimals': 3,
            'pickle_file': 'model.p.' + str(bound),
            "shared_flag": False,
            "redivide_work": False,
            "cpu_count": 1,
            "output_excel": False,
            "penalty_weight": 1e-6,
        }

        solver_opts = {"Threads": 1, "MIPGap": 1e-4}

        A = CustomPyAugmecon(model, opts, solver_opts)
        A.solve()
        solutions = A.systems
        return solutions

    def solve_details(self):
        if self.parallelization:
            with ProcessPoolExecutor() as executor:
                with tqdm(total=len(self.pairs),
                          desc="Augmecon",
                          position=1,
                          leave=False) as pbar:
                    futures = [
                        executor.submit(self.solve_range, pair)
                        for pair in self.pairs
                    ]
                    chunksize = cpu_count(True)

                    results = []
                    for chunk in _chunks(futures, chunksize):
                        delta_results = [f.result() for f in chunk]
                        [results.extend(result) for result in delta_results]
                        pbar.update(len(delta_results))
        else:
            results = []
            for pair in tqdm(
                    self.pairs,
                    desc="Augmecon",
            ):
                solutions = self.solve_range(pair)
                results.extend(solutions)

        solutions = results

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.solution_to_system, solution)
                for solution in solutions
            ]
            systems = [future.result() for future in futures]

        systems = list(filter(None, systems))

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

        temp_sols = [
            np.array([
                point["average_accuracy"], point["served"],
                point["bandwidth_reserved"]
            ]) for point in points
        ]
        mask = paretoset(np.vstack(temp_sols), sense=["max", "max", "max"])

        pareto_points = [point for (point, m) in zip(points, mask) if m]
        pareto_systems = [system for (system, m) in zip(systems, mask) if m]

        return pareto_points, pareto_systems

    # transform solution into system
    def solution_to_system(self, solution):
        system = deepcopy(self.system)
        solution = dict(solution)

        served = sum([variable for _, variable in solution.items()])
        if served == 0:
            return None

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
