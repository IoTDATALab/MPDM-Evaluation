import copy
import operator
from typing import List

from attr import attrs, attrib, asdict, astuple
from py_range_parse import Range

from .cloud import Cloud, CloudResource
from .edge import Edge, EdgeResource
from .end import End
import numpy as np


@attrs
class System(object):
    alpha = attrib(type=float)

    cloud = attrib(type=Cloud)
    edge = attrib(type=Edge)
    end = attrib(type=End)

    number = attrib(type=Range, default=None)
    accuracy = attrib(type=Range, default=None)
    bandwidth = attrib(type=Range, default=None)

    paradigms = attrib(factory=list, type=List[str])

    def served_streams(self):
        return list(
            filter(lambda stream: stream.decision is not None,
                   self.end.streams))

    def average_accuracy(self, extended_accuracies=None):
        served_streams = self.served_streams()
        if served_streams is None or len(served_streams) == 0:
            accuracies=[]
        else:
            accuracies = [stream.state.accuracy for stream in served_streams]

        if extended_accuracies:
            accuracies.extend(extended_accuracies)

        result= np.mean(accuracies) if len(accuracies)>0 else 0
        return round(result, 3)

    def median_accuracy(self,extended_accuracies=None):
        served_streams = self.served_streams()
        if served_streams is None or len(served_streams) == 0:
            accuracies = []
        else:
            accuracies = [stream.state.accuracy for stream in served_streams]

        if extended_accuracies:
            accuracies.extend(extended_accuracies)

        result = np.median(accuracies) if len(accuracies) > 0 else 0
        return round(result, 3)

    def total_accuracy(self, extended_accuracies=None):
        served_streams = self.served_streams()
        if served_streams is None or len(served_streams) == 0:
            accuracies = []
        else:
            accuracies = [stream.state.accuracy for stream in served_streams]

        if extended_accuracies:
            accuracies.extend(extended_accuracies)

        result = np.sum(accuracies) if len(accuracies) > 0 else 0
        return result

    def served_ratio(self):
        served_streams = self.served_streams()
        total_served = len(served_streams)
        total = len(self.end.streams)
        return round(total_served / total, 3)

    def bandwidth_consumed(self):
        return sum(map(lambda stream: stream.state.bandwidth,
                       self.end.streams))

    def bandwidth_usage(self):
        bandwidth_consumed = self.bandwidth_consumed()
        bandwidth_reserved = self.cloud.resources.bandwidth
        return round(
            bandwidth_consumed / (bandwidth_consumed + bandwidth_reserved), 3)

    def bandwidth_reserved(self):
        return self.cloud.resources.bandwidth

    def bandwidth_reserved_ratio(self):
        return 1 - self.bandwidth_usage()

    def add_placement(self, stream):
        if stream.decision is None:
            return
        edge_resource_demands = None
        cloud_resource_demands = None

        if stream.decision.type == "PureEdge":
            edge_resource_demands = EdgeResource(
                *astuple(stream.decision.edge_model.demands),
                stream.demands.bandwidth)
        elif stream.decision.type == "PureCloud":
            cloud_resource_demands = CloudResource(stream.demands.bandwidth)
        elif stream.decision.type == "ECS-CLB":
            edge_resource_demands = EdgeResource(
                *astuple(stream.decision.edge_model.demands),
                (1 - stream.decision.alpha) * stream.demands.bandwidth)
            cloud_resource_demands = CloudResource(stream.decision.alpha *
                                                   stream.demands.bandwidth)
        if edge_resource_demands is not None:
            stream.decision.edge_node.resources = stream.decision.edge_node.resources - edge_resource_demands
        if cloud_resource_demands is not None:
            self.cloud.resources = self.cloud.resources - cloud_resource_demands
            stream.state.update(stream.decision,
                                cloud_resource_demands.bandwidth)
        else:
            stream.state.update(stream.decision)

    def find_feasible_placement(self, stream_decision, stream, nodes=None):
        result_stream_decision = None
        nodes = copy.copy(
            self.edge.nodes) if nodes is None else copy.copy(nodes)

        bandwidth = stream.demands.bandwidth
        cloud_bandwidth = self.cloud.resources.bandwidth

        if stream_decision is None:
            return result_stream_decision

        if stream_decision.type == "PureCloud":
            if cloud_bandwidth >= bandwidth:
                result_stream_decision = stream_decision
        else:
            if len(nodes) == 0:
                return result_stream_decision
            if stream_decision.type == "ECS-CLB" and cloud_bandwidth < self.alpha * bandwidth:
                return result_stream_decision

            def check(node):
                demands = asdict(stream_decision.edge_model.demands)
                resources = asdict(node.resources)
                if stream_decision.type == "PureEdge":
                    demands['bandwidth'] = bandwidth
                elif stream_decision.type == "ECS-CLB":
                    demands['edge_bandwidth'] = bandwidth * (
                        1 - stream_decision.alpha)
                    demands['cloud_bandwidth'] = bandwidth - demands[
                        'edge_bandwidth']
                    resources['edge_bandwidth'] = resources.pop("bandwidth")
                    resources['cloud_bandwidth'] = cloud_bandwidth
                demands_vector = list(demands.values())
                resources_vector = list(resources.values())
                is_found = all([
                    a - b >= 0
                    for a, b in zip(resources_vector, demands_vector)
                ])
                return is_found, demands_vector, resources_vector

            result_index = None
            for index, node in enumerate(nodes):
                is_found, _, _ = check(node)
                if is_found:
                    result_index = index
                    break
            if result_index is not None:
                result_stream_decision = copy.deepcopy(stream_decision)
                result_stream_decision.edge_node = nodes[result_index]
        return result_stream_decision

    def find_feasible_decisions(self, node, decisions, stream):
        bandwidth = stream.demands.bandwidth

        def check(stream_decision):
            if stream_decision.type == "PureCloud":
                return False
            demands = asdict(stream_decision.edge_model.demands)
            resources = asdict(node.resources)
            if stream_decision.type == "PureEdge":
                demands['bandwidth'] = bandwidth
            elif stream_decision.type == "ECS-CLB":
                demands['edge_bandwidth'] = bandwidth * (1 -
                                                         stream_decision.alpha)
                resources['edge_bandwidth'] = resources.pop("bandwidth")
            demands_vector = list(demands.values())
            resources_vector = list(resources.values())
            is_found = all(
                [a - b >= 0 for a, b in zip(resources_vector, demands_vector)])
            return is_found

        return any(map(check, decisions))
