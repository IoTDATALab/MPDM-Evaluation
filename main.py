from copy import deepcopy
from functools import partial
from itertools import product
from os import makedirs
from os.path import exists
from time import strftime, localtime, time

from click import command, option
from epyc import Experiment
from tinydb import TinyDB
from tqdm import tqdm

from parse import load_data, parse_policy


class CustomExperiment(Experiment):
    def do(self, params):
        policy = parse_policy(params)
        points, systems = policy.solve()
        result = {
            'system_accuracies': [
                list(
                    map(lambda stream: stream.state.accuracy,
                        system.served_streams())) for system in systems
            ],
            'points':
            points,
            'constraints': {
                "number":
                tuple([policy.system.number.start, policy.system.number.end]),
                "accuracy":
                tuple([policy.system.accuracy.start, policy.system.accuracy.end]),
                "bandwidth":
                tuple([policy.system.bandwidth.start, policy.system.bandwidth.end]),
            }
        }
        return result


def run_experiment(params, db=None):
    e = CustomExperiment()
    rc = e.set(params).run()
    result = params['params']
    result.update(rc[e.RESULTS])
    result['experiment_time'] = rc[e.METADATA][e.EXPERIMENT_TIME]
    if db:
        database = TinyDB(db)
        database.insert(dict(result))
    return result


@command()
@option('--path',
        default="./tests/tests.yaml",
        help='Path to experiment setups in YAML format.')
def main(path):
    data = load_data(path)

    runs = data['params']['runs']
    runs_set = list(range(1, runs + 1))

    alphas = data['params']['alpha']
    paradigms_set = data['params']['paradigms']
    number_each_edge_types = data['params']['number_each_edge_type']
    edge_orders = data['params']['edge_order']

    policies = data['params']['policy']
    policy_list = []
    for policy in policies:
        if policy['algorithm'] in ["Heuristic", "HeuristicT","Augmecon", "WeightedSum"]:
            s_set = policy["s"]
            parts_set = policy["parts"]
            parallelization_set = policy["parallelization"]
            temp_params = list(
                product(alphas, number_each_edge_types, edge_orders,
                        paradigms_set, runs_set, s_set, parts_set,
                        parallelization_set))

            for params in temp_params:
                policy_list.append({
                    "algorithm": policy['algorithm'],
                    "alpha": params[0],
                    "number_each_edge_type": params[1],
                    "edge_order": params[2],
                    "paradigms": params[3],
                    "run": params[4],
                    "s": params[5],
                    "parts": params[6],
                    "parallelization": params[7]
                })
        else:
            nfes = policy['nfe']
            population_sizes = policy['population_size']
            parallelization_set = policy["parallelization"]

            temp_params = list(
                product(alphas, number_each_edge_types, edge_orders,
                        paradigms_set, runs_set, nfes,
                        population_sizes, parallelization_set))
            for params in temp_params:
                policy_list.append({
                    "algorithm": policy['algorithm'],
                    "alpha": params[0],
                    "number_each_edge_type": params[1],
                    "edge_order": params[2],
                    "paradigms": params[3],
                    "run": params[4],
                    "nfe": params[5],
                    "population_size": params[6],
                    "parallelization": params[7]
                })

    data.pop('params')
    setups = dict(data)

    experiments = []
    for policy in policy_list:
        experiment = deepcopy(setups)
        experiment['params'] = policy
        experiments.append(experiment)

    if not exists('plot/results'):
        makedirs('plot/results')

    timestamp = strftime('%Y%m%d%H%M%S', localtime(time()))
    db = "plot/results/results-database-" + timestamp + ".json"
    partial_run_experiment = partial(run_experiment, db=db)

    [
        partial_run_experiment(experiment)
        for experiment in tqdm(experiments, desc="Experiments", position=0)
    ]


if __name__ == '__main__':
    main()
