# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
from math import sqrt
from typing import Any, Dict
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Qt5Cairo")
import matplotlib.pyplot as plt
import itertools as it
from utils import *
import time as t

NUM_CITIES = 23
STEADY_STATE = 1000


class Tsp:

    def __init__(self, num_cities: int, seed: Any = None) -> None:
        if seed is None:
            seed = num_cities
        self._num_cities = num_cities
        self._graph = nx.DiGraph()
        np.random.seed(seed)
        for c in range(num_cities):
            self._graph.add_node(c, pos=(np.random.random(), np.random.random()))

    def distance(self, n1, n2) -> int:
        pos1 = self._graph.nodes[n1]['pos']
        pos2 = self._graph.nodes[n2]['pos']
        return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0])**2 +
                                                         (pos1[1] - pos2[1])**2))

    def evaluate_solution(self, solution: np.array) -> float:
        total_cost = 0
        tmp = solution.tolist() + [solution[0]]
        for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
            total_cost += self.distance(n1, n2)
        return total_cost

    def plot(self, path: np.array = None) -> None:
        if path is not None:
            self._graph.remove_edges_from(list(self._graph.edges))
            tmp = path.tolist() + [path[0]]
            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
                self._graph.add_edge(n1, n2)
        # Slight modifications for IDE plotting compatibility
        fig, ax = plt.subplots(figsize=(12, 5))
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink',
                ax=ax)
        if path is not None:
            ax.set_title(f"Current path: {self.evaluate_solution(path):,}")
        plt.show()

    @property
    def graph(self) -> nx.digraph:
        return self._graph


def tweak(solution: np.array, *, pm: float = .1) -> np.array:
    new_solution = solution.copy()
    p = None
    while p is None or p < pm:
        i1 = np.random.randint(0, solution.shape[0])
        i2 = np.random.randint(0, solution.shape[0])
        temp = new_solution[i1]
        new_solution[i1] = new_solution[i2]
        new_solution[i2] = temp
        p = np.random.random()
    return new_solution

# My mutations

def swap(solution: np.array, *, seed: int = None) -> np.ndarray:
    """Same as tweak mutation but always guaranteed to swap
    since replace=False.

    :param solution: the solution to mutate
    :param seed: for reproducibility
    :return: the tweaked solution
    """

    np.random.seed(seed)
    i1, i2 = np.random.choice(solution.shape[0], size=(2,), replace=False)
    solution[i1], solution[i2] = solution[i2], solution[i1]
    return solution

# insertion mutation strategies
def first_left(solution: np.array, i1: int, i2: int):
    """Swap strategy where element in i1 is inserted left of element in i2

    :param solution: the solution to mutate
    :param i1: leftward element to swap
    :param i2: rightward element to swap
    :return: the tweaked solution
    """

    tmp = solution[i1+1:i2].copy()
    solution[i2-1] = solution[i1]
    solution[i1:i2-1] = tmp
    return solution

def first_right(solution: np.array, i1: int, i2: int):
    """Swap strategy where element in i1 is inserted right of element in i2

    :param solution: the solution to mutate
    :param i1: leftward element to swap
    :param i2: rightward element to swap
    :return: the tweaked solution
    """

    tmp = solution[i1+1:i2+1].copy()
    solution[i2] = solution[i1]
    solution[i1:i2] = tmp
    return solution

def second_left(solution: np.array, i1: int, i2: int):
    """Swap strategy where element in i2 is inserted left of element in i1

    :param solution: the solution to mutate
    :param i1: leftward element to swap
    :param i2: rightward element to swap
    :return: the tweaked solution
    """

    tmp = solution[i1:i2].copy()
    solution[i1] = solution[i2]
    solution[i1+1:i2+1] = tmp
    return solution

def second_right(solution: np.array, i1: int, i2: int):
    """Swap strategy where element in i2 is inserted right of element in i1

    :param solution: the solution to mutate
    :param i1: leftward element to swap
    :param i2: rightward element to swap
    :return: the tweaked solution
    """

    tmp = solution[i1+1:i2].copy()
    solution[i1+1] = solution[i2]
    solution[i1+2:i2+1] = tmp
    return solution

def insert(solution: np.array, *, strategy = None,
           strategy_probs : np.ndarray = np.array([.25] * 4), seed: int = None) -> np.ndarray:
    """Mutation strategy in which of two alleles one is inserted next to the other and the values
    between them shifted as needed

    :param solution: the solution to mutate
    :param strategy: the strategy to adopt. If None a random one is picked.
    :param strategy_probs: If strategy is None, a random strategy is picked with these probabilities
                           in the order [first_left/right, second_left/right]
    :param seed: for reproducibility
    :return: teh mutated solution
    """

    np.random.seed(seed)
    i1, i2 = np.sort(np.random.choice(solution.shape[0], size=(2,), replace=False))
    if strategy is not None:
        solution = strategy(solution, i1, i2)
    else:
        strategy = np.random.choice(
            [first_left, first_right, second_left, second_right],
            p=strategy_probs
        )
        solution = strategy(solution, i1, i2)
    return solution

def scramble(solution: np.ndarray, *, seed: int = None):
    """Elements between 2 random alleles (included) are randomly shuffles.

    :param solution: the solution to mutate
    :param seed: for reproducibility
    :return: the mutated solution
    """
    np.random.seed(seed)
    i1, i2 = np.sort(np.random.choice(solution.shape[0], size=(2,), replace=False))
    np.random.shuffle(solution[i1:i2+1])
    return solution

def invert(solution: np.ndarray, *, seed: int = None):
    """Elements between 2 random alleles (included) are reversed in their order.

    :param solution: the solution to mutate
    :param seed: for reproducibility
    :return: the mutated solution
    """
    np.random.seed(seed)
    i1, i2 = np.sort(np.random.choice(solution.shape[0], size=(2,), replace=False))
    tmp = solution[i1:i2+1]
    solution[i1:i2+1] = tmp[::-1]
    return solution

def mutate(solution: np.array, *, pm: float = .1, strategy = swap,
           strategy_kwargs: Dict[str, Any] = None, seed: int = None) -> np.array:
    """Mutate function. Ineffective mutation given by probabilty of not entering
    the while clause

    :param solution: the solution to mutate.
    :param pm: the probability of mutation.
    :param strategy: the strategy with which to mutate.
    :param strategy_kwargs: the strategy arguments if needed.
    :param seed: for reproducibility
    :return: the mutated solution
    """
    np.random.seed(seed)
    new_solution = solution.copy()
    p = None

    while p is None or p < pm:
        if strategy_kwargs is not None:
            new_solution = strategy(new_solution, **strategy_kwargs)
            if "seed" in strategy_kwargs:
                strategy_kwargs["seed"] += 1
        else:
            new_solution = strategy(new_solution)
        p = np.random.random()

    return new_solution

def main():
    givens = []
    pops = []
    hpops = []

    rstarts = []
    rpstarts = []
    hpstarts = []
    nums_cities = np.geomspace(4, 100, 4).astype(int)

    for num_cities in nums_cities:
        print(f"\n\n# of cities: {num_cities}")
        problem = Tsp(num_cities)

        print("Given Solution")

        solution = np.array(range(num_cities))
        np.random.shuffle(solution)
        solution_cost = problem.evaluate_solution(solution)
        rstarts.append(solution_cost)
        print(f"Starting from: {solution_cost:,}")
        #problem.plot(solution)

        history = [(0, solution_cost)]
        steady_state = 0
        step = 0
        while steady_state < STEADY_STATE:
            step += 1
            steady_state += 1
            new_solution = tweak(solution, pm=.5)
            new_solution_cost = problem.evaluate_solution(new_solution)
            if new_solution_cost < solution_cost:
                solution = new_solution
                solution_cost = new_solution_cost
                history.append((step, solution_cost))
                steady_state = 0

        givens.append(solution_cost)
        print(f"Ending at:     {solution_cost:,} after {step:,} generations")
        #problem.plot(solution)


        n_individuals = max([num_cities // 2, 2])
        print(f"\nPopulation: {n_individuals}")

        population = [np.arange(n_individuals) for i in range(n_individuals)]
        for individual in population:
            np.random.shuffle(individual)
        population = [(individual, problem.evaluate_solution(individual)) for individual in population]

        best = min([fitness for _, fitness in population])
        rpstarts.append(best)
        print(f"Starting from: {best:,}")
        step = 0
        steady_state = 0
        while steady_state < STEADY_STATE:
            step += 1
            steady_state += 1
            N_NEXT = n_individuals // 2 + (n_individuals % 2)
            population = sorted(population, key=lambda x: x[1])
            population = population[:N_NEXT]
            offspring = [mutate(individual, strategy=strategy) \
                         for (individual, _), strategy in zip(population, np.random.choice(
                    [swap, scramble, invert, insert], size=(N_NEXT,)
                ))]
            offspring = [(individual, problem.evaluate_solution(individual)) for individual in offspring]
            population = population + offspring
            population = population[:n_individuals]
            min_ = min([fitness for _, fitness in population])
            if min_ < best:
                best = min_
                steady_state = 0
        population = sorted(population, key=lambda x: x[1])
        solution = population[0][0]
        solution_cost = population[0][1]
        pops.append(solution_cost)
        print(f"Ending at:     {solution_cost:,} after {step:,} generations")
        #problem.plot(solution)

        print(f"\nPopulation ({n_individuals}) + heuristics")
        nodes = nx.get_node_attributes(problem.graph, "pos")

        population = [nearest_neighbor(nodes, start_from=i) for i in range(n_individuals)]
        population = [(individual, problem.evaluate_solution(individual)) for individual in population]
        best = min([fitness for _, fitness in population])
        hpstarts.append(best)
        print(f"Starting from: {best:,}")
        step = 0
        steady_state = 0
        while steady_state < STEADY_STATE:
            step += 1
            steady_state += 1
            N_NEXT = n_individuals//2+(n_individuals%2)
            population = sorted(population, key=lambda x : x[1])
            population = population[:N_NEXT]
            offspring = [mutate(individual, strategy=strategy)\
                         for (individual, _), strategy in zip(population, np.random.choice(
                    [swap, scramble, invert, insert], size=(N_NEXT,)
                ))]
            offspring = [(individual, problem.evaluate_solution(individual)) for individual in offspring]
            population = population + offspring
            population = population[:n_individuals]
            min_ = min([fitness for _, fitness in population])
            if min_ < best:
                best = min_
                steady_state = 0
        population = sorted(population, key=lambda x: x[1])
        solution = population[0][0]
        solution_cost = population[0][1]
        hpops.append(solution_cost)
        print(f"Ending at:     {solution_cost:,} after {step:,} generations")
        #problem.plot(solution)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(nums_cities, givens, label="Vanilla")
    ax.plot(nums_cities, pops, label="RandPopulation")
    ax.plot(nums_cities, hpops, label="HeuPopulation")
    ax.plot(nums_cities, hpstarts, color="black", linestyle=":", label="HeuPopStart")
    ax.plot(nums_cities, rpstarts, color="grey", linestyle=":", label="RandPopStart")
    ax.plot(nums_cities, rstarts, color="silver", linestyle=":", label="RandStart")
    ax.set_xscale("log")
    ax.grid()
    ax.legend()
    plt.show()



def hill_climbers():
    # My Solution =====================================
    givens = []
    hgivens = []
    mmuts = []
    heumuts = []

    rstarts = []
    hstarts = []
    nums_cities = np.geomspace(4, 400, 10).astype(int)

    for num_cities in nums_cities:
        print(f"\n\n# of cities: {num_cities}")
        problem = Tsp(num_cities)

        # given solution
        print("\nGiven Solution")
        solution = np.array(range(num_cities))
        np.random.shuffle(solution)
        initial_solution = solution.copy()
        solution_cost = problem.evaluate_solution(solution)
        rstarts.append(solution_cost)
        print(f"Starting from: {solution_cost:,}")
        # problem.plot(solution)

        history = [(0, solution_cost)]
        steady_state = 0
        step = 0
        while steady_state < STEADY_STATE:
            step += 1
            steady_state += 1
            new_solution = tweak(solution, pm=.5)
            new_solution_cost = problem.evaluate_solution(new_solution)
            if new_solution_cost < solution_cost:
                solution = new_solution
                solution_cost = new_solution_cost
                history.append((step, solution_cost))
                steady_state = 0
        givens.append(solution_cost)
        print(f"Ending at:     {solution_cost:,}")
        # problem.plot(solution)

        # many mutations:
        print("\nMany mutations")

        solution = initial_solution
        solution_cost = problem.evaluate_solution(solution)
        print(f"Starting from: {solution_cost:,}")

        # problem.plot(solution)
        history = [(0, solution_cost)]
        steady_state = 0
        step = 0
        while steady_state < STEADY_STATE:
            step += 1
            steady_state += 1
            mutation_strategy = np.random.choice([swap, scramble, invert, insert])
            new_solution = mutate(solution, pm=.5, strategy=mutation_strategy)
            new_solution_cost = problem.evaluate_solution(new_solution)
            if new_solution_cost < solution_cost:
                solution = new_solution
                solution_cost = new_solution_cost
                history.append((step, solution_cost))
                steady_state = 0
        mmuts.append(solution_cost)
        print(f"Ended at:     {solution_cost:,}")
        # problem.plot(solution)

        # given solution with heuristic intialization
        print("\nGiven Solution + heuristic initialization")
        nodes = nx.get_node_attributes(problem.graph, "pos")
        solution = nearest_neighbor(nodes)
        initial_solution = solution.copy()
        solution_cost = problem.evaluate_solution(solution)
        hstarts.append(solution_cost)
        print(f"Starting from: {solution_cost:,}")
        # problem.plot(solution)

        history = [(0, solution_cost)]
        steady_state = 0
        step = 0
        while steady_state < STEADY_STATE:
            step += 1
            steady_state += 1
            new_solution = tweak(solution, pm=.5)
            new_solution_cost = problem.evaluate_solution(new_solution)
            if new_solution_cost < solution_cost:
                solution = new_solution
                solution_cost = new_solution_cost
                history.append((step, solution_cost))
                steady_state = 0
        hgivens.append(solution_cost)
        print(f"Ending at:     {solution_cost:,}")
        # problem.plot(solution)

        # heuristic initialization + many mutations:

        print("\nNearest-neighbor heuristc initialization + many mutations")

        solution = initial_solution
        solution_cost = problem.evaluate_solution(solution)
        print(f"Starting from: {solution_cost:,}")

        # problem.plot(solution)
        history = [(0, solution_cost)]
        steady_state = 0
        step = 0
        while steady_state < STEADY_STATE:
            step += 1
            steady_state += 1
            mutation_strategy = np.random.choice([swap, scramble, invert, insert])
            new_solution = mutate(solution, pm=.5, strategy=mutation_strategy)
            new_solution_cost = problem.evaluate_solution(new_solution)
            if new_solution_cost < solution_cost:
                solution = new_solution
                solution_cost = new_solution_cost
                history.append((step, solution_cost))
                steady_state = 0
        heumuts.append(solution_cost)
        print(f"Ended at:     {solution_cost:,}")
        # problem.plot(solution)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(nums_cities, givens, label="Vanilla")
    ax.plot(nums_cities, hgivens, label="Vanilla+Heuristics")
    ax.plot(nums_cities, mmuts, label="ManyMutations")
    ax.plot(nums_cities, heumuts, label="Hardcore")
    ax.plot(nums_cities, hstarts, color="black", linestyle=":", label="HeuStart")
    ax.plot(nums_cities, rstarts, color="grey", linestyle=":", label="RandStart")
    ax.set_xscale("log")
    ax.grid()
    ax.legend()
    plt.show()

def test_heuristics(): # test performance and speed of heuristic initializations
    nns, cis = [], []
    nnvs, civs = [], []
    xs = []

    for num_cities in range(4, 100):
        xs.append(num_cities)
        ci, nn = [], []
        civ, nnv = [], []
        for seed in range(5):
            problem = Tsp(num_cities, seed=seed)
            nodes = nx.get_node_attributes(problem.graph, "pos")

            t0 = t.time()
            solution = cheapest_insertion(nodes)
            t1 = t.time()
            ci.append(t1 - t0)
            civ.append(problem.evaluate_solution(solution))

            t0 = t.time()
            solution = nearest_neighbor(nodes)
            t1 = t.time()
            nn.append(t1 - t0)
            nnv.append(problem.evaluate_solution(solution))

        cis.append(np.mean(ci))
        nns.append(np.mean(nn))
        civs.append(np.mean(civ))
        nnvs.append(np.mean(nnv))

        print(num_cities)
        print(f"cheapest_insertion took: {cis[-1]*1000:.2f} ms (mean guess: {civs[-1]:.2f})")
        print(f"nearest_neighbor took:    {nns[-1]*1000:.2f} ms (mean guess: {nnvs[-1]:.2f})")

    fig, axs = plt.subplots(figsize=(14, 5), ncols=2)
    axs[0].plot(xs, cis, label="ci")
    axs[0].plot(xs, nns, label="nn")
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(xs, civs, label="ci")
    axs[1].plot(xs, nnvs, label="nn")
    axs[1].grid()
    axs[1].legend()
    plt.show()

def test_insert_mutation():
    solution = np.arange(10)
    seed = 42
    mutate(solution, pm=.5, strategy=insert, seed=seed, strategy_kwargs={"strategy":first_left, "seed":seed})
    mutate(solution, pm=.5, strategy=insert, seed=seed, strategy_kwargs={"strategy":first_right, "seed":seed})
    mutate(solution, pm=.5, strategy=insert, seed=seed, strategy_kwargs={"strategy":second_left, "seed":seed})
    mutate(solution, pm=.5, strategy=insert, seed=seed, strategy_kwargs={"strategy":second_right, "seed":seed})
    mutate(solution, pm=.5, strategy=insert, seed=seed)

def test_other_mutations():
    solution = np.arange(10)
    seed = 42
    mutate(solution, pm=.5, strategy=swap, seed=seed)
    mutate(solution, pm=.5, strategy=scramble, seed=seed)
    mutate(solution, pm=.5, strategy=invert, seed=seed)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    # test_heuristics()
    # test_insert_mutation()
    # test_other_mutations()
    main()