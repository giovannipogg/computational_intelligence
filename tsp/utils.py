import numpy as np
import itertools as it
from typing import Tuple, List, Dict


# My utils
def l2_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    x1, y1 = pos1
    x2, y2 = pos2

    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

# My initializations
def cheapest_insertion(nodes: Dict[int, Tuple[int, int]], start_from: int = 0) -> np.ndarray:
    """Starting from a 1-node sequence, i.e a non-hamiltonian cycle of length 0,
    the next node to be added is the one that minimizes the (next) cycle length

    :param nodes: the nodes of the graph
    :param start_from: the first node to consider
    :return: a sequence of nodes
    """

    L = len(nodes)  # total number of nodes
    seq = [start_from]  # sequence to be returned: initially a self-loop, i.e. a 1 node cycle
    anti_seq = [node for node in nodes.keys() if node != start_from]  # nodes not in the sequence
    l = 1  # initial length of the sequence in number of nodes
    current_length = 0.  # current length of the sequence: initially 0.

    while l != L:  # terminate when all nodes are added to seq

        lengths = []  # keep track of candidate cycles lengths
        with_ = []  # keep track which node is inserted where

        # for all candidate nodes and all candidate positions
        for node, i in it.product(anti_seq, range(l)):
            candidate = nodes[node]  # position of the candidate node
            after = nodes[seq[i]]  # position of the node after which it would be put
            before = nodes[seq[(i + 1) % l]]  # position of the node before which it would be put

            # the new candidate sub-sequence is the edge from after to candidate +
            #                                  the edge from candidate to before -
            #                                      the edge from after to before
            length = l2_distance(after, candidate) \
                     + l2_distance(candidate, before) \
                     - l2_distance(after, before)

            lengths.append(length)  # keeping track of candidate sub-sequences lengths ...
            with_.append((node, i + 1))  # ... if the candidate node were to be put in position i+1

        # then, just compute the new sequence based on the one minimizing the candidate sub-sequence length
        amin = np.argmin(lengths)
        node, pos = with_[amin]

        seq = seq[:pos] + [node] + seq[pos:]
        anti_seq.remove(node)
        current_length += lengths[amin]
        l += 1

    """Discussion:

    this initialization complexity, being N the number of nodes can be computed as

    (N-1) x 1     +     (# of candidates nodes)x(# of candidate positions)
    (N-2) x 2     +
                  ...
        1 x (N-1) =
    -----------------
    O(N^3)       
    """

    return np.array(seq)


def closest(from_: Tuple[float, float], anti_seq: List[int], nodes: Dict[int, Tuple[float, float]]) -> int:
    distances = []
    with_ = []

    for node in anti_seq:
        candidate = nodes[node]  # position of the candidate node
        distance = l2_distance(from_, candidate)  # distance to the last
        distances.append(distance)
        with_.append(node)
    amin = np.argmin(distances)
    added = with_[amin]

    return added


def forward(
        seq: List[int], anti_seq: List[int],
        nodes: Dict[int, Tuple[float, float]]) -> Tuple[List[int], int]:
    added = closest(nodes[seq[-1]], anti_seq, nodes)
    return seq + [added], added


def backward(
        seq: List[int], anti_seq: List[int],
        nodes: Dict[int, Tuple[float, float]]) -> Tuple[List[int], int]:
    added = closest(nodes[seq[0]], anti_seq, nodes)
    return [added] + seq, added


def both(
        seq: List[int], anti_seq: List[int],
        nodes: Dict[int, Tuple[float, float]]) -> Tuple[List[int], int]:
    last = closest(nodes[seq[-1]], anti_seq, nodes)
    first = closest(nodes[seq[0]], anti_seq, nodes)

    if l2_distance(nodes[seq[-1]], nodes[last]) < l2_distance(nodes[seq[0]], nodes[first]):
        return seq + [last], last
    return [first] + seq, first


def nearest_neighbor(
        nodes: Dict[int, Tuple[float, float]],
        start_from: int = 0,
        strategy=forward) -> np.ndarray:
    """Starting from a node, build partial solutions adding as a next
    node in the sequence the closest to the last.

    :param nodes:
    :param start_from:
    :return:
    """
    seq = [start_from]
    l = 1
    anti_seq = [node for node in nodes.keys() if node != start_from]
    L = len(nodes)

    while l < L:
        seq, added = strategy(seq, anti_seq, nodes)
        l += 1
        anti_seq.remove(added)

    return np.array(seq)