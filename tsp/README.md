## Travelling Salesperson Problem  V 0

### Mutations

The code is largely based on the vanilla hill-climber presented during lecture, and inspired by the book by Eiben and Smith "Introduction to Evolutionary Computation".
A solution is here represented as a permutation of nodes to be visited by the salesperson. Due to time constraints, in this version only mutation operators are implemented: specifically all mutation operators presented in the book for permutation representations.

A fair comparison is thus performed between the swap-only (vanilla) hill-climber and one with a wider variety of operators to be mutated by.

### Initialisation

Being the problem NP-Hard, it is common to see it sub-optimally solved exploiting some heuristic. Thus, in this particular case it was deemed not unhelpful trying to initialize the solution through one of these.
Two heuristics are implemented:
- "Nearest neighbor", by which, starting from a given node, a path is constructed by always adding the closest node, and
- "Cheapest insertion", by which starting from an incomplete tour of the cities the next one is added in the position minimizing the cost of the tour.
The first heuristic proved to be quite fast and effective, with computation time staying realtively low in the number of cities, whereas the second was deemed impractical due to the long time it took just initialize over few hundreds of cities.

### Putting all together

The tests performed are thus the product between one vs many mutation operators and random vs heuristic initialization. Numbers of cities tested are part of a geometric space from 4 to 400.
It can be seen that the nearest neighbor heuristic solution becomes harder to improve upon as the number of cities increases (all the other variables being fixed). Nevertheless, it does improve the performaces of the regardless of the mutations available.
On the other hand, comparing randomly initialized hill-climbers, the larger set of operators does seem to lead to an improvement in performances with respect to vanilla.

![alt text](https://github.com/giovannipogg/computational_intelligence/tree/main/tsp/Figure_1.png)


### Populations

A small experiment with partenogenetic individuals under a 'plus' strategy was conducted.
A couple of tests were performed between random vs heuristic initialization (of each individual with different starting nodes) with the extended set of mutations against the vanilla hill-climber. Due to longer computational time the steady state was limited to just 100 and so was the reange of number of cities. In general, having more randomly initialized individuals makes it possible to further improving on the simpler solution, while the power of the heuristc makes it hard for a population initialized with it to make big improvements.
