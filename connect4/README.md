### Connect4 with capped Minimax and Montecarlo playouts

Plays in terminal from folder.

Having numba installed is highly reccomendable but not necessary once the @njit decorators are removed from the `connect.py` and `minimax.py` files.

Does not implement a settings loop so to change difficulty go to minimax and change:

- SAMPLES to adjust the number of Montecarlo playouts, and 
- MAX for modifying the cap of the Minimax.
