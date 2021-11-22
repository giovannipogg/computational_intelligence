## Connect4 with capped Minimax and Montecarlo playouts

### Easier played than done... no wait...

Reccomended: having numba installed. Not necessary without the @njit decorators in the `connect.py` and `minimax.py` files.

Does not implement a settings loop so to change difficulty go to `minimax.py` and change:

- SAMPLES to adjust the number of Montecarlo playouts, and/or 
- MAX for modifying the cap of the Minimax

Warning: modiying them may significantly affect the delay of response.
