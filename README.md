# DC Circuit Simulator


## How to use it
     
Check out the live version here: [https://blank-app-igpqlv707mf.streamlit.app/](https://blank-app-igpqlv707mf.streamlit.app/)


## Methods and Variables used for solving the circuit

The script defines a directed graph `GRAPH` that represents the circuit, and assigns properties to each edge in the graph, such as the type of component (voltage source, resistor, capacitor, or inductor) and its value.

The script then performs the following steps:

1. Constructs the minimum spanning tree (MST) of the graph.
2. Assigns a unique number to each edge in the graph, with the MST edges numbered first, followed by the "red" edges (edges not in the MST).
3. Initializes the tie-set matrix and the cut-set matrix, which are used to represent the relationships between the branch currents and voltages.
4. Solves for the branch currents and voltages in the Laplace domain, and then converts the results to the time domain using inverse Laplace transforms.
5. Plots the time-domain currents and voltages for each edge in the circuit.


The `logic.py` file defines the following functions and variables:

- `GRAPH`: a directed graph that represents the electrical circuit.
- `edge_properties`: a dictionary that maps each edge in the graph to its component type and value.
- `edge_numbering`: a dictionary that maps each edge in the graph to a unique number.
- `tie_set_matrix`: a matrix that represents the relationships between the branch currents and the link currents.
- `cut_matrix`: a matrix that represents the relationships between the branch voltages and the tree voltages.
- `branch_current_symbols`: a list of symbols representing the branch currents.
- `branch_voltage_symbols`: a list of symbols representing the branch voltages.
- `I_link`: a matrix of symbols representing the link currents.
- `V_tree`: a matrix of symbols representing the tree voltages.
- `time_domain_currents`: a list of functions that represent the time-domain branch currents.
- `time_domain_voltages`: a list of functions that represent the time-domain branch voltages.



### How to run it on your own machine

1. Install the requirements

 - `networkx`
- `numpy`
- `matplotlib`
- `sympy`
- `streamlit`

You can install these dependencies using pip:

```
pip install networkx numpy matplotlib sympy streamlit
```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

## Contributing

If you would like to contribute to this Circuit Simulator, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request.




