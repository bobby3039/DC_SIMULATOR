
import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# # DIRECTED GRAPH USED. (U,V) REPRESENTS AN EDGE  FROM NODE 'U' TO NODE 'V'.
# GRAPH = nx.DiGraph()

# #GRAPH CONSTRUCTION
# GRAPH.add_edge(1, 2)
# GRAPH.add_edge(2, 3)
# GRAPH.add_edge(3, 1)

# #V->VOLTAGE , R->RESISTANCE , C->CAPACITOR . ALL IN SI UNITS
# edge_properties = {
#     (1,2): ("V", 6),
#     (2,3): ("R", 2),
#     (3,1): ("C",1),
   
# }
st.title("Logic Page")

# Check if main page was run
# if "main_run" not in st.session_state:
#     st.warning("Please run the Main Page first.")
#     st.stop()

# # Your logic here
# st.success("Main page was run. Proceeding with logic.")

if "done" in st.session_state and st.session_state.done:
    GRAPH = nx.DiGraph()
    for u, v in st.session_state.G.edges:
        GRAPH.add_edge(u, v)

    edge_properties = dict(st.session_state.edge_properties)


s, t = sp.symbols('s t', real=True) # LAPLACE AND TIME  DOMAIN SYMBOLS

for (edge,(component_type, value)) in edge_properties.items():
    if component_type == "V":
        edge_properties[edge] = ("V", value / s)
    elif component_type == "C":
        edge_properties[edge] = ("C", 1 / (value * s))
    elif component_type == "L":
        edge_properties[edge] = ("L", s * value)


#  MINIMUM SPANNING TREE (CONSIDERED AS UNDIRECTED)
# 'MST' IS UNDIRECTED. 'GRAPH' WAS DIRECTED.
MST = nx.minimum_spanning_tree(GRAPH.to_undirected())

# VISUALIZING THE GRAPH  AND MST
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# pos = nx.spring_layout(GRAPH)
# nx.draw_networkx_nodes(GRAPH, pos, node_color="lightblue", node_size=520)
# nx.draw_networkx_edges(GRAPH, pos, edge_color="grey", arrows=True ,arrowsize=30)    # DIRECTED GRAPH IS SHOWN WITH GREY COLOR( i.e with arrow)
# nx.draw_networkx_labels(GRAPH, pos, font_size=15, font_family="DejaVu Sans")
# nx.draw_networkx_edges(MST, pos, edge_color="green", width=3)     # DIRECTED GRAPH IS SHOWN WITH GREY COLOR (i.e without arrow)
# plt.axis("off")
# plt.show()
fig, ax = plt.subplots()

# Generate layout
pos = nx.spring_layout(GRAPH)

# Draw the graph
nx.draw_networkx_nodes(GRAPH, pos, node_color="lightblue", node_size=520, ax=ax)
nx.draw_networkx_edges(GRAPH, pos, edge_color="grey", arrows=True, arrowsize=30, ax=ax)
nx.draw_networkx_labels(GRAPH, pos, font_size=15, font_family="DejaVu Sans", ax=ax)
nx.draw_networkx_edges(MST, pos, edge_color="green", width=3, ax=ax)

# Hide axis
ax.axis("off")

# Show on Streamlit
st.pyplot(fig)



# NUMBERING OF EDGES. ( CONVENTION :FIRST NUMBERING IS DONE FOR MST BRANCHES . THEN FOR 'RED' BRANCHES...)
# (CONVENTION : 'RED' SYMBOL IS IS USED TO DENOTE THE BRANCHES WHICH IS NOT THE PART OF MST  i.e co-tree branches).

edge_numbering = {} #DICTIONARY  FOR (U,V)->NUMBER ASSIGNED
edge_counter = 1

#  GETTING A NEW GRAPH WHICH IS DIRECTED FORM OF MST
directed_MST = nx.DiGraph()
for (u,v) in MST.edges():
    if GRAPH.has_edge(u,v):
        directed_MST.add_edge(u,v)
    else:
        directed_MST.add_edge(v,u)
    edge_numbering[(u,v) if GRAPH.has_edge(u,v) else (v,u)] = edge_counter
    edge_counter+=1

# NUMBERING THE RED EDGES 
for edge in GRAPH.edges():
    if edge not in directed_MST.edges():
        edge_numbering[edge] = edge_counter
        edge_counter+=1

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# print("Edge numbering (MST edges first, then RED edges), including properties:")
# for (edge,number) in edge_numbering.items():
#     (component,value) = edge_properties[edge]
#     print(f" Edge {edge} : Number {number}, Component: {component}, Value: {value} ")

st.markdown("### Edge Numbering (MST edges first, then RED edges), including properties:")

for (edge, number) in edge_numbering.items():
    (component, value) = edge_properties[edge]
    st.write(f"Edge {edge} : Number {number}, Component: {component}, Value: {value}")


################################################################################################################################


# TIE-SET MATRIX INITIALISATION
num_total_edges = len(GRAPH.edges())
num_MST_edges =len(MST.edges())
num_red_edges = ( num_total_edges - num_MST_edges ) #i.e. no of co-tree edges
tie_set_matrix = np.zeros( (num_red_edges, num_total_edges), dtype=int )



st.markdown("### Cycles formed by adding each red edge to the MST:")
cycle_index = 0  # Row index for tie-set matrix
# Iterating through each red edge to find the cycles formed by adding it to the MST
for edge in GRAPH.edges():
    if edge not in directed_MST.edges():  # Red edge

        # Creating a temporary directed graph with (directed_MST + red edge)
        temp_graph = directed_MST.copy()
        temp_graph.add_edge(*edge)

        # Converting to undirected graph to detect cycle
        undirected_temp = temp_graph.to_undirected()

        try:
            cycle = nx.find_cycle(undirected_temp, source=edge[0])
            st.write(f"🔴 Red edge {edge} (Number {edge_numbering[edge]}) forms cycle: {cycle}")

            # Filling the tie-set matrix
            for (u, v) in cycle:
                if (u, v) in edge_numbering:
                    col = edge_numbering[(u, v)] - 1
                    tie_set_matrix[cycle_index][col] = -1
                elif (v, u) in edge_numbering:
                    col = edge_numbering[(v, u)] - 1
                    tie_set_matrix[cycle_index][col] = 1

            cycle_index += 1

        except nx.NetworkXNoCycle:
            st.warning(f"⚠️ Red edge {edge} (Number {edge_numbering[edge]}) does not form a cycle.")


cut_matrix = np.zeros( ( len(MST.edges()) , num_total_edges ), dtype=int )
def fill_matrix_with_identity(n, m):
    identity_matrix = np.eye(n)
    result_matrix = np.zeros((n, m), dtype=int)
    result_matrix[:, :n] = identity_matrix
    return result_matrix
    
cut_matrix = fill_matrix_with_identity(num_MST_edges,  num_total_edges)

for i in range(num_red_edges):
    for j in range(num_MST_edges):
        cut_matrix[j][num_MST_edges+ i] = -tie_set_matrix[i][j]

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# #  tie-set matrix
# print("\nTie-set matrix (rows correspond to removed edges, columns correspond to all edges):")
# print(tie_set_matrix)
# # cut-set matrix
# print("\nCut matrix:")
# print(cut_matrix)
st.markdown("### Tie-set Matrix")
st.write(tie_set_matrix)

# Display cut-set matrix (simplified)
st.markdown("### Cut-set Matrix")
st.write(cut_matrix)

################################################################################################################################

# Number of edges and the number of link currents
#num_total_edges = len(GRAPH.edges())
# num_link_currents = len(GRAPH.edges()) - len(MST.edges())

# CREATE SYMBOLS DYNAMICALLY FOR BRANCH CURRENTS AND VOLTAGES
branch_current_symbols = sp.symbols(f'i1:{num_total_edges + 1}')  # i1, i2, ..., i(num_total_edges)
branch_voltage_symbols = sp.symbols(f'v1:{num_total_edges + 1}')  # v1, v2, ..., v(num_total_edges)

# ASSIGNING SYMBOLS FOR LINK CURRENTS (UPPERCASE I) AND TREE VOLTAGES (UPPERCASE V)
I_link = sp.Matrix([sp.symbols(f'I{i}') for i in range(num_MST_edges + 1, num_total_edges + 1)])  # LINK CURRENTS
V_tree = sp.Matrix([sp.symbols(f'V{i}') for i in range(1, num_MST_edges + 1)])  # MST VOLTAGES

# RELATIONS PROVIDED:
# 1. [TIE_SET_MATRIX]^T * I_LINK = I_EACH_EDGE
I_each_edge = sp.Matrix(branch_current_symbols)  # ALL CURRENT SYMBOLS (LOWER CASE i)
tie_set_relation = sp.Eq((tie_set_matrix.T * I_link), I_each_edge)

# 2. [CUT_SET_MATRIX]^T * V_TREE = V_EACH_EDGE
V_each_edge = sp.Matrix(branch_voltage_symbols)  # ALL VOLTAGE SYMBOLS (LOWER CASE v)
cut_set_relation = sp.Eq(cut_matrix.T * V_tree, V_each_edge)

# DISPLAY RELATIONS
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# print("TIE-SET RELATION (FOR CURRENTS):")
# sp.pprint(tie_set_relation)

# print("\nCUT-SET RELATION (FOR VOLTAGES):")
# sp.pprint(cut_set_relation)

# Tie-set Relation (Currents)
st.markdown("###  Tie-set Relation (for Currents)")
# Display the single expression directly if it's not a list
if isinstance(tie_set_relation, sp.Basic):
    st.latex(sp.latex(tie_set_relation))
else:
    for expr in tie_set_relation:
        st.latex(sp.latex(expr))

# Cut-set Relation (Voltages)
st.markdown("### Cut-set Relation (for Voltages)")
# Display the single expression directly if it's not a list
if isinstance(cut_set_relation, sp.Basic):
    st.latex(sp.latex(cut_set_relation))
else:
    for expr in cut_set_relation:
        st.latex(sp.latex(expr))
##################################################################################################################################

# Adding relations for capital voltages based on edge properties
capital_V_relationships = {}  # KEY: EDGE NUMBER, VALUE: Voltage or i*R expression

for (edge, number) in edge_numbering.items():
    component, value = edge_properties[edge]
    if component == "V":  # Voltage source
        capital_V_relationships[number] = value
    else:  # Resistor
        capital_V_relationships[number] = sp.symbols(f'i{number}') * value

# Step 2: Substitute V_tree symbols with expressions from capital_V_relationships
for j in range(len(V_tree)):
    capital_index = j + 1  # Since numbering starts at 1
    if capital_index in capital_V_relationships:
        cut_set_relation = cut_set_relation.subs(V_tree[j], capital_V_relationships[capital_index])

# Step 3: Display the cut-set relation (voltage equation before current substitution)
# st.write("### Cut-set relation (for voltages):")
# st.latex(sp.latex(cut_set_relation))

# Step 4: Create a system of tie-set equations and solve for small currents
branch_current_symbols = sp.symbols(f'i1:{num_total_edges + 1}')  # i1, i2, ..., iN
tie_set_eqs = [tie_set_relation.lhs[i] - tie_set_relation.rhs[i] for i in range(len(tie_set_relation.lhs))]
small_currents_solution = sp.solve(tie_set_eqs, branch_current_symbols)

# Step 5: Substitute small currents into the cut-set relation
for j in range(len(I_each_edge)):
    if branch_current_symbols[j] in small_currents_solution:
        cut_set_relation = cut_set_relation.subs(branch_current_symbols[j], small_currents_solution[branch_current_symbols[j]])

# Step 6: Display the final modified cut-set relation
# st.write("### Modified Cut-set relation (after substituting iⱼ in terms of capital Iⱼ):")
# st.latex(sp.latex(cut_set_relation))



# ##################################################################################################################################


k = len(GRAPH.edges()) - len(MST.edges())  #NUMBER OF RED EDGES

# Extract the last k equations from the cut-set relation
cut_set_last_k_eqs = cut_set_relation.lhs[-k:]
cut_set_last_k_rhs = cut_set_relation.rhs[-k:]

# Display the last k equations (LHS and RHS)
# print(f"\nLast {k} equations of the modified Cut-set relation:")
# for (lhs, rhs) in zip(cut_set_last_k_eqs, cut_set_last_k_rhs):
#     sp.pprint(sp.Eq(lhs, rhs))



# Replace RHS elements of the last k equations based on edge properties
modified_cut_set_last_k_eqs = []

for (lhs,rhs) in zip(cut_set_last_k_eqs, cut_set_last_k_rhs):
    # Iterate through all edges and update RHS terms
    updated_rhs = rhs
    for (edge,number) in edge_numbering.items():
        (component,value) = edge_properties[edge]

        # Replace v_i based on edge properties
        v_symbol = sp.symbols(f'v{number}')  # Symbol for voltage v_i
        if component == "V":  # Voltage source
            updated_rhs = updated_rhs.subs(v_symbol, value)  
        elif component in {"R", "C", "L"}:  
            I_symbol = sp.symbols(f'I{number}')  # Symbol for current I_i
            updated_rhs = updated_rhs.subs(v_symbol, value * I_symbol)  # Replace v_i with value * I_i

    # Append the modified equation to the list
    modified_cut_set_last_k_eqs.append(sp.Eq(lhs, updated_rhs))


# print(f"\nModified Last {k} equations of the Cut-set relation (RHS replaced):")
# for eq in modified_cut_set_last_k_eqs:
#     sp.pprint(eq)
I_symbols = [sp.symbols(f'I{len(MST.edges()) + i + 1}') for i in range(k)]  # Link currents

# Extract LHS and RHS for solving
equations = [eq.lhs - eq.rhs for eq in modified_cut_set_last_k_eqs]

# Solve the equations for I_i
solutions = sp.solve(equations, I_symbols)

# Display the solutions for each I_i

print("\nSolutions for I_i (link currents) in terms of s (direction changed):")
for (I_symbol, solution) in solutions.items():
    sp.pprint(sp.Eq(I_symbol, (-1)*solution))

# all i_i (currents through edges) in terms of s
all_currents = []
for i, branch_current in enumerate(I_each_edge):
    current_expr = (-1)*tie_set_relation.lhs[i]  # Directly use the LHS of the tie-set relation
    for I_symbol, solution in solutions.items():
        current_expr = current_expr.subs(I_symbol, solution)  # Substitute independent I_i values
    all_currents.append(sp.simplify(current_expr))

#  all v_i (voltages across edges) in terms of s
all_voltages = []
for (i, branch_voltage) in enumerate(V_each_edge):
    voltage_expr = cut_set_relation.lhs[i]  # Directly use the LHS of the cut-set relation
    for (I_symbol, solution) in solutions.items():
        voltage_expr = voltage_expr.subs(I_symbol, solution)  # Substitute independent I_i values
    all_voltages.append(sp.simplify(voltage_expr))

# Display results
print("\nAll currents (i_i) through edges in terms of s:")
for i, current_expr in enumerate(all_currents, start=1):
    sp.pprint(sp.Eq(sp.symbols(f'i{i}'), current_expr))

print("\nAll voltages (v_i) across edges in terms of s:")
for i, voltage_expr in enumerate(all_voltages, start=1):
    sp.pprint(sp.Eq(sp.symbols(f'v{i}'), voltage_expr))

# ################################################################################################################################


import streamlit as st
from sympy import inverse_laplace_transform, symbols, simplify, Eq, latex

t = symbols('t', real=True, positive=True)

# TIME-DOMAIN CURRENTS i_i(t)
time_domain_currents = []
for i, current_expr in enumerate(all_currents):
    current_time_expr = inverse_laplace_transform(current_expr, s, t)
    time_domain_currents.append(simplify(current_time_expr))

# TIME-DOMAIN VOLTAGES v_i(t)
time_domain_voltages = []
for i, voltage_expr in enumerate(all_voltages):
    voltage_time_expr = inverse_laplace_transform(voltage_expr, s, t)
    time_domain_voltages.append(simplify(voltage_time_expr))

# DISPLAY RESULTS IN STREAMLIT
st.subheader("All Currents $i_i(t)$ through edges:")
for i, current_time_expr in enumerate(time_domain_currents, start=1):
    st.latex(latex(Eq(symbols(f'i{i}(t)'), current_time_expr)))

st.subheader("All Voltages $v_i(t)$ across edges:")
for i, voltage_time_expr in enumerate(time_domain_voltages, start=1):
    st.latex(latex(Eq(symbols(f'v{i}(t)'), voltage_time_expr)))

st.session_state["time_domain_currents"] = time_domain_currents
st.session_state["time_domain_voltages"] = time_domain_voltages
st.session_state["laplace_s"] = s








