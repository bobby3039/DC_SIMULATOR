import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import sympy as sp

# ------------------ Session State Initialization ------------------
if "G" not in st.session_state:
    st.session_state.G = nx.DiGraph()
if "edge_labels" not in st.session_state:
    st.session_state.edge_labels = {}
if "edge_properties" not in st.session_state:
    st.session_state.edge_properties = {}
if "done" not in st.session_state:
    st.session_state.done = False
if "node_count" not in st.session_state:
    st.session_state.node_count = 0

st.set_page_config(layout="centered")
st.title("DC CIRCUIT SIMULATOR")

# ------------------ Finalize Graph ------------------
if not st.session_state.done and st.button("Finish Graph"):
    st.session_state.done = True
    st.success("Graph finalized. Editing disabled.")

# ------------------ Add Node ------------------
if not st.session_state.done:
    if st.button("➕ Add Node"):
        st.session_state.node_count += 1
        n = st.session_state.node_count
        st.session_state.G.add_node(n)
        st.success(f"Node {n} added")

# ------------------ Add Edge ------------------
if not st.session_state.done:
    st.subheader("➕ Add Edge")
    nodes = list(st.session_state.G.nodes)
    if len(nodes) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            u = st.selectbox("From Node", nodes, key="u")
        with col2:
            v = st.selectbox("To Node", nodes, key="v")

        col3, col4 = st.columns(2)
        with col3:
            typ = st.selectbox("Component Type", ["R", "L", "C", "V"], key="type")
        with col4:
            val = st.number_input("Value", value=1.0, step=0.1, key="val")

        if st.button("➕ Add Edge"):
            if u == v:
                st.error("Self loops are not allowed")
            elif (u, v) in st.session_state.G.edges:
                st.error("Edge already exists")
            else:
                st.session_state.G.add_edge(u, v)
                label = f"{typ}={val}"
                st.session_state.edge_labels[(u, v)] = label
                st.session_state.edge_properties[(u, v)] = (typ, val)
                st.success(f"Edge {u} → {v} with {label} added")

# ------------------ Display Graph ------------------
st.subheader("📊 Circuit Graph")
if st.session_state.G.number_of_nodes() > 0:
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(st.session_state.G)
    nx.draw(st.session_state.G, pos, with_labels=True, node_color='lightgreen', node_size=800, font_size=14, ax=ax, arrows=True)
    nx.draw_networkx_edge_labels(st.session_state.G, pos, edge_labels=st.session_state.edge_labels, font_size=12, ax=ax)
    st.pyplot(fig)
else:
    st.info("Add nodes to begin drawing the circuit.")

# ------------------ Reset Graph ------------------
if st.button("🔁 Reset"):
    st.session_state.G = nx.DiGraph()
    st.session_state.edge_labels = {}
    st.session_state.edge_properties = {}
    st.session_state.done = False
    st.session_state.node_count = 0
    st.success("Graph reset.")

# ------------------ Display Data ------------------
st.subheader("📋 Graph Data")
st.write("Nodes:", list(st.session_state.G.nodes))
st.write("Edges and Components:")
for (u, v), lbl in st.session_state.edge_labels.items():
    st.write(f"{u} → {v}: {lbl}")

# if not st.session_state.done:
#     if st.button("Finish Graph"):
#         st.session_state.done = True
#         st.success("Graph finalized!")
#         st.experimental_rerun()  # Refresh to show the next step

if st.session_state.done:
    st.success("Graph is finalized.")
    st.info("👉 Now go to **'## Finalized Graph Details'** from the **sidebar** to see the summary.")
st.sidebar.title("Nav")


