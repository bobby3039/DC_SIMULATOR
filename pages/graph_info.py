import streamlit as st
import networkx as nx
 
st.set_page_config(layout="centered")
st.title("Final Graph Info Page")
st.write("This is the graph info page.")


st.title("🧾 Finalized Graph Details")

if "done" in st.session_state and st.session_state.done:
    GRAPH = nx.DiGraph()
    for u, v in st.session_state.G.edges:
        GRAPH.add_edge(u, v)

    edge_properties = dict(st.session_state.edge_properties)

    st.write("Graph Nodes:", list(GRAPH.nodes))
    st.write("Graph Edges:", list(GRAPH.edges))
    st.write("Edge Properties:")
    for edge, properties in edge_properties.items():
        st.write(f"{edge}: {properties}")
else:
    st.warning("Graph is not finalized yet. Please finalize the graph first.")
