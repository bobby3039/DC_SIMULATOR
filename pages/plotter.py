import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify

st.title("Voltage and Current vs Time")

# Check if simulation data is available
required_keys = ["time_domain_currents", "time_domain_voltages", "laplace_s"]
if not all(k in st.session_state for k in required_keys):
    st.warning("Missing data. Please simulate the circuit first.")
    st.stop()

# Retrieve from session_state
time_domain_currents_exprs = st.session_state["time_domain_currents"]
time_domain_voltages_exprs = st.session_state["time_domain_voltages"]

# Time input controls
end_time = st.number_input("End time", min_value=1.0, max_value=100.0, value=15.0)
#num_points = st.slider("Number of time points", 100, 2000, 1000)

# Define time values
t = symbols('t', real=True, positive=True)
time_values = np.linspace(0, end_time, 100000)

# Lambdify and evaluate currents
i_values = []
for expr in time_domain_currents_exprs:
    func = lambdify(t, expr, modules='numpy')
    result = func(time_values)
    i_values.append(result if np.ndim(result) else np.full_like(time_values, result))

# Lambdify and evaluate voltages
v_values = []
for expr in time_domain_voltages_exprs:
    try:
        func = lambdify(t, expr, modules='numpy')
        result = func(time_values)
        v_values.append(result if np.ndim(result) else np.full_like(time_values, result))
    except Exception as e:
        st.warning(f"Error evaluating voltage function: {e}")
        v_values.append(np.zeros_like(time_values))

# Plot currents
st.subheader("Currents $i_i(t)$ vs Time")
fig1, ax1 = plt.subplots(figsize=(15, 7.5))
for i, y in enumerate(i_values):
    ax1.plot(time_values, y, label=f'$i_{{{i+1}}}(t)$')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Current (A)')
ax1.set_title('Currents through edges')
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# Plot voltages
st.subheader("Voltages $v_i(t)$ vs Time")
fig2, ax2 = plt.subplots(figsize=(15, 7.5))
for i, y in enumerate(v_values):
    ax2.plot(time_values, y, label=f'$v_{{{i+1}}}(t)$')
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Voltage (V)')
ax2.set_title('Voltages across edges')
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)
