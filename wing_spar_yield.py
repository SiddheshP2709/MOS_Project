import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
import re
from sympy import symbols, Piecewise, integrate, And, Function, cos, sin

################################################################################

st.write("""
         ##  Yield Prediction in Aircraft Wing Spar
         #### *Fill in the required information*""")
st.sidebar.header("Yield Prediction in Aircraft Wing Spar")
st.sidebar.caption("The main aim of this project is to predict the "
"*Yield Strength* of the wing spar of an aircraft using the data provided.")
st.sidebar.markdown("The data provided consists of the following columns: "
"Other info")
################################################################################

# material = st.radio(
#     "Select the spar material",
#     ("Aluminum", "Steel", "Composite"),
#     horizontal=True,
# )
# if material == "Aluminum":
#     Sigma_yield = 275e6  
#     st.write(  Sigma_yield)
# elif material == "Steel":
#     Sigma_yield = 370e6
#     st.write( Sigma_yield)
# elif material == "Composite":
#     Sigma_yield = 200e6
#     st.write( Sigma_yield)
Sigma_yield = st.number_input("Enter the yield strength of the material (Pa)", value=275e6, format="%f", step=1e6)

st.write("""
         ##### Enter the spar dimensions""")


col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    spar_length = st.number_input("Length(m)", value=100.0, format="%f",step=0.1)
with col2:
    h = st.number_input("Total Height(m)", value=10.0, format="%f",step=0.1)
with col3:
    f = st.number_input("Flange Height(m)", value=2.50, format="%f",step=0.1)
with col4:
    w = st.number_input("Flange Width(m)", value=5.0, format="%f",step=0.1)
with col5:
    r = st.number_input("Web Width(m)", value=3.0, format="%f",step=0.1)


load_segments = []
segment_limits = []

for i in range(1, 2):  # Assuming 1 segment
    col1, col2, col3 = st.columns(3)
    with col1:
        q = st.text_input(f"q{i} ", key=f"q{i}")
    with col2:
        lower_limit = st.text_input(f"Lower limit" , key=f"lower_{i}")
        lower_limit = float(lower_limit) if lower_limit else None  # Convert to float
    with col3:
        upper_limit = st.text_input(f"Upper limit ", key=f"upper_{i}")
        upper_limit = float(upper_limit) if upper_limit else None  # Convert to float
    if q:
        load_segments.append((q, lower_limit, upper_limit))

##########################################CALCULATION
x = symbols('x')
q_piecewise = []

#piecewise function
reaction_X0 = 0
moment_X0 = 0
for q_str, lower, upper in load_segments:
    q_expr = sp.sympify(q_str)  
    q_piecewise.append((q_expr, And(x >= lower, x < upper)))
    reaction_X0 = sum(integrate(q_expr, (x, lower, upper)) for q_str, lower, upper in load_segments)
    moment_X0 = sum(integrate(q_expr * x, (x, lower, upper)) for q_str, lower, upper in load_segments)
q_piecewise.append((0, True))  
q = Piecewise(*q_piecewise)

# st.write(reaction_X0)
# st.write(moment_X0)
y_vals = np.linspace(-h/2, 0, 25)
y____ = np.sort(abs(y_vals))
y_vals = np.concatenate((y_vals, y____))



x_min = min([lim[1] for lim in load_segments]) if load_segments else 0
x_max = max([lim[2] for lim in load_segments]) if load_segments else 20
x_vals = np.linspace(x_min, x_max, 100)
x_vals_plot = np.linspace(0, spar_length, 100)
V = integrate(-q, x) + reaction_X0  # Shear force function
M = integrate(-V, x) + moment_X0 # Moment function
V_vals = np.array([float(V.subs(x, val).evalf()) for val in x_vals_plot])
M_vals = np.array([float(M.subs(x, val).evalf()) for val in x_vals_plot])

# st.write("Shear Force Function V(x):", V)
# st.write("Bending Moment Function M(x):", M)

Izz = 2*(((w*f**3)/12 + ((w*f)*(h-f)**2)/4)) + (r*(h- 2*f)**3)/12

# st.write(y_vals)

def principle_stress(sigma_x, tau_xy):
    center = sigma_x /2
    radius = (center**2 + tau_xy**2)**0.5
    sigma1 = center + radius
    sigma2 = center - radius
    tau_max = radius
    return sigma1, sigma2, tau_max


sigmas =[]
Tresca = []
sigma_xx =[]
tau_xy =[]
sigma_y = 0
for j in range(len(y_vals)):
    for i in range(x_vals_plot.shape[0]):
        sigma_x = float(M.subs(x, x_vals[i]).evalf()) * y_vals[j] / Izz
        sigma_xx.append(sigma_x)
        if abs(y_vals[j]>= (h - f)):
            T = (float(V.subs(x, x_vals_plot[i]).evalf())/(Izz*w))*((w/2)*(((h/2)**2)-(y_vals[j]**2)))
        else:
            T = (float(V.subs(x, x_vals_plot[i]).evalf())/(Izz*r))*(((w/2)*((h*f)-(f**2))) + ((r/2)*(((h/2 - f)**2)-(y_vals[j]**2))))
        tau_xy.append(T)
        sigma_x_p, sigma_y_p, tau_xy_p = principle_stress(sigma_x, T)
        sigma = (((sigma_x_p**2)- (sigma_x_p*sigma_y_p) + (sigma_y_p**2) + (3*(tau_xy_p**2)))**0.5)
        sigmas.append(sigma)
        Tresca.append(tau_xy_p)
# st.write(sigmas)
# st.write(Tresca)

# sigma_matrix = np.array(sigmas).reshape(len(y_vals), len(x_vals))


if st.button("Calculate"):

    st.write("Shear force diagram")
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals_plot, V_vals, label='Shear Force V(x)', color='blue')
    plt.fill_between(x_vals_plot, V_vals, color='blue', alpha=0.1)
    plt.xlabel("x (m)")
    plt.ylabel("Shear Force V (N)")
    plt.grid()
    st.pyplot(plt)

    st.write("Bending moment diagram")
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals_plot, M_vals, label='Bending Moment M(x)', color='red')
    plt.fill_between(x_vals_plot, M_vals, color='red', alpha=0.1)
    plt.xlabel("x (m)")
    plt.ylabel("Bending Moment M (Nm)")
    plt.grid()
    st.pyplot(plt)

    # st.write("sigma_xx plot")
    st.write("Stress Intensity Visualization (von Mises and Tresca)")

    sigma_array = np.array(sigmas).reshape(len(y_vals), len(x_vals))
    tresca_array = np.array(Tresca).reshape(len(y_vals), len(x_vals))
    sigma_xx_array = np.array(sigma_xx).reshape(len(y_vals), len(x_vals))
    tau_xy_array = np.array(tau_xy).reshape(len(y_vals), len(x_vals))
    X, Y = np.meshgrid(x_vals_plot, y_vals)

    yield_map_vm = sigma_array >= Sigma_yield
    yield_map_tresca = tresca_array >= (Sigma_yield / 2)

    # plt.figure(figsize=(10, 5))
    # cp = plt.contourf(X, Y, sigma_array, cmap='RdYlGn_r')
    # plt.contour(X, Y, yield_map_vm, levels=[0.5], colors='white', linewidths=2)
    # plt.title("Von Mises Stress with Yield Contour")
    # plt.xlabel("x (m)")
    # plt.ylabel("y (m)")
    # plt.colorbar(cp, label='Von Mises Stress (Pa)')
    # st.pyplot(plt)

    plt.figure(figsize=(15, 5))
    cp = plt.contourf(X, Y, sigma_array, levels=200, cmap='RdYlGn_r')  # Added levels=200 for finer gradation
    plt.contour(X, Y, yield_map_vm, levels=[0.5], colors='white', linewidths=2)
    plt.title("Von Mises Stress with Yield Contour")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.colorbar(cp, label='RMS of difference of Principal Stresses (Pa) (For von Mises Criterion)')
    st.pyplot(plt)


    plt.figure(figsize=(15, 5))
    cp2 = plt.contourf(X, Y, tresca_array, levels = 200 ,cmap='viridis')
    plt.contour(X, Y, yield_map_tresca, levels=[0.5], colors='white', linewidths=2)
    plt.title("Tresca Stress with Yield Contour")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.colorbar(cp2, label='Maximum Shear Stress (Pa) (For Tresca Criterion)')
    st.pyplot(plt)

    # st.write(sigma_xx_array.shape)

    plt.figure(figsize=(15, 5))
    contour1 = plt.contourf(X, Y, sigma_xx_array, levels=100, cmap='coolwarm')
    plt.colorbar(contour1, label='MPa')
    plt.title('Normal Stress ($\\sigma_{xx}$)')
    plt.xlabel('Width (m)')
    plt.ylabel('Height (m)')
    # plt.axis('equal')
    plt.tight_layout()
    # plt.show()
    st.pyplot(plt)


    plt.figure(figsize=(15, 5))
    contour2 = plt.contourf(X, Y, tau_xy_array, levels=100, cmap='PuOr')
    plt.colorbar(contour2, label='MPa')
    plt.title('Shear Stress ($\\tau_{xy}$)')
    plt.xlabel('Width (m)')
    plt.ylabel('Height (m)')
    # plt.axis('equal')
    plt.tight_layout()
    # plt.show()
    st.pyplot(plt)

    

    # st.write("Moment of Inertia Izz:", Izz)

################################################################################################################################



