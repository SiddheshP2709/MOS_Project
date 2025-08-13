# Yield Prediction in Aircraft Wing Spar

## Overview

This Streamlit-based web application was developed as part of the **ES 221 course project** by a group of five students. It performs yield prediction in an aircraft wing spar modeled as a cantilevered I-beam.

The app allows users to define a distributed load, computes the resulting stress distribution, and evaluates yielding using both **von Mises** and **Tresca** criteria.

[Access the Yield Prediction Web App]((https://aircraftyieldpredictor.streamlit.app/))

---

## Features

- Input spar geometry and material yield strength  
- Define distributed loads as symbolic expressions over specific ranges  
- Calculates shear force and bending moment distributions  
- Computes stress distribution across the I-beam cross-section  
- Evaluates yielding using von Mises and Tresca criteria  
- Visualizes load distribution, shear force, bending moment, and yield zones  

---

## How to Use

1. Enter the yield strength of the material  
2. Specify spar dimensions: length, total height, flange height & width, and web width  
3. Define the distributed load using a symbolic expression (e.g., `1000*x`) and set the limits of the load segment  
4. Click "Calculate" to generate stress distributions and yield contour plots  

---

## Output Description

- **Load Distribution Plot** – Shows the applied load along the beam  
- **Shear Force Diagram** – Visualizes shear force V(x)  
- **Bending Moment Diagram** – Visualizes moment M(x)  
- **Von Mises & Tresca Stress Maps** – Contour plots with yielding regions highlighted  
- **Normal & Shear Stress Contour Plots** – Depict stress variation across the spar  
