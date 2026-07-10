import pybamm
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Load parameters
# ------------------------------------------------
param = pybamm.ParameterValues("Chen2020")

# ------------------------------------------------
# 2. Get stoichiometry limits (CORRECT UNPACKING)
# ------------------------------------------------
x_n_min, x_n_max, x_p_min, x_p_max = \
    pybamm.lithium_ion.get_min_max_stoichiometries(param)

# ------------------------------------------------
# 3. Define SOC range
# ------------------------------------------------
soc = np.linspace(0.01, 0.99, 100)

# SOC → stoichiometry
theta_n = x_n_min + soc * (x_n_max - x_n_min)
theta_p = x_p_max - soc * (x_p_max - x_p_min)

# ------------------------------------------------
# 4. Evaluate electrode OCPs
# ------------------------------------------------
U_n = param["Negative electrode OCP [V]"](theta_n)
U_p = param["Positive electrode OCP [V]"](theta_p)

OCV = U_p - U_n

# ------------------------------------------------
# 5. Fit 5th-degree polynomial
# ------------------------------------------------
coeffs = np.polyfit(soc, OCV, deg=5)
ocv_poly = np.poly1d(coeffs)
OCV_fit = ocv_poly(soc)

# ------------------------------------------------
# 6. Plot
# ------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(soc * 100, OCV, "o", label="Actual OCV", markersize=4)
plt.plot(soc * 100, OCV_fit, "-", label="5th-degree Polynomial Fit")
plt.xlabel("State of Charge (%)")
plt.ylabel("Open Circuit Voltage (V)")
plt.title("OCV vs SOC (PyBaMM – Chen2020)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 7. SOC → Voltage function
# ------------------------------------------------
def voltage_from_soc(soc_input):
    soc_input = np.clip(soc_input, 0.0, 1.0)
    return ocv_poly(soc_input)


