from CSM.plane_truss import PlaneTrussProblem
from test_plane_truss import solved_right_triangle, solved_single_bar_diagonal
import numpy as np
#truss=PlaneTrussProblem.load_file("structure.yaml")
truss=PlaneTrussProblem.load_file("bridge.yaml")
#truss=solved_right_triangle()
print(truss.external_forces)

num_samples=20
xi_vals = np.linspace(-1, 1, num_samples)

# ── Global structural scale dimensions (used for arrows & boundary symbols) ──
all_stresses = []

for xi in xi_vals:
    all_stresses.append(truss.get_element_stress(1, xi))

mean=np.mean(all_stresses)
variance=np.var(all_stresses)
print(all_stresses)
print(mean)
print(variance)
truss.plot_plane_truss(show_deformed=True)