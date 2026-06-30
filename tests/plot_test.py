from CSM.plane_truss_experiment import PlaneEBBeamProblem
from CSM.plane_truss import PlaneTrussProblem
from test_plane_truss import solved_right_triangle, solved_single_bar_diagonal
import numpy as np
#truss=PlaneTrussProblem.load_file("bridge.yaml")
truss=PlaneEBBeamProblem.load_file("bridge.yaml")
#truss=solved_right_triangle()


num_samples=20
xi_vals = np.linspace(-1, 1, num_samples)

# ── Global structural scale dimensions (used for arrows & boundary symbols) ──

#print("global stiffness \n", truss.assemble_global_stiffness())
#print("reduced \n",truss.set_external_constraints(truss.constrained_dofs, 3 * truss.num_nodes))
#print("constraints ", truss.constrained_dofs)
#print("external forces \n", truss.external_forces)
#print("response forces \n", truss.get_reaction_forces())

all_stresses = []
element=0
for xi in xi_vals:
    all_stresses.append(truss.get_element_stress(element, xi))

displacements=[]
for xi in xi_vals:
    displacements.append(truss.get_displacement_on_element(element,xi))
#print(f"displacements on element {element} \n", displacements)
mean=np.mean(displacements)
variance=np.var(displacements)
print(f"displacements on element {element} \n", displacements)
print("mean of displacements", mean)
print("variance of displacements", variance)
truss.plot_plane_truss(show_deformed=True, scale_factor=10000, stress_to_plot="axial")# stress_to_plot="bending")