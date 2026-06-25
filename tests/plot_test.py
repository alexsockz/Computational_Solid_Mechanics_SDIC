from CSM.plane_eb_beam import PlaneEBBeamProblem
from test_plane_truss import solved_right_triangle, solved_single_bar_diagonal
import numpy as np
#truss=PlaneTrussProblem.load_file("structure.yaml")
truss=PlaneEBBeamProblem.load_file("bridge_EB.yaml")
#truss=solved_right_triangle()


num_samples=20
xi_vals = np.linspace(-1, 1, num_samples)

# ── Global structural scale dimensions (used for arrows & boundary symbols) ──

print("global stiffness \n", truss.assemble_global_stiffness())
print("reduced \n",truss.set_external_constraints(truss.constrained_dofs, 3 * truss.num_nodes))
print("constraints ", truss.constrained_dofs)
print("external forces \n", truss.external_forces)
print("response forces \n", truss.get_reaction_forces())

all_stresses = []
element=1
for xi in xi_vals:
    all_stresses.append(truss.get_element_stress(element, xi))

displacements=[]
for xi in xi_vals:
    displacements.append(truss.get_displacement_on_element(element,xi))
print(f"displacements on element {element} \n", displacements)
mean=np.mean(all_stresses)
variance=np.var(all_stresses)
print(f"stresses on element {element} \n", all_stresses)
print("mean of stresses", mean)
print("variance of stresses", variance)
truss.plot_plane_truss(show_deformed=True, stress_to_plot="bending", scale_factor=1000 )