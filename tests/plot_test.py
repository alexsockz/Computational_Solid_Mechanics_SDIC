from CSM.plane_truss import PlaneTrussProblem
from test_plane_truss import solved_right_triangle, solved_single_bar_diagonal
#truss=PlaneTrussProblem.load_file("structure.yaml")
truss=PlaneTrussProblem.load_file()
truss.plot_plane_truss(show_deformed=True, scale_factor=1e7)