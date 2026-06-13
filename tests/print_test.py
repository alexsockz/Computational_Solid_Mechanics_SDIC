from CSM.plane_truss import PlaneTrussProblem
from test_plane_truss import solved_single_bar_diagonal
#truss=PlaneTrussProblem.load_file("structure.yaml")
truss=solved_single_bar_diagonal()
truss.plot_plane_truss(show_deformed=True)