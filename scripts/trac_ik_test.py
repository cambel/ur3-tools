from trac_ik_python.trac_ik import IK

ik_solver = IK(base_link="base_link", tip_link="tool0",
                 timeout=0.005, epsilon=1e-5, solve_type="Speed",
                 urdf_string=None)

seed_state = [0.0] * ik_solver.number_of_joints

qa = [-0.131,  0.337,  0.263, -1.,  0.,  0., -0.019]

print(ik_solver.get_ik(seed_state, -0.131,  0.337,  0.263, -1.   ,  0.   ,  0.   , -0.019))

seed_state = [0.3584 , -0.80591,  2.14269, -3.55912, -1.82971,  0.43981]

print(ik_solver.get_ik(seed_state, -0.131,  0.269,  0.006, -0.383,  0.   ,  0.   , -0.924))
