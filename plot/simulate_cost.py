import numpy as np

cost_l1 = 1.0
cost_l2 = 10.0
cost_alpha = 0.00001
target_force_torque = [0.,0.,0.,0.,0.,0.]
cost_ws = [1.,1.,1.]
cost_step = -1.0
cost_goal = 200.0
distance_threshold = 4.0
max_force_torque = [25.0,25.0,25.0,2.0,2.,2.]
cost_collision = -10

def distance(dist, max_distance):
    pose = dist / max_distance
    distance_norm = None
    distance_norm = l1l2(pose)
    return max_range(distance_norm, 30, -70)

def l1l2(dist, weights=[1, 1, 1, 1, 1, 1.]):
    l1 = cost_l1 * np.array(weights)
    l2 = cost_l2 * np.array(weights)
    dist = dist
    norm = (0.5 * (dist ** 2) * l2 +
            np.log(cost_alpha + (dist ** 2)) * l1)
    return norm.sum()

def contact_force(force):
    net_force = l1l2(force, weights=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05])
    reward = 0.0 if np.any(force >= 1.0) else net_force
    return max_range(reward, 0, -15)

def actions(actions):
    last_actions = actions
    reward = np.sum(np.sqrt(last_actions**2))
    return max_range(reward, len(actions))

def distance_force_action_step_goal(dist, force, action, max_distance):
    cdistance = distance(dist, max_distance)
    cforce = contact_force(force)
    cactions = actions(action)

    reward = (np.dot(np.array([cdistance, cforce, cactions]), cost_ws))
    reward += cost_step  # penalty per step

    speed_cost = 0
    ik_cost = 0
    collision_cost = 0

    done_reward = cost_goal if np.linalg.norm(dist[:3], axis=-1) < distance_threshold else 0

    collision_bool = False

    if np.any(np.abs(force*max_force_torque) > max_force_torque):
        collision_cost += cost_collision
        collision_bool = True

    reward += ik_cost + speed_cost + collision_cost + done_reward

    return reward, collision_bool

def max_range(value, max_value, min_value=0):
    return np.interp(value, [min_value, max_value], [1, 0])
