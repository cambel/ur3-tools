#older
import sys
import copy
import rospy
import numpy as np

from gym import spaces

from ur_control.hybrid_controller import ForcePositionController
from ur_control import utils, spalg
from ur3e_openai.control import controller
from ur3e_openai.robot_envs.utils import get_value_from_range
from pyquaternion import Quaternion

class ParallelController(controller.Controller):
    def __init__(self, arm, agent_control_dt):
        controller.Controller.__init__(self, arm, agent_control_dt)
        self.force_control_model = self._init_hybrid_controller()
        self.pd_range_type = rospy.get_param(self.param_prefix+"/pd_range_type", 'sum')

    def _init_hybrid_controller(self):
        # position PD
        # proportional gain of position controller
        Kp = self.base_position_kp
        Kd = np.sqrt(Kp)
        Kp_pos = Kp * self.position_pd_exp_base
        # derivative gain of position controller
        Kd_pos = Kd * self.position_pd_exp_base
        position_pd = utils.PID(Kp=Kp_pos, Kd=Kd_pos)

        # Force PD
        # proportional gain of force controller
        Kp = self.base_force_kp
        Kp_force = Kp * self.force_pd_exp_base
        # derivative gain of force controller
        Kd_force = Kp *  0.1 * self.force_pd_exp_base
        Ki_force = Kp * 0.01 * self.force_pd_exp_base
        force_pd = utils.PID(Kp=Kp_force, Kd=Kd_force, Ki=Ki_force)
        return ForcePositionController(position_pd=position_pd, force_pd=force_pd, alpha=np.diag(self.alpha), dt=self.robot_control_dt)

    def reset(self):
        self.force_control_model.position_pd.reset()
        self.force_control_model.force_pd.reset()

    def act(self, action, target):
        assert action.shape == (self.n_actions, )
        if np.any(np.isnan(action)):
            rospy.logerr("Invalid NAN action(s)" + str(action))
            sys.exit()
        assert np.all(action >= -1.0001) and np.all(action <= 1.0001)

        # ensure that we don't change the action outside of this scope
        actions = np.copy(action)

        self.force_control_model.set_goals(position=target)
        self.set_force_signal(target)

        if self.n_actions == 9:
            actions_pds = actions[6:8]
            actions_alpha = actions[8:]
        elif self.n_actions == 14:
            actions_pds = actions[6:8]
            actions_alpha = actions[8:]
        elif self.n_actions == 19:
            actions_pds = actions[6:18]
            actions_alpha = actions[18:]
        elif self.n_actions == 24:
            actions_pds = actions[6:18]
            actions_alpha = actions[18:]

        self.set_pd_parameters(actions_pds)
        self.set_alpha(actions_alpha)

        position_action = actions[:6]
        position_action = [np.interp(position_action[i], [-1, 1], [-1*self.max_speed[i], self.max_speed[i]]) for i in range(len(position_action))]
        position_action /= self.action_scale

        # Take action
        action_result = self.ur3e_arm.set_hybrid_control(
            model=self.force_control_model,
            timeout=self.agent_control_dt,
            max_force_torque=self.max_force_torque,
            action=position_action
        )
        return action_result

    def set_force_signal(self, target_pose):
        force_signal = self.desired_force_torque.copy()
        target_q = Quaternion(np.roll(target_pose[3:], -1))
        force_signal[:3] = target_q.rotate(force_signal[:3])
        self.force_control_model.set_goals(force=force_signal)

    def set_pd_parameters(self, action):
        if len(action) == 2:
            position_kp = get_value_from_range(action[0], self.base_position_kp, self.kpd_range, mtype=self.pd_range_type)
            position_kd = np.sqrt(position_kp) * self.position_pd_exp_base
            position_kp = position_kp * self.position_pd_exp_base

            force_kp = get_value_from_range(action[1], self.base_force_kp, self.kpi_range, mtype=self.pd_range_type)
            force_ki = (force_kp * 0.01) * self.force_pd_exp_base
            force_kp = force_kp * self.force_pd_exp_base
        else:
            assert len(action) == 12
            position_kp = [get_value_from_range(act, self.base_position_kp, self.kpd_range, mtype=self.pd_range_type) for act in action[:6]]
            position_kd = np.sqrt(position_kp) * self.position_pd_exp_base
            position_kp = position_kp * self.position_pd_exp_base

            force_kp = [get_value_from_range(act, self.base_force_kp, self.kpi_range, mtype=self.pd_range_type) for act in action[6:]]
            force_ki = (np.array(force_kp) * 0.01) * self.force_pd_exp_base
            force_kp = force_kp * self.force_pd_exp_base

        self.force_control_model.position_pd.set_gains(Kp=position_kp, Kd=position_kd)
        self.force_control_model.force_pd.set_gains(Kp=force_kp, Ki=force_ki)

    def set_alpha(self, action):
        if len(action) == 1:
            alpha = get_value_from_range(action, self.alpha_base, self.alpha_range)
            alpha = alpha * np.ones(6)
        else:
            assert len(action) == 6
            alpha = [get_value_from_range(act, self.alpha_base, self.alpha_range) for act in action]

        self.force_control_model.alpha = np.diag(alpha)

    def straight_path(self, target_pose=None, desired_force_torque=None, duration=1):
        self.set_alpha([0.8])
        self.set_pd_parameters([0, 0])
        if target_pose is not None:
            self.force_control_model.set_goals(position=target_pose)
        if desired_force_torque is not None:
            self.force_control_model.set_goals(force=desired_force_torque)
        self.ur3e_arm.set_hybrid_control(
            model=self.force_control_model,
            timeout=duration,
            max_force_torque=self.max_force_torque,
            action=np.zeros(6)
        )
