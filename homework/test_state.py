###! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: AU7043_project_2_2025.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import sys
import time

sys.path.append(".")

import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from lane_changing import LaneChangingEnv
from build_state import build_state


class StateNormalizer:
    def __init__(self, clip=True, epsilon=1e-8):
        self.clip = clip
        self.epsilon = epsilon
        
        # state[0:4]
        max_ego_y=-15
        min_ego_y=-30
        max_ego_x=600
        min_ego_x=0
        
        max_ego_heading=2*np.pi
        min_ego_heading=0
        max_ego_speed=50
        min_ego_speed=0
        # state[4:10]
        max_lane_dist=10
        min_lane_dist=-10
        max_lane_heading=1
        min_lane_heading=-1
        # state[10:35]
        max_surrounding_vehicle_dist=50
        min_surrounding_vehicle_dist=0
        max_surrounding_vehicle_heading=1.5
        min_surrounding_vehicle_heading=-1.5
        
        min_rel_speed=-10
        max_rel_speed=10
        
        max_rel_x=40
        min_rel_x=-40
        
        min_rel_y=-10
        max_rel_y=10
        # state[35:37]
        max_goal_dist=400
        min_goal_dist=0
        # state[37:38]
        max_risk=1
        min_risk=0
        
        self.state_low = [min_ego_x, min_ego_y, min_ego_heading, min_ego_speed] + \
                   [min_lane_dist, min_lane_heading] * 3 + \
                   [min_rel_x, min_rel_y, min_rel_speed, min_surrounding_vehicle_dist, min_surrounding_vehicle_heading] * 5 + \
                   [min_goal_dist, min_lane_heading] + \
                   [min_risk]
        self.state_low = np.array(self.state_low)
        self.state_high = [max_ego_x, max_ego_y, max_ego_heading, max_ego_speed] + \
                    [max_lane_dist, max_lane_heading] * 3 + \
                    [max_rel_x, max_rel_y, max_rel_speed, max_surrounding_vehicle_dist, max_surrounding_vehicle_heading] * 5 + \
                    [max_goal_dist, max_lane_heading] + \
                    [max_risk]
        self.state_high = np.array(self.state_high)
    def normalize(self, state):
        state = np.array(state)
        norm = 2 * ((state - self.state_low) / (self.state_high - self.state_low + self.epsilon)) - 1
        if self.clip:
            norm = np.clip(norm, -1, 1)
        return norm

    def denormalize(self, norm_state):
        return norm_state * (self.state_high - self.state_low) + self.state_low
    
class MyLaneChangingModel:
    def __init__(self):
        pass

    def step(self):  # customize the input for your car following model
        steering = 0
        accel = 0.4
        # steering = np.random.uniform(low=-0.5, high=0.5)
        # accel = np.random.uniform(low=-4, high=2)
        return steering, accel


def main(level="easy"):
    if level == "easy":
        max_step = 500
    elif level == "medium":
        max_step = 400
    elif level == "hard":
        max_step = 300

    env = LaneChangingEnv(render_mode="human", max_step=max_step)
    observation, infos = env.reset()

    # The infors include the traffic status, the ego vehicle status, the other vehicles status, and the centerlines
    logging.info(f"Infos: {infos.keys()}")

    lane_changing_model = MyLaneChangingModel()

    for step in range(max_step + 10):
        env.render()

        action = lane_changing_model.step()
        observation, infos = env.step(action)
        time.sleep(2)
        # ------------------------------------------------------------------------------------------------------------------
        state = build_state(observation, infos) 
        state_normalizer = StateNormalizer()  
        normalized_state = state_normalizer.normalize(state)  
        
        print(f"ego_state_features={state[0:4]}")
        # print(f"lane_features={state[4:10]}")
        # print(f"surrounding_vehicles={state[10:35]}")
        print(f"goal_features={state[35:37]}")
        print(f"risk_features={state[37:38]}")
        print(f"与车道的横向距离={state[4],state[6],state[8]}") 
        print(f"与车道的角度差={state[5],state[7],state[9]}")
        print(f"与别的车辆的rel_x={state[10],state[15],state[20],state[25],state[30]}")
        print(f"与别的车辆的rel_y={state[11],state[16],state[21],state[26],state[31]}")
        print(f"与别的车辆的rel_speed={state[12],state[17],state[22],state[27],state[32]}")
        print(f"与别的车辆的角度差={state[14],state[19],state[24],state[29],state[34]}")
        print(f"与别的车辆的距离={state[13],state[18],state[23],state[28],state[33]}")
        print("--------------------------------------------")
        
        print(f"ego_state_features={normalized_state[0:4]}")
        # print(f"lane_features={state[4:10]}")
        # print(f"surrounding_vehicles={state[10:35]}")
        print(f"goal_features={normalized_state[35:37]}")
        print(f"risk_features={normalized_state[37:38]}")
        print(f"与车道的横向距离={normalized_state[4],normalized_state[6],normalized_state[8]}") 
        print(f"与车道的角度差={normalized_state[5],normalized_state[7],normalized_state[9]}")
        print(f"与别的车辆的rel_x={normalized_state[10],normalized_state[15],normalized_state[20],normalized_state[25],normalized_state[30]}")
        print(f"与别的车辆的rel_y={normalized_state[11],normalized_state[16],normalized_state[21],normalized_state[26],normalized_state[31]}")
        print(f"与别的车辆的rel_speed={normalized_state[12],normalized_state[17],normalized_state[22],normalized_state[27],normalized_state[32]}")
        print(f"与别的车辆的角度差={normalized_state[14],normalized_state[19],normalized_state[24],normalized_state[29],normalized_state[34]}")
        print(f"与别的车辆的距离={normalized_state[13],normalized_state[18],normalized_state[23],normalized_state[28],normalized_state[33]}")
        print("--------------------------------------------")
        # ------------------------------------------------------------------------------------------------------------------
        logging.debug(infos["status"].name)

        if infos["status"].name not in ["NORMAL", "COMPLETED"]:
            raise RuntimeError(
                f"Simulation failed with status: {infos['status'].name} at step {step}."
            )
        elif infos["status"].name == "COMPLETED":
            logging.info(f"Simulation completed successfully at step {step}.")
            break

def test():
    env = LaneChangingEnv(render_mode="rgb_array", max_step=100)
    count = 0
    for i in range(100):
        observation, infos = env.reset()
        state = build_state(observation, infos) 
        state_normalizer = StateNormalizer()  
        normalized_state = state_normalizer.normalize(state)  
        print(state[0])
        if state[0]>600:
            count+=1
    print(f"一共有{count}次")
if __name__ == "__main__":
    # np.random.seed(0)  # define the random seed to reproduce the scenario
    main()
    # test()
