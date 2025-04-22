###! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: lane_changing_wrapper.py
# @Description: 将LaneChangingEnv适配到gymnasium接口
# @Author: Tactics2D Team
# @Version:

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lane_changing import LaneChangingEnv
from build_state import build_state

class LaneChangingWrapper(gym.Env):
    """
    将LaneChangingEnv包装成符合gymnasium接口的环境
    """
    
    def __init__(self, render_mode="rgb_array", max_step=500, difficulty="easy"):
        """
        初始化环境
        
        参数:
        - render_mode: 渲染模式，可选 "human" 或 "rgb_array"
        - max_step: 最大步数
        - difficulty: 难度级别，可选 "easy", "medium", "hard"
        """
        super().__init__()
        
        # 设置最大步数
        if difficulty == "easy":
            self.max_step = 500
        elif difficulty == "medium":
            self.max_step = 400
        elif difficulty == "hard":
            self.max_step = 300
        else:
            self.max_step = max_step
            
        # 创建原始环境
        self.env = LaneChangingEnv(render_mode=render_mode, max_step=self.max_step)
        
        # 定义观察空间和动作空间
        self.observation_space = spaces.Box(
            low=np.full((38,), -np.inf, dtype=np.float32), 
            high=np.full((38,), np.inf, dtype=np.float32), 
            shape=(38,),  # 状态向量维度
            dtype=np.float32
        )
        
        # 动作空间：转向角和加速度
        self.action_space = spaces.Box(
            low=np.array([-0.5, -4.0], dtype=np.float32),  # 最小转向角和加速度
            high=np.array([0.5, 2.0], dtype=np.float32),   # 最大转向角和加速度
            dtype=np.float32
        )
        
        # 保存当前状态
        self.current_observation = None
        self.current_infos = None
        self._max_episode_steps = self.max_step
        
    def reset(self, seed=None, options=None):
        """
        重置环境
        
        返回:
        - state: 状态向量
        - info: 环境信息
        """
        observation, infos = self.env.reset(seed=seed)
        self.current_observation = observation
        self.current_infos = infos
        
        # 构建状态向量
        state = build_state(observation, infos)
        
        return state, infos
        
    def step(self, action):
        """
        执行动作
        
        参数:
        - action: 动作，包含转向角和加速度
        
        返回:
        - next_state: 下一个状态向量
        - reward: 奖励
        - terminated: 是否终止
        - truncated: 是否截断
        - info: 环境信息
        """
        # 执行动作
        observation, infos = self.env.step(action)
        
        # 构建状态向量
        next_state = build_state(observation, infos)
        
        # 计算奖励
        reward = self._compute_reward(self.current_observation, self.current_infos, 
                                     action, observation, infos)
        
        # 判断是否结束
        status = infos["status"].name
        terminated = status in ["COMPLETED", "COLLIDED", "TIME_EXCEEDED", "OUT_OF_ROAD"]
        truncated = False
        
        # 更新当前状态
        self.current_observation = observation
        self.current_infos = infos
        
        return next_state, reward, terminated, truncated, infos
    
    def render(self):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        return self.env.close()
    
    def _compute_reward(self, observation, infos, action, next_observation, next_infos):
        """
        计算奖励函数
        
        参数:
        - observation: 当前观察
        - infos: 当前环境信息
        - action: 执行的动作
        - next_observation: 执行动作后的观察
        - next_infos: 执行动作后的环境信息
        
        返回:
        - reward: 奖励值
        """
        # 解析动作
        steering, accel = action
        
        # 解析状态信息
        ego_state = infos["ego_state"]
        next_ego_state = next_infos["ego_state"]
        other_states = infos["other_states"]
        next_other_states = next_infos["other_states"]
        centerlines = infos["centerlines"]
        status = next_infos["status"].name
        
        # 初始化奖励
        reward = 0.0
        
        # 1. 安全奖励 - 避免碰撞和危险情况
        if status == "COLLIDED":
            return -100.0  # 碰撞严重惩罚
        elif status == "OUT_OF_ROAD":
            return -50.0   # 驶出道路惩罚
        
        # 计算与其他车辆的最小距离
        min_distance = float('inf')
        for vehicle_id, state in next_other_states.items():
            distance = np.sqrt(
                (next_ego_state.location[0] - state.location[0])**2 + 
                (next_ego_state.location[1] - state.location[1])**2
            )
            min_distance = min(min_distance, distance)
        
        # 安全距离奖励 - 距离越近惩罚越大，但有一个安全阈值
        safety_threshold = 10.0  # 安全距离阈值
        if min_distance < safety_threshold:
            safety_reward = -5.0 * (1.0 - min_distance / safety_threshold)**2
        else:
            safety_reward = 0.0
        
        reward += safety_reward
        
        # 2. 目标奖励 - 接近目标
        # 假设目标在前方200米处
        destination = (
            ego_state.location[0] + 200 * np.cos(ego_state.heading),
            ego_state.location[1] + 200 * np.sin(ego_state.heading)
        )
        
        # 计算当前和下一步到目标的距离
        current_distance = np.sqrt(
            (ego_state.location[0] - destination[0])**2 + 
            (ego_state.location[1] - destination[1])**2
        )
        next_distance = np.sqrt(
            (next_ego_state.location[0] - destination[0])**2 + 
            (next_ego_state.location[1] - destination[1])**2
        )
        
        # 距离减少则给予奖励
        progress_reward = 0.1 * (current_distance - next_distance)
        reward += progress_reward
        
        # 3. 车道奖励 - 保持在合适的车道上
        # 找到最近的车道
        min_lateral_distance = float('inf')
        for lane_id, centerline in centerlines.items():
            from build_state import calculate_lane_lateral_distance
            lateral_distance = calculate_lane_lateral_distance(next_ego_state.location, centerline)
            min_lateral_distance = min(min_lateral_distance, lateral_distance)
        
        # 车道居中奖励 - 距离车道中心越近奖励越高
        lane_reward = 0.5 * np.exp(-min_lateral_distance)
        reward += lane_reward
        
        # 4. 速度奖励 - 维持合理的行驶速度
        target_speed = 20.0  # 目标速度
        speed_diff = abs(next_ego_state.speed - target_speed)
        speed_reward = 0.2 * np.exp(-0.2 * speed_diff)  # 速度接近目标值时奖励高
        reward += speed_reward
        
        # 5. 舒适奖励 - 避免急转弯和急加减速
        comfort_reward = -0.1 * (abs(steering) + 0.1 * abs(accel))
        reward += comfort_reward
        
        # 6. 换道成功奖励
        # 检测是否完成换道
        def get_current_lane(location, centerlines):
            min_distance = float('inf')
            current_lane = None
            
            for lane_id, centerline in centerlines.items():
                from build_state import calculate_lane_lateral_distance
                distance = calculate_lane_lateral_distance(location, centerline)
                if distance < min_distance:
                    min_distance = distance
                    current_lane = lane_id
            
            return current_lane
            
        current_lane = get_current_lane(ego_state.location, centerlines)
        next_lane = get_current_lane(next_ego_state.location, centerlines)
        
        if current_lane != next_lane and min_lateral_distance < 1.0:
            # 如果换到了新车道且位置合适，给予额外奖励
            reward += 5.0
        
        # 7. 任务完成奖励
        if status == "COMPLETED":
            reward += 50.0
        
        return reward