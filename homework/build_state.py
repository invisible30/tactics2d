import numpy as np

def normalize_angle(angle):
    """将角度归一化到[-π, π]范围"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def calculate_lane_lateral_distance(point, centerline):
    """计算点到车道的横向距离，正值表示在车道右侧，负值表示在车道左侧"""
    if len(centerline) < 2:
        return float('inf')
        
    # 假设centerline只有两个点，分别是车道的起点和终点
    start_point = centerline[0]
    end_point = centerline[1]
    
    # 计算车道方向向量
    lane_dir_x = end_point[0] - start_point[0]
    lane_dir_y = end_point[1] - start_point[1]
    
    # 归一化车道方向
    lane_dir_len = np.sqrt(lane_dir_x**2 + lane_dir_y**2)
    if lane_dir_len > 0:
        lane_dir_x /= lane_dir_len
        lane_dir_y /= lane_dir_len
        
    # 计算点到起点的向量
    vec_x = point[0] - start_point[0]
    vec_y = point[1] - start_point[1]
    
    # 计算横向距离（点到直线的距离）
    # 使用向量叉积计算，保留符号
    lateral_distance = vec_x * lane_dir_y - vec_y * lane_dir_x
    
    return lateral_distance

def calculate_collision_risk(ego_state, other_states):
    """计算前方碰撞风险"""
    min_ttc = float('inf')
    
    # 获取自车朝向的单位向量
    ego_dir_x = np.cos(ego_state.heading)
    ego_dir_y = np.sin(ego_state.heading)
    
    for vehicle_id, state in other_states.items():
        # 计算相对位置
        rel_x = state.location[0] - ego_state.location[0]
        rel_y = state.location[1] - ego_state.location[1]
        
        # 计算前向距离
        forward_distance = rel_x * ego_dir_x + rel_y * ego_dir_y
        
        # 如果车辆在前方
        if forward_distance > 0:
            # 计算横向距离
            lateral_distance = abs(rel_x * ego_dir_y - rel_y * ego_dir_x)
            
            # 如果在同一车道上
            if lateral_distance < 2.0:
                # 计算相对速度
                rel_speed = ego_state.speed - state.speed
                
                # 如果有碰撞风险
                if rel_speed > 0:
                    # 计算碰撞时间
                    ttc = forward_distance / rel_speed
                    min_ttc = min(min_ttc, ttc)
    
    # 将最小碰撞时间转换为风险值
    if min_ttc != float('inf'):
        return 1.0 / min_ttc
    else:
        return 0

def build_state(observation, infos, destination=None):
    """
    构建状态向量
    
    参数:
    - observation: 环境观察
    - infos: 环境信息
    - destination: 目标位置，如果为None则使用默认目标
    
    返回:
    - state: 状态向量
    """
    ego_state = infos["ego_state"]
    other_states = infos["other_states"]
    centerlines = infos["centerlines"]
    
    # 如果没有提供目标位置，使用默认目标
    if destination is None:
        # 假设目标在前方200米处
        destination = (
            ego_state.location[0] + 200 * np.cos(ego_state.heading),
            ego_state.location[1] + 200 * np.sin(ego_state.heading)
        )
    
    # 1. 车辆自身状态
    ego_state_features = [
        ego_state.location[0], 
        ego_state.location[1],
        ego_state.heading-np.pi,
        ego_state.speed
    ]
    
    # 确定是否在上方车道（y > -15）
    is_upper_lane = ego_state.location[1] > -15
    
    # 2. 车道信息 - 根据自车位置选择车道
    lane_features = []
    selected_lanes = {}
    
    # 根据中心线的y坐标排序车道
    sorted_lanes = []
    for lane_id, centerline in centerlines.items():
        # 使用中心线的y坐标平均值作为排序依据
        avg_y = (centerline[0][1] + centerline[1][1]) / 2
        
        # 根据自车位置筛选车道
        if (is_upper_lane and avg_y > -15) or (not is_upper_lane and avg_y <= -15):
            sorted_lanes.append((avg_y, lane_id, centerline))
    
    # 按y坐标排序（上方车道从大到小，下方车道从小到大）
    sorted_lanes.sort(reverse=is_upper_lane)
    
    # 选择最多三条车道
    lane_count = 0
    for _, lane_id, centerline in sorted_lanes:
        lane_direction = np.arctan2(centerline[1][1] - centerline[0][1], 
                                   centerline[1][0] - centerline[0][0])
        lateral_distance = calculate_lane_lateral_distance(ego_state.location, centerline)
        heading_diff = normalize_angle(ego_state.heading - lane_direction)
        lane_features.extend([lateral_distance, heading_diff])
        selected_lanes[lane_id] = centerline
        lane_count += 1
        
        # 最多保留三条车道
        if lane_count >= 3:
            break
    
    # 如果车道不足三条，用默认值填充
    while len(lane_features) < 6:  # 每条车道2个特征，最多3条车道
        lane_features.extend([100, 0])  # 大距离值表示没有车道
    
    # 3. 周围车辆信息 - 根据自车位置选择车辆
    surrounding_vehicles = []
    vehicle_distances = []
    
    for vehicle_id, state in other_states.items():
        # 计算车辆相对位置
        rel_x = state.location[0] - ego_state.location[0]
        rel_y = state.location[1] - ego_state.location[1]
        
        # 根据自车位置筛选车辆
        if (is_upper_lane and state.location[1] > -15) or (not is_upper_lane and state.location[1] <= -15):
            # 计算距离
            distance = np.sqrt(rel_x**2 + rel_y**2)
            
            # 计算车辆相对角度
            rel_angle = normalize_angle(np.arctan2(rel_y, rel_x) - ego_state.heading)
            
            # 如果角度差大于1，忽略
            if abs(rel_angle) > 1:
                continue
                
            vehicle_distances.append((distance, vehicle_id))
    
    # 按距离排序
    vehicle_distances.sort()
    
    # 取最近的5辆车
    for i in range(min(5, len(vehicle_distances))):
        distance, vehicle_id = vehicle_distances[i]
        state = other_states[vehicle_id]
        
        rel_x = state.location[0] - ego_state.location[0]
        rel_y = state.location[1] - ego_state.location[1]
        rel_speed = ego_state.speed - state.speed
        # rel_speed = state.speed
        
        rel_angle = normalize_angle(np.arctan2(rel_y, rel_x) - ego_state.heading)
        
        surrounding_vehicles.extend([rel_x, rel_y, rel_speed, distance, rel_angle])
    
    # 如果车辆不足5辆，用0填充
    while len(surrounding_vehicles) < 5 * 5:
        surrounding_vehicles.extend([0, 0, 0, 100, 0])  # 大距离值表示没有车
    
    # 4. 目标信息
    distance_to_destination = np.sqrt(
        (ego_state.location[0] - destination[0])**2 + 
        (ego_state.location[1] - destination[1])**2
    )
    angle_to_destination = normalize_angle(
        np.arctan2(
            destination[1] - ego_state.location[1],
            destination[0] - ego_state.location[0]
        ) - ego_state.heading
    )
    goal_features = [distance_to_destination, angle_to_destination]
    
    # 5. 碰撞风险评估 - 只考虑后面车道上的车辆
    filtered_states = {k: v for k, v in other_states.items() if k in [vid for _, vid in vehicle_distances]}
    forward_collision_risk = calculate_collision_risk(ego_state, filtered_states)
    risk_features = [forward_collision_risk]
    
    # 整合所有特征
    state = np.concatenate([
        ego_state_features,
        lane_features,
        surrounding_vehicles,
        goal_features,
        risk_features
    ])
    
    return state