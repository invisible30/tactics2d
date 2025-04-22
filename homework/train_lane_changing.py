###! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: train_lane_changing.py
# @Description: 使用PPO训练换道算法
# @Author: Tactics2D Team
# @Version:

import os
import sys
import argparse

def main():
    """
    使用PPO训练换道算法的启动脚本
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用PPO训练换道算法')
    parser.add_argument('--difficulty', type=str, default='easy', choices=['easy', 'medium', 'hard'], 
                        help='难度级别: easy, medium, hard')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--load', action='store_true', help='是否加载预训练模型')
    parser.add_argument('--model_idx', type=int, default=100, help='加载哪个模型')
    parser.add_argument('--steps', type=int, default=int(1e6), help='训练步数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--write', action='store_true', help='是否使用TensorBoard记录训练曲线')
    
    args = parser.parse_args()
    
    # 构建main.py的命令行参数
    cmd_args = [
        'python', 'd:\\E\\tactics2d\\homework\\main.py',
        '--EnvIdex', '6',  # 使用LaneChanging环境
        '--difficulty', args.difficulty,
        '--render', 'True' if args.render else 'False',
        '--Loadmodel', 'True' if args.load else 'False',
        '--ModelIdex', str(args.model_idx),
        '--Max_train_steps', str(args.steps),
        '--seed', str(args.seed),
        '--write', 'True' if args.write else 'False',
        '--Distribution', 'Beta',  # 使用Beta分布
        '--T_horizon', '2048',  # 轨迹长度
        '--save_interval', '50000',  # 保存间隔
        '--eval_interval', '5000',  # 评估间隔
    ]
    
    # 执行命令
    os.system(' '.join(cmd_args))

if __name__ == '__main__':
    main()