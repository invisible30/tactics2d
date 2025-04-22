from datetime import datetime
import os, shutil
import argparse
import torch
import gymnasium as gym
import numpy as np

from utils import str2bool, Action_adapter, Reward_adapter, evaluate_policy
from PPO import PPO_agent
from lane_changing_wrapper import LaneChangingWrapper

# 添加状态归一化器类
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

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=6, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3, LaneChanging')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Distribution', type=str, default='Beta', help='Should be one of Beta ; GS_ms  ;  GS_m')
parser.add_argument('--Max_train_steps', type=int, default=int(5e7), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(5e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--difficulty', type=str, default='easy', help='Difficulty level for LaneChanging: easy, medium, hard')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def main():
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v3','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3', 'LaneChanging']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3', 'LaneChanging']

    # Build Env
    if opt.EnvIdex == 6:  # LaneChanging环境
        env = LaneChangingWrapper(render_mode="human" if opt.render else "rgb_array", difficulty=opt.difficulty)
        eval_env = LaneChangingWrapper(render_mode="rgb_array", difficulty=opt.difficulty)
        # 创建状态归一化器
        state_normalizer = StateNormalizer()
    else:
        env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
        eval_env = gym.make(EnvName[opt.EnvIdex])
        state_normalizer = None
    
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])
    opt.max_steps = env._max_episode_steps
    print('Env:',EnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,
          '  max_a:',opt.max_action,'  min_a:',env.action_space.low[0], 'max_steps', opt.max_steps)

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Use tensorboard to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Beta dist maybe need larger learning rate, Sometimes helps
    # if Dist[distnum] == 'Beta' :
    #     kwargs["a_lr"] *= 2
    #     kwargs["c_lr"] *= 4

    if not os.path.exists('model'): os.mkdir('model')
    agent = PPO_agent(**vars(opt)) # transfer opt to dictionary, and use it to init PPO_agent
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        while True:
            ep_r = evaluate_policy(env, agent, opt.max_action, 1, state_normalizer)
            print(f'Env:{EnvName[opt.EnvIdex]}, Episode Reward:{ep_r}')
    else:
        traj_lenth, total_steps = 0, 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & trian'''
            while not done:
                '''Interact with Env'''
                # 对于LaneChanging环境，对状态进行归一化
                if opt.EnvIdex == 6 and state_normalizer is not None:
                    s_normalized = state_normalizer.normalize(s)
                    a, logprob_a = agent.select_action(s_normalized, deterministic=False)
                else:
                    a, logprob_a = agent.select_action(s, deterministic=False)
                    
                act = Action_adapter(a,opt.max_action) #[0,1] to [-max,max]
                s_next, r, dw, tr, info = env.step(act) # dw: dead&win; tr: truncated
                
                # 对于LaneChanging环境，不需要适配奖励
                if opt.EnvIdex != 6:
                    r = Reward_adapter(r, opt.EnvIdex)
                    
                done = (dw or tr)

                '''Store the current transition'''
                # 对于LaneChanging环境，存储归一化后的状态
                if opt.EnvIdex == 6 and state_normalizer is not None:
                    s_next_normalized = state_normalizer.normalize(s_next)
                    agent.put_data(s_normalized, a, r, s_next_normalized, logprob_a, done, dw, idx=traj_lenth)
                else:
                    agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx=traj_lenth)
                    
                s = s_next

                traj_lenth += 1
                total_steps += 1
                # print(f"total_steps:{total_steps}")
                '''Update if its time'''
                if traj_lenth % opt.T_horizon == 0:
                    agent.train()
                    traj_lenth = 0

                '''Record & log'''
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, opt.max_action, turns=3, state_normalizer=state_normalizer) # 传入归一化器
                    if opt.write: writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:',EnvName[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)

                '''Save model'''
                if total_steps % opt.save_interval==0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))

        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()





