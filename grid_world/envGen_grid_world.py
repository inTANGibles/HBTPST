import pandas as pd
import numpy as np
from grid_world.grid_world import GridWorld
from grid_world.data_parser import DataParser
from grid_world import grid_utils,grid_plot
from utils_tool import utils
from DMEIRL.value_iteration import value_iteration
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from PIL import Image
import os
from tqdm import tqdm
import sys
sys.path.append("../")
from DMEIRL.DeepMEIRL_FC import DeepMEIRL_FC
import random


class GridWorld_envGen(GridWorld):
    '''
    World used to work with custom "gym env", aiming to generate new region environments according to target svf.
    
    its main function contains:
    1.calculate original svf & init_prob from real pedestrian trajs,
    2.parse original region environments,
    3.calculate current svf from changed region environments,
    4.calculate difference between current svf and target svf
    '''
    def __init__(self,width,height,
                 envs_img_folder_path,
                 experts_traj_filePath,
                 model_path,# nn model that convert features of particuler state to reward
                 model_n_input,#demo:4
                 model_layers,#demo:16,32,32,16
                 target_svf_delta:dict = {},#key:state_active, value:delta
                 trans_prob = 0.6,
                 discount = 0.98,
                 ):
        self.width = width
        self.height = height

        model = DeepMEIRL_FC(n_input=model_n_input,layers=model_layers)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        self.model = model

        super().__init__(width=width,height=height,
                         environments_img_folderPath=envs_img_folder_path,
                         expert_traj_filePath=experts_traj_filePath,
                         trans_prob=trans_prob,
                         discount=discount)

        #----parse original region environments----
        self.envs_arr_origin = self.parser.environments_arr #dim0: env type, dim1(2D): env value

        #----calculate original svf & init_prob----
        self.prob_initial_state = self.__getInitialStatesProb()
        #self.SVF_origin = self.StateVisitationFrequency()
        self.SVF_origin_simu = self.Expected_StateVisitationFrequency(self.parser.environments_arr)
        #self.ShowSVF(self.SVF_origin,'Original SVF')
        self.ShowReward(self.reward_now,0,1)
        self.ShowSVF(self.SVF_origin_simu,'Simulated SVF')
        self.ShowSVFOrigin()
        if len(target_svf_delta)>0:
            self.SVF_target = self.GetTargetSVF(target_svf_delta)
            self.ShowSVF(self.SVF_target,'Target SVF')

    # def ShowInitialization(self):
    #     # ----calculate original svf & init_prob----
    #     self.prob_initial_state = self.__getInitialStatesProb()
    #     # self.SVF_origin = self.StateVisitationFrequency()
    #     self.SVF_origin_simu = self.Expected_StateVisitationFrequency(self.parser.environments_arr)
    #     # self.ShowSVF(self.SVF_origin,'Original SVF')
    #     self.ShowReward(self.reward_now)
    #     self.ShowSVF(self.SVF_origin_simu, 'Simulated SVF')


    def GetTargetSVF(self,target_svf_delta:dict):
        target_svf = self.SVF_origin_simu.clone()
        for state,delta in target_svf_delta.items():
            s = self.state_fid[state]
            target_svf[s] += delta
        return target_svf
    
    def ShowSVFOrigin(self):
        self.SVF_origin = self.StateVisitationFrequency()
        self.ShowSVF(self.SVF_origin,'Original SVF')

    def StateVisitationFrequency(self):
        svf = torch.zeros(self.n_states_active,dtype=torch.float32).to(device)
        for traj in self.experts.trajs:
            for s , *_ in traj:
                index = self.state_fid[s]
                svf[index] += 1
        return svf/len(self.experts.trajs)
    
    def RefreshInitState(self):
        self.prob_initial_state = self.__getInitialStatesProb().cpu().numpy()
        self.ShowGrid3DBarChart(self.prob_initial_state,title='initial state probability')
    
    def Expected_StateVisitationFrequency(self,envs_arr):
        envs_arr = np.array(envs_arr)
        if envs_arr.shape[1] != self.height or envs_arr.shape[2] != self.width:
            raise ValueError("envs_arr shape not match")
        #get features
        features_arr = self.parser.GetFeaturesFromEnvs2DArray(envs_arr)
        state_features = self.GetStatesValueFromArr(features_arr)
        features_arr_active,_,_ = self.GetAvtiveFeatureArr(state_features)
        features = torch.from_numpy(features_arr_active).float().to(device)
        #get rewards
        rewards = self.model(features).flatten()
        self.reward_now = rewards.detach().cpu().numpy()
        #compute exp_svf
        policy = value_iteration(0.001,self,rewards.detach(),self.discount,demo=True)
        #probability of visiting the initial state
        policy = policy.cpu().numpy()
        #print("Expected_StateVisitationFrequency start")
        with torch.no_grad():
            #Compute 𝜇
            d = torch.from_numpy(np.transpose(self.dynamics_fid,(2,1,0))).float().to(device)
            mu = self.prob_initial_state.repeat(self.experts.traj_avg_length,1)
            x = (policy[:,:,np.newaxis]*self.dynamics_fid).sum(1)
            x = torch.from_numpy(x).float().to(device)
            for t in range(1,self.experts.traj_avg_length):
                mu[t,:] = torch.matmul(mu[t-1,:],x)

        return mu.sum(dim = 0)
    
    
        
    
    def CalActionReward(self,envs_arr):
        exp_svf = self.Expected_StateVisitationFrequency(envs_arr)
        return -self.__calSVFLoss(exp_svf).cpu().numpy()
        

    def __calSVFLoss(self,exp_svf):
        compare = nn.MSELoss()
        with torch.no_grad():
            loss = compare(self.SVF_origin,exp_svf)
        return loss

    def __getInitialStatesProb(self):
        prob_initial_state = torch.zeros(self.n_states_active,dtype=torch.float32).to(device)
        for traj in self.experts.trajs:
            index = self.state_fid[traj[0][0]]
            prob_initial_state[index] += 1
        prob_initial_state = prob_initial_state/self.experts.trajs_count
        return prob_initial_state
    
    #-------------------------traj--------------------------
    def reset(self,random_init = False):
        if random_init:
            index = np.random.randint(self.n_states_active)
            self.state = self.fid_state[index]
        else:
            if len(self.prob_initial_state)>0:
                self.state = random.choices(range(len(self.prob_initial_state)),weights=self.prob_initial_state)[0]
            else:
                self.state = 0
        self.state = self.fid_state[self.state]
        return self.state
    
    def step(self, a):
        index = self.state_fid[self.state]
        probs = self.dynamics_fid[index, a, :]
        index = np.random.choice(self.n_states_active, p=probs)
        self.state = self.fid_state[index]
        return self.state

    def GenerateTrajs(self,traj_count,traj_length,save = False):
        reward = torch.from_numpy(self.reward_now).to(device)
        policy = value_iteration(0.0001,self,reward,self.discount).argmax(1)
        policy = policy.cpu().numpy()
        trajs = []
        for i in tqdm(range(traj_count)):
            traj = []
            state = self.reset(random_init=False)
            for j in range(traj_length):
                index = self.state_fid[state]
                action = policy[index]
                next_state = self.step(action)
                traj.append((state,action,next_state))
                state = next_state
            trajs.append(traj)
        m = np.array(range(1,(len(trajs)+1)))
        df_trajs = pd.DataFrame({'m':m,'trajs':trajs})
        if save:
            df_trajs.to_csv(f'learned_trajs_{utils.date}.csv',index=False)
        self.df_trajs = df_trajs
        return df_trajs
    
    def PrintTrajs(self,index = -1,save_path=''):
        for ind,row in self.df_trajs.iterrows():
            if index != -1:
                if ind != index:
                    continue
            x = []
            y = []
            t = []
            for i,pair in enumerate(row.trajs):
                coord = self.StateToCoord(pair[0])
                x.append(coord[0])
                y.append(coord[1])
                t.append(i)
            grid_plot.PrintTraj3D(x,y,t)

    def ShowTrajs(self, trajs_df=None, title='Trajectories'):
        """
        显示轨迹的可视化

        参数:
        trajs_df: 轨迹数据框，如果为None则使用self.df_trajs
        title: 图表标题
        """
        if trajs_df is None:
            if not hasattr(self, 'df_trajs'):
                print("没有可用的轨迹数据")
                return
            trajs_df = self.df_trajs

        ts = []  # t0:x1,t1:y1,t3:x2,t4:y2,t5:counts

        trajs = trajs_df['trajs'].tolist()
        for traj in trajs:
            for i in range(len(traj) - 1):
                t1 = traj[i]
                t2 = traj[i + 1]
                x1, y1 = self.StateToCoord(t1[0])
                x2, y2 = self.StateToCoord(t2[0])
                # x1 += 0.1
                # y1 += 0.1
                # x2 += 0.1
                # y2 += 0.1

                # 检查是否已存在相同的轨迹段
                found = False
                for tt in ts:
                    if tt[0] == x1 and tt[1] == y1 and tt[2] == x2 and tt[3] == y2:
                        tt[4] += 1
                        found = True
                        break
                if not found:
                    ts.append([x1, y1, x2, y2, 1])

        grid_plot.ShowTraj(ts, self.width*15, self.height*15, title=title)


    def ShowExpertTrajs(self):
        """
        显示专家轨迹
        """
        if hasattr(self, 'experts') and hasattr(self.experts, 'df_trajs'):
            self.ShowTrajs(self.experts.df_trajs, title='Expert Trajectories')
        else:
            print("没有可用的专家轨迹数据")


    def ShowLearnedTrajs(self):
        """
        显示学习到的轨迹
        """
        if hasattr(self, 'df_trajs'):
            self.ShowTrajs(self.df_trajs, title='Learned Trajectories')
        else:
            print("没有可用的学习轨迹数据")

        
        