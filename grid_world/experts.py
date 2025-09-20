import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import utils
from grid_world import grid_utils,grid_plot
import math
from datetime import datetime
import pickle
import os
from datetime import datetime
current_time = datetime.now()
date = str(current_time.month)+str(current_time.day)



class Experts:
    def __init__(self,width,height,trajs_file_path = None,df_trajs = None,bias = 0):
        self.width = width
        self.height = height
        
        if trajs_file_path:
            self.df_trajs_all = self.ReadExpertTrajs(trajs_file_path)
        elif len(df_trajs)>0:
            self.df_trajs_all = df_trajs
        
        self.trajs_all = self.df_trajs_all['trajs'].tolist()
        self.traj_all_avg_length = int(np.mean(self.df_trajs_all['trajs'].apply(lambda x:len(x))))+bias
        print(f"trajs all avg length: {self.traj_all_avg_length}")
        self.mac_list = self.df_trajs_all['m'].tolist()
        self.traj_all_lens = self.df_trajs_all['trajs'].apply(lambda x:len(x)).tolist()

        self.clusterReaded = False
        #clustered result
        self.df_trajs = self.df_trajs_all
        self.trajs = self.trajs_all
        self.trajs_count = len(self.trajs)
        self.traj_avg_length = self.traj_all_avg_length
        self.cluster_now = -1
        self.bias = bias
        

    def ReadExpertTrajs(self,trajs_file_path):
        df_trajs_all = pd.read_csv(trajs_file_path)
        df_trajs_all['trajs'] = df_trajs_all['trajs'].apply(lambda x:eval(x))
        print(f"Get experts trajs num:{len(df_trajs_all)}")
        return df_trajs_all
        
    def GetExpertTraj(self,m):
        return self.df_trajs[self.df_trajs['m']==m]['trajs'].tolist()[0]
    
    def ChangeTrajLenBias(self,b):
        self.traj_all_avg_length = int(np.mean(self.df_trajs_all['trajs'].apply(lambda x:len(x))))+b
        self.traj_avg_length = self.traj_all_avg_length
        self.bias = b

    
    def GetClusteredTrajAvgLength(self,cluster_now):
        if self.cluster_now != cluster_now:
            self.cluster_now = cluster_now
            self.traj_clusterd_avg_length = int(np.mean(self.df_trajs[self.df_trajs['cluster']==cluster_now]['trajs'].apply(lambda x:len(x))))
            return self.traj_clusterd_avg_length
        else:
            return self.traj_clusterd_avg_length
    
    def ShowTraj(self,m):
        traj_grid = np.zeros((self.height,self.width))
        trajs = self.GetExpertTraj(m)
        for t in trajs:
            state = t[0]
            x,y = grid_utils.StateToCoord(state,self.width)
            traj_grid[y,x] = 1
        grid_plot.ShowGridWorld(traj_grid)

    def GetExpertsMovingCenter(self):
        m_centers = []
        m_x = []
        m_y = []
        for m in self.mac_list:
            traj = self.GetExpertTraj(m)
            x,y = self.__getTrajMovingCenter(traj)
            m_centers.append((x,y))
            m_x.append(x)
            m_y.append(y)
        return m_centers,m_x,m_y

    def __getTrajMovingCenter(self,traj):
        traj_x = []
        traj_y = []
        for t in traj:
            x,y = grid_utils.StateToCoord(t[0],self.width)
            traj_x.append(x)
            traj_y.append(y)
        traj_x = np.array(traj_x)
        traj_y = np.array(traj_y)
        return np.mean(traj_x),np.mean(traj_y)

    def ReadCluster(self, c_result):
        """
        Map cluster results to expert trajectories.

        Parameters:
        c_result: DataFrame containing 'mac' and 'cluster' columns from cluster result file
        """
        # Create mapping dictionary from base ID to cluster
        cluster_map = {}
        for idx, row in c_result.iterrows():
            mac_id = str(row['mac'])
            cluster_map[mac_id] = row['cluster']

        # Match cluster for each trajectory
        c = []
        for i, r in self.df_trajs_all.iterrows():
            m_id = str(r['m'])
            # Extract base ID (remove fragment number after +)
            if '+' in m_id:
                base_id = m_id.split('+')[0]
            else:
                base_id = m_id

            if base_id in cluster_map:
                c.append(cluster_map[base_id])
            else:
                c.append(-1)

        self.df_trajs_all['cluster'] = c
        self.clusterReaded = True

    def ApplyCluster(self, c_set):
        """
        Apply cluster filtering to select specific cluster groups.

        Parameters:
        c_set: indexes of applied clusters to filter
        """
        if not self.clusterReaded:
            raise ValueError("Must read cluster result first")

        self.df_trajs = self.df_trajs_all[self.df_trajs_all['cluster'].isin(c_set)].reset_index(drop=True)
        self.trajs = self.df_trajs['trajs'].tolist()

        # Calculate average trajectory length for selected clusters
        if len(self.df_trajs) == 0:
            self.traj_avg_length = 0
        else:
            traj_lengths = self.df_trajs['trajs'].apply(
                lambda x: len(x) if x is not None and isinstance(x, list) else 0)
            mean_length = np.mean(traj_lengths)

            if np.isnan(mean_length):
                self.traj_avg_length = int(np.median(traj_lengths)) + self.bias
            else:
                self.traj_avg_length = int(mean_length) + self.bias

