import torch
from torch.nn import Module
import torch.nn as nn
from .encoders.trajectron import Trajectron
from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
import torch.nn.functional as F
import pdb

class EnvPredicter(nn.Module):
    def __init__(self, pre_time_step = 1):
        super(EnvPredicter, self).__init__()
        self.time = pre_time_step

        # self.data_embed['wind'] = nn.Linear(1, embed_dim)
        # self.data_embed['intensity_class'] = nn.Linear(6, embed_dim)
        # self.data_embed['move_velocity'] = nn.Linear(1, embed_dim)
        # self.data_embed['future_direction24'] = nn.Linear(1, embed_dim)  # -1    OK
        # self.data_embed['future_inte_change24'] = nn.Linear(1, embed_dim)  # -1  OK  16*9

        self.embed_dim = 16
        self.key_num = 5

        self.data_embed = nn.ModuleDict()
        self.data_embed['wind'] = nn.Sequential(
            nn.Linear(self.embed_dim, 1)        )
        self.data_embed['intensity_class'] = nn.Sequential(
            nn.Linear(self.embed_dim, 6)        )
        self.data_embed['move_velocity'] = nn.Sequential(
            nn.Linear(self.embed_dim ,1)        )
        self.data_embed['future_direction24'] = nn.Sequential(
            nn.Linear(self.embed_dim, 1)        )
        self.data_embed['future_inte_change24'] = nn.Sequential(
            nn.Linear(self.embed_dim, 1)         )


    def forward(self, encoded_env_data):
        env_data_y_list = []
        k = 0
        for key in self.data_embed:
            res = torch.sigmoid(self.data_embed[key](encoded_env_data[:,k*self.embed_dim:(k+1)*self.embed_dim])) #[B,6] [B,1]
            k = k+1
            env_data_y_list.append(res)

        return env_data_y_list

class AutoEncoder(Module):

    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder #=Trajectron

        #config.diffnet：TransformerConcatLinear

        self.diffnet = getattr(diffusion, config.diffnet) #models.diffusion = diffusion

        self.diffusion = DiffusionTraj(
            net = self.diffnet(point_dim=4,
                               # context_dim=config.context_dim,
                               context_dim=256,
                               tf_layer=config.tf_layer, residual=False),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'
            )
        )

        self.get_env_data_y = EnvPredicter(pre_time_step=1)

    def encode(self, batch, node_type):
        x_gph,encoded_age,encoded_env_data = self.encoder.get_latent(batch, node_type) # batch[1]中x x是[256,8,12]  拆分成两个[256,8,6]然后跑两遍 别拆
        return x_gph,encoded_age,encoded_env_data #[B,256] [B,4]

    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        #print(f"Using {sampling}")
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x,encoded_age,encoded_env_data = self.encoder.get_latent(batch, node_type)

        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,encoded_age,encoded_env_data,sample,bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)#torch.Size([20, 1, 12, 2]) sample20
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()

    def get_env_loss_loss(self,env_data_y_hat,env_data_y0,device,B):
        #1,6,1,1,1
        key = ['wind','intensity_class','move_velocity','future_direction24','future_inte_change24']
        env_data_y = []
        for key_i in key:
            env_key_B = torch.tensor(env_data_y0[0][key_i]).unsqueeze(0).unsqueeze(0)
            for i in range(B - 1):
                env_key_B = torch.cat((env_key_B, torch.tensor(env_data_y0[i + 1][key_i]).unsqueeze(0).unsqueeze(0)),
                                      dim=0)  # [B,6]
            if(key_i=='intensity_class'):
                env_key_B = env_key_B.squeeze(1)

            env_data_y.append(env_key_B)


        loss_key = 0
        for i in range(len(env_data_y)):
            loss_key = loss_key+F.mse_loss(env_data_y[i].to(device).to(dtype=torch.float32), env_data_y_hat[i], reduction='mean')

        return loss_key/len(env_data_y)

    def get_env_loss(self,env_data_x, encoded_age, encoded_env_data, env_data_y):
        '''

        :param env_data_x:
        :param encoded_age: [B,4]
        :param encoded_env_data: [B,80]
        :param env_data_y:
        :return:
        '''
        # 预测
        env_data_y_hat = self.get_env_data_y(encoded_env_data) #长度为5的list
        # 与gt算loss
        loss = self.get_env_loss_loss(env_data_y_hat,env_data_y[0],encoded_env_data.device,encoded_env_data.size(0))
        return loss

    def get_loss(self, batch, node_type):
        (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st,
         neighbors_edge_value, robot_traj_st_t, map_tuple
         , env_data_x, env_data_y
         , timestep_range_x, timestep_range_y
         , gph_all_hpa_data_x, gph_all_hpa_data_y
         , gph_all_hpa_data_x_norm_t
         , wind_data_x, wind_data_y
         , wind_data_x_norm, wind_data_y_norm
         , q_data_x_norm, t_data_x_norm
         , vo_cen_x, vo_dis_x
         ) = batch
        x_gph,encoded_age,encoded_env_data = self.encode(batch,node_type) # torch.Size([256, 256]) torch.Size([256, 256]) torch.Size([256, 576])
        mechanism_traj = torch.randn((x_gph.size(0),y_t.size(1),128))
        mechanism_inten = torch.randn((x_gph.size(0),y_t.size(1),128))
        loss = self.diffusion.get_loss(y_t.cuda(), x_gph, encoded_age, encoded_env_data
                                       ,mechanism_traj, mechanism_inten)
        return loss
