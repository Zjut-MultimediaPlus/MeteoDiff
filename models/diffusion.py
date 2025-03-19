import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import pdb

class VarianceSchedule(Module):
#var_sched = VarianceSchedule(num_steps=100,beta_T=5e-2,mode='linear')
    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()
    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas #

class TransformerFusion(nn.Module):
    def __init__(self, input_size, num_heads=4, num_layers=4):
        super(TransformerFusion, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size*2, num_heads, dim_feedforward=input_size * 4),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_size*2, input_size)

    def forward(self, x1, x2):
        # 将输入张量沿着第2维拼接
        fused_input = torch.cat((x1, x2), dim=2)

        # 使用Transformer进行特征融合
        fused_output = self.transformer(fused_input)
        fused_output = fused_output.mean(dim=0)  # 取平均作为融合后的特征

        # 使用全连接层进行输出
        output = self.fc(fused_output)

        return output


class AddMechanism(nn.Module):
    def __init__(self, input_size=128):
        super(AddMechanism, self).__init__()
        # self.gph_mapping = nn.Linear(fusion_size, input_size)  # 映射 gph 编码到与 x 编码相同的维度
        # self.fusion_layer = nn.Linear(input_size + input_size, input_size)  # 输入维度为两个 x 编码的维度

        self.transformer_fusion_model = TransformerFusion(input_size)

    def forward(self, x, mechan):
        fusion = self.transformer_fusion_model(x, mechan) #B,256
        residual = x + fusion*0.001
        return residual

class DiffusionTraj(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net  # TransformerConcatLinear net = self.diffnet(point_dim=4, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False)
        self.var_sched = var_sched

    def get_loss(self, x_0, context, encoded_age,encoded_env_data,mechanism_traj,mechanism_inten, t=None):
        batch_size, T ,point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t] #torch.Size([256])
        beta = self.var_sched.betas[t].cuda() #torch.Size([256])
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()
        e_rand = torch.randn_like(x_0).cuda()

        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context,
                           encoded_age = encoded_age,encoded_env_data=encoded_env_data
                           , mechanism_traj = mechanism_traj, mechanism_inten = mechanism_inten
                           )

        first_add = 1.3
        if (T == 4):
            Wt = [first_add, 1, 1, 1]
        elif (T == 3):
            Wt = [first_add, 1, 1]
        elif (T == 2):
            Wt = [first_add, 1]
        elif (T == 1):
            Wt = [first_add]

        loss = 0
        for i in range(e_theta.size(1)): #T
            loss_i = Wt[i]*F.mse_loss(e_theta[:,i,:].view(-1, point_dim), e_rand[:,i,:].view(-1, point_dim), reduction='mean')
            loss+=loss_i
        loss = loss/e_theta.size(1)

        return loss


    def sample(self, num_points, context,
               encoded_age,encoded_env_data,sample, bestof, point_dim=4, flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        traj_list = []
        for i in range(sample):
            batch_size = context.size(0)
            if bestof: #true
                x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
            else:
                x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)
            traj = {self.var_sched.num_steps: x_T} # x_T：torch.Size([3, 4, 2])
            stride = step
            for t in range(self.var_sched.num_steps, 0, -stride):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[t-stride]
                sigma = self.var_sched.get_sigmas(t, flexibility)
                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]#torch.Size([3, 4, 2])
                beta = self.var_sched.betas[[t]*batch_size]
                mechanism_traj = torch.randn((x_T.size(0), x_T.size(1), 128))
                mechanism_inten = torch.randn((x_T.size(0), x_T.size(1), 128))  # vo850
                e_theta = self.net(x_t, beta=beta, context=context
                                   ,encoded_age = encoded_age,encoded_env_data=encoded_env_data
                                   ,mechanism_traj = mechanism_traj, mechanism_inten = mechanism_inten
                                   )
                if sampling == "ddpm":#true
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                elif sampling == "ddim":
                    x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta
                else:
                    pdb.set_trace()
                traj[t-stride] = x_next.detach()
                traj[t] = traj[t].cpu()
                if not ret_traj:
                   del traj[t]
            if ret_traj:
                traj_list.append(traj)
            else:#false
                traj_list.append(traj[0])
        return torch.stack(traj_list)

class TransformerConcatLinear(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual # False
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1_2 = ConcatSquashLinear(context_dim, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, point_dim, context_dim+3)

        self.concat5 = ConcatSquashLinear(16, point_dim, context_dim+3)
        self.linear_true = Linear(128, point_dim)

        self.concat_env_age_x_traj = ConcatSquashLinear(2, 128, 32)
        self.concat_env_age_x_inten = ConcatSquashLinear(1, 64, 36)
        self.concat_env_age_x_wind = ConcatSquashLinear(1, 64, 20)

        self.add_traj_mechanism = AddMechanism()
        self.add_inten_mechanism = AddMechanism()

        self.concat1 = ConcatSquashLinear(point_dim, 2 * context_dim,
                                          context_dim + 3)
        #_type1
        self.concat_env_age_x_traj_type1 = ConcatSquashLinear(2, 2, 32)
        self.concat_env_age_x_inten_type1 = ConcatSquashLinear(1, 1, 36)
        self.concat_env_age_x_wind_type1 = ConcatSquashLinear(1, 1, 20)

        self.concat1_type2 = ConcatSquashLinear(point_dim, 2 * context_dim, 347)
        self.concat3_type2 = ConcatSquashLinear(2 * context_dim, context_dim, 347)
        self.concat4_type2 = ConcatSquashLinear(context_dim, context_dim // 2, 347)
        self.linear_type2 = ConcatSquashLinear(context_dim // 2, point_dim, 347)

        self.encode_dif_u_v = DifUVChangeTC(num_features=4*101*101,x_dim=context_dim//2)

    def get_new(self, x, beta, context
                , encoded_age,encoded_env_data
                          ,mechanism_traj,mechanism_inten):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1) torch.Size([B, 1, 1])
        context = context.view(batch_size, 1, -1)  # (B, 1, F) torch.Size([B, 1, 256])--torch.Size([B, 1, 384])

        encoded_age = encoded_age.view(batch_size, 1, -1)  # (B,1,4)
        env_traj = encoded_env_data[0].to(x.device).view(batch_size, 1, -1)
        env_inten = encoded_env_data[1].to(x.device).view(batch_size, 1, -1)
        env_wind = encoded_env_data[2].to(x.device).view(batch_size, 1, -1)

        # 轨迹
        ctx_emb_env_age_traj = env_traj
        ctx_emb_env_age_traj = ctx_emb_env_age_traj / 100
        # 强度
        ctx_emb_env_age_inten = torch.cat([env_inten, encoded_age], dim=-1)
        ctx_emb_env_age_inten = ctx_emb_env_age_inten / 100
        # 风速
        ctx_emb_env_age_wind = torch.cat([env_wind, encoded_age], dim=-1)  # (B, 1, 20)
        ctx_emb_env_age_wind = ctx_emb_env_age_wind / 100

        # [B,4,128]
        x_traj = self.concat_env_age_x_traj(ctx_emb_env_age_traj,
                                            x[:, :, 0:2])  # (B, 1, 84) (B,4,4)  输出torch.Size([256, 4, 256])
        # [B,4,64]
        x_pres = self.concat_env_age_x_inten(ctx_emb_env_age_inten, x[:, :, 2:3])
        # [B,4,64]
        x_wind = self.concat_env_age_x_wind(ctx_emb_env_age_wind, x[:, :, 3:4])
        x_inten = torch.concat((x_pres, x_wind), dim=2)

        x = torch.cat((x_traj, x_inten), dim=2)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3) torch.Size([B, 1, 259])

        x = self.concat1_2(ctx_emb, x)  # [B, 4, 512]
        # 【B,4,512】
        final_emb = x.permute(1, 0, 2)  # torch.Size([4, B, 512])
        final_emb = self.pos_emb(final_emb)  # torch.Size([4, B, 512])
        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)  # torch.Size([B, 4, 512])
        trans = self.concat3(ctx_emb, trans)  # torch.Size([B, 4, 256])
        trans = self.concat4(ctx_emb, trans)  # torch.Size([B, 4, 128])
        trans = self.linear(ctx_emb, trans)  # torch.Size([B, 4, 4])
        return trans
    def forward(self, x, beta, context
                , encoded_age,encoded_env_data
                ,mechanism_traj,mechanism_inten
                # ,dif_d,dif_v
                ):
        trans = self.get_new(x, beta, context
                , encoded_age,encoded_env_data,mechanism_traj,mechanism_inten)
        return trans

class TransformerLinear(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.residual = residual

        self.pos_emb = PositionalEncoding(d_model=128, dropout=0.1, max_len=24)
        self.y_up = nn.Linear(2, 128)
        self.ctx_up = nn.Linear(context_dim+3, 128)
        self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.linear = nn.Linear(128, point_dim)

    def forward(self, x, beta, context):

        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        ctx_emb = self.ctx_up(ctx_emb)
        emb = self.y_up(x)
        final_emb = torch.cat([ctx_emb, emb], dim=1).permute(1,0,2)
        #pdb.set_trace()
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb)  # 13 * b * 128
        trans = trans[1:].permute(1,0,2)   # B * 12 * 128, drop the first one which is the z
        return self.linear(trans)

class LinearDecoder(Module):
    def __init__(self):
            super().__init__()
            self.act = F.leaky_relu
            self.layers = ModuleList([
                #nn.Linear(2, 64),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
                nn.Linear(256, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 12)
                #nn.Linear(2, 64),
                #nn.Linear(2, 64),
            ])
    def forward(self, code):

        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out
