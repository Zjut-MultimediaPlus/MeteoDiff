import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .components import *
from .model_utils import *
import models.encoders.dynamics as dynamic_module
from environment.scene_graph import DirectedEdge
from models.encoders.wind_encoder_predictor import GetWindEncByTransformer,WindTransformerVideoPredictor
from .utils import *
import pdb
from utils.trajectron_hypers import get_traj_hypers

class Combined_gph_x_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Combined_gph_x_Encoder, self).__init__()
        self.x_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.gph_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gph_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, gph):
        encoded_x = self.x_encoder(x)
        encoded_gph = self.gph_encoder(gph)

        combined_encoding = torch.cat((encoded_x, encoded_gph), dim=-1)
        return combined_encoding

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolingModel(nn.Module):
    def __init__(self, out_channels):
        super(PoolingModel, self).__init__()
        self.out_channels = out_channels

        # 使用两个不同的池化层，一个最大池化，一个平均池化
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 1x1 卷积层，用于调整通道数
        self.conv1x1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x):
        # 应用最大池化和平均池化
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)

        # 拼接两个池化结果
        pooled = torch.cat((max_pooled, avg_pooled), dim=1)
#torch.Size([32, 128, 25, 25])
        # 应用 1x1 卷积来调整通道数
        output = self.conv1x1(pooled)

        return output

class SpatialAttention1(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attention_map1 = self.conv1(x)  #x:torch.Size([32, 64, 50, 50]) map:torch.Size([32, 1, 50, 50])
        attention_map1_transposed = torch.transpose(attention_map1, 2, 3)  # 交换通道维度
        attention_map2 = self.conv2(x)

        attention_map = attention_map1_transposed*attention_map2

        attention_weights = torch.sigmoid(attention_map) #torch.Size([32, 1, 50, 50])
        return x * attention_weights

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x):

        # 执行第一个1x1卷积
        output1 = self.conv1(x)

        # 执行第二个1x1卷积并进行转置
        output2 = self.conv2(x)
        output2_transposed = torch.transpose(output2, 1, 2)  # 交换通道维度

        # 执行softmax操作
        softmax_output2_transposed = F.softmax(output2_transposed, dim=1)

        # 执行第三个1x1卷积
        output3 = self.conv3(x)

        # 将softmax的结果与第三个卷积的输出相乘
        output = torch.mul(softmax_output2_transposed, output3)

        # 加回原始输入
        output = output + x

        return output

class SelfAttentionLSTM8(nn.Module):
    def __init__(self, in_channels=1):
        super(SelfAttentionLSTM8, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1) #通道1--1
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.conv7 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.conv10 = nn.Conv2d(in_channels*3, in_channels, kernel_size=1)
        self.conv11 = nn.Conv2d(in_channels*3, in_channels, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels*3, in_channels, kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels*3, in_channels, kernel_size=1)

    def sa_conv_lstm(self, x, en_1d): ##torch.Size([8, 32, 1, 16, 16]) #[B,1,16,16]
        # #看sa-conv-lstm的
        # M,H：每一个小的 都是（B，256）
        memory = torch.zeros_like(x[0])  # [32, 1, 16, 16]
        H = torch.zeros_like(x[0])
        C = torch.randn_like(x[0]) * 1e-6

        for i in range(x.size(0)):  # 8次循环
            a_xh = torch.sigmoid(self.conv10(torch.cat((H, x[i], en_1d), dim=1)))  #到单通道吗
            ca_xh = C*a_xh
            ga = torch.sigmoid(self.conv11(torch.cat((H, x[i], en_1d), dim=1)))
            gv = torch.tanh(self.conv12(torch.cat((H, x[i], en_1d), dim=1)))
            C = ca_xh + ga*gv
            a_xh1 = torch.sigmoid(self.conv13(torch.cat((H, x[i], en_1d), dim=1)))
            H = a_xh1*torch.tanh(C)
            memory, H = self.self_attention_memory(memory, H)  # H:torch.Size([32, 1, 16, 16])
        return H

    def self_attention_memory(self, m, h): #[32, 1, 16, 16]
        vh = self.conv1(h)
        kh = self.conv2(h)
        qh = self.conv3(h)
        qh = torch.transpose(qh, 2, 3)
        ah = F.softmax(kh*qh,dim=-1) #基本全是0.0625 0.0624
        zh = vh*ah

        km = self.conv4(m)
        vm = self.conv5(m)
        am = F.softmax(qh*km,dim=-1)
        zm = vm*am
        z0 = torch.cat((zh, zm), dim=1)
        z = self.conv6(z0)
        hz = torch.cat((h, z), dim=1)

        ot = torch.sigmoid(self.conv7(hz))  #到单通道吗
        gt = torch.tanh(self.conv8(hz))
        it = torch.sigmoid(self.conv9(hz))

        gi = gt*it
        mf = (1-it)*m
        mt = gi+mf
        ht = ot*mt

        return mt,ht

    def forward(self, x, en_1d): #torch.Size([32, 8, 1, 16, 16]) #[B,1,16,16]
        B,_,_,_,_ = x.size()  #最好还是B,T,C,H,W
        x = x.permute(1, 0, 2, 3, 4) #torch.Size([8, 32, 1, 16, 16])
        H = self.sa_conv_lstm(x, en_1d)#[B,1,16,16]
        flattened_tensor = H.view(B, -1)
        return flattened_tensor #(B,256) 特别趋同

class SpaceCenterAttention(nn.Module):
    def __init__(self, input_size_x=100, input_size_y=100, block_size=20):
        super(SpaceCenterAttention, self).__init__()
        self.pool_num = 3
        self.block_size = block_size  #20
        self.block_bianchang = int(input_size_x//block_size)
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y

    def get_block_tensor(self,x,big_area_size):#(B,10,10) 100块 # 100,60,20
        B,_,_ = x.size() #b,100,100
        for k in range(B):
            original_tensor = x[k]
            for i in range(0, big_area_size, self.block_size):
                for j in range(0, big_area_size, self.block_size):
                    block = original_tensor[i:i + self.block_size, j:j + self.block_size]#[20,20]
                    block_tensor_one = block.unsqueeze(0) #[1,20,20]
                    if(k==0 and i==0 and j==0):
                        block_tensor = block_tensor_one
                    else:
                        block_tensor = torch.concat((block_tensor_one, block_tensor),dim=0) #[B*5*5,20,20]
        return block_tensor  #[B*5*5,20,20]

    def add_attention_to_x(self, zh_all_conv_out, x):# zh_all_conv_out:[B,block_num,20,20] x:torch.Size([B, 100, 100])
        B,H,W = x.size()
        block_size = self.block_size  #20
        block_bianchang = int(H//block_size)
        # 初始化原始张量
        zh_100_100 = torch.zeros(B, H, W)
        zh_100_100 = zh_100_100.to(x.device)

        zh_all_conv_out = zh_all_conv_out.permute(1,0,2,3) #[block_num,B,20,20]

        #这个太慢了
        for i in range(block_bianchang):
            for j in range(block_bianchang):
                zh_100_100[:, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = \
                zh_all_conv_out[block_bianchang * i + j]

        return x+zh_100_100

    def get_qkv(self,block_tensor):  #[B*5*5,20,20]
        block_tensor = block_tensor.unsqueeze(1)  ###[B*5*5,1,20,20]
        conv1 = nn.Conv2d(1, 1, kernel_size=1).to(block_tensor.device)
        conv2 = nn.Conv2d(1, 1, kernel_size=1).to(block_tensor.device)
        conv3 = nn.Conv2d(1, 1, kernel_size=1).to(block_tensor.device)

        Q = conv1(block_tensor)  # #[B*5*5,1,20,20]
        Q = torch.transpose(Q, 2, 3)
        K = conv2(block_tensor)
        V = conv3(block_tensor)
        return Q,K,V#[B*5*5,1,20,20]

    def space_center_area(self, x):  ##x:torch.Size([B, 100, 100])
        #分块 10*10一块 共10*10=100块
        B=x.size(0)
        big_area_size = x.size(1) # 100,60,20
        block_tensor = self.get_block_tensor(x,big_area_size) #[B*5*5,20,20]
        Q,K,V = self.get_qkv(block_tensor) ## #[B*5*5,1,20,20]
        # 用Q,K,V算更新值 直接更新V后，再把V分开算算
        _,C,H,W = Q.size()
        Q = Q.reshape(B, -1, C, H, W) #[B,5*5,1,20,20]
        K = K.reshape(B, -1, C, H, W)
        V = V.reshape(B, -1, C, H, W)

        Q = Q.permute(1, 0, 2, 3, 4)#[5*5,B,1,20,20]
        K = K.permute(1, 0, 2, 3, 4)
        V = V.permute(1, 0, 2, 3, 4)

        block_num = Q.size(0) #5*5

        for i in range(block_num):
            q = Q[i]  # [B,1,20,20]
            v = V[i]  # [B,1,20,20]
            for j in range(block_num):
                k = K[j]  # [B,1,20,20]
                a = F.softmax(q * k, dim=-1)
                zh = a * v  #[B,1,20,20]
                zh = zh.unsqueeze(0)  # [1,B,1,20,20]
                if (i == 0 and j == 0):
                    zh_block = zh
                else:
                    zh_block = torch.concat((zh_block, zh), dim=0)  # [block_num*block_num,B,1,20,20]
        zh_block = zh_block.reshape(block_num*block_num*B, C, H, W)
        conv4 = nn.Conv2d(1, 1, kernel_size=1).to(zh_block.device)
        zh_block_conv_out = conv4(zh_block)  ## [block_num*block_num*B,1,20,20]
        zh_block_conv_out = zh_block_conv_out.reshape(block_num, block_num, B, C, H, W) ## [block_num,block_num,B,1,20,20]
        zh_block_conv_out = zh_block_conv_out.permute(2,0,1,3,4,5) # [B, block_num,block_num,1,20,20]
        zh_block_conv_out_aver = zh_block_conv_out.mean(dim=2)  ## [B,block_num,1,20,20]

        zh_block_conv_out_aver_all = zh_block_conv_out_aver.reshape(B, -1, H, W)## [B,block_num,20,20]

        x = self.add_attention_to_x(zh_block_conv_out_aver_all,x)# zh_all_conv_out:[B,5*5,20,20] x:torch.Size([B, 100, 100])
        # 返回更新值x
        return x

    def space_center_attention(self, x):  ##x:torch.Size([B, 100, 100])
        for i in range(self.pool_num):
            start = i*20              #0  20  40
            end = x.size(1) - start #100  80  60
            x_temp = x[:, start:end, start:end]
            x[:, start:end, start:end] = self.space_center_area(x_temp)  #[B, 100, 100]  [B, 60, 60]  [B, 20, 20]

        return x  #x:torch.Size([B, 100, 100])

    def forward(self, x):##x:torch.Size([B, 8, 100, 100])
        x = x.permute(1, 0, 2, 3) #[8, B, 100, 100]
        for t in range(x.size(0)):
            x[t] = self.space_center_attention(x[t])  #torch.Size([B, 100, 100])
        x = x.permute(1, 0, 2, 3)
        return x  #[B, 8, 100, 100]

class TimeAwareEncoderForGPH10(nn.Module):
    def __init__(self, point_num, image_size=100, hidden_dim=256):
        super(TimeAwareEncoderForGPH10, self).__init__()
        self.point_num = point_num
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.conv_layers2 = nn.Sequential(  #torch.Size([32, 1, 100, 100])
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=64),   #[B,64,50,50]门控 x*G     G=sig(conv(x))

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=128),  #[B,128,25,25]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            SpatialAttention1(in_channels=256)  #torch.Size([B, 256, 12, 12])

        )

        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3)

        self.linear = nn.Linear(12, 256)

        # self.rnn = nn.GRU(
        #     input_size = 256 * (image_size // 8) ** 2,
        #     # input_size=32 * (image_size // 4) ** 2,
        #                   hidden_size=hidden_dim, batch_first=True)

        self.encode_time = SelfAttentionLSTM8(in_channels=1)

    def get_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        B,t,H,W = x.size()
        x = x.permute(1,0,2,3)
        shape = (t-1,B,H,W)
        # 创建空张量
        x_differ = torch.empty(shape)
        for i in range(t-1):
            x_differ[i] = x[i+1]-x[i]
        x_differ = x_differ.permute(1,0,2,3)
        return x_differ #（差值也可以增强 跨步数 额不太精准）[B, 7, 100, 100]

    def get_t_center_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :, 50, 50].unsqueeze(2).unsqueeze(3).expand_as(x)
        return x_t_center #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]

    def get_t_center_differ2(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :, 50, 50].unsqueeze(2).unsqueeze(3).expand_as(x)
        epsilon = 1e-12
        reciprocal_tensor = 1.0 / (x_t_center + epsilon)

        # 使用sigmoid函数将数值映射到0到1的范围内
        normalized_tensor = torch.sigmoid(reciprocal_tensor) #可能存在正负  这块等下调试看看

        return normalized_tensor #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]

    # 110
    def forward(self, x, node_history_encoded):  #[B, 8, 100, 100]  #[B,256] 过去的所有点 都在
        batch_size, _, _, _ = x.size() #x:torch.Size([B, 8, 100, 100])

        # 获取时刻间差值
        x_differ = self.get_differ(x) #[B, 7, 100, 100]
        # 获取中心点差值
        x_t_center = self.get_t_center_differ(x) ##[B, 8, 100, 100]
        # 2d数据编码
        x_list = []
        for t in range(self.point_num): #一起编码呗 #torch.Size([B, 1, 100, 100])
            x_t = torch.cat((x[:, t, :, :].unsqueeze(1), x_t_center[:, t, :, :].unsqueeze(1)), dim=1) # Get the data for the current time step

            x_t = self.conv_layers2(x_t)  # torch.Size([B, C 256, H 12, W 12])

            # 首先 reshape 成 [B, 16, 48, 48]
            x_t = x_t.view(batch_size, 16, 48, 48)

            # 编码成 [B, 1, 16, 16]，这里使用卷积层来实现
            x_t = self.conv(x_t)  # 用 3x3 的卷积核 torch.Size([32, 1, 16, 16])
            x_list.append(x_t)

        x = torch.stack(x_list, dim=1)  # torch.Size([32, 8, 1, 16, 16])

        # 1d数据简单重新组建形状
        node_history_encoded = node_history_encoded.reshape(batch_size,16,16) ##[B,16,16]
        node_history_encoded = node_history_encoded.unsqueeze(1) #[B,1,16,16]
        hidden = self.encode_time(x, node_history_encoded) #torch.Size([32, 8, 1, 16, 16]) #[B,1,16,16]

        return hidden #(B,256)
class TimeAwareEncoderForGPH10ZUV(nn.Module):
    def __init__(self, point_num, image_size=100, hidden_dim=256):
        super(TimeAwareEncoderForGPH10ZUV, self).__init__()
        self.point_num = point_num
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.conv_layers2 = nn.Sequential(  #torch.Size([32, 1, 100, 100])
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=64),   #[B,64,50,50]门控 x*G     G=sig(conv(x))

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=128),  #[B,128,25,25]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            SpatialAttention1(in_channels=256)  #torch.Size([B, 256, 12, 12])

        )

        # self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3)
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3, padding=4)

        self.linear = nn.Linear(12, 256)

        # self.rnn = nn.GRU(
        #     input_size = 256 * (image_size // 8) ** 2,
        #     # input_size=32 * (image_size // 4) ** 2,
        #                   hidden_size=hidden_dim, batch_first=True)

        self.encode_time = SelfAttentionLSTM8(in_channels=1)

    def get_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        B,t,H,W = x.size()
        x = x.permute(1,0,2,3)
        shape = (t-1,B,H,W)
        # 创建空张量
        x_differ = torch.empty(shape)
        for i in range(t-1):
            x_differ[i] = x[i+1]-x[i]
        x_differ = x_differ.permute(1,0,2,3)
        return x_differ

    def get_t_center_differ(self, x):
        x_t_center = x-x[:, :,:, 50, 50].unsqueeze(3).unsqueeze(4).expand_as(x)
        return x_t_center

    # 110
    def forward(self, x, node_history_encoded):  ## [B,T,10,100,100]  #[B,256]
        batch_size, _, _, _,_ = x.size()

        x_t_center = self.get_t_center_differ(x) ## [B,T,10,100,100]
        x_list = []
        for t in range(self.point_num):
            x_t = torch.cat((x[:, t, :, :, :], x_t_center[:, t, :, :, :]), dim=1) #[B,20,100,100]

            x_t = self.conv_layers2(x_t.float())  # torch.Size([B, 256, 10, 10])  [B,20,100,100]

            # reshape  [B, 16, 48, 48]
            x_t = x_t.view(batch_size, 16, 40, 40)

            # [B, 1, 16, 16]
            x_t = self.conv(x_t)  #  3x3  torch.Size([32, 1, 16, 16])
            x_list.append(x_t)

        x = torch.stack(x_list, dim=1)  # torch.Size([32, 8, 1, 16, 16])

        # 1d数据简单重新组建形状
        node_history_encoded = node_history_encoded.reshape(batch_size,16,16) ##[B,16,16]
        node_history_encoded = node_history_encoded.unsqueeze(1) #[B,1,16,16]
        hidden = self.encode_time(x, node_history_encoded) #torch.Size([32, 8, 1, 16, 16]) #[B,1,16,16]

        return hidden #(B,256)

class TimeAwareEncoderForGPH10ZUVVMS(nn.Module):
    def __init__(self, point_num, image_size=100, hidden_dim=256):
        super(TimeAwareEncoderForGPH10ZUVVMS, self).__init__()
        self.point_num = point_num
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.conv_layers2 = nn.Sequential(  #torch.Size([32, 1, 100, 100])
            nn.Conv2d(in_channels=22, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=64),   #[B,64,50,50]门控 x*G     G=sig(conv(x))

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=128),  #[B,128,25,25]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            SpatialAttention1(in_channels=256)  #torch.Size([B, 256, 12, 12])

        )

        # self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3)
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3, padding=4)

        self.linear = nn.Linear(12, 256)

        # self.rnn = nn.GRU(
        #     input_size = 256 * (image_size // 8) ** 2,
        #     # input_size=32 * (image_size // 4) ** 2,
        #                   hidden_size=hidden_dim, batch_first=True)

        self.encode_time = SelfAttentionLSTM8(in_channels=1)

    def get_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        B,t,H,W = x.size()
        x = x.permute(1,0,2,3)
        shape = (t-1,B,H,W)
        # 创建空张量
        x_differ = torch.empty(shape)
        for i in range(t-1):
            x_differ[i] = x[i+1]-x[i]
        x_differ = x_differ.permute(1,0,2,3)
        return x_differ #（差值也可以增强 跨步数 额不太精准）[B, 7, 100, 100]

    def get_t_center_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :,:, 50, 50].unsqueeze(3).unsqueeze(4).expand_as(x)
        return x_t_center #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]

    def get_t_center_differ2(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :, 50, 50].unsqueeze(2).unsqueeze(3).expand_as(x)
        epsilon = 1e-12
        reciprocal_tensor = 1.0 / (x_t_center + epsilon)

        # 使用sigmoid函数将数值映射到0到1的范围内
        normalized_tensor = torch.sigmoid(reciprocal_tensor) #可能存在正负  这块等下调试看看

        return normalized_tensor #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]

    # 110
    def forward(self, x, node_history_encoded):  ## [B,T,10,100,100]  #[B,256] 过去的所有点 都在
        batch_size, _, _, _,_ = x.size()

        # 获取中心点差值
        x_t_center = self.get_t_center_differ(x) ## [B,T,10,100,100]
        # 2d数据编码
        x_list = []
        for t in range(self.point_num):
            x_t = torch.cat((x[:, t, :, :, :], x_t_center[:, t, :, :, :]), dim=1) #[B,20,100,100]

            x_t = self.conv_layers2(x_t.float())  # torch.Size([B, 256, 10, 10])  [B,20,100,100]

            # 首先 reshape 成 [B, 16, 48, 48]
            x_t = x_t.view(batch_size, 16, 40, 40)

            # 编码成 [B, 1, 16, 16]，这里使用卷积层来实现
            x_t = self.conv(x_t)  # 用 3x3 的卷积核 torch.Size([32, 1, 16, 16])
            x_list.append(x_t)

        x = torch.stack(x_list, dim=1)  # torch.Size([32, 8, 1, 16, 16])

        # 1d数据简单重新组建形状
        node_history_encoded = node_history_encoded.reshape(batch_size,16,16) ##[B,16,16]
        node_history_encoded = node_history_encoded.unsqueeze(1) #[B,1,16,16]
        hidden = self.encode_time(x, node_history_encoded) #torch.Size([32, 8, 1, 16, 16]) #[B,1,16,16]

        return hidden #(B,256)


class TimeAwareEncoderForGPH10ZUVVMSVOQT(nn.Module):
    def __init__(self, point_num, image_size=100, hidden_dim=256):
        super(TimeAwareEncoderForGPH10ZUVVMSVOQT, self).__init__()
        self.point_num = point_num
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.conv_layers2 = nn.Sequential(  #torch.Size([32, 1, 100, 100])
            nn.Conv2d(in_channels=38, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=64),   #[B,64,50,50]门控 x*G     G=sig(conv(x))

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=128),  #[B,128,25,25]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            SpatialAttention1(in_channels=256)  #torch.Size([B, 256, 12, 12])

        )

        # self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3)
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3, padding=4)

        self.linear = nn.Linear(12, 256)

        # self.rnn = nn.GRU(
        #     input_size = 256 * (image_size // 8) ** 2,
        #     # input_size=32 * (image_size // 4) ** 2,
        #                   hidden_size=hidden_dim, batch_first=True)

        self.encode_time = SelfAttentionLSTM8(in_channels=1)

    def get_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        B,t,H,W = x.size()
        x = x.permute(1,0,2,3)
        shape = (t-1,B,H,W)
        # 创建空张量
        x_differ = torch.empty(shape)
        for i in range(t-1):
            x_differ[i] = x[i+1]-x[i]
        x_differ = x_differ.permute(1,0,2,3)
        return x_differ #（差值也可以增强 跨步数 额不太精准）[B, 7, 100, 100]

    def get_t_center_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :,:, 50, 50].unsqueeze(3).unsqueeze(4).expand_as(x)
        return x_t_center #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]

    def get_t_center_differ2(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :, 50, 50].unsqueeze(2).unsqueeze(3).expand_as(x)
        epsilon = 1e-12
        reciprocal_tensor = 1.0 / (x_t_center + epsilon)

        # 使用sigmoid函数将数值映射到0到1的范围内
        normalized_tensor = torch.sigmoid(reciprocal_tensor) #可能存在正负  这块等下调试看看

        return normalized_tensor #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]

    # 110
    def forward(self, x, node_history_encoded):  ## [B,T,10,100,100]  #[B,256] 过去的所有点 都在
        batch_size, _, _, _,_ = x.size()

        # 获取中心点差值
        x_t_center = self.get_t_center_differ(x) ## [B,T,10,100,100]
        # 2d数据编码
        x_list = []
        for t in range(self.point_num):
            x_t = torch.cat((x[:, t, :, :, :], x_t_center[:, t, :, :, :]), dim=1) #[B,20,100,100]

            x_t = self.conv_layers2(x_t.float())  # torch.Size([B, 256, 10, 10])  [B,20,100,100]

            # 首先 reshape 成 [B, 16, 48, 48]
            x_t = x_t.view(batch_size, 16, 40, 40)

            # 编码成 [B, 1, 16, 16]，这里使用卷积层来实现
            x_t = self.conv(x_t)  # 用 3x3 的卷积核 torch.Size([32, 1, 16, 16])
            x_list.append(x_t)

        x = torch.stack(x_list, dim=1)  # torch.Size([32, 8, 1, 16, 16])

        # 1d数据简单重新组建形状
        node_history_encoded = node_history_encoded.reshape(batch_size,16,16) ##[B,16,16]
        node_history_encoded = node_history_encoded.unsqueeze(1) #[B,1,16,16]
        hidden = self.encode_time(x, node_history_encoded) #torch.Size([32, 8, 1, 16, 16]) #[B,1,16,16]

        return hidden #(B,256)

class TimeAwareEncoderForGPH10ZUVVMSVO(nn.Module):
    def __init__(self, point_num, image_size=100, hidden_dim=256):
        super(TimeAwareEncoderForGPH10ZUVVMSVO, self).__init__()
        self.point_num = point_num
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.conv_layers2 = nn.Sequential(  #torch.Size([32, 1, 100, 100])
            nn.Conv2d(in_channels=28, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=64),   #[B,64,50,50]门控 x*G     G=sig(conv(x))

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=128),  #[B,128,25,25]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            SpatialAttention1(in_channels=256)  #torch.Size([B, 256, 12, 12])

        )

        # self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3)
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3, padding=4)

        self.linear = nn.Linear(12, 256)

        # self.rnn = nn.GRU(
        #     input_size = 256 * (image_size // 8) ** 2,
        #     # input_size=32 * (image_size // 4) ** 2,
        #                   hidden_size=hidden_dim, batch_first=True)

        self.encode_time = SelfAttentionLSTM8(in_channels=1)

    def get_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        B,t,H,W = x.size()
        x = x.permute(1,0,2,3)
        shape = (t-1,B,H,W)
        # 创建空张量
        x_differ = torch.empty(shape)
        for i in range(t-1):
            x_differ[i] = x[i+1]-x[i]
        x_differ = x_differ.permute(1,0,2,3)
        return x_differ #（差值也可以增强 跨步数 额不太精准）[B, 7, 100, 100]

    def get_t_center_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :,:, 50, 50].unsqueeze(3).unsqueeze(4).expand_as(x)
        return x_t_center #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]

    def get_t_center_differ2(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :, 50, 50].unsqueeze(2).unsqueeze(3).expand_as(x)
        epsilon = 1e-12
        reciprocal_tensor = 1.0 / (x_t_center + epsilon)

        # 使用sigmoid函数将数值映射到0到1的范围内
        normalized_tensor = torch.sigmoid(reciprocal_tensor) #可能存在正负  这块等下调试看看

        return normalized_tensor #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]

    # 110
    def forward(self, x, node_history_encoded):  ## [B,T,10,100,100]  #[B,256] 过去的所有点 都在
        batch_size, _, _, _,_ = x.size()

        # 获取中心点差值
        x_t_center = self.get_t_center_differ(x) ## [B,T,10,100,100]
        # 2d数据编码
        x_list = []
        for t in range(self.point_num):
            x_t = torch.cat((x[:, t, :, :, :], x_t_center[:, t, :, :, :]), dim=1) #[B,20,100,100]

            x_t = self.conv_layers2(x_t.float())  # torch.Size([B, 256, 10, 10])  [B,20,100,100]

            # 首先 reshape 成 [B, 16, 48, 48]
            x_t = x_t.view(batch_size, 16, 40, 40)

            # 编码成 [B, 1, 16, 16]，这里使用卷积层来实现
            x_t = self.conv(x_t)  # 用 3x3 的卷积核 torch.Size([32, 1, 16, 16])
            x_list.append(x_t)

        x = torch.stack(x_list, dim=1)  # torch.Size([32, 8, 1, 16, 16])

        # 1d数据简单重新组建形状
        node_history_encoded = node_history_encoded.reshape(batch_size,16,16) ##[B,16,16]
        node_history_encoded = node_history_encoded.unsqueeze(1) #[B,1,16,16]
        hidden = self.encode_time(x, node_history_encoded) #torch.Size([32, 8, 1, 16, 16]) #[B,1,16,16]

        return hidden #(B,256)



class TimeAwareEncoderForGPH10ALL(nn.Module):
    def __init__(self, point_num, image_size=100, hidden_dim=256):
        super(TimeAwareEncoderForGPH10ALL, self).__init__()
        self.point_num = point_num
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.conv_layers2 = nn.Sequential(  #torch.Size([32, 1, 100, 100])
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=64),   #[B,64,50,50]门控 x*G     G=sig(conv(x))

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=128),  #[B,128,25,25]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            SpatialAttention1(in_channels=256)  #torch.Size([B, 256, 12, 12])

        )
        self.conv_layers3 = nn.Sequential(  # torch.Size([B, 4, 100, 100])
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # torch.Size([B, 32, 50, 50])

            SpatialAttention1(in_channels=32),  # [B, 32, 50, 50]

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # [B, 64, 25, 25]

            SpatialAttention1(in_channels=64),  # [B,64,25,25]

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            SpatialAttention1(in_channels=128)  # torch.Size([B, 128, 12, 12])

        )

        self.conv = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=3)

        self.linear = nn.Linear(12, 256)

        # self.rnn = nn.GRU(
        #     input_size = 256 * (image_size // 8) ** 2,
        #     # input_size=32 * (image_size // 4) ** 2,
        #                   hidden_size=hidden_dim, batch_first=True)

        self.encode_time = SelfAttentionLSTM8(in_channels=1)

    def get_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        B,t,H,W = x.size()
        x = x.permute(1,0,2,3)
        shape = (t-1,B,H,W)
        # 创建空张量
        x_differ = torch.empty(shape)
        for i in range(t-1):
            x_differ[i] = x[i+1]-x[i]
        x_differ = x_differ.permute(1,0,2,3)
        return x_differ #（差值也可以增强 跨步数 额不太精准）[B, 7, 100, 100]

    def get_t_center_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :, :, 50, 50].unsqueeze(3).unsqueeze(4).expand_as(x)
        return x_t_center #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]

    # 110
    def forward(self, x, node_history_encoded):  #[B, 8, 2, 101, 101]
        batch_size, _, _, _, _ = x.size()
        x_t_center = self.get_t_center_differ(x) ##[B, 8, 100, 100]
        # 2d数据编码
        x_list = []
        for t in range(self.point_num): #一起编码呗 #torch.Size([B, 1, 100, 100])
            x_t = torch.cat((x[:, t, :, :,:], x_t_center[:, t, :, :,:]), dim=1) # Get the data for the current time step
            x_t = self.conv_layers3(x_t)  # torch.Size([B, 256, 12, 12]) --[B, 128, 12, 12]

            # 首先 reshape 成 [B, 16, 48, 48]
            x_t = x_t.view(batch_size, 8, 48, 48)

            # 编码成 [B, 1, 16, 16]，这里使用卷积层来实现
            x_t = self.conv(x_t)  # 用 3x3 的卷积核 torch.Size([B, 1, 16, 16])
            x_list.append(x_t)

        x = torch.stack(x_list, dim=1)  # torch.Size([32, 8, 1, 16, 16])

        # 1d数据简单重新组建形状
        node_history_encoded = node_history_encoded.reshape(batch_size,16,16) ##[B,16,16]
        node_history_encoded = node_history_encoded.unsqueeze(1) #[B,1,16,16]
        hidden = self.encode_time(x, node_history_encoded) #torch.Size([32, 8, 1, 16, 16]) #[B,1,16,16]

        return hidden #(B,256)


class TimeAwareEncoderForUV(nn.Module):
    def __init__(self, num_feature):
        super(TimeAwareEncoderForUV, self).__init__()
        self.encode_time = SelfAttentionLSTM8(in_channels=1)
        self.encode_his_to_his = GetWindEncByTransformer(num_features=num_feature)
        self.encode_his_to_his_direct = GetWindEncByTransformer(num_features=num_feature)
        self.get_future_from_his = WindTransformerVideoPredictor(num_features=num_feature, T1=8, T2=4)
        self.get_future_from_his_direct = WindTransformerVideoPredictor(num_features=num_feature, T1=8, T2=4)

    def get_wind_speed_direction(self, wind):
        #[Batchsize, Time, 4, 101, 101]
        u_wind = wind[:,:,:,:,:,0]  # 示例数据
        v_wind = wind[:,:,:,:,:,1]
        # 计算风速
        wind_speed = torch.sqrt(u_wind ** 2 + v_wind ** 2)

        # 计算风向 (arctan2 处理象限)
        wind_direction = torch.atan2(v_wind, u_wind)

        # 将风向从弧度转换为度，并将范围转换到 [0, 360]
        wind_direction_deg = torch.rad2deg(wind_direction)
        wind_direction_deg = (wind_direction_deg + 360) % 360  # 0.2-90
        # @@@wind_direction_deg 转成tensor
        return wind_speed,wind_direction_deg

    def get_t_center_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :, 50, 50].unsqueeze(2).unsqueeze(3).expand_as(x)
        return x_t_center #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]

    def update_track_with_wind(track, wind_speed_at_track, wind_direction_at_track, time_step=1):
        # transformer_encoder
        # 以简化方式利用风速和风向更新轨迹
        new_track = np.copy(track)
        for b in range(track.shape[0]):
            for t in range(1, track.shape[1]):
                avg_speed = np.mean(wind_speed_at_track[b, t - 1])  # 使用平均风速
                avg_direction = np.mean(wind_direction_at_track[b, t - 1])  # 使用平均风向
                # 计算经度和纬度的增量 (根据风速和风向的影响)
                delta_lat = avg_speed * np.cos(np.radians(avg_direction)) * time_step
                delta_lon = avg_speed * np.sin(np.radians(avg_direction)) * time_step
                # 更新经纬度
                new_track[b, t, 0] = track[b, t - 1, 0] + delta_lon  # 经度
                new_track[b, t, 1] = track[b, t - 1, 1] + delta_lat  # 纬度
        return new_track

    def cal_dif_last_his_and_pred(self,history_encoded,future_encoded):
        diff_8_to_1 = history_encoded[:, -1, :] - future_encoded[:, 0, :]
        future_diffs = future_encoded[:, 1:, :] - future_encoded[:, :-1, :]
        future_diffs = torch.concat((diff_8_to_1.unsqueeze(1), future_diffs), dim=1)
        return future_diffs

    # 110
    def forward(self, wind, future_data, node_history_encoded):  #[B, 8, 100, 100]  #[B,256] 过去的所有点 都在
        wind = wind.to(node_history_encoded.device)
        B, T1, C, H, W, _ = wind.shape
        hyperparams = get_traj_hypers()
        T2 = hyperparams['prediction_horizon']
        num_features = C * H * W
        # [B,T,4,101,101]
        # his
        wind_speed, wind_direction_deg = self.get_wind_speed_direction(wind)
        future_wind_speed, future_wind_direction = self.get_wind_speed_direction(future_data)

        #================================================================================================
        #his
        #wind speed
        history_data_value = wind_speed.reshape(B, T1, num_features).float()
        history_encoded = self.encode_his_to_his(history_data_value)  # 输出形状为 [B, T1, num_features]
        history_encoded = history_encoded.reshape(B, T1, C, H, W)
        # direct
        history_data_dir = wind_direction_deg.reshape(B, T1, num_features).float()
        history_encoded_dir = self.encode_his_to_his_direct(history_data_dir)  # 输出形状为 [B, T1, num_features]
        history_encoded_dir = history_encoded_dir.reshape(B, T1, C, H, W)

        #future===============================================================
        fu_v, fu_d = self.get_wind_speed_direction(future_data)
        #value
        fu_v = fu_v.reshape(B, T2, num_features).float()
        fu_enc_v_gt = self.encode_his_to_his(fu_v)
        fu_enc_v_gt = fu_enc_v_gt.reshape(B, T2, C, H, W)
        #direct
        fu_d = fu_d.reshape(B, T2, num_features).float()
        fu_enc_d_gt = self.encode_his_to_his_direct(fu_d)
        fu_enc_d_gt = fu_enc_d_gt.reshape(B, T2, C, H, W)

        # his-->future
        fu_enc_v = self.get_future_from_his(history_encoded)  # [B,t2,4,101,101]
        fu_enc_d = self.get_future_from_his_direct(history_encoded_dir)

        # 第三步：计算差值
        enc_diffs_v = self.cal_dif_last_his_and_pred(history_encoded, fu_enc_v)
        enc_diffs_d = self.cal_dif_last_his_and_pred(history_encoded_dir, fu_enc_d)

        # 预测特征的误差
        dif_fu_v_d = (fu_enc_v_gt - fu_enc_v) + (fu_enc_d_gt - fu_enc_d)
        # 获取时刻间差值
        # x_differ = self.get_differ(x) #[B, 7, 100, 100]
        # # 获取中心点差值
        # x_t_center = self.get_t_center_differ(x) ##[B, 8, 100, 100]
        # 2d数据编码
        # x_list = []
        # for t in range(self.point_num): #一起编码呗 #torch.Size([B, 1, 100, 100])
        #     x_t = torch.cat((x[:, t, :, :].unsqueeze(1), x_t_center[:, t, :, :].unsqueeze(1)), dim=1) # Get the data for the current time step
        #
        #     x_t = self.conv_layers2(x_t)  # torch.Size([32, 256, 12, 12])
        #
        #     # 首先 reshape 成 [B, 16, 48, 48]
        #     x_t = x_t.view(batch_size, 16, 48, 48)
        #
        #     # 编码成 [B, 1, 16, 16]，这里使用卷积层来实现
        #     x_t = self.conv(x_t)  # 用 3x3 的卷积核 torch.Size([32, 1, 16, 16])
        #     x_list.append(x_t)
        #
        # x = torch.stack(x_list, dim=1)  # torch.Size([32, 8, 1, 16, 16])

        # 1d数据简单重新组建形状
        # node_history_encoded = node_history_encoded.reshape(batch_size,16,16) ##[B,16,16]
        # node_history_encoded = node_history_encoded.unsqueeze(1) #[B,1,16,16]
        # hidden = self.encode_time(x, node_history_encoded) #torch.Size([32, 8, 1, 16, 16]) #[B,1,16,16]

        return enc_diffs_v,enc_diffs_d #(B,256)



class TimeAwareEncoderForGPH200(nn.Module):
    def __init__(self, point_num, image_size=100, hidden_dim=256):
        super(TimeAwareEncoderForGPH200, self).__init__()
        self.point_num = point_num
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.conv_layers2 = nn.Sequential(  #torch.Size([32, 1, 100, 100])
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=64),   #[B,64,50,50]门控 x*G     G=sig(conv(x))

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            SpatialAttention1(in_channels=128),  #[B,128,25,25]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            SpatialAttention1(in_channels=256)  #torch.Size([B, 256, 12, 12])

        )

        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3)

        self.linear = nn.Linear(12, 256)

        # self.rnn = nn.GRU(
        #     input_size = 256 * (image_size // 8) ** 2,
        #     # input_size=32 * (image_size // 4) ** 2,
        #                   hidden_size=hidden_dim, batch_first=True)

        self.encode_time = SelfAttentionLSTM8(in_channels=1)

    def get_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        B,t,H,W = x.size()
        x = x.permute(1,0,2,3)
        shape = (t-1,B,H,W)
        # 创建空张量
        x_differ = torch.empty(shape)
        for i in range(t-1):
            x_differ[i] = x[i+1]-x[i]
        x_differ = x_differ.permute(1,0,2,3)
        return x_differ #（差值也可以增强 跨步数 额不太精准）[B, 7, 100, 100]

    def get_t_center_differ(self, x): #x:torch.Size([B, 8, 100, 100])
        x_t_center = x-x[:, :, :, 50, 50].unsqueeze(3).unsqueeze(4).expand_as(x)
        return x_t_center #（差值也可以增强 跨步数 额不太精准）[B, 8, 100, 100]
    # 110
    def forward(self, x, node_history_encoded):  #[B, 8, 100, 100]  #[B,256] 过去的所有点 都在
        x = torch.Tensor(x)
        batch_size, _, _, _, _ = x.size()
        # 获取中心点差值
        x_t_center = self.get_t_center_differ(x) # [B, 8, 4, 101, 101]
        # 2d数据编码
        x_list = []
        for t in range(self.point_num): #一起编码呗 #torch.Size([B, 1, 100, 100])
            x_t = torch.cat((x[:, t, :, :, :], x_t_center[:, t, :, :, :]), dim=1) #[B,8,101,101] Get the data for the current time step
            x_t = x_t.to(node_history_encoded.device)
            x_t = self.conv_layers2(x_t)  # torch.Size([32, 256, 12, 12])

            # 首先 reshape 成 [B, 16, 48, 48]
            x_t = x_t.view(batch_size, 16, 48, 48)

            # 编码成 [B, 1, 16, 16]，这里使用卷积层来实现
            x_t = self.conv(x_t)  # 用 3x3 的卷积核 torch.Size([32, 1, 16, 16])
            x_list.append(x_t)

        x = torch.stack(x_list, dim=1)  # torch.Size([32, 8, 1, 16, 16])

        # 1d数据简单重新组建形状
        node_history_encoded = node_history_encoded.reshape(batch_size,16,16) ##[B,16,16]
        node_history_encoded = node_history_encoded.unsqueeze(1) #[B,1,16,16]
        hidden = self.encode_time(x, node_history_encoded) #torch.Size([32, 8, 1, 16, 16]) #[B,1,16,16]

        return hidden #(B,256)


class EncoderForAge(nn.Module):
    def __init__(self):
        super(EncoderForAge, self).__init__()
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=3)
        self.linear = nn.Linear(4, 4)

    # 110
    def forward(self, timestep_range_x):  #[B, 8, 100, 100]  #[B,256] 过去的所有点 都在
        t = timestep_range_x[:, 1:2].to(dtype=torch.float32) #[B,1]
        ty = torch.randn(timestep_range_x.size(0), 4)
        for i in range(4):
            ty[:,i:i+1]=t+1+i

        t1 = (ty/50).to(t.device).to(dtype=torch.float32)
        out = self.linear(t1)
        return torch.tanh(out) # [B,4] [-1,1]

class TimeAwareEncoderForENV5Wind(nn.Module):
    def __init__(self, point_num=8, hidden_dim=256):
        super(TimeAwareEncoderForENV5Wind, self).__init__()
        self.time = point_num

        embed_dim = 16
        self.data_embed = nn.ModuleDict()
        self.data_embed['wind'] = nn.Linear(1, embed_dim)

        # self.env_encoder = LSTM_for_env(out_feature=256)

    def forward(self, env_data_t, x_encoding): #x_encoding:[B,256]
        #终于到这一步了env_data dict,0-7时刻，每个里面B个dict,13项 env_data[t][b1]
        B=x_encoding.size(0)
        embed = []
        t=7
        env_data = env_data_t[t]
        embed_list = []
        for key in self.data_embed:
            list_single_key = []
            for i in range(B):
                list_single_key.append(env_data[i][key])

            tensor_single_key = torch.tensor(list_single_key).to(x_encoding.device).to(dtype=torch.float32)  # [B,6]
            if (len(tensor_single_key.shape) == 1):
                tensor_single_key = tensor_single_key.unsqueeze(1)

            now_embed = self.data_embed[key](tensor_single_key)
            embed_list.append(now_embed)

        emb_env = torch.cat(embed_list, dim=1) #[B,80]

        return emb_env #[B,80]

# TimeAwareEncoderForENV5Trajectory
# TimeAwareEncoderForENV5Intensity
class TimeAwareEncoderForENV5Trajectory(nn.Module):
    def __init__(self, point_num=8, hidden_dim=256):
        super(TimeAwareEncoderForENV5Trajectory, self).__init__()
        self.time = point_num

        embed_dim = 16
        self.data_embed = nn.ModuleDict()
        self.data_embed['move_velocity'] = nn.Linear(1, embed_dim)
        self.data_embed['future_direction24'] = nn.Linear(1, embed_dim)  # -1    OK

        # self.env_encoder = LSTM_for_env(out_feature=256)

    def forward(self, env_data_t, x_encoding): #x_encoding:[B,256]
        #终于到这一步了env_data dict,0-7时刻，每个里面B个dict,13项 env_data[t][b1]
        B=x_encoding.size(0)
        embed = []
        t=7
        env_data = env_data_t[t]
        embed_list = []
        for key in self.data_embed:
            list_single_key = []
            for i in range(B):
                list_single_key.append(env_data[i][key])

            tensor_single_key = torch.tensor(list_single_key).to(x_encoding.device).to(dtype=torch.float32)  # [B,6]
            if (len(tensor_single_key.shape) == 1):
                tensor_single_key = tensor_single_key.unsqueeze(1)

            if(key=='future_direction24'):
                # tensor_single_key[:, :] = 1  #1已测完
                now_embed = self.data_embed[key](tensor_single_key)
            else:
                now_embed = self.data_embed[key](tensor_single_key)

            embed_list.append(now_embed)

        emb_env = torch.cat(embed_list, dim=1) #[B,32]

        return emb_env #[B,32]


class TimeAwareEncoderForENV5Intensity(nn.Module):
    def __init__(self, point_num=8, hidden_dim=256):
        super(TimeAwareEncoderForENV5Intensity, self).__init__()
        self.time = point_num

        embed_dim = 16
        self.data_embed = nn.ModuleDict()
        self.data_embed['intensity_class'] = nn.Linear(6, embed_dim)
        self.data_embed['future_inte_change24'] = nn.Linear(1, embed_dim)  # -1  OK  16*9

        # self.env_encoder = LSTM_for_env(out_feature=256)

    def forward(self, env_data_t, x_encoding): #x_encoding:[B,256]
        #终于到这一步了env_data dict,0-7时刻，每个里面B个dict,13项 env_data[t][b1]
        B=x_encoding.size(0)
        embed = []
        t=7
        env_data = env_data_t[t]
        embed_list = []
        for key in self.data_embed:
            list_single_key = []
            for i in range(B):
                list_single_key.append(env_data[i][key])

            tensor_single_key = torch.tensor(np.array(list_single_key)).to(x_encoding.device).to(dtype=torch.float32)  # [B,6]
            if (len(tensor_single_key.shape) == 1):
                tensor_single_key = tensor_single_key.unsqueeze(1)

            if (key == 'future_inte_change24'):
                # tensor_single_key[:, :] = 1  #固定1
                now_embed = self.data_embed[key](tensor_single_key)
            else:
                now_embed = self.data_embed[key](tensor_single_key)

            embed_list.append(now_embed)

        emb_env = torch.cat(embed_list, dim=1) #[B,32]

        return emb_env #[B,32]

class LSTM_for_env(nn.Module):
    def __init__(self, out_feature=256):
        super(LSTM_for_env, self).__init__()
        self.out_feature = out_feature
        in_channels = 1
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 通道1--1
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.conv7 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.conv10 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.conv11 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def sa_conv_lstm(self, x, en_1d):  ##torch.Size([8, 32, 1, 16, 16]) #[B,1,16,16]
        # #看sa-conv-lstm的
        # M,H：每一个小的 都是（B，256）
        memory = torch.zeros_like(x[0])  # [32, 1, 16, 16]
        H = torch.zeros_like(x[0])
        C = torch.randn_like(x[0]) * 1e-6

        for i in range(x.size(0)):  # 8次循环
            a_xh = torch.sigmoid(self.conv10(torch.cat((H, x[i], en_1d), dim=1)))  # 到单通道吗
            ca_xh = C * a_xh
            ga = torch.sigmoid(self.conv11(torch.cat((H, x[i], en_1d), dim=1)))
            gv = torch.tanh(self.conv12(torch.cat((H, x[i], en_1d), dim=1)))
            C = ca_xh + ga * gv
            a_xh1 = torch.sigmoid(self.conv13(torch.cat((H, x[i], en_1d), dim=1)))
            H = a_xh1 * torch.tanh(C)
            memory, H = self.self_attention_memory(memory, H)  # H:torch.Size([32, 1, 16, 16])
        return H

    def self_attention_memory(self, m, h):  # [32, 1, 16, 16]
        vh = self.conv1(h)
        kh = self.conv2(h)
        qh = self.conv3(h)
        qh = torch.transpose(qh, 2, 3)
        ah = F.softmax(kh * qh, dim=-1)  # 基本全是0.0625 0.0624
        zh = vh * ah

        km = self.conv4(m)
        vm = self.conv5(m)
        am = F.softmax(qh * km, dim=-1)
        zm = vm * am
        z0 = torch.cat((zh, zm), dim=1)
        z = self.conv6(z0)
        hz = torch.cat((h, z), dim=1)

        ot = torch.sigmoid(self.conv7(hz))  # 到单通道吗
        gt = torch.tanh(self.conv8(hz))
        it = torch.sigmoid(self.conv9(hz))

        gi = gt * it
        mf = (1 - it) * m
        mt = gi + mf
        ht = ot * mt

        return mt, ht

    def forward(self, x, en_1d):
        #x: [T,B,256] torch.Size([8, 256, 256])
        # en_1d: [B,256]
        T, B, _ = x.size()
        x = x.reshape(T,B,16,16)  #[T,B,16,16]
        x = x.unsqueeze(2)   #[T,B,1,16,16]   [8,256,1,16,16]
        x = x/10
        en_1d = en_1d.reshape(B,16,16)
        en_1d = en_1d.unsqueeze(1)  #[B,1,16,16]
        H = self.sa_conv_lstm(x, en_1d)  # [B,1,16,16]
        flattened_tensor = H.view(B, -1)
        return flattened_tensor  # (B,256) 特别趋同

class LSTM_for_env2(nn.Module):
    def __init__(self, out_feature=256):
        super(LSTM_for_env2, self).__init__()
        self.out_feature = out_feature
        in_channels = 1
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 通道1--1
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.conv7 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.conv10 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.conv11 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def sa_conv_lstm(self, x, en_1d):  ##torch.Size([8, 32, 1, 16, 16]) #[B,1,16,16]
        # #看sa-conv-lstm的
        # M,H：每一个小的 都是（B，256）
        memory = torch.zeros_like(x)  # [32, 1, 16, 16]
        H = torch.zeros_like(x)
        C = torch.randn_like(x) * 1e-6

        a_xh = torch.sigmoid(self.conv10(torch.cat((H, x, en_1d), dim=1)))  # 到单通道吗
        ca_xh = C * a_xh
        ga = torch.sigmoid(self.conv11(torch.cat((H, x, en_1d), dim=1)))
        gv = torch.tanh(self.conv12(torch.cat((H, x, en_1d), dim=1)))
        C = ca_xh + ga * gv
        a_xh1 = torch.sigmoid(self.conv13(torch.cat((H, x, en_1d), dim=1)))
        H = a_xh1 * torch.tanh(C)
        memory, H = self.self_attention_memory(memory, H)  # H:torch.Size([32, 1, 16, 16])
        return H

    def self_attention_memory(self, m, h):  # [32, 1, 16, 16]
        vh = self.conv1(h)
        kh = self.conv2(h)
        qh = self.conv3(h)
        qh = torch.transpose(qh, 2, 3)
        ah = F.softmax(kh * qh, dim=-1)  # 基本全是0.0625 0.0624
        zh = vh * ah

        km = self.conv4(m)
        vm = self.conv5(m)
        am = F.softmax(qh * km, dim=-1)
        zm = vm * am
        z0 = torch.cat((zh, zm), dim=1)
        z = self.conv6(z0)
        hz = torch.cat((h, z), dim=1)

        ot = torch.sigmoid(self.conv7(hz))  # 到单通道吗
        gt = torch.tanh(self.conv8(hz))
        it = torch.sigmoid(self.conv9(hz))

        gi = gt * it
        mf = (1 - it) * m
        mt = gi + mf
        ht = ot * mt

        return mt, ht

    def forward(self, x, en_1d):
        #x: [B,256]
        # en_1d: [B,256]
        B, _ = x.size()
        x = x.reshape(B,16,16)  #[B,16,16]
        x = x.unsqueeze(1)   #[B,1,16,16]
        x = x/10
        en_1d = en_1d.reshape(B,16,16)
        en_1d = en_1d.unsqueeze(1)  #[B,1,16,16]
        H = self.sa_conv_lstm(x, en_1d)  # [B,1,16,16]
        flattened_tensor = H.view(B, -1)
        return flattened_tensor  # (B,256) 特别趋同

class TimeAwareEncoderForENV(nn.Module):
    def __init__(self, point_num=8, hidden_dim=256):
        super(TimeAwareEncoderForENV, self).__init__()
        self.time = point_num

        embed_dim = 16
        self.data_embed = nn.ModuleDict()
        self.data_embed['area'] = nn.Linear(6, embed_dim)
        self.data_embed['wind'] = nn.Linear(1, embed_dim)
        self.data_embed['intensity_class'] = nn.Linear(6, embed_dim)
        self.data_embed['move_velocity'] = nn.Linear(1, embed_dim)
        self.data_embed['month'] = nn.Linear(12, embed_dim)
        # self.data_embed['locationg'] 太大
        self.data_embed['location_long'] = nn.Linear(36, embed_dim)
        self.data_embed['location_lat'] = nn.Linear(12, embed_dim)
        # self.data_embed['history_direction12'] = nn.Linear(8, embed_dim) #this   有的是-1
        # self.data_embed['history_direction24'] = nn.Linear(8, embed_dim) #-1     有的是-1
        self.data_embed['future_direction24'] = nn.Linear(1, embed_dim)  # -1    OK
        # self.data_embed['history_inte_change24'] = nn.Linear(4, embed_dim) #-1   有的是-1
        self.data_embed['future_inte_change24'] = nn.Linear(1, embed_dim)  # -1  OK  16*9

        self.linear144to256 = nn.Linear(144, hidden_dim)   #144,256

        self.env_encoder = LSTM_for_env(out_feature=256)

    def forward(self, env_data_t, x_encoding): #x_encoding:[B,256]
        #终于到这一步了env_data dict,0-7时刻，每个里面B个dict,13项 env_data[t][b1]
        B=x_encoding.size(0)
        embed = []
        for t in range(self.time):
            env_data = env_data_t[t]
            embed_list = []
            for key in self.data_embed:
                list_single_key = []
                for i in range(B):
                    list_single_key.append(env_data[i][key])

                tensor_single_key = torch.tensor(list_single_key).to(x_encoding.device).to(dtype=torch.float32)  #[B,6]
                if (len(tensor_single_key.shape)==1) :
                    tensor_single_key = tensor_single_key.unsqueeze(1)

                now_embed = self.data_embed[key](tensor_single_key)
                embed_list.append(now_embed)

            emb_256 = self.linear144to256(torch.cat(embed_list, dim=1))  #torch.Size([256, 144])
            embed.append(emb_256)  #list:一个是torch.Size([256, 144])

        tensor_embed = torch.stack(embed) #[T,B,144] torch.Size([8, 256, 144])

        feature = self.env_encoder(tensor_embed,x_encoding)  #[B,256]
        return feature #[B,256]

class TimeAwareEncoderForENV4(nn.Module):
    def __init__(self, point_num=8, hidden_dim=256):
        super(TimeAwareEncoderForENV4, self).__init__()
        self.time = point_num

        embed_dim = 16
        self.data_embed = nn.ModuleDict()
        self.data_embed['area'] = nn.Linear(6, embed_dim)
        self.data_embed['wind'] = nn.Linear(1, embed_dim)
        self.data_embed['intensity_class'] = nn.Linear(6, embed_dim)
        self.data_embed['move_velocity'] = nn.Linear(1, embed_dim)
        self.data_embed['month'] = nn.Linear(12, embed_dim)
        # self.data_embed['locationg'] 太大
        self.data_embed['location_long'] = nn.Linear(36, embed_dim)
        self.data_embed['location_lat'] = nn.Linear(12, embed_dim)
        # self.data_embed['history_direction12'] = nn.Linear(8, embed_dim) #this   有的是-1
        # self.data_embed['history_direction24'] = nn.Linear(8, embed_dim) #-1     有的是-1
        self.data_embed['future_direction24'] = nn.Linear(1, embed_dim)  # -1    OK
        # self.data_embed['history_inte_change24'] = nn.Linear(4, embed_dim) #-1   有的是-1
        self.data_embed['future_inte_change24'] = nn.Linear(1, embed_dim)  # -1  OK  16*9

        self.old = nn.Linear(1, embed_dim)

        self.linear160to256 = nn.Linear(160, hidden_dim)   #144,256

        self.env_encoder = LSTM_for_env(out_feature=256)

    def forward(self, env_data_t, x_encoding,timestep_range_x,timestep_range_y): #x_encoding:[B,256]
        #终于到这一步了env_data dict,0-7时刻，每个里面B个dict,13项 env_data[t][b1]
        B=x_encoding.size(0)
        embed = []

        t1 = timestep_range_x[:,0:1].to(x_encoding.device).to(dtype=torch.float32)
        t2 = t1.clone()

        for t in range(self.time):
            env_data = env_data_t[t]
            embed_list = []
            for key in self.data_embed:
                list_single_key = []
                for i in range(B):
                    list_single_key.append(env_data[i][key])

                tensor_single_key = torch.tensor(list_single_key).to(x_encoding.device).to(dtype=torch.float32)  #[B,6]
                if (len(tensor_single_key.shape)==1) :
                    tensor_single_key = tensor_single_key.unsqueeze(1)

                now_embed = self.data_embed[key](tensor_single_key)
                embed_list.append(now_embed)

            old = self.old(t2)
            embed_list.append(old)
            with torch.no_grad():
                t2=t2+1

            emb_256 = self.linear160to256(torch.cat(embed_list, dim=1))  #torch.Size([256, 144])
            embed.append(emb_256)  #list:一个是torch.Size([256, 144])

        tensor_embed = torch.stack(embed) #[T,B,144] torch.Size([8, 256, 144])

        feature = self.env_encoder(tensor_embed,x_encoding)  #[B,256]
        return feature #[B,256]

class TimeAwareEncoderForENV2(nn.Module):
    def __init__(self, point_num=8, hidden_dim=256):
        super(TimeAwareEncoderForENV2, self).__init__()
        self.time = point_num

        embed_dim = 16
        self.data_embed = nn.ModuleDict()
        self.data_embed['area'] = nn.Linear(6, embed_dim)
        self.data_embed['wind'] = nn.Linear(1, embed_dim)
        self.data_embed['intensity_class'] = nn.Linear(6, embed_dim)
        self.data_embed['move_velocity'] = nn.Linear(1, embed_dim)
        self.data_embed['month'] = nn.Linear(12, embed_dim)
        # self.data_embed['locationg'] 太大
        self.data_embed['location_long'] = nn.Linear(36, embed_dim)
        self.data_embed['location_lat'] = nn.Linear(12, embed_dim)

        self.data_embed['history_direction12'] = nn.Linear(8, embed_dim) #this   有的是-1
        self.data_embed['history_direction24'] = nn.Linear(8, embed_dim) #-1     有的是-1
        self.data_embed['future_direction24'] = nn.Linear(1, embed_dim)  # -1    OK

        self.data_embed['history_inte_change24'] = nn.Linear(4, embed_dim) #-1   有的是-1
        self.data_embed['future_inte_change24'] = nn.Linear(1, embed_dim)  # -1  OK  16*9

        self.linear192to256 = nn.Linear(192, hidden_dim)   #144,256

        self.env_encoder = LSTM_for_env(out_feature=256)

    def forward(self, env_data_t, x_encoding): #x_encoding:[B,256]
        #终于到这一步了env_data dict,0-7时刻，每个里面B个dict,13项 env_data[t][b1]
        B=x_encoding.size(0)
        embed = []
        env_data = env_data_t[self.time-1]
        embed_list = []
        for key in self.data_embed:
            list_single_key = []
            for i in range(B):
                list_single_key.append(env_data[i][key])

            tensor_single_key = torch.tensor(list_single_key).to(x_encoding.device).to(dtype=torch.float32)  # [B,6]
            if (len(tensor_single_key.shape) == 1):
                tensor_single_key = tensor_single_key.unsqueeze(1)

            now_embed = self.data_embed[key](tensor_single_key)
            embed_list.append(now_embed)

        emb_256 = self.linear192to256(torch.cat(embed_list, dim=1))  # torch.Size([256, 256])

        return emb_256 #[B,256]


class TimeAwareEncoderForENV3(nn.Module):
    def __init__(self, point_num=8, hidden_dim=256):
        super(TimeAwareEncoderForENV3, self).__init__()
        self.time = point_num

        embed_dim = 16
        self.data_embed = nn.ModuleDict()
        self.data_embed['area'] = nn.Linear(6, embed_dim)
        self.data_embed['wind'] = nn.Linear(1, embed_dim)
        self.data_embed['intensity_class'] = nn.Linear(6, embed_dim)
        self.data_embed['move_velocity'] = nn.Linear(1, embed_dim)
        self.data_embed['month'] = nn.Linear(12, embed_dim)
        # self.data_embed['locationg'] 太大
        self.data_embed['location_long'] = nn.Linear(36, embed_dim)
        self.data_embed['location_lat'] = nn.Linear(12, embed_dim)

        self.data_embed['history_direction12'] = nn.Linear(8, embed_dim) #this   有的是-1
        self.data_embed['history_direction24'] = nn.Linear(8, embed_dim) #-1     有的是-1
        self.data_embed['future_direction24'] = nn.Linear(1, embed_dim)  # -1    OK

        self.data_embed['history_inte_change24'] = nn.Linear(4, embed_dim) #-1   有的是-1
        self.data_embed['future_inte_change24'] = nn.Linear(1, embed_dim)  # -1  OK  16*9

        self.linear192to256 = nn.Linear(192, hidden_dim)   #144,256

        self.env_encoder = LSTM_for_env2(out_feature=256)

    def forward(self, env_data_t, x_encoding): #x_encoding:[B,256]
        #终于到这一步了env_data dict,0-7时刻，每个里面B个dict,13项 env_data[t][b1]
        B=x_encoding.size(0)
        embed = []
        env_data = env_data_t[self.time-1]
        embed_list = []
        for key in self.data_embed:
            list_single_key = []
            for i in range(B):
                list_single_key.append(env_data[i][key])

            tensor_single_key = torch.tensor(list_single_key).to(x_encoding.device).to(dtype=torch.float32)  # [B,6]
            if (len(tensor_single_key.shape) == 1):
                tensor_single_key = tensor_single_key.unsqueeze(1)

            now_embed = self.data_embed[key](tensor_single_key)
            embed_list.append(now_embed)

        emb_256 = self.linear192to256(torch.cat(embed_list, dim=1))  # torch.Size([256, 256])
        emb_256_add_1d = self.env_encoder(emb_256, x_encoding) #(B,256)

        return emb_256_add_1d #[B,256]

class TimeAwareEncoderForENV1(nn.Module):
    def __init__(self, point_num=8, hidden_dim=256):
        super(TimeAwareEncoderForENV1, self).__init__()
        self.time = point_num

        embed_dim = 16
        self.data_embed = nn.ModuleDict()
        self.data_embed['area'] = nn.Linear(6, embed_dim)
        self.data_embed['wind'] = nn.Linear(1, embed_dim)
        self.data_embed['intensity_class'] = nn.Linear(6, embed_dim)
        self.data_embed['move_velocity'] = nn.Linear(1, embed_dim)
        self.data_embed['month'] = nn.Linear(12, embed_dim*3)
        # self.data_embed['locationg'] 太大
        self.data_embed['location_long'] = nn.Linear(36, embed_dim*4)
        self.data_embed['location_lat'] = nn.Linear(12, embed_dim*3)
        # self.data_embed['history_direction12'] = nn.Linear(8, embed_dim) #this   有的是-1
        # self.data_embed['history_direction24'] = nn.Linear(8, embed_dim) #-1     有的是-1
        self.data_embed['future_direction24'] = nn.Linear(1, embed_dim)  # -1    OK
        # self.data_embed['history_inte_change24'] = nn.Linear(4, embed_dim) #-1   有的是-1
        self.data_embed['future_inte_change24'] = nn.Linear(1, embed_dim)  # -1  OK  16*16

        # self.linear160to256 = nn.Linear(160, hidden_dim)   #160,256

        self.env_encoder = LSTM_for_env(out_feature=256)

    def forward(self, env_data_t, x_encoding): #x_encoding:[B,256]
        #终于到这一步了env_data dict,0-7时刻，每个里面B个dict,13项 env_data[t][b1]
        B=x_encoding.size(0)
        embed = []
        for t in range(self.time):
            env_data = env_data_t[t]
            embed_list = []
            for key in self.data_embed:
                list_single_key = []
                for i in range(B):
                    list_single_key.append(env_data[i][key])

                tensor_single_key = torch.tensor(list_single_key).to(x_encoding.device).to(dtype=torch.float32)  #[B,6]
                if (len(tensor_single_key.shape)==1) :
                    tensor_single_key = tensor_single_key.unsqueeze(1)

                now_embed = self.data_embed[key](tensor_single_key)
                embed_list.append(now_embed)

            # emb_256 = self.linear160to256(torch.cat(embed_list, dim=1))  #torch.Size([256, 144])
            emb_256 = torch.cat(embed_list, dim=1) #torch.Size([256, 256])
            embed.append(emb_256)  #list:一个是torch.Size([256, 144])

        tensor_embed = torch.stack(embed) #[T,B,144] torch.Size([8, 256, 144])

        feature = self.env_encoder(tensor_embed,x_encoding)  #[B,256]
        return feature #[B,256]

class TransformerFusionModel(nn.Module):
    def __init__(self, input_size, num_heads=4, num_layers=4):
        super(TransformerFusionModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, num_heads, dim_feedforward=input_size * 4),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x1, x2, D):
        # 将输入张量沿着第2维拼接
        fused_input = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0), D.unsqueeze(0)), dim=0)

        # 使用Transformer进行特征融合
        fused_output = self.transformer(fused_input)
        fused_output = fused_output.mean(dim=0)  # 取平均作为融合后的特征

        # 使用全连接层进行输出
        output = self.fc(fused_output)

        return output

class TransformerFusionModel2(nn.Module):
    def __init__(self, input_size, num_heads=4, num_layers=4):
        super(TransformerFusionModel2, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, num_heads, dim_feedforward=input_size * 4),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x1, x2, x3):
        # 将输入张量沿着第2维拼接
        fused_input = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0), x2.unsqueeze(0)), dim=0)

        # 使用Transformer进行特征融合
        fused_output = self.transformer(fused_input)
        fused_output = fused_output.mean(dim=0)  # 取平均作为融合后的特征

        # 使用全连接层进行输出
        output = self.fc(fused_output)

        return output


class AttentionFusionWithResidual(nn.Module):
    def __init__(self, input_size, fusion_size):
        super(AttentionFusionWithResidual, self).__init__()
        # self.gph_mapping = nn.Linear(fusion_size, input_size)  # 映射 gph 编码到与 x 编码相同的维度
        # self.fusion_layer = nn.Linear(input_size + input_size, input_size)  # 输入维度为两个 x 编码的维度

        self.transformer_fusion_model = TransformerFusionModel(input_size)

    def forward(self, x, gph, D_encode):
        # mapped_gph = self.gph_mapping(gph)  # 映射 gph 编码到与 x 编码相同的维度  mapped_gph的均值是x的三分之一，方差无敌小
        # mapped_gph = gph/10
        fusion = self.transformer_fusion_model(x, gph, D_encode) #B,256
        residual = x + fusion*0.001
        return residual

class AttentionFusionWithResidual2(nn.Module):
    def __init__(self, input_size, fusion_size):
        super(AttentionFusionWithResidual2, self).__init__()
        # self.gph_mapping = nn.Linear(fusion_size, input_size)  # 映射 gph 编码到与 x 编码相同的维度
        # self.fusion_layer = nn.Linear(input_size + input_size, input_size)  # 输入维度为两个 x 编码的维度

        self.transformer_fusion_model = TransformerFusionModel2(input_size)

    def forward(self, x, gph_500, gph_200):
        # mapped_gph = self.gph_mapping(gph)  # 映射 gph 编码到与 x 编码相同的维度  mapped_gph的均值是x的三分之一，方差无敌小
        # mapped_gph = gph/10
        fusion = self.transformer_fusion_model(x, gph_500, gph_200) #B,256
        residual = x + fusion*0.001
        return residual


# 正在写
class FusionAttentionModel(nn.Module):
    def __init__(self, num_layer):
        super(FusionAttentionModel, self).__init__()
        self.num_layer = num_layer
        self.gate_for_gph = Use_1d_choose_F(input_size=256)
        self.gate_for_env = Use_1d_choose_F(input_size=256)
        self.fc = nn.Linear(256*3, 256)

    def forward(self, x, gph, env):
        fusion = x
        num_Layer = self.num_layer
        for i in range(num_Layer):
            gph_new = self.gate_for_gph(fusion, gph)  # [B,256]
            env_new = self.gate_for_env(fusion, env)
            fusion = self.fc(torch.concat((gph_new,fusion,env_new),dim = 1)) # [B,256]

        return fusion


class AttentionFusionWithResidual6(nn.Module):
    def __init__(self, input_size, fusion_size):
        super(AttentionFusionWithResidual6, self).__init__()
        self.fusion_attention_model = FusionAttentionModel(num_layer=4)

    def forward(self, x, gph, env):
        fusion_1d_2d_env = self.fusion_attention_model(x, gph, env/10) #B,256

        residual = x + fusion_1d_2d_env*0.001 #可以考虑0.01 0.1 都可以
        return residual

class Use_1d_choose_F(nn.Module):
    def __init__(self, input_size):
        super(Use_1d_choose_F, self).__init__()
        in_channels = 1
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x, gph):
        B,_ = x.size()
        x1 = x.reshape(B,16,16)
        x1 = x1.unsqueeze(1)  #(B,1,16,16)
        gph1 = gph.reshape(B, 16, 16)
        gph1 = gph1.unsqueeze(1)  # (B,1,16,16)

        attention_map1 = self.conv1(x1)  # x:torch.Size([32, 64, 50, 50]) map:torch.Size([32, 1, 50, 50])
        attention_map1_transposed = torch.transpose(attention_map1, 2, 3)  # 交换通道维度
        attention_map2 = self.conv2(gph1)

        attention_map = attention_map1_transposed * attention_map2

        attention_weights = torch.sigmoid(attention_map)  # torch.Size([32, 1, 50, 50])

        res = x1 * attention_weights  #[B,1,16,16]
        res = res.reshape(B,256)

        return res #(B,256)

class AttentionFusionWithResidual3(nn.Module):
    def __init__(self, input_size, fusion_size):
        super(AttentionFusionWithResidual3, self).__init__()
        # self.gph_mapping = nn.Linear(fusion_size, input_size)  # 映射 gph 编码到与 x 编码相同的维度
        # self.fusion_layer = nn.Linear(input_size + input_size, input_size)  # 输入维度为两个 x 编码的维度

        self.transformer_fusion_model = TransformerFusionModel(input_size)
        self.transformer_fusion_model2 = TransformerFusionModel(input_size)

        self.gate_for_gph = Use_1d_choose_F(input_size=256)
        self.gate_for_env = Use_1d_choose_F(input_size=256)

    def forward(self, x, gph, env): #(B, 256)
        B,_ = x.size()
        gph_new = self.gate_for_gph(x,gph)
        env_new = self.gate_for_env(x,env)

        fusion_x_gph = self.transformer_fusion_model(x, gph_new) #B,256
        fusion_x_env = self.transformer_fusion_model2(x, env_new) #B,256 0.45  0.67?

        residual = x + fusion_x_gph*0.001 + fusion_x_env*0.001
        return residual

class AttentionFusionWithResidual4(nn.Module):
    def __init__(self, input_size, fusion_size):
        super(AttentionFusionWithResidual4, self).__init__()
        # self.gph_mapping = nn.Linear(fusion_size, input_size)  # 映射 gph 编码到与 x 编码相同的维度
        # self.fusion_layer = nn.Linear(input_size + input_size, input_size)  # 输入维度为两个 x 编码的维度

        self.transformer_fusion_model = TransformerFusionModel(input_size)
        self.transformer_fusion_model2 = TransformerFusionModel(input_size)

        # self.gate_for_gph = Use_1d_choose_F(input_size=256)
        self.gate_for_env = Use_1d_choose_F(input_size=256)

    def forward(self, x, gph, env): #(B, 256)
        B,_ = x.size()
        env_new = self.gate_for_env(x,env)

        fusion_x_gph = self.transformer_fusion_model(x, gph) #B,256
        fusion_x_env = self.transformer_fusion_model2(x, env_new) #B,256 0.45  0.67?

        residual = x + fusion_x_gph*0.001 + fusion_x_env*0.001
        return residual

class Fusion_for_2d_guide_1d_2(nn.Module):
    def __init__(self,input_dim1=128, input_dim2=256, output_dim=128):
        super(Fusion_for_2d_guide_1d_2, self).__init__()
        self.fc1 = nn.Linear(input_dim1, output_dim)
        self.fc2 = nn.Linear(input_dim2, output_dim)
        self.relu = nn.ReLU()

        self.attention_weights = nn.Linear(output_dim, 1)


    def forward(self, node_history_encoded,encoded_gph_data2): #[B,128],[B,256]
        # encoded1 = self.relu(self.fc1(node_history_encoded))
        encoded1 = node_history_encoded #[B,128]
        encoded2 = self.fc2(encoded_gph_data2) #[B,128]
        G = torch.sigmoid(self.attention_weights(encoded1))
        encoded2_change = G * encoded2  #使用F1d对F2d做一下筛选 再加回F1d

        fused_encoding = encoded1 + encoded2_change*0.01  # 这里使用相加作为示例
        return fused_encoding

class Enc_2d_for_1d_enc(nn.Module):
    def __init__(self,input_dim=256, output_dim=12, copy_t=8):
        super(Enc_2d_for_1d_enc, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.t = copy_t

    def forward(self,encoded_gph_data2): #[B,256]
        x = self.fc1(encoded_gph_data2)  #[B,12]
        x = x.unsqueeze(0) #[1,B,12]
        x_8 = x
        for i in range(self.t-1):
            x_8 = torch.cat((x, x_8), dim=0) #[8,B,12]
        return x_8.permute(1,0,2)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)  # 可以根据需要调整输出维度

    def forward(self, x):
        # x: [B, T, 4]
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [B, T, hidden_size]
        # 使用最后一个时间步的隐藏状态作为输出
        out = self.fc(h_n[-1])  # h_n: [num_layers, B, hidden_size]
        return out

class MultimodalGenerativeCVAE(object):
    def __init__(self,
                 env,
                 node_type,
                 model_registrar,
                 hyperparams,
                 device,
                 edge_types,
                 ):
        self.hyperparams = hyperparams
        self.env = env
        self.node_type = node_type
        self.model_registrar = model_registrar
        self.log_writer = None
        self.device = device
        self.edge_types = [edge_type for edge_type in edge_types if edge_type[0] is node_type]
        self.curr_iter = 0

        self.node_modules = dict()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state'][node_type]
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()]))
        if self.hyperparams['incl_robot_node']:
            self.robot_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[env.robot_type].values()])
            )
        ##########
        self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))

        edge_types_str = [DirectedEdge.get_str_from_types(*edge_type) for edge_type in self.edge_types]
        # 这是一个创建图形模型的方法。图形模型用于建模实体之间的时空关系，并在 CVAE 中用于生成轨迹样本。
        self.create_graphical_model(edge_types_str)

        dynamic_class = getattr(dynamic_module, hyperparams['dynamic'][self.node_type]['name'])
        dyn_limits = hyperparams['dynamic'][self.node_type]['limits']
        # 创建动态模型：self.dynamic，该模型用于预测实体的未来状态，根据历史轨迹和环境的动态特性。
        self.dynamic = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
                                     self.model_registrar, self.x_size, self.node_type)

        self.npl_rate = self.hyperparams['npl_rate']
        #初始化损失函数：self.NPairLoss，该损失函数用于计算轨迹预测的损失。
        self.NPairLoss = NPairLoss(self.hyperparams['tao'])

        self.fusion_model = AttentionFusionWithResidual(input_size=256,
                                                   fusion_size=256).to(self.device)
        self.fusion_model2 = AttentionFusionWithResidual2(input_size=256,
                                                        fusion_size=256).to(self.device)

        self.encoder_for_gph_500 = TimeAwareEncoderForGPH10(point_num=8, image_size=100,
                                                        hidden_dim=256).to(self.device)
        self.encoder_for_gph_200 = TimeAwareEncoderForGPH10(point_num=8, image_size=100,
                                                        hidden_dim=256).to(self.device)

        self.encoder_for_z200_z500_uv = TimeAwareEncoderForGPH10ZUV(point_num=8, image_size=100,
                                                            hidden_dim=256).to(self.device)
        # self.encoder_for_z200_z500_uv_vms = TimeAwareEncoderForGPH10ZUVVMS(point_num=8, image_size=100,
        #                                                             hidden_dim=256).to(self.device)
        self.encoder_for_z200_z500_uv_vms_vo_q_t = TimeAwareEncoderForGPH10ZUVVMSVOQT(point_num=8, image_size=100,
                                                                    hidden_dim=256).to(self.device)
        self.encoder_for_z200_z500_uv_vms_vo = TimeAwareEncoderForGPH10ZUVVMSVO(point_num=8, image_size=100,
                                                                    hidden_dim=256).to(self.device)
        self.LSTMmodel = LSTMModel(input_size=4, hidden_size=256, num_layers=1).to(self.device)

        self.encoder_for_gph_200_500 = TimeAwareEncoderForGPH10ALL(point_num=8, image_size=100,
                                                            hidden_dim=256).to(self.device)

        self.encoder_for_wind = TimeAwareEncoderForUV(num_feature=4*101*101).to(self.device)

        self.encoder_for_age = EncoderForAge().to(self.device)

        self.gph_guide_1d = Fusion_for_2d_guide_1d_2(input_dim1=128, input_dim2=256, output_dim=128
                                                   ).to(self.device)   # [B,128],[B,256]
        self.enc_2d_for_1d_enc = Enc_2d_for_1d_enc(input_dim=256, output_dim=12,copy_t=8).to(self.device)

        self.encoder_for_env_wind = TimeAwareEncoderForENV5Wind(point_num=8, hidden_dim=256).to(self.device)
        self.encoder_for_env_trajectory = TimeAwareEncoderForENV5Trajectory(point_num=8, hidden_dim=256).to(self.device)
        self.encoder_for_env_intensity = TimeAwareEncoderForENV5Intensity(point_num=8, hidden_dim=256).to(self.device)


    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):
        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule(self.node_type + '/node_history_encoder',
                           model_if_absent=nn.LSTM(input_size=self.state_length, #暂时是6 它会自己改的 那就好
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))

        self.add_submodule(self.node_type + '/gph_data_encoder',
                           model_if_absent=nn.LSTM(input_size=32,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))
        #'/gph_encoder2'
        self.add_submodule(self.node_type + '/gph_encoder2',
                           model_if_absent=nn.LSTM(input_size=2500,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))

        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule(self.node_type + '/node_future_encoder',
                           model_if_absent=nn.LSTM(input_size=self.pred_state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True))
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule(self.node_type + '/node_future_encoder/initial_h',
                           model_if_absent=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))
        self.add_submodule(self.node_type + '/node_future_encoder/initial_c',
                           model_if_absent=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))

        ############################
        #   Robot Future Encoder   #
        ############################
        # We'll create this here, but then later check if we're next to the robot.
        # Based on that, we'll factor this into the computation graph (or not).
        if self.hyperparams['incl_robot_node']:
            self.add_submodule('robot_future_encoder',
                               model_if_absent=nn.LSTM(input_size=self.robot_state_length,
                                                       hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                       bidirectional=True,
                                                       batch_first=True))
            # These are related to how you initialize states for the robot future encoder.
            self.add_submodule('robot_future_encoder/initial_h',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))
            self.add_submodule('robot_future_encoder/initial_c',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))

        if self.hyperparams['edge_encoding']:
            if self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=nn.LSTM(input_size=self.hyperparams['enc_rnn_dim_edge'],
                                                           hidden_size=self.hyperparams['enc_rnn_dim_edge_influence'],
                                                           bidirectional=True,
                                                           batch_first=True))

                self.eie_output_dims = 4 * self.hyperparams['enc_rnn_dim_edge_influence']

            elif self.hyperparams['edge_influence_combine_method'] == 'attention': #这个
                # Chose additive attention because of https://arxiv.org/pdf/1703.03906.pdf
                # We calculate an attention context vector using the encoded edges as the "encoder"
                # (that we attend _over_)
                # and the node history encoder representation as the "decoder state" (that we attend _on_).
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=AdditiveAttention(
                                       encoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_edge_influence'],
                                       decoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_history']))

                self.eie_output_dims = self.hyperparams['enc_rnn_dim_edge_influence']


        if self.hyperparams['use_map_encoding']:
            if self.node_type in self.hyperparams['map_encoder']:
                me_params = self.hyperparams['map_encoder'][self.node_type]
                self.add_submodule(self.node_type + '/map_encoder',
                                   model_if_absent=CNNMapEncoder(me_params['map_channels'],
                                                                 me_params['hidden_channels'],
                                                                 me_params['output_size'],
                                                                 me_params['masks'],
                                                                 me_params['strides'],
                                                                 me_params['patch_size']))


        self.latent = DiscreteLatent(self.hyperparams, self.device)

        # Node History Encoder
        x_size = self.hyperparams['enc_rnn_dim_history']
        if self.hyperparams['edge_encoding']:
            #              Edge Encoder
            x_size += self.eie_output_dims
        if self.hyperparams['incl_robot_node']:
            #              Future Conditional Encoder
            x_size += 4 * self.hyperparams['enc_rnn_dim_future']
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            #              Map Encoder
            x_size += self.hyperparams['map_encoder'][self.node_type]['output_size']

        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/p_z_x',
                               model_if_absent=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size

        self.add_submodule(self.node_type + '/hx_to_z',
                           model_if_absent=nn.Linear(hx_size, self.latent.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/q_z_xy',
                               #                                           Node Future Encoder
                               model_if_absent=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']

        self.add_submodule(self.node_type + '/hxy_to_z',
                           model_if_absent=nn.Linear(hxy_size, self.latent.z_dim))

        ####################
        #   Decoder LSTM   #
        ####################
        if self.hyperparams['incl_robot_node']:
            decoder_input_dims = self.pred_state_length + self.robot_state_length + z_size + x_size
        else:
            decoder_input_dims = self.pred_state_length + z_size + x_size

        self.add_submodule(self.node_type + '/decoder/state_action',
                           model_if_absent=nn.Sequential(
                               nn.Linear(self.state_length, self.pred_state_length)))

        self.add_submodule(self.node_type + '/decoder/rnn_cell',
                           model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_h',
                           model_if_absent=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))

        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_log_pis',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_mus',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_log_sigmas',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_corrs',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))

        self.x_size = x_size
        self.z_size = z_size

    def create_edge_models(self, edge_types):
        for edge_type in edge_types:
            neighbor_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[edge_type.split('->')[1]].values()]))
            if self.hyperparams['edge_state_combine_method'] == 'pointnet':
                self.add_submodule(edge_type + '/pointnet_encoder',
                                   model_if_absent=nn.Sequential(
                                       nn.Linear(self.state_length, 2 * self.state_length),
                                       nn.ReLU(),
                                       nn.Linear(2 * self.state_length, 2 * self.state_length),
                                       nn.ReLU()))

                edge_encoder_input_size = 2 * self.state_length + self.state_length

            elif self.hyperparams['edge_state_combine_method'] == 'attention':
                self.add_submodule(self.node_type + '/edge_attention_combine',
                                   model_if_absent=TemporallyBatchedAdditiveAttention(
                                       encoder_hidden_state_dim=self.state_length,
                                       decoder_hidden_state_dim=self.state_length))
                edge_encoder_input_size = self.state_length + neighbor_state_length

            else:
                edge_encoder_input_size = self.state_length + neighbor_state_length

            self.add_submodule(edge_type + '/edge_encoder',
                               model_if_absent=nn.LSTM(input_size=edge_encoder_input_size,
                                                       hidden_size=self.hyperparams['enc_rnn_dim_edge'],
                                                       batch_first=True))

    def create_graphical_model(self, edge_types):
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_node_models()

        #####################
        #   Edge Encoders   #
        #####################
        if self.hyperparams['edge_encoding']:
            self.create_edge_models(edge_types)

        for name, module in self.node_modules.items():
            module.to(self.device)

    def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
        value_scheduler = None
        rsetattr(self, name + '_scheduler', value_scheduler)
        if creation_condition:
            annealer_kws['device'] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + '_annealer', value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, value_annealer(0).clone().detach())
            dummy_optimizer = optim.Optimizer([rgetattr(self, name)], {'lr': value_annealer(0).clone().detach()})
            rsetattr(self, name + '_optimizer', dummy_optimizer)

            value_scheduler = CustomLR(dummy_optimizer,
                                       value_annealer)
            rsetattr(self, name + '_scheduler', value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(name='kl_weight',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['kl_weight_start'],
                                      'finish': self.hyperparams['kl_weight'],
                                      'center_step': self.hyperparams['kl_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams[
                                          'kl_sigmoid_divisor']
                                  })

        self.create_new_scheduler(name='latent.temp',
                                  annealer=exp_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['tau_init'],
                                      'finish': self.hyperparams['tau_final'],
                                      'rate': self.hyperparams['tau_decay_rate']
                                  })

        self.create_new_scheduler(name='latent.z_logit_clip',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['z_logit_clip_start'],
                                      'finish': self.hyperparams['z_logit_clip_final'],
                                      'center_step': self.hyperparams['z_logit_clip_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
                                          'z_logit_clip_divisor']
                                  },
                                  creation_condition=self.hyperparams['use_z_logit_clipping'])
#***********
    # 用于管理所有的步进式变化参数（annealed_vars）并自动更新它们的值。
    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars): #annealed_vars：调用它的几个函数里没发现这个参数
            if rgetattr(self, annealed_var + '_scheduler') is not None:
                # First we step the scheduler.
                with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                    warnings.simplefilter("ignore")
                    rgetattr(self, annealed_var + '_scheduler').step()

                # Then we set the annealed vars' value.
                rsetattr(self, annealed_var, rgetattr(self, annealed_var + '_optimizer').param_groups[0]['lr'])

        self.summarize_annealers()

    def summarize_annealers(self):
        if self.log_writer is not None:
            for annealed_var in self.annealed_vars:
                if rgetattr(self, annealed_var) is not None:
                    self.log_writer.add_scalar('%s/%s' % (str(self.node_type), annealed_var.replace('.', '/')),
                                               rgetattr(self, annealed_var), self.curr_iter)
#主要改这个
    def get_vms(self,u_4_hPa,v_4_hPa): # [B,T,4,100,100]
        u200 = u_4_hPa[:, :, 0:1, :, :]
        u850 = u_4_hPa[:, :, 3:4, :, :]
        v200 = v_4_hPa[:, :, 0:1, :, :]
        v850 = v_4_hPa[:, :, 3:4, :, :]
        delta_u = u200-u850
        delta_v = v200-v850
        vms = torch.sqrt(delta_u**2 + delta_v**2) #[B,T1,1,100,100]
        # print(vms.size())
        return vms

    def get_data_norm(self, gph_200_500):  # gph_200_500: array 8,4,101,101

        # if(gph_data_x_norm.shape[1]!=2):
        #     print("gph_data_x_norm data error:", gph_data_x_norm.shape[1])
        rel_state_gph = gph_200_500[-1]  # 全0
        mean_gph = rel_state_gph
        std_gph = np.zeros_like(gph_200_500[0])
        std_gph = std_gph + 3  # 原为3
        gph_data_x_norm_t = np.where(np.isnan(gph_200_500), np.array(np.nan),
                                     (gph_200_500 - mean_gph) / std_gph)  # -90，90
        return gph_data_x_norm_t
    def compute_vorticity(self, u_wind, v_wind, dx, dy):
        """
        计算涡度
        u_wind: u风 [B, T1, L, H, W]
        v_wind: v风 [B, T1, L, H, W]
        dx: x方向（经度）上的距离
        dy: y方向（纬度）上的距离
        返回: vorticity [B, T1, L, H, W]
        """
        # 计算相对于x（经度）的v风偏导数
        dv_dx = np.gradient(v_wind, dx, axis=-1)

        # 计算相对于y（纬度）的u风偏导数
        du_dy = np.gradient(u_wind, dy, axis=-2)

        # 计算涡度
        vorticity = dv_dx - du_dy
        return vorticity

    def get_vo(self,u_wind,v_wind):
        # 假设平均纬度为 30 度 (可以根据具体区域调整)
        avg_latitude = 15.0  # 台风活动常见的纬度范围

        # 固定的纬度方向距离 (dy)
        dy = 111.32 * 0.25  # 0.25度纬度分辨率

        # 计算在平均纬度下的经度方向距离 (dx)
        dx = 111.32 * 0.25 * np.cos(np.radians(avg_latitude))  # 固定纬度下的 dx

        # 提取200 hPa、500 hPa 和 850 hPa层
        u_selected = u_wind[:, :, [0, 1, 3], :, :]  # 提取200、500、850 hPa
        v_selected = v_wind[:, :, [0, 1, 3], :, :]

        # 计算200、500、850 hPa的涡度
        vorticity_selected = self.compute_vorticity(u_selected, v_selected, dx, dy)

        # vorticity_selected 的形状为 [B, T1, 3, 81, 81]，对应200、500、850 hPa的涡度
        return vorticity_selected

    def calculate_alignment_score(self, tensor1, tensor2):
        """
        计算两个 (B, T, 3, 2) 张量之间的对齐分数。

        参数:
        tensor1: 第一个张量，形状为 (B, T, 3, 2)
        tensor2: 第二个张量，形状为 (B, T, 3, 2)

        返回:
        对齐分数，形状为 (B, T, 1)
        """
        # 计算三个坐标的中心点
        center1 = torch.mean(tensor1, dim=2)  # (B, T, 2)
        center2 = torch.mean(tensor2, dim=2)  # (B, T, 2)

        # 计算中心点之间的欧几里得距离
        distances = torch.norm(center1 - center2, dim=2)  # (B, T)

        # 计算对齐分数（可以根据需要进行调整，这里我们取倒数以获得更高的分数表示更好的对齐）
        alignment_score = 1 / (1 + distances)  # (B, T)

        # 添加一个额外维度
        alignment_score = alignment_score.unsqueeze(2)  # (B, T, 1)

        return alignment_score

    def get_vo_distance(self, vo, vo_cen, vo_dis):  # [B, T1, 3, 81, 81]
        Dcen200 = 1/ (1+vo_dis[:,:,0]) #[B,3] 靠近台风眼的程度，值越大，越靠近
        Dcen850 = 1/ (1+vo_dis[:,:,2])

        cen_200 = vo_cen[:,:,0,:,:] #(B,T,3,2)
        cen_500 = vo_cen[:,:,1,:,:]
        cen_850 = vo_cen[:,:,2,:,:]

        D28 = self.calculate_alignment_score(cen_200, cen_850) #(B,T,1)
        D58 = self.calculate_alignment_score(cen_500, cen_850)

        return Dcen200, Dcen850, D28, D58

    def compute_vorticity_gradient(self, vorticity):
        """
        计算涡度梯度
        vorticity: 涡度 [B, T1, L, H, W]
        dx: x方向（经度）上的距离
        dy: y方向（纬度）上的距离
        返回: vorticity_gradient [B, T1, L, H, W, 2]
        """
        # 固定的纬度方向距离 (dy)
        dy = 111.32 * 0.25  # 0.25度纬度分辨率
        avg_latitude = 15.0  # 台风活动常见的纬度范围
        # 计算在平均纬度下的经度方向距离 (dx)
        dx = 111.32 * 0.25 * np.cos(np.radians(avg_latitude))  # 固定纬度下的 dx

        dv_d_x = np.gradient(vorticity, dx, axis=-1)  # 东西方向的涡度梯度
        dv_d_y = np.gradient(vorticity, dy, axis=-2)  # 南北方向的涡度梯度
        return dv_d_x, dv_d_y # 东西方向的涡度梯度, 南北方向的涡度梯度

    def obtain_encoded_tensors(self,
                               mode,
                               inputs,  #x
                               inputs_st,
                               labels,  #y
                               labels_st,
                               first_history_indices,
                               neighbors,
                               neighbors_edge_value,
                               robot,
                               map
                               ,env_data_x,env_data_y
                               , timestep_range_x, timestep_range_y
                               , gph_all_hpa_data_x, gph_all_hpa_data_y
                               ,gph_all_hpa_data_x_norm_t
                               , wind_data_x, wind_data_y
                               , wind_data_x_norm , wind_data_y_norm
                               , q_data_x_norm, t_data_x_norm
                               , vo_cen_x, vo_dis_x

                               ) -> (torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor):
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """
        #pdb.set_trace()
        # print("gph_all_hpa_data_x_norm_t.size:", gph_all_hpa_data_x_norm_t.size())
        x, x_r_t, y_e, y_r, y = None, None, None, None, None
        initial_dynamics = dict()

        node_history = inputs
        node_present_state = inputs[:, -1]
        node_pos = inputs[:, -1, 0:2]
        node_vel = inputs[:, -1, 2:4]

        node_inten = inputs[:, -1, 6:8]
        node_inten_v = inputs[:, -1, 8:10]

        node_history_st = inputs_st
        node_present_state_st = inputs_st[:, -1]
        node_pos_st = inputs_st[:, -1, 0:2]
        node_vel_st = inputs_st[:, -1, 2:4]

        n_s_t0 = node_present_state_st

        initial_dynamics['pos'] = node_pos #torch.Size([256, 2])
        initial_dynamics['vel'] = node_vel

        initial_dynamics['inten'] = node_inten  # torch.Size([256, 2])
        initial_dynamics['inten_v'] = node_inten_v

        self.dynamic.set_initial_condition(initial_dynamics) #

        if self.hyperparams['incl_robot_node']:
            x_r_t, y_r = robot[..., 0, :], robot[..., 1:, :]


        # torch.Size([256, 128])
        node_history_encoded = self.encode_node_history(mode,
                                                        node_history_st, #torch.Size([256, 8, 12])
                                                        first_history_indices)
        edge = torch.zeros_like(node_history_encoded)  #size:

        x_concat_list = list()

        x_concat_list.append(edge)
        x_concat_list.append(node_history_encoded)
        x_encoding = torch.cat(x_concat_list, dim=1)  #size?
        x_encoding = x_encoding.to(self.device)

        #===================================================================
        # gph_200
        # gph_200_norm = gph_all_hpa_data_x_norm_t[:,:,0:1,:,:] #[B,T,4,101,101]
        # s1,s2,s3,s4,s5=gph_all_hpa_data_x_norm_t.size()
        # gph_200_norm = gph_200_norm.reshape(s1,s2,s4,s5) #[B,T,101,101]
        # gph_200_norm = gph_200_norm.to(self.device)
        # gph_200_norm = gph_200_norm.float()
        #
        # encoded_gph_data_200 = self.encoder_for_gph_200(gph_200_norm, x_encoding)  # (B,256) best
        # encoded_gph_data_200 = encoded_gph_data_200.to(self.device)

        # ===================================================================
        # gph_500
        # gph_500_norm = gph_all_hpa_data_x_norm_t[:, :, 1:2, :, :]  # [B,T,4,101,101]
        # s1, s2, s3, s4, s5 = gph_all_hpa_data_x_norm_t.size()
        # gph_500_norm = gph_500_norm.reshape(s1, s2, s4, s5)  # [B,T,101,101]
        # gph_500_norm = gph_500_norm.to(self.device)
        # gph_500_norm = gph_500_norm.float()
        #
        # encoded_gph_data_500 = self.encoder_for_gph_500(gph_500_norm, x_encoding)  # (B,256) best
        # encoded_gph_data_500 = encoded_gph_data_500.to(self.device)

        # ===================================================================
        # gph_200_500
        gph_200_500_norm = gph_all_hpa_data_x_norm_t  # [B,T,2,100,100]
        gph_200_500_norm = gph_200_500_norm.to(self.device)
        gph_200_500_norm = gph_200_500_norm.float()

        # print("gph_200_500_norm.size:",gph_200_500_norm.size())
        # encoded_gph_data_200_500 = self.encoder_for_gph_200_500(gph_200_500_norm, x_encoding)
        '''wind_data_x: torch.Size([256, 8, 2, 4, 81, 81])'''
        u_4_hPa_norm = wind_data_x_norm[:, :, 0,] # [B, T1, 4, 81, 81]
        u_4_hPa_norm = torch.flip(u_4_hPa_norm, dims=[2]) # 气压层200-500-700-850
        v_4_hPa_norm = wind_data_x_norm[:, :, 1,]
        v_4_hPa_norm = torch.flip(v_4_hPa_norm, dims=[2])
        wind_data_norm = torch.concat((u_4_hPa_norm,v_4_hPa_norm),dim=2).to(gph_200_500_norm.device) # [B,T,8,100,100]
        z200_z500_uv = torch.concat((gph_200_500_norm,wind_data_norm),dim=2) # [B,T,10,100,100]

        # add-ERA5-7-200gph-uv2 开始
        # encoded_z200_z500_uv = self.encoder_for_z200_z500_uv(z200_z500_uv,x_encoding)
        # add-ERA5-7-200gph-uv2 结束

        u_4_hPa = wind_data_x[:, :, 0, ]  # [B, T1, 4, 81, 81]
        u_4_hPa = torch.flip(u_4_hPa, dims=[2])  # 气压层200-500-700-850
        v_4_hPa = wind_data_x[:, :, 1, ]
        v_4_hPa = torch.flip(v_4_hPa, dims=[2])
        # vms = self.get_vms(u_4_hPa,v_4_hPa)  #[B,T1,1,100,100]
        # vms_norm = torch.tensor(self.get_data_norm(vms)).to(gph_200_500_norm.device)
        # vms_norm = vms_norm*0.02

        vo = torch.tensor(self.get_vo(u_4_hPa,v_4_hPa)) #[B, T1, 3, 81, 81]，对应200、500、850 hPa的涡度
        vo = vo*5 # 正负值都扩大一些
        vo = vo.to(gph_200_500_norm.device)

        # Dcen = vo_dis
        '''(B,T) (B,T) (B,T,1) (B,T,1)  '''
        Dcen200, Dcen850, D28, D58 = self.get_vo_distance(vo, vo_cen_x, vo_dis_x)
        D = torch.concat((Dcen200.unsqueeze(2), Dcen850.unsqueeze(2), D28, D58),dim=2) #(B,T,4)
        D = D.float().to(gph_200_500_norm.device)
        D_encode = self.LSTMmodel(D) # (B,16)


        # # 东西方向的涡度梯度, 南北方向的涡度梯度
        # vo_grad_x, vo_grad_y = self.compute_vorticity_gradient(vo)

        # vo_grad_x = torch.tensor(vo_grad_x).to(vo.device) # [B,t1,3,81,81]
        # vo_grad_y = torch.tensor(vo_grad_y).to(vo.device) # [B,t1,3,81,81]

        # [B,T,20,81,81]
        # z200_z500_uv = torch.concat((z200_z500_uv),dim=2)

        # encoded_z200_z500_uv_vms = self.encoder_for_z200_z500_uv_vms(z200_z500_uv_vms,x_encoding)
        encoded_z200_z500_uv = self.encoder_for_z200_z500_uv(z200_z500_uv, x_encoding)


        # ===================================================================
        # wind_u_v

        # dif_v,dif_d = self.encoder_for_wind(, x_encoding)

        encoded_env_data = []
        encoded_env_data_trajectory = self.encoder_for_env_trajectory(env_data_x, x_encoding)  # [B,32]
        encoded_env_data_intensity = self.encoder_for_env_intensity(env_data_x, x_encoding)  # [B,32]
        encoded_env_data_wind = self.encoder_for_env_wind(env_data_x, x_encoding)  # [B,80]
        encoded_env_data.append(encoded_env_data_trajectory)
        encoded_env_data.append(encoded_env_data_intensity)
        encoded_env_data.append(encoded_env_data_wind)


        timestep_range_x = timestep_range_x.to(self.device)
        timestep_range_y = timestep_range_y.to(self.device)
        encoded_age = self.encoder_for_age(timestep_range_x) #[B,4]

        # env_data_x_t = env_data_x.float()#@@@应该有问题
        # encoded_env_data = self.encoder_for_env(env_data_x, x_encoding) #[B,256]
        # encoded_env_data = self.encoder_for_env2(env_data_x, x_encoding,timestep_range_x,timestep_range_y)  # [B,256]

        node_present = node_present_state_st  # [bs, state_dim]

        if mode != ModeKeys.PREDICT:
            y = labels_st


        if self.hyperparams['edge_encoding']:
            node_edges_encoded = list()
            for edge_type in self.edge_types:
                # Encode edges for given edge type
                encoded_edges_type = self.encode_edge(mode,
                                                      node_history, #torch.Size([256, 8, 6])
                                                      node_history_st,
                                                      edge_type,
                                                      neighbors[edge_type],
                                                      neighbors_edge_value[edge_type],
                                                      first_history_indices)
                node_edges_encoded.append(encoded_edges_type)  # List of [bs/nbs, enc_rnn_dim]


        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            if self.log_writer and (self.curr_iter + 1) % 500 == 0:
                map_clone = map.clone()
                map_patch = self.hyperparams['map_encoder'][self.node_type]['patch_size']
                map_clone[:, :, map_patch[1]-5:map_patch[1]+5, map_patch[0]-5:map_patch[0]+5] = 1.
                self.log_writer.add_images(f"{self.node_type}/cropped_maps", map_clone,
                                           self.curr_iter, dataformats='NCWH')

            encoded_map = self.node_modules[self.node_type + '/map_encoder'](map * 2. - 1., (mode == ModeKeys.TRAIN))
            do = self.hyperparams['map_encoder'][self.node_type]['dropout']
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))

        '''
        encoded_gph_data : torch.randn(256, 128)
        x_encoding : torch.randn(256, 256)
        '''

        # fused_x = self.fusion_model(x_encoding, encoded_gph_data2)  # 均值 大三倍 方差大四倍

        # fused_x = self.fusion_model(x_encoding, encoded_gph_data_200_500)
        # fused_x = self.fusion_model(x_encoding, encoded_z200_z500_uv)
        # fused_x = self.fusion_model(x_encoding, encoded_z200_z500_uv_vms)

        fused_x = self.fusion_model(x_encoding, encoded_z200_z500_uv, D_encode)

        x_gph = fused_x
        return x_gph,encoded_age,encoded_env_data #[B,256] [B,4]

# mode:ModeKeys.TRAIN   node_hist:torch.Size([256, 8, 12]) torch.Size([256])每个节点在场景中首次有数据的时间步（索引）
    def encode_node_history(self, mode, node_hist, first_history_indices):
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[self.node_type + '/node_history_encoder'],
                                                      original_seqs=node_hist,
                                                      lower_indices=first_history_indices)
        # torch.Size([255, 8, 128])
        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)
        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_node_history2(self, mode, node_hist, encoded_gph_data2, first_history_indices):
        #简单处理encoded_gph_data2:[B,256]
        encoded_gph_data2_copy8 = self.enc_2d_for_1d_enc(encoded_gph_data2)

        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        outputs, _ = run_lstm_on_variable_length_seqs2(self.node_modules[self.node_type + '/node_history_encoder'],
                                                      original_seqs=node_hist,
                                                       encoded_gph_data = encoded_gph_data2_copy8,
                                                      lower_indices=first_history_indices)
        # torch.Size([255, 8, 128])
        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)
        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_gph_data(self, mode, node_hist, first_history_indices):
        #[Batchsize,point num,100,100]---[Batchsize,128]

        # 定义编码器
        encoder100to32 = nn.Sequential(
            nn.Flatten(),  # 将输入展平成一维向量
            nn.Linear(100 * 100, 256),  # 编码成大小为 [8, 12] 的表示
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.Flatten(),
            nn.Linear(64, 32)
        )
        node_hist = node_hist.to(torch.float32)

        # 把node_hist里nan的值变成均值 这样其实不好 暂时先这样
        # node_hist_copy = node_hist.clone()
        # nan_indices = torch.isnan(node_hist_copy)
        # mean_value = torch.mean(node_hist_copy[~nan_indices])
        # node_hist_copy[nan_indices] = mean_value
        # node_hist = node_hist_copy

        node_hist = node_hist.to(self.device)
        encoder100to32 = encoder100to32.to(self.device)

        # 对每个样本逐个进行编码
        encoded_samples = []
        for sample in node_hist:
            encoded_sample = encoder100to32(sample)
            encoded_samples.append(encoded_sample)

        # 将编码后的样本堆叠成批次
        node_hist = torch.stack(encoded_samples) #对一个torch.Size([171, 8, 32])做标准化处理，均值为0，方差为1

        # mean = node_hist.mean()
        # std = node_hist.std()
        # mean = mean.to(self.device)
        # std = std.to(self.device)

        # 标准化处理
        # node_hist = (node_hist - mean) / std


        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[self.node_type + '/gph_data_encoder'],
                                                      original_seqs=node_hist, # node_hist:shape torch.Size([171, 8, 32])
                                                      lower_indices=first_history_indices)
        # torch.Size([255, 8, 128])
        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)
        out = outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

        # encoder128to32 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128, 64),
        #     nn.Flatten(),
        #     nn.Linear(64, 32)
        # )
        # encoder128to32 = encoder128to32.to(self.device)
        # outs = encoder128to32(out)

        # tensor_128_256 = torch.cat((out, out), dim=1)

        return out #这里输出nan了 所以检查下输入 真的有nan 别人也有 为啥人家不报错？ 仔细看看

    # 这段代码是一个编码边信息的函数，用于将边的历史状态和邻居的状态编码成一个边的特征表示。下面是函数的主要步骤：
    # 1. 根据给定的邻居信息和边类型，获取每个邻居的状态。如果邻居为空，则创建一个全零张量作为邻居状态。
    # 2. 根据设定的边状态组合方法，将邻居状态进行组合。可以选择的组合方法有"sum"（求和）、"max"（取最大值）和"mean"（取平均值）。
    # 3. 将组合后的邻居状态与当前边的历史状态连接起来，形成联合历史。
    # 4. 使用LSTM模型对联合历史进行处理，得到输出序列。
    # 5. 对输出序列进行dropout操作。
    # 6. 根据每个序列的最后一个索引位置，提取相应的输出作为最终的边特征表示。
    # 7. 如果设定了动态边（dynamic_edges），则将最终的边特征表示与边的掩码（edge_mask）相乘，得到最终的编码边特征表示。
    # 总之，这个函数通过将边的历史状态和邻居的状态编码成一个边的特征表示，用于在图神经网络中对边进行建模和处理。
    # 具体的编码方式和组合方法可以根据具体任务和需求进行调整和选择。
    def encode_edge(self,
                    mode,
                    node_history,
                    node_history_st,
                    edge_type,
                    neighbors,
                    neighbors_edge_value,
                    first_history_indices):

        max_hl = self.hyperparams['maximum_history_length']

        edge_states_list = list()  # list of [#of neighbors, max_ht, state_dim]
        for i, neighbor_states in enumerate(neighbors):  # Get neighbors for timestep in batch
            if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
                neighbor_state_length = int(
                    np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
                )
                edge_states_list.append(torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device))
            else:
                edge_states_list.append(torch.stack(neighbor_states, dim=0).to(self.device))

        if self.hyperparams['edge_state_combine_method'] == 'sum':
            # Used in Structural-RNN to combine edges as well.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(torch.clamp(torch.sum(edge_value.to(self.device),
                                                                           dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        elif self.hyperparams['edge_state_combine_method'] == 'max':
            # Used in NLP, e.g. max over word embeddings in a sentence.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.max(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(torch.clamp(torch.max(edge_value.to(self.device),
                                                                           dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        elif self.hyperparams['edge_state_combine_method'] == 'mean':
            # Used in NLP, e.g. mean over word embeddings in a sentence.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.mean(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(torch.clamp(torch.mean(edge_value.to(self.device),
                                                                            dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        outputs, _ = run_lstm_on_variable_length_seqs(
            self.node_modules[DirectedEdge.get_str_from_types(*edge_type) + '/edge_encoder'],
            original_seqs=joint_history,
            lower_indices=first_history_indices
        )

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)
        ret = outputs[torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence]
        if self.hyperparams['dynamic_edges'] == 'yes':
            return ret * combined_edge_masks
        else:
            return ret

    def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules[self.node_type + '/node_future_encoder/initial_h']
        initial_c_model = self.node_modules[self.node_type + '/node_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules[self.node_type + '/node_future_encoder'](node_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

    def encode_robot_future(self, mode, robot_present, robot_future) -> torch.Tensor:
        """
        Encodes the robot future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param robot_present: Current state of the robot. [bs, state]
        :param robot_future: Future states of the robot. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules['robot_future_encoder/initial_h']
        initial_c_model = self.node_modules['robot_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(robot_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(robot_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules['robot_future_encoder'](robot_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

    def q_z_xy(self, mode, x, y_e) -> torch.Tensor:
        r"""
        .. math:: q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :return: Latent distribution of the CVAE.
        """
        xy = torch.cat([x, y_e], dim=1)

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            dense = self.node_modules[self.node_type + '/q_z_xy']
            h = F.dropout(F.relu(dense(xy)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = xy

        to_latent = self.node_modules[self.node_type + '/hxy_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def p_z_x(self, mode, x):
        r"""
        .. math:: p_\theta(z \mid \mathbf{x}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :return: Latent distribution of the CVAE.
        """
        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            dense = self.node_modules[self.node_type + '/p_z_x']
            h = F.dropout(F.relu(dense(x)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = x

        to_latent = self.node_modules[self.node_type + '/hx_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def project_to_GMM_params(self, tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        log_pis = self.node_modules[self.node_type + '/decoder/proj_to_GMM_log_pis'](tensor)
        mus = self.node_modules[self.node_type + '/decoder/proj_to_GMM_mus'](tensor)
        log_sigmas = self.node_modules[self.node_type + '/decoder/proj_to_GMM_log_sigmas'](tensor)
        corrs = torch.tanh(self.node_modules[self.node_type + '/decoder/proj_to_GMM_corrs'](tensor))
        return log_pis, mus, log_sigmas, corrs

    def p_y_xz(self, mode, x, x_nr_t, y_r, n_s_t0, z_stacked, prediction_horizon,
               num_samples, num_components=1, gmm_mode=False):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.node_type + '/decoder/state_action'](n_s_t0)

        state = initial_state
        if self.hyperparams['incl_robot_node']:
            input_ = torch.cat([zx,
                                a_0.repeat(num_samples*num_components, 1),
                                x_nr_t.repeat(num_samples*num_components, 1)], dim=1)
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples*num_components, 1)], dim=1)

        for j in range(ph):
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t.reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
            corrs.append(
                corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, num_components))

            if self.hyperparams['incl_robot_node']:
                dec_inputs = [zx, a_t, y_r[:, j].repeat(num_samples * num_components, 1)]
            else:
                dec_inputs = [zx, a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)

        a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                       torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(corrs, [num_samples, -1, ph, num_components]))

        if self.hyperparams['dynamic'][self.node_type]['distribution']:
            y_dist = self.dynamic.integrate_distribution(a_dist, x)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.traj_sample()
            sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, sampled_future
        else:
            return y_dist

    # 总的来说，encoder方法是CVAE模型中负责编码输入数据的部分，
    # 它通过计算后验概率分布和先验概率分布，并进行采样，
    # 将输入数据映射到潜在空间中的向量表示。
    # 同时，通过计算KL散度，可以用于优化模型的训练过程。
    def encoder(self, mode, x, y_e, num_samples=None):
        """
        Encoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :param num_samples: Number of samples from the latent space during Prediction.
        :return: tuple(z, kl_obj)
            WHERE
            - z: Samples from the latent space.
            - kl_obj: KL Divergenze between q and p
        """
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams['k']
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams['k_eval']
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        self.latent.q_dist = self.q_z_xy(mode, x, y_e)
        self.latent.p_dist = self.p_z_x(mode, x)

        z = self.latent.sample_q(sample_ct, mode)

        if mode == ModeKeys.TRAIN:
            kl_obj = self.latent.kl_q_p(self.log_writer, '%s' % str(self.node_type), self.curr_iter)
            if self.log_writer is not None:
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'kl'), kl_obj, self.curr_iter)
        else:
            kl_obj = None

        return z, kl_obj

    def decoder(self, mode, x, x_nr_t, y, y_r, n_s_t0, z, labels, prediction_horizon, num_samples):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        y_dist = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                             prediction_horizon, num_samples, num_components=num_components)
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        if self.hyperparams['log_histograms'] and self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_yt_xz'), log_p_yt_xz, self.curr_iter)

        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        return log_p_y_xz

#********
    def get_latent(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot,
                   map,
                   prediction_horizon
                   , env_data_x, env_data_y
                   , timestep_range_x, timestep_range_y
                   , gph_all_hpa_data_x, gph_all_hpa_data_y
                   , gph_all_hpa_data_x_norm_t
                   , wind_data_x, wind_data_y
                   ,wind_data_x_norm,wind_data_y_norm
                   ,q_data_x_norm,t_data_x_norm
                   ,vo_cen_x, vo_dis_x
                   ) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        x_gph,encoded_age,encoded_env_data = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=labels,
                                                                     labels_st=labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map
                                                                     , env_data_x=env_data_x, env_data_y=env_data_y
                                                                     , timestep_range_x=timestep_range_x,
                                                                     timestep_range_y=timestep_range_y
                                                                         , gph_all_hpa_data_x=gph_all_hpa_data_x,
                                                                         gph_all_hpa_data_y=gph_all_hpa_data_y
                                                                         ,gph_all_hpa_data_x_norm_t=gph_all_hpa_data_x_norm_t
                                                                         , wind_data_x=wind_data_x
                                                                         , wind_data_y=wind_data_y
                                                                         , wind_data_x_norm = wind_data_x_norm
                                                                         , wind_data_y_norm = wind_data_y_norm
                                                                         , q_data_x_norm = q_data_x_norm
                                                                         , t_data_x_norm=t_data_x_norm
                                                                         , vo_cen_x = vo_cen_x
                                                                         , vo_dis_x = vo_dis_x)
        return x_gph,encoded_age,encoded_env_data #[B,256] [B,4]

#********
    def train_loss(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot,
                   map,
                   prediction_horizon) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=labels,
                                                                     labels_st=labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map)
        # 得到潜在向量z和KL散度kl。
        z, kl = self.encoder(mode, x, y_e)

        # 进行解码，得到对数概率log_p_y_xz。
        log_p_y_xz = self.decoder(mode, x, x_nr_t, y, y_r, n_s_t0, z,
                                  labels,  # Loss is calculated on unstandardized label
                                  prediction_horizon,
                                  self.hyperparams['k'])

        eye_mat = torch.eye(self.latent.p_dist.event_shape[-1], device=self.device)
        argmax_idxs = torch.argmax(self.latent.p_dist.probs, dim=2)
        x_target_onehot = torch.squeeze(eye_mat[argmax_idxs])
        x_target = torch.argmax(x_target_onehot, -1)

        # DisDis 计算归一化对比度损失
        nploss = self.NPairLoss(x, x_target)

        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)

        mutual_inf_q = mutual_inf_mc(self.latent.q_dist)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist)

        ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
        loss = -ELBO + self.npl_rate * nploss

        if self.hyperparams['log_histograms'] and self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_y_xz'),
                                          log_p_y_xz_mean,
                                          self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_q'),
                                       mutual_inf_q,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_p'),
                                       mutual_inf_p,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'log_likelihood'),
                                       log_likelihood,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
                                       loss,
                                       self.curr_iter)
            if self.hyperparams['log_histograms']:
                self.latent.summarize_for_tensorboard(self.log_writer, str(self.node_type), self.curr_iter)
        return loss
#**********
    def eval_loss(self,
                  inputs,
                  inputs_st,
                  first_history_indices,
                  labels,
                  labels_st,
                  neighbors,
                  neighbors_edge_value,
                  robot,
                  map,
                  prediction_horizon) -> torch.Tensor:
        """
        Calculates the evaluation loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL

        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=labels,
                                                                     labels_st=labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map)

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        ### Importance sampled NLL estimate
        z, _ = self.encoder(mode, x, y_e)  # [k_eval, nbs, N*K]
        z = self.latent.sample_p(1, mode, full_dist=True)
        y_dist, _ = self.p_y_xz(ModeKeys.PREDICT, x, x_nr_t, y_r, n_s_t0, z,
                                prediction_horizon, num_samples=1, num_components=num_components)
        # We use unstandardized labels to compute the loss
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)
        nll = -log_likelihood

        return nll

    def predict(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon,
                num_samples,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                pcmd=False):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :param pcmd: If True: Sort the outputs for pcmd.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, x_nr_t, _, y_r, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                   inputs=inputs,
                                                                   inputs_st=inputs_st,
                                                                   labels=None,
                                                                   labels_st=None,
                                                                   first_history_indices=first_history_indices,
                                                                   neighbors=neighbors,
                                                                   neighbors_edge_value=neighbors_edge_value,
                                                                   robot=robot,
                                                                   map=map)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        _, our_sampled_future = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode)

        if pcmd:
            _, indices = torch.sort(self.latent.p_dist.probs, dim=2, descending=True)
            sorted_future = torch.zeros_like(our_sampled_future)
            for i in range(inputs.shape[0]):
                sorted_future[:, i] = our_sampled_future[:, i][indices[i, 0]]
            return sorted_future
        else:
            return our_sampled_future
