import torch
import torch.nn as nn

class GetWindEncByTransformer(nn.Module):
    def __init__(self, num_features, num_layers=4, nhead=8, d_model=512):
        super(GetWindEncByTransformer, self).__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Position encoding for time steps
        self.pos_encoder = nn.Embedding(5000, self.num_features)

        self.fc1 = nn.Linear(num_features, d_model)
        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Linear layer to map back to original feature size
        self.fc = nn.Linear(d_model, num_features)

    def forward(self, x):
        # x shape: [B, T, num_features], reshape to [T, B, num_features] for transformer
        B, T, _ = x.shape

        # Add positional encoding for time steps
        time_indices = torch.arange(T).unsqueeze(0).repeat(B, 1).to(x.device) #[B,T]
        x = x + self.pos_encoder(time_indices) #[8,8,512]
        x = self.fc1(x)

        # Apply transformer encoding
        encoded = self.transformer_encoder(x.permute(1, 0, 2))  # [T, B, num_features]

        # Pass through linear layer to project back to feature space
        output = self.fc(encoded).permute(1, 0, 2)  # [B, T, num_features]

        return output

class WindTransformerVideoPredictor(nn.Module):
    def __init__(self, num_features, d_model=512, num_layers=4, nhead=8, T1=8, T2=4):
        super(WindTransformerVideoPredictor, self).__init__()
        self.T1 = T1
        self.T2 = T2
        self.num_features = num_features
        self.d_model = d_model

        # Transformer Encoder-Decoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Linear layers to project input and output to d_model dimension
        self.input_fc = nn.Linear(num_features, d_model)
        self.output_fc = nn.Linear(d_model, num_features)

    def forward(self, x):
        # x shape: [B, T1, 4, 101, 101]
        B, T1, C, H, W = x.shape
        x = x.view(B, T1, -1)  # Flatten to [B, T1, num_features]

        # Project input to d_model
        x = self.input_fc(x)  # [B, T1, d_model]

        # Transpose for Transformer: [T1, B, d_model]
        x = x.permute(1, 0, 2)

        # Encoder step: Encode history T1 steps
        memory = self.encoder(x)  # [T1, B, d_model]

        # Prepare decoder input: a tensor of zeros with shape [T2, B, d_model]
        decoder_input = torch.zeros(self.T2, B, self.d_model).to(x.device)

        # Decoder step: Decode future T2 steps
        decoded_output = self.decoder(decoder_input, memory)  # [T2, B, d_model]

        # Project decoded output back to original feature space
        decoded_output = self.output_fc(decoded_output)  # [T2, B, num_features]

        # Reshape back to [B, T2, 4, 101, 101]
        decoded_output = decoded_output.permute(1, 0, 2).view(B, self.T2, C, H, W)

        return decoded_output


if __name__ == '__main__':
    # 初始化示例数据  # 批次大小、历史时刻数、预测时刻数
    B, T1, T2, C, H, W = 8, 8, 4, 4, 101, 101
    num_features = C * H * W  # 特征维度
    history_data = torch.rand(B, T1, num_features)  # 历史8个时刻的数据
    future_data = torch.rand(B, T2, num_features)

    # 创建Transformer模型实例
    encode_his_to_his = GetWindEncByTransformer(num_features)

    # 第一步：历史8个时刻的编码
    history_encoded = encode_his_to_his(history_data)  # 输出形状为 [B, T1, num_features]
    history_encoded = history_encoded.reshape(B, T1, C, H, W)

    future_encoded_gt = encode_his_to_his(future_data)
    future_encoded_gt = future_encoded_gt.reshape(B, T2, C, H, W)

    get_future_from_his = WindTransformerVideoPredictor(num_features=num_features, T1=T1, T2=T2)

    future_encoded = get_future_from_his(history_encoded)  # [B,t2,4,101,101]

    # 第三步：计算差值
    diff_8_to_1 = history_encoded[:, -1, :] - future_encoded[:, 0, :]
    future_diffs = future_encoded[:, 1:, :] - future_encoded[:, :-1, :]
    future_diffs = torch.concat((diff_8_to_1.unsqueeze(1), future_diffs), dim=1)
    print("将来各个时刻之间的差值:", future_diffs.shape)

    # 预测特征的误差
    loss_wind_pred = future_encoded_gt - future_encoded


