import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from utils.Adapter_to_premodel import CustomGPT2Model
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class Adapter(nn.Module):
    def __init__(self, in_feat, hid_dim, dropout=0.1,skip=True):
        super().__init__()
        self.D_fc1 = nn.Linear(in_feat, hid_dim)
        self.D_fc2 = nn.Linear(hid_dim, in_feat)

        self.act = nn.GELU()
        self.skip = skip
        self.drop = nn.Dropout(dropout)


    def forward(self, x):
        #X[1024,6,1280]
        x_intermediate = self.D_fc1(x)
        x_intermediate = self.act(x_intermediate)
        x_intermediate = self.D_fc2(x_intermediate)
        x_output = self.drop(x_intermediate)
        if self.skip:
            x_output = x_output + x

        return x_output
class SpectModule(nn.Module):
    def __init__(self, freq_len, adapter_len,dropout):
        super().__init__()
        self.adapter_len = adapter_len
        self.freq_len = freq_len

        self.drop = nn.Dropout(dropout)

        # 更好的初始化
        self.weight_r = nn.Parameter(torch.rand(freq_len, adapter_len // 2))
        self.weight_i = nn.Parameter(torch.rand(freq_len, adapter_len // 2))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight_r)
        nn.init.xavier_uniform_(self.weight_i)
    def forward(self, x):
        B, M, N, P = x.shape
        x = rearrange(x, 'b m n p -> b m p n')
        x_ft = torch.fft.rfft(x, dim=-1)
        x_real = x_ft.real
        x_imag = x_ft.imag


        x_real = torch.einsum("bmpf, fd->bmpd", x_real, self.weight_r)
        x_imag = torch.einsum("bmpf, fd->bmpd", x_imag, self.weight_i)

        x_ft = torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1))

        if self.adapter_len == N:
            res = torch.fft.irfft(x_ft, dim=-1, n=N)
        else:
            res = torch.fft.irfft(x_ft, dim=-1, n=self.adapter_len)


        res = rearrange(res, 'b m p n -> b m n p')

        return self.drop(res)


class SpectBlock(nn.Module):
    def __init__(self, in_feat, freq_len,  adapter_len=6,dropout_1=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(in_feat)
        self.ln_2 = nn.LayerNorm(in_feat)
        self.attn = SpectModule(freq_len // 2 + 1, adapter_len,dropout_1)

    def forward(self, x):
        x = self.attn(self.ln_1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class FFT_adapter(nn.Module):
    def __init__(self, n_layer, in_feat, seq_len,dropout_1=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([SpectBlock(in_feat, seq_len,dropout_1) for i in range(n_layer)])

    def forward(self, x):
        res_list = []
        for i, block in enumerate(self.blocks):
            res_list.append(block(x))
        return res_list


#————————————————————————————————————————————————————————————————————————————————————————————————
class Ex_en_to_gpt(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu


    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])



        x_glb_attn = torch.reshape(x_glb_attn,(x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)# [1024*1, 512] → [1024, 1, 512]
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)



        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.pred_len = configs.pred_len
        self.d_ff = configs.d_ff
        self.gpt_layers = configs.gpt_layers
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        if self.stride > 1 or self.patch_size > 1:
            self.patch_num += 1
        self.use_fft_adapter=configs.use_fft_adapter
        self.gpt_enc_in = configs.gpt_enc_in
        self.en_pred_len = configs.en_pred_len
        self.n_vars = configs.n_vars
        self.en_model = configs.en_model
        self.en_patch_len = configs.en_patch_len
        self.en_heads = configs.en_heads
        self.en_dff = configs.en_dff
        self.en_patch_num = int(configs.seq_len // self.en_patch_len)
        self.head_nf = self.en_model * (self.en_patch_num + 1)

        self.en_embedding = EnEmbedding(self.n_vars,self.en_model, self.en_patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(self.seq_len, self.en_model, configs.embed, configs.freq,configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,output_attention=False),self.en_model,self.en_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,output_attention=False),self.en_model, self.en_heads),
                    self.en_model,
                    self.en_dff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
            ],
            norm_layer=torch.nn.LayerNorm(self.en_model)
        )
        self.ex_en_to_gpt =Ex_en_to_gpt(configs.enc_in, self.head_nf, self.en_pred_len,head_dropout=configs.dropout)


        gpt2_path = '/tmp/zfh_1/Long-term_Forecasting/GPT2-lage/AI-ModelScope/gpt2-large/'
        config = GPT2Config.from_pretrained(gpt2_path)
    

        self.gpt2 = CustomGPT2Model(config)
    

        use_pretrained = getattr(configs, 'use_pretrained_gpt2', True)
    
        if use_pretrained:
            print("\n" + "=" * 80)
            print("✓ 加载预训练GPT-2权重")
            print("=" * 80)
            pretrained_model = GPT2Model.from_pretrained(
                gpt2_path,
                output_attentions=True,
                output_hidden_states=True
            )
            self.gpt2.load_state_dict(pretrained_model.state_dict(), strict=False)
            del pretrained_model
        else:
            print("\n" + "=" * 80)
            print("✗ 使用随机初始化GPT-2 (消融实验)")
            print("=" * 80)
            self._initialize_gpt2_randomly()
    

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]


        for i in range(configs.gpt_layers):
            self.gpt2.h[i].scale = configs.scale
            if hasattr(configs, 'T_type') and configs.T_type == 1:
                self.gpt2.h[i].T_adapter = Adapter(configs.d_model, configs.adapter_dim, configs.dropout,skip=False)
                self.gpt2.h[i].T_adapter_gate = torch.nn.Parameter(torch.zeros(1, self.patch_num, 1))

            if hasattr(configs, 'C_type') and configs.C_type == 1:
                self.gpt2.h[i].C_adapter = Adapter(configs.d_model,configs.adapter_dim,configs.dropout,skip=False)
                self.gpt2.h[i].C_num = self.gpt_enc_in
                self.gpt2.h[i].C_adapter_gate = torch.nn.Parameter(torch.zeros(1, self.gpt_enc_in, 1))




        spect_layers =configs.gpt_layers

        self.fft_adapter = FFT_adapter(spect_layers, self.gpt_enc_in, self.patch_num, configs.dropout)
        self.adapter_in_layer = nn.ModuleList([nn.Linear(configs.patch_len, configs.d_model) for i in range(spect_layers)])


        print(f"Patch num: {self.patch_num}")
        print(f"Spect adapter layers: {spect_layers}")
        print(f"GPT layers: {configs.gpt_layers}")

        self.in_layer = nn.Linear(configs.patch_len, configs.d_model)
        self.proj_layer = nn.Linear( configs.d_model, self.d_ff)
        self.out_conv = nn.Conv1d(in_channels=self.patch_num, out_channels=configs.pred_len, kernel_size=1)




        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'adapter' in name or 'adapter_gate' in name:
                param.requires_grad = True
            elif 'ln' in name:
                param.requires_grad = True
            elif 'wpe' in name:
                param.requires_grad = False
            else:
                param.requires_grad = False

        self.param_stats = self.calculate_model_parameters()



    def calculate_model_parameters(self):
        """计算并打印模型各部分的参数量"""

        print("\n" + "=" * 80)
        print("模型参数统计")
        print("=" * 80)

        # 1. 嵌入层参数
        print("\n【嵌入层】")
        en_embedding_params = sum(p.numel() for p in self.en_embedding.parameters())
        ex_embedding_params = sum(p.numel() for p in self.ex_embedding.parameters())
        print(f"  EnEmbedding:        {en_embedding_params:>12,} ({en_embedding_params / 1e6:.2f}M)")
        print(f"  ExEmbedding:        {ex_embedding_params:>12,} ({ex_embedding_params / 1e6:.2f}M)")

        # 2. 编码器参数
        print("\n【编码器】")
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"  总参数:             {encoder_params:>12,} ({encoder_params / 1e6:.2f}M)")
        print(f"  可训练参数:         {encoder_trainable:>12,} ({encoder_trainable / 1e6:.2f}M)")

        # 3. 转换层参数
        print("\n【转换层】")
        ex_en_to_gpt_params = sum(p.numel() for p in self.ex_en_to_gpt.parameters())
        print(f"  Ex_en_to_gpt:       {ex_en_to_gpt_params:>12,} ({ex_en_to_gpt_params / 1e6:.2f}M)")

        # 4. GPT-2参数详细统计
        print("\n【GPT-2模型】")
        gpt2_total = sum(p.numel() for p in self.gpt2.parameters())
        gpt2_trainable = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        gpt2_frozen = gpt2_total - gpt2_trainable


        adapter_params = 0
        adapter_gate_params = 0
        ln_params = 0

        for name, param in self.gpt2.named_parameters():
            if 'adapter_gate' in name:
                adapter_gate_params += param.numel()
            elif 'adapter' in name:
                adapter_params += param.numel()
            elif 'ln' in name and param.requires_grad:
                ln_params += param.numel()

        print(f"  GPT-2总参数:        {gpt2_total:>12,} ({gpt2_total / 1e6:.2f}M)")
        print(f"  └─ 冻结参数:        {gpt2_frozen:>12,} ({gpt2_frozen / 1e6:.2f}M)")
        print(f"  └─ 可训练参数:      {gpt2_trainable:>12,} ({gpt2_trainable / 1e6:.2f}M)")
        print(f"     ├─ T/C Adapter:  {adapter_params:>12,} ({adapter_params / 1e6:.2f}M)")
        print(f"     ├─ Adapter Gate: {adapter_gate_params:>12,} ({adapter_gate_params / 1e6:.2f}M)")
        print(f"     └─ LayerNorm:    {ln_params:>12,} ({ln_params / 1e6:.2f}M)")

        print("\n【FFT适配器】")
        fft_adapter_params = sum(p.numel() for p in self.fft_adapter.parameters())
        adapter_in_layer_params = sum(p.numel() for p in self.adapter_in_layer.parameters())
        print(f"  FFT Adapter:        {fft_adapter_params:>12,} ({fft_adapter_params / 1e6:.2f}M)")
        print(f"  Adapter In Layer:   {adapter_in_layer_params:>12,} ({adapter_in_layer_params / 1e6:.2f}M)")

        # 6. 输入输出层参数
        print("\n【输入输出层】")
        in_layer_params = sum(p.numel() for p in self.in_layer.parameters())
        proj_layer_params = sum(p.numel() for p in self.proj_layer.parameters())
        out_conv_params = sum(p.numel() for p in self.out_conv.parameters())
        print(f"  In Layer:           {in_layer_params:>12,} ({in_layer_params / 1e6:.2f}M)")
        print(f"  Proj Layer:         {proj_layer_params:>12,} ({proj_layer_params / 1e6:.2f}M)")
        print(f"  Out Conv:           {out_conv_params:>12,} ({out_conv_params / 1e6:.2f}M)")

        # 7. 总计
        print("\n" + "=" * 80)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"【总参数量】")
        print(f"  总参数:             {total_params:>12,} ({total_params / 1e6:.2f}M)")
        print(f"  可训练参数:         {trainable_params:>12,} ({trainable_params / 1e6:.2f}M)")
        print(f"  冻结参数:           {frozen_params:>12,} ({frozen_params / 1e6:.2f}M)")
        print(f"  可训练比例:         {trainable_params / total_params * 100:>11.2f}%")
        print("=" * 80 + "\n")
        with open(self.configs.cost_path, 'a',
                  newline='', encoding='utf-8') as file:
            file.write(f"{self.configs.csv_name}\n")
            file.write(f"可训练参数:         {trainable_params:>12,} ({trainable_params / 1e6:.2f}M)\n")
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_ratio': trainable_params / total_params
        }
    def _initialize_gpt2_randomly(self):
        """随机初始化GPT-2权重"""
        for name, module in self.gpt2.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        print("GPT-2随机初始化完成!")

    def forward(self, x_enc, x_mark_enc=None):

        a = x_enc.clone()
        ex_x=x_enc[:,:,self.n_vars:]
        en_embed, n_vars = self.en_embedding(x_enc[:, :, 0:self.n_vars].permute(0, 2, 1))

        ex_embed = self.ex_embedding(x_enc[:,:,self.n_vars:], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2],enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.ex_en_to_gpt(enc_out)
        x_enc = dec_out.permute(0, 2, 1)

        b = torch.cat([x_enc, ex_x], dim=2)

        B, L, M = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = rearrange(x_enc, 'b l m -> b m l')
        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)


        if self.use_fft_adapter:
            fft_adapter_list = self.fft_adapter(x_enc)
            adapters = []
            for i in range(self.gpt_layers - len(fft_adapter_list)):
                adapters.append(None)
            for i in range(len(fft_adapter_list)):
                fft_adapter_list[i] = self.adapter_in_layer[i](fft_adapter_list[i])
                fft_adapter_list[i] = rearrange(fft_adapter_list[i], 'b m n p -> (b m) n p')
                adapters.append(fft_adapter_list[i])
        else:
            adapters = [None] * self.gpt_layers



        x = rearrange(x_enc, 'b m n p -> (b m) n p')
        outputs = self.in_layer(x)
        outputs = self.gpt2(inputs_embeds=outputs, adapters=adapters).last_hidden_state
        outputs = self.proj_layer(outputs)
        outputs = self.out_conv(outputs)
        outputs = outputs.mean(dim=-1)
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        # 反标准化
        outputs = outputs * stdev
        outputs = outputs + means

        ex_x = ex_x[:, -self.pred_len:, :]
        outputs = torch.cat([outputs,ex_x],dim=2)
        return outputs,a,b

