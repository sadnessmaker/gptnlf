from transformers.models.gpt2.modeling_gpt2 import  GPT2Block
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

# 自定义GPT2Block类，修改forward方法以支持适配器
class CustomGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.T_adapter = None
        self.C_adapter = None
        self.T_adapter_gate = None
        self.C_adapter_gate = None
        self.C_num = None
        self.scale = 1.0

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            adapter_hidden_states=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # 应用时间适配器
        if hasattr(self, 'T_adapter') and self.T_adapter is not None:
            if adapter_hidden_states is not None:
                # 确保维度匹配
                # print(f"attn_output shape: {attn_output.shape}")
                # print(f"adapter_hidden_states shape: {adapter_hidden_states.shape}")
                # print(f"T_adapter_gate shape: {self.T_adapter_gate.shape}")

                # 检查并调整适配器输出的维度
                if adapter_hidden_states.shape != attn_output.shape:
                    # 如果序列长度不匹配，进行调整
                    if adapter_hidden_states.shape[1] != attn_output.shape[1]:
                        # 使用插值或截断来匹配序列长度
                        target_seq_len = attn_output.shape[1]
                        if adapter_hidden_states.shape[1] > target_seq_len:
                            # 截断
                            adapter_hidden_states = adapter_hidden_states[:, :target_seq_len, :]
                        else:
                            # 填充或重复
                            pad_len = target_seq_len - adapter_hidden_states.shape[1]
                            adapter_hidden_states = F.pad(adapter_hidden_states, (0, 0, 0, pad_len), 'replicate')

                adapter_output = self.T_adapter(adapter_hidden_states)

                # 调整gate的维度以匹配adapter_output
                gate = self.T_adapter_gate
                if gate.shape[1] != adapter_output.shape[1]:
                    # 重新调整gate的大小
                    gate = gate.expand(-1, adapter_output.shape[1], -1)

                adapter_output = adapter_output * gate

                # 确保最终的维度匹配
                if adapter_output.shape != attn_output.shape:
                    print(
                        f"Warning: adapter_output shape {adapter_output.shape} doesn't match attn_output shape {attn_output.shape}")
                    # 尝试调整到相同的形状
                    if adapter_output.shape[1] != attn_output.shape[1]:
                        target_seq_len = attn_output.shape[1]
                        if adapter_output.shape[1] > target_seq_len:
                            adapter_output = adapter_output[:, :target_seq_len, :]
                        else:
                            pad_len = target_seq_len - adapter_output.shape[1]
                            adapter_output = F.pad(adapter_output, (0, 0, 0, pad_len), 'replicate')

                attn_output = attn_output + adapter_output

        # residual connection
        hidden_states = attn_output + residual

        # 应用通道适配器
        if hasattr(self, 'C_adapter') and self.C_adapter is not None:
            # 重排维度以应用通道适配器
            B, N, D = hidden_states.shape
            if hasattr(self, 'C_num') and self.C_num is not None:
                try:
                    # 更安全的维度重塑
                    if N % self.C_num == 0:
                        hidden_states_reshaped = hidden_states.view(B * self.C_num, N // self.C_num, D)
                        c_adapter_output = self.C_adapter(hidden_states_reshaped)

                        # 调整C_adapter_gate的维度
                        c_gate = self.C_adapter_gate
                        if c_gate.shape[1] != self.C_num:
                            c_gate = c_gate.expand(-1, self.C_num, -1)

                        c_adapter_output = c_adapter_output * c_gate.view(1, 1, 1)
                        hidden_states = hidden_states + c_adapter_output.view(B, N, D)
                    else:
                        # 如果无法整除，直接应用适配器
                        c_adapter_output = self.C_adapter(hidden_states)
                        c_gate = self.C_adapter_gate.mean()
                        hidden_states = hidden_states + c_adapter_output * c_gate
                except Exception as e:
                   # print(f"Channel adapter error: {e}")
                    # 直接应用适配器作为后备方案
                    c_adapter_output = self.C_adapter(hidden_states)
                    c_gate = self.C_adapter_gate.mean()
                    hidden_states = hidden_states + c_adapter_output * c_gate

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


# 自定义GPT2Model类，修改forward方法以支持适配器
class CustomGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        # 替换所有的GPT2Block为CustomGPT2Block
        self.h = nn.ModuleList([CustomGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            adapters=None,  # 新增参数

    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取对应层的适配器隐藏状态
            adapter_hidden_state = None
            if adapters is not None and i < len(adapters) and adapters[i] is not None:
                adapter_hidden_state = adapters[i]

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                adapter_hidden_states=adapter_hidden_state,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

