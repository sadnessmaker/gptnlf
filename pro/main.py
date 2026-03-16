
import argparse
import os
import torch

from exp.exp_short_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args, print_hyperparameters
import random
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='GPTNET')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='short_term_forecast',
                        help='task name, options:[short_term_forecast, few_shot_forecast]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='GPTNET_1', help='model id')
    parser.add_argument('--model', type=str, required=False, default='GPTNET_1',
                        help='model name, options: [TEMPO,GPTNET_1,GPT4TS,LSTM,Autoformer, Transformer, TimesNet,GPTNET_1,PatchTST,DLinear,iTransformer,Informer,Reformer]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='RYE', help='dataset type Energy ')
    parser.add_argument('--root_path', type=str,
                        default='/root/autodl-tmp/GPTNET/datasets/OPSD/GE/',
                        help='root path of the data file Correlation_analysis/OPSD')
    parser.add_argument('--data_path', type=str, default='GE_net_load.csv', help='data file net_load_1.csv / BE_data_time_gai_1.csv / GE_net_load.csv   /france_net_1.csv Rye_dan_1.csv')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='net_load', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='/root/autodl-tmp/GPTNET/checkpoints/',
                        help='location of model checkpoints/tmp/zfh_1/net_load_forecasting/checkpoints/ ')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=10, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    #en-ex_transformer
    parser.add_argument('--gpt_enc_in', type=int, default=4, help='外部嵌入目标输入编码器数,rye/at=4,france=5')
    parser.add_argument('--en_pred_len', type=int, default=48, help='外部嵌入输出长度')
    parser.add_argument('--n_vars', type=int, default=4, help='目标变量数,rye/at=4,france=5')
    parser.add_argument('--en_model', type=int, default=512, help='外部嵌入模型维度长度')
    parser.add_argument('--en_patch_len', type=int, default=16, help='外部嵌入patch长度16 8')
    parser.add_argument('--en_heads', type=int, default=8, help='外部嵌入头数')
    parser.add_argument('--en_dff', type=int, default=512, help='外部嵌入中间维度')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=12, help='encoder input size rye=16 ,france=11,AT,GE=12,BE=13')
    parser.add_argument('--dec_in', type=int, default=12, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=12, help='output size')
    parser.add_argument('--d_model', type=int, default=1280, help='dimension of model=[small:768,lage:1280]')
    parser.add_argument('--n_heads', type=int, default=18, help='num of heads=[small:8,lage:18]')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1280, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0, help='dropout0')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')



    # optimization
    parser.add_argument('--feture_loss', type=int, default=4, help='模型损失优化的变量数,rye/at=4,france=5')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs30')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data256')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate,lstm=0.01，其他=0.0001,DLinear=0.001 少样本：Transformer、Ieformer、Reformer=0.0001，LSTM=0.01，GPTNET_1=0.0005其他=0.001')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    # parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

   # hyperparameters


    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    #GPT-2
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')  # LLAMA, GPT2, BERT
    parser.add_argument('--gpt_layers', type=int, default=4, help='gpt_layers')
    parser.add_argument('--adapter_layer', type=int, default=2, help='adapter_layer')
    parser.add_argument('--spect_adapter_layer', type=int, default=2, help='spect_adapter_layer')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--T_type', type=int, default=1, help='T_type')
    parser.add_argument('--C_type', type=int, default=1, help='C_type')
    parser.add_argument('--use_fft_adapter', type=int, default=1, help='fft_type')

    parser.add_argument('--adapter_dim', type=int, default=64, help='adapter_dim32')
   # parser.add_argument('--rank', type=int, default=16, help='rank')
    parser.add_argument('--adapter_dropout', type=float, default=0, help='adapter dropout')
    parser.add_argument('--llm_dim', type=int, default='1280',
                        help='LLM model dimension')  # LLama7b:4096; GPT2-small:768;GPT2-large:1280; BERT-base:768
    parser.add_argument('--stride', type=int, default=8, help='stride 8  4')
    parser.add_argument('--padding', type=int, default=8, help='padding 8  4')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length 16  8')
    parser.add_argument('--llm_layers', type=int, default=1)
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--align_const', type=float, default=0.4)

    parser.add_argument('--percent', type=float, default=1, help='proportion of in-distribution downstream dataset')

    #其他
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2025, help="Randomization seed")

    #tempo
    # parser.add_argument('--pool', action='store_true', help='whether use prompt pool')
    # parser.add_argument('--pretrain', type=int, default=1)
    # parser.add_argument('--is_gpt', type=int, default=1)
    # parser.add_argument('--prompt', type=int, default=0)
    # parser.add_argument('--use_token', type=int, default=0)
    # parser.add_argument('--freeze', type=int, default=1)
    # parser.add_argument('--num_nodes', type=int, default=1)
    # parser.add_argument('--loss_func', type=str, default='mse')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False
   # set gpu id
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # print arguments
    # print_args(args)
    # print_hyperparameters(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'few_shot_forecast':
        Exp = Exp_Few_Shot_Forecast
    else:
     print("请指定任务")
       # Exp = Exp_Long_Term_Forecast
    rmse_list=[]
    mae_list=[]
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_fs{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.percent,
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            rmse,mae=exp.test(setting)

            rmse_list.append(rmse)
            mae_list.append(mae)
            torch.cuda.empty_cache()
    else:
        ii = 0
        exp = Exp(args)
        setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_fs{}_{}'.format(
            args.task_name,
            args.vlm_type,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.percent,
            ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)

        torch.cuda.empty_cache()
    rmse_avg = np.mean(rmse_list)
    mae_avg = np.mean(mae_list)
    print(">>>>>>>>>>>>>>>>my test rmse and mae<<<<<<<<<<<<<<<<<<<")
    print('rmse_after_avg: {},mae_after_avg: {}'.format(rmse_avg, mae_avg))
    print('rmse_after_0: {},mae_after_0: {}'.format(rmse_list[0], mae_list[0]))
# #     # print('rmse_after_1: {},mae_after_1: {}'.format(rmse_list[1], mae_list[1]))
# #     # print('rmse_after_2: {},mae_after_2: {}'.format(rmse_list[2], mae_list[2]))
# #
# #     # with open('/tmp/zfh_1/net_load_forecasting/PIC/methods_pic/AT/zhibiao.txt', 'a') as f:#'/tmp/zfh_1/net_load_forecasting/PIC/methods_pic/zhibiao_methods.txt'
# #     #     f.write(f"\ngptnet间接_19_GPTNET_1:\nRMSE: {rmse_avg:.6f}\nMAE: {mae_avg:.6f}\n")#gptnet_adapter
