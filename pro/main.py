# import argparse
# import os
# import torch
# import itertools
# import json
# from datetime import datetime
#
# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_few_shot_forecasting import Exp_Few_Shot_Forecast
#
# from utils.print_args import print_args, print_hyperparameters
# import random
# import numpy as np
#
#
# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
#
#
# def grid_search(args, param_grid):
#     """
#     执行网格搜索
#     """
#     # 获取所有参数组合
#     param_names = list(param_grid.keys())
#     param_values = list(param_grid.values())
#     param_combinations = list(itertools.product(*param_values))
#
#     print(f"开始网格搜索，共有 {len(param_combinations)} 种参数组合")
#
#     results = []
#     best_rmse = float('inf')
#     best_params = None
#
#     for i, param_combo in enumerate(param_combinations):
#         print(f"\n=== 参数组合 {i + 1}/{len(param_combinations)} ===")
#
#         # 更新参数
#         current_args = argparse.Namespace(**vars(args))
#         for param_name, param_value in zip(param_names, param_combo):
#             setattr(current_args, param_name, param_value)
#
#         # 打印当前参数组合
#         print("当前参数:")
#         for param_name, param_value in zip(param_names, param_combo):
#             print(f"  {param_name}: {param_value}")
#
#         try:
#             # 运行实验
#             rmse_list = []
#             mae_list = []
#
#             if current_args.task_name == 'long_term_forecast':
#                 Exp = Exp_Long_Term_Forecast
#             elif current_args.task_name == 'few_shot_forecast':
#                 Exp = Exp_Few_Shot_Forecast
#             else:
#                 Exp = Exp_Long_Term_Forecast
#
#             for ii in range(current_args.itr):
#                 exp = Exp(current_args)
#                 setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_fs{}_{}'.format(
#                     current_args.task_name,
#                     current_args.model_id,
#                     current_args.model,
#                     current_args.data,
#                     current_args.features,
#                     current_args.seq_len,
#                     current_args.label_len,
#                     current_args.pred_len,
#                     current_args.d_model,
#                     current_args.percent,
#                     ii)
#
#                 print(f'>>>>>>>开始训练: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#                 exp.train(setting)
#                 print(f'>>>>>>>测试: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#                 rmse, mae = exp.test(setting)
#
#                 rmse_list.append(rmse)
#                 mae_list.append(mae)
#                 torch.cuda.empty_cache()
#
#             # 计算平均指标
#             avg_rmse = np.mean(rmse_list)
#             avg_mae = np.mean(mae_list)
#             std_rmse = np.std(rmse_list)
#             std_mae = np.std(mae_list)
#
#             # 保存结果
#             result = {
#                 'params': dict(zip(param_names, param_combo)),
#                 'rmse_mean': avg_rmse,
#                 'mae_mean': avg_mae,
#                 'rmse_std': std_rmse,
#                 'mae_std': std_mae,
#                 'rmse_list': rmse_list,
#                 'mae_list': mae_list
#             }
#             results.append(result)
#
#             print(f"结果: RMSE={avg_rmse:.6f}(±{std_rmse:.6f}), MAE={avg_mae:.6f}(±{std_mae:.6f})")
#
#             # 更新最佳结果
#             if avg_rmse < best_rmse:
#                 best_rmse = avg_rmse
#                 best_params = dict(zip(param_names, param_combo))
#                 print("*** 发现新的最佳参数组合! ***")
#
#         except Exception as e:
#             print(f"参数组合失败: {e}")
#             result = {
#                 'params': dict(zip(param_names, param_combo)),
#                 'error': str(e)
#             }
#             results.append(result)
#
#     return results, best_params, best_rmse
#
#
# def save_grid_search_results_text(results, best_params, best_rmse, save_path):
#     """
#     保存网格搜索结果为纯文本格式
#     """
#     # 创建保存目录
#     os.makedirs(save_path, exist_ok=True)
#
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     # 1. 保存最佳参数
#     best_params_file = os.path.join(save_path, f"best_params_{timestamp}.txt")
#     with open(best_params_file, 'w', encoding='utf-8') as f:
#         f.write("=" * 60 + "\n")
#         f.write("最佳参数配置\n")
#         f.write("=" * 60 + "\n")
#         f.write(f"时间: {timestamp}\n")
#         f.write(f"最佳RMSE: {float(best_rmse):.6f}\n\n")
#         f.write("参数设置:\n")
#         f.write("-" * 30 + "\n")
#         for key, value in best_params.items():
#             f.write(f"{key:20s}: {value}\n")
#
#     # 2. 保存详细结果
#     detailed_file = os.path.join(save_path, f"detailed_results_{timestamp}.txt")
#     with open(detailed_file, 'w', encoding='utf-8') as f:
#         f.write("=" * 80 + "\n")
#         f.write("网格搜索详细结果\n")
#         f.write("=" * 80 + "\n")
#         f.write(f"时间: {timestamp}\n")
#         f.write(f"总实验数: {len(results)}\n\n")
#
#         # 分离成功和失败的实验
#         successful_results = []
#         failed_results = []
#
#         for result in results:
#             if 'error' in result:
#                 failed_results.append(result)
#             else:
#                 successful_results.append(result)
#
#         f.write(f"成功实验: {len(successful_results)} 个\n")
#         f.write(f"失败实验: {len(failed_results)} 个\n\n")
#
#         # 保存成功的实验
#         if successful_results:
#             # 按RMSE排序
#             successful_results.sort(key=lambda x: float(x['rmse_mean']))
#
#             f.write("成功实验结果 (按RMSE排序):\n")
#             f.write("=" * 80 + "\n")
#
#             for i, result in enumerate(successful_results):
#                 f.write(f"\n实验 {i + 1}:\n")
#                 f.write("-" * 50 + "\n")
#                 f.write(f"RMSE均值: {float(result['rmse_mean']):.6f}\n")
#                 f.write(f"RMSE标准差: {float(result['rmse_std']):.6f}\n")
#                 f.write(f"MAE均值: {float(result['mae_mean']):.6f}\n")
#                 f.write(f"MAE标准差: {float(result['mae_std']):.6f}\n")
#                 f.write("参数配置:\n")
#                 for key, value in result['params'].items():
#                     f.write(f"  {key}: {value}\n")
#
#                 # 每次运行的结果
#                 if 'rmse_list' in result and 'mae_list' in result:
#                     f.write("各次运行结果:\n")
#                     for j, (rmse_val, mae_val) in enumerate(zip(result['rmse_list'], result['mae_list'])):
#                         f.write(f"  运行{j + 1}: RMSE={float(rmse_val):.6f}, MAE={float(mae_val):.6f}\n")
#
#         # 保存失败的实验
#         if failed_results:
#             f.write(f"\n\n失败实验:\n")
#             f.write("=" * 80 + "\n")
#             for i, result in enumerate(failed_results):
#                 f.write(f"\n失败实验 {i + 1}:\n")
#                 f.write(f"参数: {result['params']}\n")
#                 f.write(f"错误: {result['error']}\n")
#
#     # 3. 保存摘要
#     summary_file = os.path.join(save_path, f"summary_{timestamp}.txt")
#     with open(summary_file, 'w', encoding='utf-8') as f:
#         f.write("网格搜索结果摘要\n")
#         f.write("=" * 50 + "\n")
#         f.write(f"时间: {timestamp}\n")
#         f.write(f"最佳RMSE: {float(best_rmse):.6f}\n\n")
#
#         f.write("最佳参数:\n")
#         f.write("-" * 20 + "\n")
#         for key, value in best_params.items():
#             f.write(f"{key}: {value}\n")
#
#         if successful_results:
#             f.write(f"\n前5名结果:\n")
#             f.write("-" * 20 + "\n")
#             for i, result in enumerate(successful_results[:5]):
#                 f.write(f"第{i + 1}名: RMSE={float(result['rmse_mean']):.6f}\n")
#
#         f.write(f"\n统计信息:\n")
#         f.write(f"成功实验: {len(successful_results)}\n")
#         f.write(f"失败实验: {len(failed_results)}\n")
#         f.write(f"总实验数: {len(results)}\n")
#
#     # 4. 保存CSV表格
#     csv_file = os.path.join(save_path, f"results_{timestamp}.csv")
#     with open(csv_file, 'w', encoding='utf-8') as f:
#         if successful_results:
#             # 表头
#             param_names = list(successful_results[0]['params'].keys())
#             headers = ['排名', 'RMSE均值', 'RMSE标准差', 'MAE均值', 'MAE标准差'] + param_names
#             f.write(','.join(headers) + '\n')
#
#             # 数据行
#             for i, result in enumerate(successful_results):
#                 row = [
#                     str(i + 1),
#                     f"{float(result['rmse_mean']):.6f}",
#                     f"{float(result['rmse_std']):.6f}",
#                     f"{float(result['mae_mean']):.6f}",
#                     f"{float(result['mae_std']):.6f}"
#                 ]
#                 for param_name in param_names:
#                     row.append(str(result['params'][param_name]))
#                 f.write(','.join(row) + '\n')
#
#     print(f"\n结果已保存到以下文件:")
#     print(f"  最佳参数: {best_params_file}")
#     print(f"  详细结果: {detailed_file}")
#     print(f"  结果摘要: {summary_file}")
#     print(f"  CSV表格: {csv_file}")
#
#     return best_params_file, detailed_file, summary_file, csv_file
#
#
# if __name__ == '__main__':
#     fix_seed = 2025
#     random.seed(fix_seed)
#     torch.manual_seed(fix_seed)
#     np.random.seed(fix_seed)
#
#     parser = argparse.ArgumentParser(description='GPTNET with Grid Search')
#
#     # 添加网格搜索开关
#     parser.add_argument('--use_grid_search', type=str2bool, default=True,
#                         help='是否使用网格搜索优化超参数')
#     parser.add_argument('--grid_search_save_path', type=str,
#                         default='/tmp/zfh_1/net_load_forecasting/grid_search_results/BE/',
#                         help='网格搜索结果保存路径')
#
#     # basic config
#     parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
#                         help='task name, options:[long_term_forecast, short_term_forecast, few_shot_forecast]')
#     parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
#     parser.add_argument('--model_id', type=str, required=False, default='GPTNET_1', help='model id')
#     parser.add_argument('--model', type=str, required=False, default='GPTNET_1',
#                         help='model name, options: [GPT4TS,LSTM,Autoformer, Transformer, TimesNet,GPTNET_1,PatchTST,DLinear,iTransformer,Informer,Reformer]')
#
#     # data loader
#     parser.add_argument('--data', type=str, required=False, default='RYE', help='dataset type Energy ')
#     parser.add_argument('--root_path', type=str,
#                         default='/tmp/zfh_1/net_load_forecasting/datasets/OPSD/BE/',
#                         help='root path of the data file')
#     parser.add_argument('--data_path', type=str, default='BE_net_load.csv', help='data file')
#     parser.add_argument('--features', type=str, default='M',
#                         help='forecasting task, options:[M, S, MS]')
#     parser.add_argument('--target', type=str, default='net_load', help='target feature in S or MS task')
#     parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
#     parser.add_argument('--checkpoints', type=str, default='/tmp/zfh_1/net_load_forecasting/checkpoints/',
#                         help='location of model checkpoints')
#
#     # forecasting task
#     parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
#     parser.add_argument('--label_len', type=int, default=10, help='start token length')
#     parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
#     parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
#     parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
#
#     # en-ex_transformer
#     parser.add_argument('--gpt_enc_in', type=int, default=4, help='外部嵌入目标输入编码器数')
#     parser.add_argument('--en_pred_len', type=int, default=48, help='外部嵌入输出长度')
#     parser.add_argument('--n_vars', type=int, default=4, help='目标变量数')
#     parser.add_argument('--en_model', type=int, default=512, help='外部嵌入模型维度长度')
#     parser.add_argument('--en_patch_len', type=int, default=16, help='外部嵌入patch长度')
#     parser.add_argument('--en_heads', type=int, default=8, help='外部嵌入头数')
#     parser.add_argument('--en_dff', type=int, default=512, help='外部嵌入中间维度')
#
#     # model define
#     parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
#     parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
#     parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
#     parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
#     parser.add_argument('--enc_in', type=int, default=13, help='encoder input size')
#     parser.add_argument('--dec_in', type=int, default=13, help='decoder input size')
#     parser.add_argument('--c_out', type=int, default=13, help='output size')
#     parser.add_argument('--d_model', type=int, default=1280, help='dimension of model')
#     parser.add_argument('--n_heads', type=int, default=18, help='num of heads')
#     parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
#     parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
#     parser.add_argument('--d_ff', type=int, default=1280, help='dimension of fcn')
#     parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
#     parser.add_argument('--factor', type=int, default=3, help='attn factor')
#     parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=True)
#     parser.add_argument('--dropout', type=float, default=0, help='dropout')
#     parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
#     parser.add_argument('--activation', type=str, default='gelu', help='activation')
#     parser.add_argument('--channel_independence', type=int, default=1, help='channel independence for FreTS model')
#     parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition')
#     parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize')
#
#     # optimization
#     parser.add_argument('--feture_loss', type=int, default=4, help='模型损失优化的变量数')
#     parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
#     parser.add_argument('--itr', type=int, default=1, help='experiments times')
#     parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
#     parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='Exp', help='exp description')
#     parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
#     parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
#
#     # GPU
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--gpu', type=int, default=0, help='gpu')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
#     parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
#
#     # GPT-2
#     parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')
#     parser.add_argument('--gpt_layers', type=int, default=6, help='gpt_layers')
#     parser.add_argument('--adapter_layer', type=int, default=2, help='adapter_layer')
#     parser.add_argument('--spect_adapter_layer', type=int, default=2, help='spect_adapter_layer')
#     parser.add_argument('--scale', type=int, default=1)
#     parser.add_argument('--T_type', type=int, default=1, help='T_type')
#     parser.add_argument('--C_type', type=int, default=1, help='C_type')
#     parser.add_argument('--use_fft_adapter', type=int, default=1, help='fft_type')
#     parser.add_argument('--adapter_dim', type=int, default=32, help='adapter_dim')
#     parser.add_argument('--rank', type=int, default=16, help='rank')
#     parser.add_argument('--adapter_dropout', type=float, default=0, help='adapter dropout')
#     parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')
#     parser.add_argument('--stride', type=int, default=8, help='stride')
#     parser.add_argument('--padding', type=int, default=8, help='padding')
#     parser.add_argument('--patch_len', type=int, default=16, help='patch length')
#     parser.add_argument('--llm_layers', type=int, default=1)
#     parser.add_argument('--prompt_domain', type=int, default=0, help='')
#     parser.add_argument('--align_const', type=float, default=0.4)
#     parser.add_argument('--percent', type=float, default=1, help='proportion of in-distribution downstream dataset')
#
#     # 其他
#     parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
#     parser.add_argument('--seed', type=int, default=2025, help="Randomization seed")
#
#     args = parser.parse_args()
#     args.use_gpu = True if torch.cuda.is_available() else False
#
#     # set gpu id
#     if args.use_gpu and args.use_multi_gpu:
#         args.devices = args.devices.replace(' ', '')
#         device_ids = args.devices.split(',')
#         args.device_ids = [int(id_) for id_ in device_ids]
#         args.gpu = args.device_ids[0]
#
#     # 定义网格搜索参数空间
#     if args.use_grid_search:
#         param_grid = {
#             'learning_rate': [0.0001, 0.0005, 0.001],
#             'batch_size': [128, 256],
#             'dropout': [0.0, 0.1],
#             'adapter_dim': [16, 32, 64],
#             'gpt_layers': [4, 6],
#
#             # 'en_model': [256, 512, 768],
#             # 'en_patch_len': [8, 16, 32],
#             # 'd_model': [768, 1280],
#             # 'n_heads': [8, 16, 18],
#         }
#
#         print("开始网格搜索超参数优化...")
#         print("搜索空间:")
#         for key, values in param_grid.items():
#             print(f"  {key}: {values}")
#
#         # 执行网格搜索
#         results, best_params, best_rmse = grid_search(args, param_grid)
#
#         # 保存结果
#         save_grid_search_results_text(results, best_params, best_rmse, args.grid_search_save_path)
#
#         print(f"\n网格搜索完成!")
#         print(f"最佳RMSE: {best_rmse:.6f}")
#         print(f"最佳参数: {best_params}")
#
#     else:
#         # 原始训练流程
#         if args.task_name == 'long_term_forecast':
#             Exp = Exp_Long_Term_Forecast
#         elif args.task_name == 'few_shot_forecast':
#             Exp = Exp_Few_Shot_Forecast
#         else:
#             print("请指定任务")
#
#         rmse_list = []
#         mae_list = []
#
#         if args.is_training:
#             for ii in range(args.itr):
#                 exp = Exp(args)
#                 setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_fs{}_{}'.format(
#                     args.task_name,
#                     args.model_id,
#                     args.model,
#                     args.data,
#                     args.features,
#                     args.seq_len,
#                     args.label_len,
#                     args.pred_len,
#                     args.d_model,
#                     args.percent,
#                     ii)
#
#                 print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
#                 exp.train(setting)
#                 print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#                 rmse, mae = exp.test(setting)
#
#                 rmse_list.append(rmse)
#                 mae_list.append(mae)
#                 torch.cuda.empty_cache()
#         else:
#             ii = 0
#             exp = Exp(args)
#             setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_fs{}_{}'.format(
#                 args.task_name,
#                 args.vlm_type,
#                 args.model_id,
#                 args.model,
#                 args.data,
#                 args.features,
#                 args.seq_len,
#                 args.label_len,
#                 args.pred_len,
#                 args.d_model,
#                 args.percent,
#                 ii)
#
#             print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#             exp.test(setting, test=1)
#             torch.cuda.empty_cache()
#
#         if rmse_list:
#             rmse_avg = np.mean(rmse_list)
#             mae_avg = np.mean(mae_list)
#             print(">>>>>>>>>>>>>>>>my test rmse and mae<<<<<<<<<<<<<<<<<<<")
#             print('rmse_after_avg: {},mae_after_avg: {}'.format(rmse_avg, mae_avg))
#             print('rmse_after_0: {},mae_after_0: {}'.format(rmse_list[0], mae_list[0]))




import argparse
import os
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_few_shot_forecasting import Exp_Few_Shot_Forecast

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
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, few_shot_forecast]')
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
