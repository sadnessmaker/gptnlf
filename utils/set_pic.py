# import time
# import os
# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# def Pearson_Correlation(data):
#     """基于Person相关系数的相关性矩阵"""
#     plt.figure(figsize=[10, 6])#[10, 6]
#     matrix_corr = data.corr(method='pearson')  # DF (8, 8)
#     matrix_corr_flipped = pd.DataFrame(
#         np.flipud(matrix_corr.values),
#         index=matrix_corr.index[::-1],
#         columns=matrix_corr.columns
#     )
#     # sns.heatmap: annot:是否在方格中写入数据
#     sns.heatmap(matrix_corr_flipped, vmin=-1, vmax=1, cmap="coolwarm", linewidths=0.5, annot=True)
#     plt.savefig('D:/LLM/net_load_forecasting/PIC/pearson/pearson_AT_4.png', dpi=300, bbox_inches='tight')
#     #plt.savefig('D:/LLM/net_load_forecasting/PIC/pearson/pearson_energy_all.png',dpi=300, bbox_inches='tight')
#     # plt.savefig('D:/LLM/net_load_forecasting/PIC/pearson/pearson_rye_all.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()
# def Spearman(data):
#
#     plt.figure(figsize=[10,6])
#     corr=data.corr(method='spearman')
#     matrix_corr_flipped = pd.DataFrame(
#         np.flipud(corr.values),
#         index=corr.index[::-1],
#         columns=corr.columns
#     )
#     # sns.heatmap: annot:是否在方格中写入数据
#     sns.heatmap(matrix_corr_flipped, vmin=-1, vmax=1, cmap="coolwarm", linewidths=0.5, annot=True)
#     plt.savefig('D:/LLM/net_load_forecasting/PIC/spearman/spearman_AT_4.png', dpi=300, bbox_inches='tight')
#     #plt.savefig('D:/LLM/net_load_forecasting/PIC/spearman/spearman_energy_all.png',dpi=300, bbox_inches='tight')
#     #plt.savefig('D:/LLM/net_load_forecasting/PIC/spearman/spearman_rye_all.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()
# #
# if __name__ == '__main__':
#
#     #奥地利的相关性分析
#     data=pd.read_csv('D:/LLM/net_load_forecasting/datasets/OPSD/net_load_1.csv')
#     #data = data[3626:4346]  # [2882:6554]
#     data.drop('date',axis = 1,inplace = True)
#     data=data.rename(columns={
#                                 'net_load':'n_l',
#                               'AT_load_actual_entsoe_transparency':'AT_ac',
#                               'AT_solar_generation_actual':'AT_solar',
#                               'AT_wind_onshore_generation_actual':'AT_wind'})
#     data=data[['n_l','AT_ac', 'AT_solar', 'AT_wind']]
#
#     Pearson_Correlation(data)
#     Spearman(data)
# #
#
#
#    # #综合能源数据集的相关性分析
#    #  data = pd.read_csv('D:/LLM/net_load_forecasting/datasets/enery/energy_net_load.csv')
#    #  # data = data[3626:4346]
#    #  data.drop('date', axis=1, inplace=True)
#    #  # data = data.rename(columns={
#    #  #     'AT_load_actual_entsoe_transparency': 'AT_ac',
#    #  #     'AT_solar_generation_actual': 'AT_solar',
#    #  #     'AT_wind_onshore_generation_actual': 'AT_wind',
#    #  #     'AT_load_forecast_entsoe_transparency': 'AT_fc',
#    #  #     'AT_price_day_ahead': 'AT_price',
#    #  #     'AT_temperature': 'AT_tem',
#    #  #     'AT_radiation_direct_horizontal': 'AT_DNI',
#    #  #     'AT_radiation_diffuse_horizontal': 'AT_DHI'})
#    #  Pearson_Correlation(data)
#    #  Spearman(data)
#
#
    # # 挪威数据集的相关性分析
    # data = pd.read_csv('D:/LLM/net_load_forecasting/datasets/Rye/Rye_dan.csv')
    # # data = data[3626:4346]#[2882:6554]
    # data.drop('date', axis=1, inplace=True)
    # data = data.rename(columns={
    #     'net_load': 'n_l',
    #     'consumption': 'rye_ac',
    #     'pv_production': 'rye_solar',
    #     'wind_production': 'rye_wind'})
    # data = data[['n_l', 'rye_ac', 'rye_solar', 'rye_wind']]
    # Pearson_Correlation(data)
    # Spearman(data)
#
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from minepy import MINE
#
# # 读取数据（假设数据文件名为data.csv）
# df = pd.read_csv('D:/LLM/net_load_forecasting/datasets/Rye/Rye_all_2.csv')
# #D:/LLM/net_load_forecasting/datasets/enery/energy_net_load_3.csv
# #D:/LLM/net_load_forecasting/datasets/Rye/Rye.csv
# #D:/LLM/net_load_forecasting/datasets/OPSD/net_load.csv
# #df = df[3626:4346]
# # df=df.rename(columns={ 'AT_load_actual_entsoe_transparency':'AT_ac',
# #                               'AT_solar_generation_actual':'AT_solar',
# #                               'AT_wind_onshore_generation_actual':'AT_wind'})
# # 删除时间列，保留需要分析的变量
# df = df.drop(columns=df.columns[0])
#
# # 计算最大信息系数（MIC）矩阵
# variables = df.columns.tolist()
# mic_matrix = np.zeros((len(variables), len(variables)))
#
# mine = MINE(alpha=0.6, c=15)
# for i, col1 in enumerate(variables):
#     for j, col2 in enumerate(variables):
#         mine.compute_score(df[col1], df[col2])
#         mic_matrix[i, j] = mine.mic()
#
#
# # 旋转矩阵（关键修改）
# rotated_matrix = np.rot90(mic_matrix, k=1)  # 逆时针旋转90度
#
# # 调整标签顺序（与旋转后的矩阵匹配）
# rotated_labels = list(reversed(variables))
#
# # 创建热力图
# plt.figure(figsize=(30, 15))
# ax=sns.heatmap(
#     rotated_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="Blues",
#     xticklabels=variables,
#     yticklabels=rotated_labels,
#     linewidths=0.5,
#     vmin=0, vmax=1,
#     annot_kws={'size':18}
#
# )
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=18)
#
# plt.title("Energy Heatmap",fontsize=18)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.tight_layout()
# plt.savefig('D:/LLM/net_load_forecasting/PIC/MIC/Rye_MIC_all.png')
# plt.show()


'''#画折线图
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('D:\\LLM\\net_load_forecasting\\PIC\\yuceandzhengshi.csv')

plt.figure(figsize=(15, 8))
plt.subplot(1, 1, 1)
plt.plot(data['true_load'], color="#7C7979", label="True",markersize=3)
plt.plot(data['pred_net_2'], color="#D75B4E", label="Pred",markersize=3)

plt.xlabel("Hour")
plt.ylabel("Value(kw)")
plt.title("net_load Prediction Results")
plt.legend()
plt.tight_layout()
plt.savefig("D:\\LLM\\net_load_forecasting\\PIC\\rye_P_T.png")
print('已保存')
plt.show()
'''
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import seaborn as sns
import numpy as np
# Set academic style
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.size': 15,
    'font.family': 'serif',
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'axes.grid': True,
    'grid.alpha': 0.3
})
# 读取数据
df = pd.read_csv('D:\\LLM\\net_load_forecasting\\datasets\\OPSD\\net_load_4.csv')
df=df[3626:4346]
df['date'] = pd.to_datetime(df['date'])
df=df.rename(columns={ 'AT_load_actual_entsoe_transparency':'consumption',
                              'AT_solar_generation_actual':'pv_production',
                              'AT_wind_onshore_generation_actual':'wind_production'})
# 创建图表
#fig, ax = plt.subplots(figsize=(18, 6))Rye
fig, ax = plt.subplots(figsize=(20, 6))
# Plot with academic colors and styles
ax.plot(df['date'].values, df['consumption'].values,
        label='load Consumption', linewidth=2.5, color='#d62728', alpha=0.8)
ax.plot(df['date'].values, df['pv_production'].values,
        label='Solar Production', linewidth=2.5, color='#ff7f0e', alpha=0.8)
ax.plot(df['date'].values, df['wind_production'].values,
        label='Wind Production', linewidth=2.5, color='#2ca02c', alpha=0.8)
ax.plot(df['date'].values, df['net_load'].values,
        label='Net Load', linewidth=2.5, color='#1f77b4', alpha=0.8)

# Add zero reference line
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
all_values = np.concatenate([
    df['consumption'].values,
    df['pv_production'].values,
    df['wind_production'].values,
    df['net_load'].values
])
y_min, y_max = np.min(all_values), np.max(all_values)
y_range = y_max - y_min

# 扩展y轴范围，为图例预留空间（增加20%的上边距）
ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.25 * y_range)
ax.set_xlabel('Time', fontweight='bold')
ax.set_ylabel('Load Value (MW)', fontweight='bold')

# ax.set_title('Temporal Analysis of Energy Production and Consumption\nJanuary 2-3, 2020',
#              fontweight='bold', pad=20)
ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right',ncol=4)#,bbox_to_anchor=(0.98, 0.15)
plt.subplots_adjust(top=0.85)
ax.grid(True, alpha=0.3)

# Format x-axis
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('D:\\LLM\\net_load_forecasting\\PIC\\huitu\\AT\\AT_four_data_pic.png')
plt.show()
