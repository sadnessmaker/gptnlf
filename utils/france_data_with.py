import pandas as pd

# df1 = pd.read_csv('D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\france_net_house1.csv')
# df1['Date_1'] = pd.to_datetime(df1['Date_1'])
# hourly_data1 = df1[(df1['Date_1'].dt.minute == 0) & (df1['Date_1'].dt.second == 0)].drop_duplicates(subset=['Date_1'])
# result_1 = hourly_data1[(hourly_data1['Date_1'] >= '2020-05-01') & (hourly_data1['Date_1'] <= '2020-12-28 23:00:00')]
# print(result_1)
# result_1.to_csv("D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\time_ok_1\\house_1.csv",index=False)
#
# df2 = pd.read_csv('D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\france_net_house2.csv')
# df2['Date_2'] = pd.to_datetime(df2['Date_2'])
# hourly_data2 = df2[(df2['Date_2'].dt.minute == 0) & (df2['Date_2'].dt.second == 0)].drop_duplicates(subset=['Date_2'])
# result_2 = hourly_data2[(hourly_data2['Date_2'] >= '2020-05-01') & (hourly_data2['Date_2'] <= '2020-12-28 23:00:00')]
# #df2['Date_2'] = pd.to_datetime(df2['Date_2'], format='%d/%m/%Y %H:%M').dt.strftime('%Y/%-m/%-d %-H:%M')
# print(result_2)
# result_2.to_csv("D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\time_ok_1\\house_2.csv",index=False)
#
# df3 = pd.read_csv('D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\france_net_house3.csv')
# df3['Date_3'] = pd.to_datetime(df3['Date_3'])
# hourly_data3 = df3[(df3['Date_3'].dt.minute == 0) & (df3['Date_3'].dt.second == 0)].drop_duplicates(subset=['Date_3'])
# result_3 = hourly_data3[(hourly_data3['Date_3'] >= '2020-05-01') & (hourly_data3['Date_3'] <= '2020-12-28 23:00:00')]
# print(result_3)
# result_3.to_csv("D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\time_ok_1\\house_3.csv",index=False)
#
# df4 = pd.read_csv('D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\france_net_ren.csv')
# df4['Date_4'] = pd.to_datetime(df4['Date_4'])
# hourly_data4 = df4[(df4['Date_4'].dt.minute == 0) & (df4['Date_4'].dt.second == 0)].drop_duplicates(subset=['Date_4'])
# result_4 = hourly_data4[(hourly_data4['Date_4'] >= '2020-05-01') & (hourly_data4['Date_4'] <= '2020-12-28 23:00:00')]
# print(result_4)
# result_4.to_csv("D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\time_ok_1\\Renergy.csv",index=False)

df5 = pd.read_csv('D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\time_ok_0\\Meteo_dataset_ 分列_filled.csv')
df5['Date_5'] = pd.to_datetime(df5['Date_5'])
hourly_data5 = df5[(df5['Date_5'].dt.minute == 0) & (df5['Date_5'].dt.second == 0)].drop_duplicates(subset=['Date_5'])
result_5 = hourly_data5[(hourly_data5['Date_5'] >= '2020-05-01') & (hourly_data5['Date_5'] <= '2020-12-28 23:00:00')]
print(result_5)
result_5.to_csv("D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\time_ok_0\\Meteo_dataset_ 分列_filled_1.csv",index=False)
#
# import pandas as pd
# from datetime import datetime, timedelta
# import numpy as np
# def check_time_series_data(data):
#     """
#     检查时间序列数据中的缺失时间点和重复数据
#
#     Parameters:
#     data: 可以是DataFrame或者包含时间数据的列表/字符串
#     """
#
#     # 如果输入是字符串，先转换为DataFrame
#     if isinstance(data, str):
#         lines = data.strip().split('\n')
#         # 解析数据
#         parsed_data = []
#         for line in lines:
#             parts = line.split()
#             if len(parts) >= 3:  # 确保有足够的列
#                 date_time = parts[0] + ' ' + parts[1]  # 合并日期和时间
#                 parsed_data.append({
#                     'datetime': date_time,
#                     'house_1_shiji': float(parts[2]) if parts[2] != '0' else 0,
#                     'house_1_jisuan': float(parts[3]) if len(parts) > 3 else 0
#                 })
#
#         df = pd.DataFrame(parsed_data)
#         df['datetime'] = pd.to_datetime(df['datetime'])
#
#     elif isinstance(data, pd.DataFrame):
#         df = data.copy()
#         # 假设第一列是时间列
#         time_col = df.columns[0]
#         df['datetime'] = pd.to_datetime(df[time_col])
#
#     else:
#         print("不支持的数据格式")
#         return
#
#     # 按时间排序
#     df = df.sort_values('datetime').reset_index(drop=True)
#
#     print("=== 时间序列数据检查结果 ===\n")
#
#     # 1. 检查重复时间点
#     print("1. 重复时间点检查:")
#     duplicates = df[df.duplicated(subset=['datetime'], keep=False)]
#     if len(duplicates) > 0:
#         print(f"发现 {len(duplicates)} 个重复时间点:")
#         for datetime_val in duplicates['datetime'].unique():
#             dup_rows = duplicates[duplicates['datetime'] == datetime_val]
#             print(f"  时间: {datetime_val}")
#             for idx, row in dup_rows.iterrows():
#                 print(f"    行 {idx}: {row.to_dict()}")
#         print()
#     else:
#         print("  ✓ 未发现重复时间点\n")
#
#     # 2. 检查缺失时间点
#     print("2. 缺失时间点检查:")
#
#     # 获取时间范围
#     start_time = df['datetime'].min()
#     end_time = df['datetime'].max()
#
#     # 生成完整的时间序列（假设是每小时一次）
#     expected_times = pd.date_range(start=start_time, end=end_time, freq='h')
#     actual_times = set(df['datetime'])
#     expected_times_set = set(expected_times)
#
#     # 找出缺失的时间点
#     missing_times = expected_times_set - actual_times
#
#     if missing_times:
#         missing_times_sorted = sorted(list(missing_times))
#         print(f"发现 {len(missing_times)} 个缺失时间点:")
#         for missing_time in missing_times_sorted:
#             print(f"  缺失: {missing_time}")
#         print()
#     else:
#         print("  ✓ 未发现缺失时间点\n")
#
#     # 3. 数据概览
#     print("3. 数据概览:")
#     print(f"  数据时间范围: {start_time} 到 {end_time}")
#     print(f"  实际数据点数: {len(df)}")
#     print(f"  预期数据点数: {len(expected_times)}")
#     print(f"  数据完整性: {len(df) / len(expected_times) * 100:.1f}%")
#
#     # 4. 时间间隔分析
#     print("\n4. 时间间隔分析:")
#     df_sorted = df.sort_values('datetime')
#     time_diffs = df_sorted['datetime'].diff().dropna()
#
#     if len(time_diffs) > 0:
#         most_common_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.iloc[0]
#         print(f"  最常见时间间隔: {most_common_interval}")
#
#         # 找出异常间隔
#         unusual_intervals = time_diffs[time_diffs != most_common_interval]
#         if len(unusual_intervals) > 0:
#             print(f"  发现 {len(unusual_intervals)} 个异常时间间隔:")
#             for idx, interval in unusual_intervals.items():
#                 prev_time = df_sorted.iloc[idx - 1]['datetime']
#                 curr_time = df_sorted.iloc[idx]['datetime']
#                 print(f"    {prev_time} -> {curr_time}: {interval}")
#
#     return df
#
# sample_data = pd.read_csv('D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\Meteo_dataset_ 分列.csv')
#
#
# # 运行检查
# df_result = check_time_series_data(sample_data)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
import pandas as pd
from datetime import datetime, timedelta
# def process_hourly_data(data_string):#缺失求均值
#     """
#     处理数据，提取整点数据
#     """
#
#
#
#     # 或者如果是从文件读取，使用：
#     df = pd.read_csv('your_file.csv')
#
#     # 转换日期时间列
#     df['Date_2'] = pd.to_datetime(df['Date_2'], format='%Y/%m/%d %H:%M')
#
#     # 设置时间范围
#     start_time = datetime(2020, 5, 1, 0, 0)
#     end_time = datetime(2020, 12, 28, 23, 0)
#
#     # 生成所有需要的整点时间
#     hourly_times = []
#     current_time = start_time
#     while current_time <= end_time:
#         hourly_times.append(current_time)
#         current_time += timedelta(hours=1)
#
#     # 创建结果DataFrame
#     result_df = pd.DataFrame({'Date_1': hourly_times})
#     result_df['house_1_shiji'] = None
#     result_df['house_1_jisuan'] = None
#
#     # 处理每个整点时间
#     for i, target_time in enumerate(hourly_times):
#         # 查找完全匹配的整点数据
#         exact_match = df[df['Date_1'] == target_time]
#
#         if len(exact_match) > 0:
#             # 如果有重复数据，取第一个
#             result_df.loc[i, 'house_1_shiji'] = exact_match.iloc[0]['house_1_shiji']
#             result_df.loc[i, 'house_1_jisuan'] = exact_match.iloc[0]['house_1_jisuan']
#         else:
#             # 如果没有完全匹配，计算该小时内所有数据的平均值
#             hour_start = target_time
#             hour_end = target_time + timedelta(hours=1)
#
#             # 查找该小时内的所有数据（不包括下一个整点）
#             hour_data = df[(df['Date_1'] >= hour_start) & (df['Date_1'] < hour_end)]
#
#             if len(hour_data) > 0:
#                 # 计算该小时内数据的平均值
#                 avg_shiji = hour_data['house_1_shiji'].mean()
#                 avg_jisuan = hour_data['house_1_jisuan'].mean()
#                 result_df.loc[i, 'house_1_shiji'] = avg_shiji
#                 result_df.loc[i, 'house_1_jisuan'] = avg_jisuan
#             else:
#                 # 如果该小时内没有任何数据，使用前后数据进行插值或设为NaN
#                 result_df.loc[i, 'house_1_shiji'] = None
#                 result_df.loc[i, 'house_1_jisuan'] = None
#
#     return result_df
#
#
# def process_hourly_data_from_file(file_path):
#     """
#     从文件读取数据并处理
#     """
#     # 读取CSV文件
#     df = pd.read_csv(file_path)
#
#     # 转换日期时间列
#     df['Date_2'] = pd.to_datetime(df['Date_2'], format='%Y/%m/%d %H:%M')
#
#     # 设置时间范围
#     start_time = datetime(2020, 5, 1, 0, 0)
#     end_time = datetime(2020, 12, 28, 23, 0)
#
#     # 生成所有需要的整点时间
#     hourly_times = []
#     current_time = start_time
#     while current_time <= end_time:
#         hourly_times.append(current_time)
#         current_time += timedelta(hours=1)
#
#     # 创建结果DataFrame
#     result_df = pd.DataFrame({'Date_2': hourly_times})
#     result_df['house_2_shiji'] = None
#     result_df['house_2_jisuan'] = None
#
#     # 处理每个整点时间
#     for i, target_time in enumerate(hourly_times):
#         # 查找完全匹配的整点数据
#         exact_match = df[df['Date_2'] == target_time]
#
#         if len(exact_match) > 0:
#             # 如果有重复数据，取第一个
#             result_df.loc[i, 'house_2_shiji'] = exact_match.iloc[0]['house_2_shiji']
#             result_df.loc[i, 'house_2_jisuan'] = exact_match.iloc[0]['house_2_jisuan']
#         else:
#             # 如果没有完全匹配，查找该整点后面最近的数据
#             after_target = df[df['Date_2'] > target_time].sort_values('Date_2')
#
#             if len(after_target) > 0:
#                 # 取整点后面第一个数据
#                 result_df.loc[i, 'house_2_shiji'] = after_target.iloc[0]['house_2_shiji']
#                 result_df.loc[i, 'house_2_jisuan'] = after_target.iloc[0]['house_2_jisuan']
#             else:
#                 # 如果后面没有数据，用前面最近的数据
#                 before_target = df[df['Date_2'] < target_time].sort_values('Date_2', ascending=False)
#                 if len(before_target) > 0:
#                     result_df.loc[i, 'house_2_shiji'] = before_target.iloc[0]['house_2_shiji']
#                     result_df.loc[i, 'house_2_jisuan'] = before_target.iloc[0]['house_2_jisuan']
#
#     return result_df

# def process_hourly_data_from_file(file_path):#可再生能源
#     """
#     从文件读取数据并处理
#     """
#     # 读取CSV文件
#     df = pd.read_csv(file_path)
#
#     # 转换日期时间列
#     df['Date_4'] = pd.to_datetime(df['Date_4'], format='%Y/%m/%d %H:%M')
#
#     # 设置时间范围
#     start_time = datetime(2020, 5, 1, 0, 0)
#     end_time = datetime(2020, 12, 28, 23, 0)
#
#     # 生成所有需要的整点时间
#     hourly_times = []
#     current_time = start_time
#     while current_time <= end_time:
#         hourly_times.append(current_time)
#         current_time += timedelta(hours=1)
#
#     # 创建结果DataFrame
#     result_df = pd.DataFrame({'Date_4': hourly_times})
#     result_df['Pout'] = None
#     result_df['Ppv1'] = None
#     result_df['Ppv2'] = None
#     result_df['Upv1'] = None
#     result_df['Upv2'] = None
#     result_df['Ubat'] = None
#     result_df['Ibat'] = None
#     result_df['Tbat'] = None
#
#     # 处理每个整点时间
#     for i, target_time in enumerate(hourly_times):
#         # 查找完全匹配的整点数据
#         exact_match = df[df['Date_4'] == target_time]
#
#         if len(exact_match) > 0:
#             # 如果有重复数据，取第一个
#             result_df.loc[i, 'Pout'] = exact_match.iloc[0]['Pout']
#             result_df.loc[i, 'Ppv1'] = exact_match.iloc[0]['Ppv1']
#             result_df.loc[i, 'Ppv2'] = exact_match.iloc[0]['Ppv2']
#             result_df.loc[i, 'Upv1'] = exact_match.iloc[0]['Upv1']
#             result_df.loc[i, 'Upv2'] = exact_match.iloc[0]['Upv2']
#             result_df.loc[i, 'Ubat'] = exact_match.iloc[0]['Ubat']
#             result_df.loc[i, 'Ibat'] = exact_match.iloc[0]['Ibat']
#             result_df.loc[i, 'Tbat'] = exact_match.iloc[0]['Tbat']
#         else:
#             # 如果没有完全匹配，查找该整点后面最近的数据
#             after_target = df[df['Date_4'] > target_time].sort_values('Date_4')
#
#             if len(after_target) > 0:
#                 # 取整点后面第一个数据
#                 result_df.loc[i, 'Pout'] = after_target.iloc[0]['Pout']
#                 result_df.loc[i, 'Ppv1'] = after_target.iloc[0]['Ppv1']
#                 result_df.loc[i, 'Ppv2'] = after_target.iloc[0]['Ppv2']
#                 result_df.loc[i, 'Upv1'] = after_target.iloc[0]['Upv1']
#                 result_df.loc[i, 'Upv2'] = after_target.iloc[0]['Upv2']
#                 result_df.loc[i, 'Ubat'] = after_target.iloc[0]['Ubat']
#                 result_df.loc[i, 'Ibat'] = after_target.iloc[0]['Ibat']
#                 result_df.loc[i, 'Tbat'] = after_target.iloc[0]['Tbat']
#
#             else:
#                 # 如果后面没有数据，用前面最近的数据
#                 before_target = df[df['Date_4'] < target_time].sort_values('Date_4', ascending=False)
#                 if len(before_target) > 0:
#                     result_df.loc[i, 'Pout'] = before_target.iloc[0]['Pout']
#                     result_df.loc[i, 'Ppv1'] = before_target.iloc[0]['Ppv1']
#                     result_df.loc[i, 'Ppv2'] = before_target.iloc[0]['Ppv2']
#                     result_df.loc[i, 'Upv1'] = before_target.iloc[0]['Upv1']
#                     result_df.loc[i, 'Upv2'] = before_target.iloc[0]['Upv2']
#                     result_df.loc[i, 'Ubat'] = before_target.iloc[0]['Ubat']
#                     result_df.loc[i, 'Ibat'] = before_target.iloc[0]['Ibat']
#                     result_df.loc[i, 'Tbat'] = before_target.iloc[0]['Tbat']
#
#     return result_df
# def process_hourly_data_from_file(file_path):#气象
#     """
#     从文件读取数据并处理
#     """
#     # 读取CSV文件
#     df = pd.read_csv(file_path)
#
#     # 转换日期时间列
#     df['Date_5'] = pd.to_datetime(df['Date_5'], format='%Y/%m/%d %H:%M')
#
#     # 设置时间范围
#     start_time = datetime(2020, 5, 1, 0, 0)
#     end_time = datetime(2020, 12, 28, 23, 0)
#
#     # 生成所有需要的整点时间
#     hourly_times = []
#     current_time = start_time
#     while current_time <= end_time:
#         hourly_times.append(current_time)
#         current_time += timedelta(hours=1)
#
#     # 创建结果DataFrame
#     result_df = pd.DataFrame({'Date_5': hourly_times})
#     result_df['Gincl'] = None
#     result_df['Ghi'] = None
#     result_df['Tpv'] = None
#     result_df['Ta'] = None
#     result_df['Rh'] = None
#     result_df['Wd'] = None
#     result_df['Ws'] = None
#     result_df['Press'] = None
#
#     # 处理每个整点时间
#     for i, target_time in enumerate(hourly_times):
#         # 查找完全匹配的整点数据
#         exact_match = df[df['Date_5'] == target_time]
#
#         if len(exact_match) > 0:
#             # 如果有重复数据，取第一个
#             result_df.loc[i, 'Gincl'] = exact_match.iloc[0]['Gincl']
#             result_df.loc[i, 'Ghi'] = exact_match.iloc[0]['Ghi']
#             result_df.loc[i, 'Tpv'] = exact_match.iloc[0]['Tpv']
#             result_df.loc[i, 'Ta'] = exact_match.iloc[0]['Ta']
#             result_df.loc[i, 'Rh'] = exact_match.iloc[0]['Rh']
#             result_df.loc[i, 'Wd'] = exact_match.iloc[0]['Wd']
#             result_df.loc[i, 'Ws'] = exact_match.iloc[0]['Ws']
#             result_df.loc[i, 'Press'] = exact_match.iloc[0]['Press']
#         else:
#             # 如果没有完全匹配，查找该整点后面最近的数据
#             after_target = df[df['Date_5'] > target_time].sort_values('Date_5')
#
#             if len(after_target) > 0:
#                 # 取整点后面第一个数据
#                 result_df.loc[i, 'Gincl'] = after_target.iloc[0]['Gincl']
#                 result_df.loc[i, 'Ghi'] = after_target.iloc[0]['Ghi']
#                 result_df.loc[i, 'Tpv'] = after_target.iloc[0]['Tpv']
#                 result_df.loc[i, 'Ta'] = after_target.iloc[0]['Ta']
#                 result_df.loc[i, 'Rh'] = after_target.iloc[0]['Rh']
#                 result_df.loc[i, 'Wd'] = after_target.iloc[0]['Wd']
#                 result_df.loc[i, 'Ws'] = after_target.iloc[0]['Ws']
#                 result_df.loc[i, 'Press'] = after_target.iloc[0]['Press']
#
#             else:
#                 # 如果后面没有数据，用前面最近的数据
#                 before_target = df[df['Date_5'] < target_time].sort_values('Date_5', ascending=False)
#                 if len(before_target) > 0:
#                     result_df.loc[i, 'Gincl'] = before_target.iloc[0]['Gincl']
#                     result_df.loc[i, 'Ghi'] = before_target.iloc[0]['Ghi']
#                     result_df.loc[i, 'Tpv'] = before_target.iloc[0]['Tpv']
#                     result_df.loc[i, 'Ta'] = before_target.iloc[0]['Ta']
#                     result_df.loc[i, 'Rh'] = before_target.iloc[0]['Rh']
#                     result_df.loc[i, 'Wd'] = before_target.iloc[0]['Wd']
#                     result_df.loc[i, 'Ws'] = before_target.iloc[0]['Ws']
#                     result_df.loc[i, 'Press'] = before_target.iloc[0]['Press']
#
#     return result_df
# # 运行示例
# if __name__ == "__main__":
#     # 处理示例数据
#     result = process_hourly_data_from_file("D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\time_ok_0\\Meteo_dataset_ 分列_filled.csv")
#
#     # 显示前几行结果
#     print("处理后的整点数据:")
#     print(result.head(10))
#     print(f"\n总共处理了 {len(result)} 个整点数据")
#
#     # 保存结果到CSV文件
#     result.to_csv('D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\hourly_data_result.csv', index=False)
#     print("\n结果已保存到 hourly_data_result.csv")
#
#     # 如果要从文件读取数据，使用：
#     # result = process_hourly_data_from_file('your_input_file.csv')

# 数据填充————————————————————————————————————————————————————————————————————————————————————————————
# import pandas as pd
# import numpy as np
# def fill_missing_values_with_duplicate_time_handling(csv_file_path, date_column='Date_5', output_file_path=None):
#     """
#     读取CSV文件并处理缺失值，特别处理重复时间点的情况
#
#     参数:
#     csv_file_path: 输入CSV文件路径
#     date_column: 时间列的列名
#     output_file_path: 输出CSV文件路径（可选）
#     """
#
#     try:
#         # 读取CSV文件
#         df = pd.read_csv(csv_file_path)
#         print(f"成功读取CSV文件，数据形状: {df.shape}")
#         print(f"原始数据缺失值统计:")
#         print(df.isnull().sum())
#
#         # 显示原始数据的前几行
#         print(f"\n原始数据前5行:")
#         print(df.head())
#
#         # 创建副本用于填充
#         df_filled = df.copy()
#
#         # 如果有时间列，先按时间排序
#         if date_column in df.columns:
#             # 尝试转换时间格式
#             try:
#                 df_filled[date_column] = pd.to_datetime(df_filled[date_column])
#                 df_filled = df_filled.sort_values(by=date_column).reset_index(drop=True)
#                 print(f"数据已按 {date_column} 列排序")
#             except:
#                 print(f"警告: 无法解析时间列 {date_column}，将按原始顺序处理")
#
#         # 获取除时间列外的其他列
#         data_columns = [col for col in df_filled.columns if col != date_column]
#
#         # 处理每个数据列的缺失值
#         for col in data_columns:
#             df_filled[col] = fill_column_with_duplicate_time_logic(df_filled, col, date_column)
#
#         print(f"\n填充后数据缺失值统计:")
#         print(df_filled.isnull().sum())
#
#         # 显示填充后数据的前几行
#         print(f"\n填充后数据前5行:")
#         print(df_filled.head())
#
#         # 保存填充后的数据
#         if output_file_path is None:
#             if csv_file_path.endswith('.csv'):
#                 output_file_path = csv_file_path.replace('.csv', '_filled.csv')
#             else:
#                 output_file_path = csv_file_path + '_filled.csv'
#
#         df_filled.to_csv(output_file_path, index=False)
#         print(f"\n填充后的数据已保存到: {output_file_path}")
#
#         return df_filled
#
#     except FileNotFoundError:
#         print(f"错误: 找不到文件 {csv_file_path}")
#         return None
#     except Exception as e:
#         print(f"处理过程中出现错误: {str(e)}")
#         return None
#
#
# def fill_column_with_duplicate_time_logic(df, column, date_column):
#     """
#     对单个列进行缺失值填充，考虑重复时间点的情况
#
#     参数:
#     df: 数据框
#     column: 要填充的列名
#     date_column: 时间列名
#     """
#
#     filled_series = df[column].copy()
#
#     # 找到所有缺失值的位置
#     missing_indices = filled_series.isnull()
#
#     for idx in df[missing_indices].index:
#         if pd.isnull(filled_series.iloc[idx]):
#             current_time = df.iloc[idx][date_column]
#
#             # 情况1: 检查是否存在相同时间点的非缺失值
#             same_time_mask = (df[date_column] == current_time) & (~df[column].isnull())
#             same_time_values = df[same_time_mask]
#
#             if not same_time_values.empty:
#                 # 在相同时间点中找到最近的有效值
#                 # 优先选择索引最接近当前位置的值
#                 closest_idx = same_time_values.index[np.argmin(np.abs(same_time_values.index - idx))]
#                 filled_series.iloc[idx] = df.iloc[closest_idx][column]
#                 print(f"列 {column} 索引 {idx}: 用同时间点索引 {closest_idx} 的值填充")
#             else:
#                 # 情况2: 如果同时间点没有有效值，使用后向填充
#                 for next_idx in range(idx + 1, len(df)):
#                     if not pd.isnull(filled_series.iloc[next_idx]):
#                         filled_series.iloc[idx] = filled_series.iloc[next_idx]
#                         print(f"列 {column} 索引 {idx}: 用后续索引 {next_idx} 的值填充")
#                         break
#                 else:
#                     # 如果后面没有有效值，使用前向填充
#                     for prev_idx in range(idx - 1, -1, -1):
#                         if not pd.isnull(filled_series.iloc[prev_idx]):
#                             filled_series.iloc[idx] = filled_series.iloc[prev_idx]
#                             print(f"列 {column} 索引 {idx}: 用前面索引 {prev_idx} 的值填充")
#                             break
#
#     return filled_series
#
#
# def analyze_duplicate_timestamps(df, date_column='Date_5'):
#     """
#     分析数据中的重复时间点情况
#     """
#     if date_column not in df.columns:
#         print(f"警告: 找不到时间列 {date_column}")
#         return
#
#     # 统计重复时间点
#     time_counts = df[date_column].value_counts()
#     duplicates = time_counts[time_counts > 1]
#
#     print(f"\n时间点重复情况分析:")
#     print(f"总时间点数: {len(time_counts)}")
#     print(f"重复时间点数: {len(duplicates)}")
#
#     if len(duplicates) > 0:
#         print(f"前10个重复最多的时间点:")
#         print(duplicates.head(10))
#
#         # 显示一个重复时间点的详细情况
#         sample_time = duplicates.index[0]
#         sample_data = df[df[date_column] == sample_time]
#         print(f"\n时间点 {sample_time} 的所有记录:")
#         print(sample_data)
#     else:
#         print("没有发现重复时间点")
#
#
# def advanced_fill_missing_values(csv_file_path, date_column='Date_5',
#                                  method='duplicate_aware', output_file_path=None):
#     """
#     高级缺失值填充方法
#
#     参数:
#     csv_file_path: CSV文件路径
#     date_column: 时间列名
#     method: 填充方法 ('duplicate_aware', 'backward', 'forward', 'interpolate')
#     output_file_path: 输出文件路径
#     """
#
#     try:
#         df = pd.read_csv(csv_file_path)
#         print(f"使用 {method} 方法进行缺失值填充")
#
#         # 先分析重复时间点情况
#         analyze_duplicate_timestamps(df, date_column)
#
#         if method == 'duplicate_aware':
#             # 使用我们的自定义方法
#             result = fill_missing_values_with_duplicate_time_handling(csv_file_path, date_column, output_file_path)
#         elif method == 'backward':
#             # 简单后向填充
#             df_filled = df.bfill()
#         elif method == 'forward':
#             # 前向填充
#             df_filled = df.ffill()
#         elif method == 'interpolate':
#             # 插值填充（仅适用于数值列）
#             numeric_columns = df.select_dtypes(include=[np.number]).columns
#             df_filled = df.copy()
#             df_filled[numeric_columns] = df_filled[numeric_columns].interpolate()
#
#         if method != 'duplicate_aware':
#             if output_file_path is None:
#                 output_file_path = csv_file_path.replace('.csv', f'_{method}_filled.csv')
#             df_filled.to_csv(output_file_path, index=False)
#             print(f"填充后的数据已保存到: {output_file_path}")
#             result = df_filled
#
#         return result
#
#     except Exception as e:
#         print(f"处理过程中出现错误: {str(e)}")
#         return None
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 使用示例
#     csv_file = "D:\\LLM\\dataset_all\\净电负荷\\净电负荷\\法国留尼汪岛\\4. 独立户用微电网数据集\\time_ok_0\\Meteo_dataset_ 分列.csv"  # 请替换为您的CSV文件路径
#
#     # 方法1: 使用重复时间点感知的填充方法（推荐）
#     print("=== 使用重复时间点感知填充方法 ===")
#     filled_data = advanced_fill_missing_values(csv_file,
#                                                date_column='Date_5',  # 根据您的数据调整列名
#                                                method='duplicate_aware')
#
#     # 方法2: 直接调用主要函数
#     # filled_data = fill_missing_values_with_duplicate_time_handling(csv_file, 'Date_5')
#
#     if filled_data is not None:
#         print("\n=== 处理完成 ===")
#         print("✓ 重复时间点中的缺失值已用最近的重复值填充")
#         print("✓ 其他缺失值已用后续有效值填充")
#         print("✓ 数据已保存到新文件")
#
#         # 验证填充结果
#         remaining_nulls = filled_data.isnull().sum().sum()
#         if remaining_nulls == 0:
#             print("✓ 所有缺失值已成功填充")
#         else:
#             print(f"⚠ 仍有 {remaining_nulls} 个缺失值未填充")