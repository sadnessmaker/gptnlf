import pandas as pd
from datetime import datetime
def extraction_data(input_data_path,output_data_path):
    load_data = pd.read_csv(input_data_path)
    #提取的特征列
    data = load_data[['utc_timestamp','AT_load_actual_entsoe_transparency','AT_load_forecast_entsoe_transparency','AT_price_day_ahead','AT_solar_generation_actual','AT_wind_onshore_generation_actual']]
    #data.reset_index()
    data.to_csv(output_data_path,index=False)

def convert_time_format(time_str):
        """将ISO 8601时间格式转换为标准格式"""
        try:
            # 解析ISO格式时间 (包含时区信息)
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            # 转换为本地时间并格式化为字符串 (去掉时区信息)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            # 如果解析失败，返回原始字符串
            return time_str
def extrac_row(input_data_path,index_column,start,end,output_data_path):
        data_data=pd.read_csv(input_data_path)

    # if index_column not in data_data.columns:
    #     raise ValueError(f"列名 '{index_column}' 不存在于CSV文件中")

        # 转换时间格式
        data_data[index_column] = data_data[index_column].apply(convert_time_format)

        # 将时间列转换为datetime类型以便比较
        data_data['utc_timestamp'] = pd.to_datetime(data_data[index_column])

        # 将输入的时间字符串转换为datetime对象
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # 提取指定时间范围内的数据
        mask = (data_data['utc_timestamp'] >= start_dt) & (data_data['utc_timestamp'] <= end_dt)
        subtable = data_data[mask].copy()



        # 可选：保存结果到新CSV文件
        if output_data_path:
            subtable.to_csv(output_data_path, index=False)
            print(f"处理后的子表已保存至: {output_data_path}")

        return subtable

def weather_process(input,output,start,end,column):
    df=pd.read_csv(input)
    data=df[['utc_timestamp','AT_temperature','AT_radiation_direct_horizontal','AT_radiation_diffuse_horizontal']]
    data[column] = data[column].apply(convert_time_format)

    # 将时间列转换为datetime类型以便比较
    data['utc_timestamp'] = pd.to_datetime(data[column])

    # 将输入的时间字符串转换为datetime对象
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # 提取指定时间范围内的数据
    mask = (data['utc_timestamp'] >= start_dt) & (data['utc_timestamp'] <= end_dt)
    subtable = data[mask].copy()

    # 可选：保存结果到新CSV文件
    if output:
        subtable.to_csv(output, index=False)
        print(f"处理后的子表已保存至: {output}")

    return subtable
# if __name__ == '__main__':
#     pass
#     #extraction_data提取指定列
#     # input_data_path='D:/LLM/net_load_forecasting/datasets/OPSD/time_series_60min_singleindex.csv'
#     # out_data_path='D:/LLM/net_load_forecasting/datasets/OPSD/AT_data.csv'
#     #extraction_data(input_data_path,out_data_path)
#
#
#
#     # #extrac_raw选指定行
#     # input_data_path = 'D:/LLM/net_load_forecasting/datasets/OPSD/AT_data.csv'
#     # out_data_path = 'D:/LLM/net_load_forecasting/datasets/OPSD/AT_data_18-19.csv'
#     # index_column='utc_timestamp'
#     # start='2018-01-01 00:00:00'
#     # end='2019-12-31 23:00:00'
#     # result=extrac_row(input_data_path,index_column,start,end,out_data_path)
#     # print(result)
#
#     #天气处理
#     # input_data_path = 'D:/LLM/net_load_forecasting/datasets/OPSD/weather_data.csv'
#     # out_data_path = 'D:/LLM/net_load_forecasting/datasets/OPSD/weather_AT_18-19.csv'
#     # start='2018-01-01 00:00:00'
#     # end='2019-12-31 23:00:00'
#     # index_column='utc_timestamp'
#     # weather_process(input_data_path,out_data_path,start,end,index_column)
#
#     # #数据合并负荷数据+天气
#     # load_data=pd.read_csv('D:/LLM/net_load_forecasting/datasets/OPSD/AT_data_18-19.csv')
#     # weather_data=pd.read_csv('D:/LLM/net_load_forecasting/datasets/OPSD/weather_AT_18-19.csv')
#     # extracted_column=['AT_temperature','AT_radiation_direct_horizontal','AT_radiation_diffuse_horizontal']
#     # extracted_data=weather_data[extracted_column]
#     # #横向添加列
#     # reasult=pd.concat([load_data,extracted_data],axis=1)
#     # reasult.to_csv('D:/LLM/net_load_forecasting/datasets/OPSD/final_net_load.csv',index=False)
#
#     #换成指定列的顺序
#     # load_data = pd.read_csv('D:/LLM/net_load_forecasting/datasets/OPSD/finally_net_load.csv')
#     # load_data=load_data[['utc_timestamp','net_load','AT_load_actual_entsoe_transparency','AT_solar_generation_actual',
#     #                      'AT_wind_onshore_generation_actual','AT_load_forecast_entsoe_transparency',
#     #                      'AT_price_day_ahead','AT_temperature','AT_radiation_direct_horizontal','AT_radiation_diffuse_horizontal']]
#     # load_data.to_csv('D:/LLM/net_load_forecasting/datasets/OPSD/finally_net_load.csv',index=False)
#     import pandas as pd
#     import numpy as np
#
#
#     def check_csv_null_values(file_path):
#         """
#         检查CSV文件中的空值和NaN值
#
#         参数:
#         file_path (str): CSV文件路径
#
#         返回:
#         dict: 包含空值统计信息的字典
#         """
#         try:
#             # 读取CSV文件
#             df = pd.read_csv(file_path)
#
#             print(f"文件: {file_path}")
#             print(f"数据形状: {df.shape}")
#             print("-" * 50)
#
#             # 检查每列的空值数量
#             null_counts = df.isnull().sum()
#
#             # 检查每列的空值百分比
#             null_percentages = (df.isnull().sum() / len(df)) * 100
#
#             # 创建汇总报告
#             null_summary = pd.DataFrame({
#                 '空值数量': null_counts,
#                 '空值百分比': null_percentages.round(2)
#             })
#
#             # 只显示有空值的列
#             has_nulls = null_summary[null_summary['空值数量'] > 0]
#
#             if len(has_nulls) > 0:
#                 print("发现空值的列:")
#                 print(has_nulls)
#                 print("-" * 50)
#
#                 # 显示总体空值统计
#                 total_cells = df.shape[0] * df.shape[1]
#                 total_nulls = df.isnull().sum().sum()
#                 print(f"总单元格数: {total_cells}")
#                 print(f"总空值数: {total_nulls}")
#                 print(f"总空值百分比: {(total_nulls / total_cells) * 100:.2f}%")
#
#                 # 显示包含空值的行数
#                 rows_with_nulls = df.isnull().any(axis=1).sum()
#                 print(f"包含空值的行数: {rows_with_nulls}")
#                 print(f"包含空值的行百分比: {(rows_with_nulls / len(df)) * 100:.2f}%")
#
#             else:
#                 print("✅ 没有发现空值或NaN值")
#
#             # 检查特定的空值类型
#             print("\n" + "=" * 50)
#             print("详细空值类型检查:")
#
#             # 检查空字符串
#             empty_strings = (df == '').sum()
#             if empty_strings.sum() > 0:
#                 print("\n空字符串 ('') 统计:")
#                 print(empty_strings[empty_strings > 0])
#
#             # 检查只包含空格的字符串
#             whitespace_only = df.apply(lambda x: x.astype(str).str.strip() == '').sum()
#             if whitespace_only.sum() > 0:
#                 print("\n只包含空格的字符串统计:")
#                 print(whitespace_only[whitespace_only > 0])
#
#             return {
#                 'null_summary': null_summary,
#                 'has_nulls': len(has_nulls) > 0,
#                 'total_nulls': total_nulls,
#                 'null_percentage': (total_nulls / total_cells) * 100
#             }
#
#         except FileNotFoundError:
#             print(f"错误: 找不到文件 '{file_path}'")
#             return None
#         except Exception as e:
#             print(f"错误: {str(e)}")
#             return None
#
#
#     def check_specific_columns(file_path, columns):
#         """
#         检查指定列的空值情况
#
#         参数:
#         file_path (str): CSV文件路径
#         columns (list): 要检查的列名列表
#         """
#         try:
#             df = pd.read_csv(file_path)
#
#             print(f"检查指定列的空值情况:")
#             print("-" * 30)
#
#             for col in columns:
#                 if col in df.columns:
#                     null_count = df[col].isnull().sum()
#                     null_pct = (null_count / len(df)) * 100
#                     print(f"{col}: {null_count} 个空值 ({null_pct:.2f}%)")
#
#                     # 显示空值位置（前10个）
#                     if null_count > 0:
#                         null_indices = df[df[col].isnull()].index.tolist()[:10]
#                         print(f"  空值位置(前10个): {null_indices}")
#                 else:
#                     print(f"警告: 列 '{col}' 不存在")
#
#         except Exception as e:
#             print(f"错误: {str(e)}")






if __name__ == "__main__":
        # 检查整个CSV文件
        # file_path = "D:/LLM/net_load_forecasting/datasets/OPSD/finally_net_load.csv"  # 替换为你的CSV文件路径
        # result = check_csv_null_values(file_path)

        # 检查特定列
        # check_specific_columns(file_path, ['column1', 'column2'])

        # 如果需要处理空值，可以使用以下方法：
        # df = pd.read_csv(file_path)
        #
        # # 删除包含空值的行
        # df_cleaned = df.dropna()
        #
        # # 用特定值填充空值
        # df_filled = df.fillna(0)  # 用0填充数值列
        # df_filled = df.fillna('未知')  # 用'未知'填充字符串列
        #
        # # 用前一个值填充
        # df_forward_filled = df.fillna(method='ffill')
        #
        # # 用后一个值填充
        # df_backward_filled = df.fillna(method='bfill')

        # #_________________________BE________________________改时间
        # 原始文件路径
        # 读取CSV文件


        df = pd.read_csv("D:\\LLM\\net_load_forecasting\\datasets\\OPSD\\BE\\BE_orgin\\BE.csv")

        # 将时间列解析为datetime类型
        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'], utc=True)

        # 转换为所需格式字符串 YYYY-MM-DD HH:MM:SS
        df['utc_timestamp'] = df['utc_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # 设定时间范围
        start_time = "2018-01-01 00:00:00"
        end_time = "2019-12-31 23:00:00"

        # 筛选指定时间段的数据
        filtered_df = df[(df['utc_timestamp'] >= start_time) & (df['utc_timestamp'] <= end_time)].copy()

        # 对筛选后的数据进行线性插值（使用前后非空值的线性插值）
        filtered_df['BE_wind_generation_actual'] = filtered_df['BE_wind_generation_actual'].interpolate(
            method='linear',
            limit_direction='both'  # 可以向前向后填充
        )

        # 保存到新的CSV文件
        filtered_df.to_csv("D:\\LLM\\net_load_forecasting\\datasets\\OPSD\\BE\\BE_orgin\\BE_time_nofill_1.csv", index=False)

        print("筛选完成，保存到 BE_time_nofill_1.csv")
        # import pandas as pd
        # import numpy as np
        #
        # # 读取数据
        # df = pd.read_csv(r"D:\LLM\net_load_forecasting\datasets\OPSD\BE\BE_orgin\BE.csv")
        #
        # # 解析时间
        # df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'], utc=True)
        #
        # # 设定时间范围
        # start_time = pd.to_datetime("2018-01-01 00:00:00", utc=True)
        # end_time = pd.to_datetime("2019-12-31 23:00:00", utc=True)
        #
        # # 筛选数据
        # filtered_df = df[(df['utc_timestamp'] >= start_time) & (df['utc_timestamp'] <= end_time)].copy()
        #
        # # 查看填充前的空值情况
        # print("填充前的空值统计：")
        # print(filtered_df.isna().sum())
        #
        # # ===== 对指定列使用前后平均值填充 =====
        # # 假设要填充 'AT' 列（替换成你需要的列名）
        # column_name = 'BE_wind_generation_actual'
        #
        # # 先前向填充创建临时列
        # filtered_df['forward'] = filtered_df[column_name].ffill()
        # filtered_df['backward'] = filtered_df[column_name].bfill()
        #
        # # 取平均值填充空值
        # filtered_df[column_name] = filtered_df[column_name].fillna(
        #     (filtered_df['forward'] + filtered_df['backward']) / 2
        # )
        #
        # # 删除临时列
        # filtered_df = filtered_df.drop(['forward', 'backward'], axis=1)
        #
        # # 查看填充后的空值情况
        # print("\n填充后的空值统计：")
        # print(filtered_df.isna().sum())
        #
        # # 转换时间格式
        # filtered_df['utc_timestamp'] = filtered_df['utc_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        #
        # # 保存
        # output_path = r"D:\LLM\net_load_forecasting\datasets\OPSD\BE\BE_orgin\BE_time_avg_filled.csv"
        # filtered_df.to_csv(output_path, index=False)
        #
        # print(f"\n筛选完成，共 {len(filtered_df)} 条记录")
        # print(f"保存到: {output_path}")
        #气象的时间处理
        # df = pd.read_csv("D:\\LLM\\net_load_forecasting\\datasets\\OPSD\\GE\\天气\\处理\\DE-wind_speed_pop_wtd-merra2.csv")
        #
        # # 将时间列解析为datetime类型（自动识别 '2014-12-31T23:00:00Z'）
        # df['time'] = pd.to_datetime(df['time'], utc=True)
        #
        # # 转换为所需格式字符串 YYYY-MM-DD HH:MM:SS
        # df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        #
        # # 设定时间范围
        # start_time = "2018-01-01 00:00:00"
        # end_time = "2019-12-31 23:00:00"
        #
        # # 筛选指定时间段的数据
        # filtered_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        #
        # # 只保留时间列和 AT 列
        # filtered_df = filtered_df[['time', 'DE']]
        #
        # # 保存到新的CSV文件
        # filtered_df.to_csv("D:\\LLM\\net_load_forecasting\\datasets\\OPSD\\GE\\天气\\处理\\timeover\\DE-wind_speed.csv", index=False)
        #
        # print("筛选完成，保存到 BE_data_time.csv")