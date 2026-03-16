import pandas as pd
import numpy as np


def check_missing_values(csv_file_path):
    """
    检查CSV文件中每个特征的空值情况

    参数:
    csv_file_path: CSV文件路径

    返回:
    打印每个特征的空值统计信息
    """

    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
        print(f"数据集形状: {df.shape}")
        print(f"总行数: {len(df)}")
        print(f"总列数: {len(df.columns)}")
        print("-" * 50)

        # 检查每列的空值情况
        missing_info = []

        for column in df.columns:
            # 计算各种类型的"空值"
            null_count = df[column].isnull().sum()  # NaN, None
            empty_str_count = (df[column] == '').sum()  # 空字符串
            whitespace_count = df[column].astype(str).str.strip().eq('').sum()  # 只有空格的字符串

            total_missing = null_count + empty_str_count + whitespace_count
            missing_percentage = (total_missing / len(df)) * 100

            missing_info.append({
                'column': column,
                'null_values': null_count,
                'empty_strings': empty_str_count,
                'whitespace_only': whitespace_count,
                'total_missing': total_missing,
                'missing_percentage': missing_percentage,
                'data_type': str(df[column].dtype)
            })

        # 创建结果DataFrame并排序
        result_df = pd.DataFrame(missing_info)
        result_df = result_df.sort_values('missing_percentage', ascending=False)

        # 打印详细结果
        print("详细空值检查结果:")
        print("=" * 80)
        for _, row in result_df.iterrows():
            print(f"特征: {row['column']}")
            print(f"  数据类型: {row['data_type']}")
            print(f"  空值(NaN/None): {row['null_values']}")
            print(f"  空字符串: {row['empty_strings']}")
            print(f"  仅空格: {row['whitespace_only']}")
            print(f"  总缺失值: {row['total_missing']}")
            print(f"  缺失比例: {row['missing_percentage']:.2f}%")
            print("-" * 40)

        # 总结统计
        print("\n总结统计:")
        print("=" * 50)
        total_features = len(df.columns)
        features_with_missing = len(result_df[result_df['total_missing'] > 0])
        features_without_missing = total_features - features_with_missing

        print(f"总特征数: {total_features}")
        print(f"有缺失值的特征数: {features_with_missing}")
        print(f"无缺失值的特征数: {features_without_missing}")

        if features_with_missing > 0:
            print(f"最高缺失比例: {result_df['missing_percentage'].max():.2f}%")
            print(f"平均缺失比例: {result_df['missing_percentage'].mean():.2f}%")

            # 显示缺失值最多的前5个特征
            print("\n缺失值最多的特征 (前5个):")
            top_missing = result_df.head(5)
            for _, row in top_missing.iterrows():
                if row['total_missing'] > 0:
                    print(f"  {row['column']}: {row['total_missing']} ({row['missing_percentage']:.2f}%)")
        else:
            print("✅ 所有特征都没有缺失值！")

        return result_df

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{csv_file_path}'")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    # # 替换为您的CSV文件路径
    # csv_file_path = "/tmp/zfh_1/net_load_forecasting/datasets/france_net/france_net.csv"
    #
    # # 执行空值检查
    # missing_data_report = check_missing_values(csv_file_path)

    # 如果需要，可以将结果保存到文件
    # if missing_data_report is not None:
    #     missing_data_report.to_csv("missing_values_report.csv", index=False)
    #     print("\n报告已保存到 'missing_values_report.csv'")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # ===== 读取CSV =====
    # 假设文件名是 data.csv，日期列叫 "date"，数值列叫 "value"
    df = pd.read_csv("D:\\LLM\\net_load_forecasting\\datasets\\Rye\\Rye_dan.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)
    ts = df["wind_production"]

    # ===== 可视化分析 =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 原始序列 + 滚动均值/标准差
    ts.plot(ax=axes[0, 0], color="blue", alpha=0.7, label="原始序列")
    ts.rolling(30).mean().plot(ax=axes[0, 0], color="red", label="30日滚动均值")
    ts.rolling(30).std().plot(ax=axes[0, 0], color="green", label="30日滚动标准差")
    axes[0, 0].set_title("时间序列 + 滚动统计")
    axes[0, 0].legend()

    # 2. 自相关函数 (ACF)
    plot_acf(ts.dropna(), lags=40, ax=axes[0, 1])
    axes[0, 1].set_title("自相关函数 (ACF)")

    # 3. 偏自相关函数 (PACF)
    plot_pacf(ts.dropna(), lags=40, ax=axes[1, 0], method="ywm")
    axes[1, 0].set_title("偏自相关函数 (PACF)")

    # 4. 频域分析 (FFT)
    fft_vals = np.fft.fft(ts - ts.mean())
    fft_freq = np.fft.fftfreq(len(ts), d=1)  # d=1 表示采样间隔为1
    axes[1, 1].stem(fft_freq[:len(ts) // 2], np.abs(fft_vals)[:len(ts) // 2], use_line_collection=True)
    axes[1, 1].set_title("频谱分析 (FFT)")
    axes[1, 1].set_xlabel("频率")
    axes[1, 1].set_ylabel("幅值")

    plt.tight_layout()
    plt.show()


# 额外的快速检查函数
def quick_missing_check(csv_file_path):
    """
    快速检查版本 - 只显示有缺失值的特征
    """
    try:
        df = pd.read_csv(csv_file_path)

        # 检查空值
        missing_summary = df.isnull().sum()
        missing_features = missing_summary[missing_summary > 0]

        if len(missing_features) == 0:
            print("✅ 没有发现空值！")
        else:
            print("⚠️  发现以下特征有空值:")
            for feature, count in missing_features.items():
                percentage = (count / len(df)) * 100
                print(f"  {feature}: {count} 个空值 ({percentage:.2f}%)")

        return missing_features

    except Exception as e:
        print(f"错误: {str(e)}")
        return None

# 使用快速检查
# quick_missing_check("your_file.csv")