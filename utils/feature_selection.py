import pandas as pd
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import csv
import os
warnings.filterwarnings('ignore')

#df = pd.read_csv("/tmp/zfh_1/net_load_forecasting/datasets/Rye/Rye_all.csv")#rye
df = pd.read_csv("/tmp/zfh_1/net_load_forecasting/datasets/OPSD/net_load.csv")#AT
# 定义目标变量和特征变量
#target_vars = ['net_load', 'consumption', 'pv_production', 'wind_production']#rye
# target_vars = ['net_load', 'KW', 're_energy']
target_vars = ['net_load', 'AT_load_actual_entsoe_transparency', 'AT_solar_generation_actual', 'AT_wind_onshore_generation_actual']#AT
feature_vars = [col for col in df.columns if col not in ['date'] + target_vars]

print("=== 气象特征选择分析 ===")
print(f"目标变量: {target_vars}")
print(f"候选特征数量: {len(feature_vars)}")
print(f"特征变量: {feature_vars}")


class WeatherFeatureSelector:
    def __init__(self, df, target_vars, feature_vars, save_path='/tmp/zfh_1/net_load_forecasting/PIC/feature/AT'):
        self.df = df
        self.target_vars = target_vars
        self.feature_vars = feature_vars
        self.results = {}
        self.save_path = save_path
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)

    def calculate_correlation(self):
        """计算皮尔逊相关系数"""
        correlation_results = {}

        for target in self.target_vars:
            correlations = []
            for feature in self.feature_vars:
                try:
                    # 检查是否存在方差
                    if self.df[feature].var() == 0 or self.df[target].var() == 0:
                        corr = 0
                        p_value = 1.0
                    else:
                        corr, p_value = pearsonr(self.df[feature], self.df[target])

                    correlations.append({
                        'feature': feature,
                        'correlation': corr,
                        'abs_correlation': abs(corr),
                        'p_value': p_value
                    })
                except Exception as e:
                    correlations.append({
                        'feature': feature,
                        'correlation': 0,
                        'abs_correlation': 0,
                        'p_value': 1.0
                    })

            # 按绝对相关系数排序
            correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            correlation_results[target] = correlations

        return correlation_results
    def calculate_f_score(self):
        """计算F统计量"""
        f_score_results = {}

        for target in self.target_vars:
            try:
                # 准备数据
                X = self.df[self.feature_vars].fillna(0)
                y = self.df[target].fillna(0)

                # 计算F统计量
                f_scores, p_values = f_regression(X, y)

                f_results = []
                for i, feature in enumerate(self.feature_vars):
                    f_results.append({
                        'feature': feature,
                        'f_score': f_scores[i],
                        'p_value': p_values[i]
                    })

                # 按F统计量排序
                f_results.sort(key=lambda x: x['f_score'], reverse=True)
                f_score_results[target] = f_results

            except Exception as e:
                print(f"计算 {target} 的F统计量时出错: {e}")
                f_score_results[target] = []

        return f_score_results
#_____________________________________________________________________
    def calculate_spearman_correlation(self):
        """计算斯皮尔曼等级相关系数"""
        spearman_results = {}

        for target in self.target_vars:
            correlations = []
            for feature in self.feature_vars:
                try:
                    if self.df[feature].var() == 0 or self.df[target].var() == 0:
                        corr = 0
                        p_value = 1.0
                    else:
                        corr, p_value = spearmanr(self.df[feature], self.df[target])

                    correlations.append({
                        'feature': feature,
                        'spearman_corr': corr,
                        'abs_spearman_corr': abs(corr),
                        'p_value': p_value
                    })
                except Exception as e:
                    correlations.append({
                        'feature': feature,
                        'spearman_corr': 0,
                        'abs_spearman_corr': 0,
                        'p_value': 1.0
                    })

            correlations.sort(key=lambda x: x['abs_spearman_corr'], reverse=True)
            spearman_results[target] = correlations

        return spearman_results

    def calculate_kendall_correlation(self):
        """计算肯德尔等级相关系数"""
        kendall_results = {}

        for target in self.target_vars:
            correlations = []
            for feature in self.feature_vars:
                try:
                    if self.df[feature].var() == 0 or self.df[target].var() == 0:
                        corr = 0
                        p_value = 1.0
                    else:
                        corr, p_value = kendalltau(self.df[feature], self.df[target])

                    correlations.append({
                        'feature': feature,
                        'kendall_corr': corr,
                        'abs_kendall_corr': abs(corr),
                        'p_value': p_value
                    })
                except Exception as e:
                    correlations.append({
                        'feature': feature,
                        'kendall_corr': 0,
                        'abs_kendall_corr': 0,
                        'p_value': 1.0
                    })

            correlations.sort(key=lambda x: x['abs_kendall_corr'], reverse=True)
            kendall_results[target] = correlations

        return kendall_results
#_________________________________________________________________________________
    def calculate_distance_correlation(self):
        """计算距离相关性（简化版）"""

        def dcorr(x, y):
            """计算距离相关系数"""
            try:
                # 计算距离矩阵
                n = len(x)
                a = np.abs(x[:, None] - x[None, :])
                b = np.abs(y[:, None] - y[None, :])

                # 中心化距离矩阵
                A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
                B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

                # 计算距离相关系数
                dcov2_xy = (A * B).sum() / (n * n)
                dcov2_xx = (A * A).sum() / (n * n)
                dcov2_yy = (B * B).sum() / (n * n)

                dcorr = np.sqrt(dcov2_xy / np.sqrt(dcov2_xx * dcov2_yy)) if (dcov2_xx * dcov2_yy) > 0 else 0
                return dcorr
            except:
                return 0

        distance_results = {}

        for target in self.target_vars:
            correlations = []
            for feature in self.feature_vars:
                try:
                    x = self.df[feature].fillna(0).values
                    y = self.df[target].fillna(0).values

                    dcorr_value = dcorr(x, y)

                    correlations.append({
                        'feature': feature,
                        'distance_corr': dcorr_value
                    })
                except Exception as e:
                    correlations.append({
                        'feature': feature,
                        'distance_corr': 0
                    })

            correlations.sort(key=lambda x: x['distance_corr'], reverse=True)
            distance_results[target] = correlations

        return distance_results

    def calculate_mutual_info(self):
        """计算互信息"""
        mutual_info_results = {}

        for target in self.target_vars:
            try:
                # 准备数据
                X = self.df[self.feature_vars].fillna(0)
                y = self.df[target].fillna(0)

                # 计算互信息
                mi_scores = mutual_info_regression(X, y, random_state=42)

                mi_results = []
                for i, feature in enumerate(self.feature_vars):
                    mi_results.append({
                        'feature': feature,
                        'mutual_info': mi_scores[i]
                    })

                # 按互信息排序
                mi_results.sort(key=lambda x: x['mutual_info'], reverse=True)
                mutual_info_results[target] = mi_results

            except Exception as e:
                print(f"计算 {target} 的互信息时出错: {e}")
                mutual_info_results[target] = []

        return mutual_info_results

    def save_top_features_to_csv(self, top_n=10):
        """将前N个最相关特征保存到CSV文件"""
        for target in self.target_vars:
            # 创建保存文件的路径
            csv_file = os.path.join(self.save_path, f'{target}_top_{top_n}_features.csv')

            with open(csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # 写入标题行
                writer.writerow([f'{target} - 前{top_n}个最相关特征分析结果'])
                writer.writerow([])  # 空行

                # 皮尔逊相关系数
                writer.writerow(['皮尔逊相关系数排名'])
                writer.writerow(['排名', '特征名称', '相关系数', 'p值'])
                if target in self.results['correlation']:
                    for i, item in enumerate(self.results['correlation'][target][:top_n]):
                        writer.writerow(
                            [i + 1, item['feature'], f"{item['correlation']:.4f}", f"{item['p_value']:.4f}"])
                writer.writerow([])  # 空行

                # F统计量
                writer.writerow(['F统计量排名'])
                writer.writerow(['排名', '特征名称', 'F统计量', 'p值'])
                if target in self.results['f_score']:
                    for i, item in enumerate(self.results['f_score'][target][:top_n]):
                        writer.writerow([i + 1, item['feature'], f"{item['f_score']:.4f}", f"{item['p_value']:.4f}"])
                writer.writerow([])  # 空行

                # 斯皮尔曼等级相关系数
                writer.writerow(['斯皮尔曼等级相关系数排名'])
                writer.writerow(['排名', '特征名称', '相关系数', 'p值'])
                if target in self.results['spearman_results']:
                    for i, item in enumerate(self.results['spearman_results'][target][:top_n]):
                        writer.writerow(
                            [i + 1, item['feature'], f"{item['spearman_corr']:.4f}", f"{item['p_value']:.4f}"])
                writer.writerow([])  # 空行

                # 肯德尔等级相关系数
                writer.writerow(['肯德尔等级相关系数排名'])
                writer.writerow(['排名', '特征名称', '相关系数', 'p值'])
                if target in self.results['kendall_results']:
                    for i, item in enumerate(self.results['kendall_results'][target][:top_n]):
                        writer.writerow(
                            [i + 1, item['feature'], f"{item['kendall_corr']:.4f}", f"{item['p_value']:.4f}"])
                writer.writerow([])  # 空行

                # 互信息
                writer.writerow(['互信息排名'])
                writer.writerow(['排名', '特征名称', '互信息值'])
                if target in self.results['mutual_info']:
                    for i, item in enumerate(self.results['mutual_info'][target][:top_n]):
                        writer.writerow([i + 1, item['feature'], f"{item['mutual_info']:.4f}"])
                writer.writerow([])  # 空行

                # 距离相关系数
                writer.writerow(['距离相关系数排名'])
                writer.writerow(['排名', '特征名称', '距离相关系数'])
                if target in self.results['distance_results']:
                    for i, item in enumerate(self.results['distance_results'][target][:top_n]):
                        writer.writerow([i + 1, item['feature'], f"{item['distance_corr']:.4f}"])

                print(f"已保存 {target} 的前{top_n}个特征分析结果到: {csv_file}")

    def run_analysis(self):
        """运行完整的特征选择分析"""
        print("\n=== 1. 皮尔逊相关系数分析 ===")
        correlation_results = self.calculate_correlation()

        print("\n=== 2. F统计量分析 ===")
        f_score_results = self.calculate_f_score()

        print("\n=== 3. 斯皮尔曼等级相关系数分析 ===")
        spearman_results = self.calculate_spearman_correlation()

        print("\n=== 4. 肯德尔等级相关系数分析 ===")
        kendall_results = self.calculate_kendall_correlation()

        print("\n=== 5. 互信息分析 ===")
        mutual_info_results = self.calculate_mutual_info()

        print("\n=== 6. 距离相关系数分析 ===")
        distance_results = self.calculate_distance_correlation()



        # 存储结果
        self.results = {
            'correlation': correlation_results,'f_score':f_score_results,
            'spearman_results': spearman_results,'kendall_results':kendall_results,
            'mutual_info': mutual_info_results,'distance_results':distance_results


        }

        return self.results

    def print_top_features(self, top_n=10):
        """打印每个目标变量的前N个最相关特征"""
        for target in self.target_vars:
            print(f"\n=== {target} 的前{top_n}个最相关特征 ===")

            # 皮尔逊相关系数
            print(f"\n【皮尔逊相关系数】")
            if target in self.results['correlation']:
                for i, item in enumerate(self.results['correlation'][target][:top_n]):
                    print(f"{i + 1:2d}. {item['feature']:<30} 相关系数: {item['correlation']:>7.4f} (p={item['p_value']:.4f})")

            # F统计量
            print(f"\n【F统计量】")
            if target in self.results['f_score']:
                for i, item in enumerate(self.results['f_score'][target][:top_n]):
                    print(f"{i + 1:2d}. {item['feature']:<30} F统计量: {item['f_score']:>7.4f} (p={item['p_value']:.4f})")

            # 斯皮尔曼等级相关系数
            print(f"\n【斯皮尔曼等级相关系数】")
            if target in self.results['spearman_results']:
                for i, item in enumerate(self.results['spearman_results'][target][:top_n]):
                    print(f"{i + 1:2d}. {item['feature']:<30} 相关系数: {item['spearman_corr']:>7.4f} (p={item['p_value']:.4f})")

            # 肯德尔等级相关系数
            print(f"\n【肯德尔等级相关系数】")
            if target in self.results['kendall_results']:
                for i, item in enumerate(self.results['kendall_results'][target][:top_n]):
                    print(f"{i + 1:2d}. {item['feature']:<30} 相关系数: {item['kendall_corr']:>7.4f} (p={item['p_value']:.4f})")
            # 互信息
            print(f"\n【互信息】")
            if target in self.results['mutual_info']:
                for i, item in enumerate(self.results['mutual_info'][target][:top_n]):
                    print(f"{i + 1:2d}. {item['feature']:<30} 互信息: {item['mutual_info']:>7.4f}")

            # 距离相关系数
            print(f"\n【距离相关系数】")
            if target in self.results['distance_results']:
                for i, item in enumerate(self.results['distance_results'][target][:top_n]):
                    print(f"{i + 1:2d}. {item['feature']:<30} 互信息: {item['distance_corr']:>7.4f}")

    def save_consensus_features_to_csv(self, top_n=4):
        """将综合排名前N个特征保存到CSV文件"""
        csv_file = os.path.join(self.save_path, f'consensus_top_{top_n}_features.csv')

        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # 写入标题行
            writer.writerow([f'综合特征重要性排名 (前{top_n}个)'])
            writer.writerow([])  # 空行

            for target in self.target_vars:
                writer.writerow([f'{target} - 综合排名'])
                writer.writerow(['排名', '特征名称', '平均分', '方法数'])

                # 收集各方法的排名
                feature_scores = {}

                # 各种方法的贡献
                methods = ['correlation', 'f_score', 'spearman_results', 'kendall_results', 'mutual_info',
                           'distance_results']

                for method in methods:
                    if target in self.results[method]:
                        for i, item in enumerate(self.results[method][target][:top_n]):
                            feature = item['feature']
                            if feature not in feature_scores:
                                feature_scores[feature] = {'count': 0, 'total_score': 0}
                            feature_scores[feature]['count'] += 1
                            feature_scores[feature]['total_score'] += (top_n - i)

                # 计算综合得分并排序
                consensus_features = []
                for feature, scores in feature_scores.items():
                    avg_score = scores['total_score'] / scores['count']
                    consensus_features.append({
                        'feature': feature,
                        'avg_score': avg_score,
                        'method_count': scores['count']
                    })

                consensus_features.sort(key=lambda x: (x['method_count'], x['avg_score']), reverse=True)


                # 写入结果
                for i, item in enumerate(consensus_features[:top_n]):
                    writer.writerow([i + 1, item['feature'], f"{item['avg_score']:.2f}", item['method_count']])

                writer.writerow([])  # 空行

        print(f"已保存综合排名前{top_n}个特征到: {csv_file}")

    def get_consensus_features(self, top_n=4):
        """获取多种方法一致认为重要的特征"""
        print(f"\n=== 综合特征重要性排名 (前{top_n}个) ===")

        for target in self.target_vars:
            print(f"\n【{target}】")

            # 收集各方法的排名
            feature_scores = {}

            # 皮尔逊相关系数贡献
            if target in self.results['correlation']:
                for i, item in enumerate(self.results['correlation'][target][:top_n]):
                    feature = item['feature']
                    if feature not in feature_scores:
                        feature_scores[feature] = {'count': 0, 'total_score': 0}
                    feature_scores[feature]['count'] += 1
                    feature_scores[feature]['total_score'] += (top_n - i)  # 排名越高分数越高

            # F统计量贡献
            if target in self.results['f_score']:
                for i, item in enumerate(self.results['f_score'][target][:top_n]):
                    feature = item['feature']
                    if feature not in feature_scores:
                        feature_scores[feature] = {'count': 0, 'total_score': 0}
                    feature_scores[feature]['count'] += 1
                    feature_scores[feature]['total_score'] += (top_n - i)

            # 斯皮尔曼等级相关系数贡献
            if target in self.results['spearman_results']:
                for i, item in enumerate(self.results['spearman_results'][target][:top_n]):
                    feature = item['feature']
                    if feature not in feature_scores:
                        feature_scores[feature] = {'count': 0, 'total_score': 0}
                    feature_scores[feature]['count'] += 1
                    feature_scores[feature]['total_score'] += (top_n - i)  # 排名越高分数越高

            # 肯德尔等级相关系数贡献
            if target in self.results['kendall_results']:
                for i, item in enumerate(self.results['kendall_results'][target][:top_n]):
                    feature = item['feature']
                    if feature not in feature_scores:
                        feature_scores[feature] = {'count': 0, 'total_score': 0}
                    feature_scores[feature]['count'] += 1
                    feature_scores[feature]['total_score'] += (top_n - i)  # 排名越高分数越高

            # 互信息贡献
            if target in self.results['mutual_info']:
                for i, item in enumerate(self.results['mutual_info'][target][:top_n]):
                    feature = item['feature']
                    if feature not in feature_scores:
                        feature_scores[feature] = {'count': 0, 'total_score': 0}
                    feature_scores[feature]['count'] += 1
                    feature_scores[feature]['total_score'] += (top_n - i)

            # 距离相关系数
            if target in self.results['distance_results']:
                for i, item in enumerate(self.results['distance_results'][target][:top_n]):
                    feature = item['feature']
                    if feature not in feature_scores:
                        feature_scores[feature] = {'count': 0, 'total_score': 0}
                    feature_scores[feature]['count'] += 1
                    feature_scores[feature]['total_score'] += (top_n - i)

            #print("feature_scores:",feature_scores)
            # 计算综合得分并排序
            consensus_features = []
            for feature, scores in feature_scores.items():
                avg_score = scores['total_score'] / scores['count']
                # 先将原始平均分保留两位小数
                avg_score = round(avg_score, 2)
                consensus_features.append({
                    'feature': feature,
                    'avg_score': avg_score,
                    'method_count': scores['count']
                })

            consensus_features.sort(key=lambda x: (x['method_count'], x['avg_score']), reverse=True)
            # 获取前top_n个特征
            top_features = consensus_features[:top_n]

            # 线性标准化：使前top_n个特征的分数和为1
            if top_features:
                total_score = sum(item['avg_score'] for item in top_features)
                if total_score > 0:
                    # 先计算标准化分数
                    for item in top_features:
                        item['normalized_score'] = item['avg_score'] / total_score

                    # 确保分数和严格等于1（处理浮点数精度问题）
                    actual_sum = sum(item['normalized_score'] for item in top_features)
                    if abs(actual_sum - 1.0) > 1e-10:  # 如果误差超过极小值
                        # 调整最大分数的特征，确保总和为1
                        max_item = max(top_features, key=lambda x: x['normalized_score'])
                        max_item['normalized_score'] += (1.0 - actual_sum)
                else:
                    # 如果总分为0，平均分配
                    for item in top_features:
                        item['normalized_score'] = 1.0 / len(top_features)

            # 打印结果
            print("排名  特征名称                        标准化分数   原始平均分   方法数")
            print("-" * 75)
            total_normalized = 0
            for i, item in enumerate(top_features):
                normalized_score = item.get('normalized_score', 0)
                total_normalized += normalized_score
                print(
                    f"{i + 1:2d}.  {item['feature']:<30} {normalized_score:>8.2f}   {item['avg_score']:>8.2f}   {item['method_count']:>4d}")

            print("-" * 75)
            print(f"     标准化分数总和:{' ' * 20} {total_normalized:>8.2f}")

            # 调试信息：显示实际精确的总和
            if top_features:
                precise_sum = sum(item.get('normalized_score', 0) for item in top_features)
                print(f"     (精确总和: {precise_sum:.10f})")
            print()
            print(top_features)
            # # 打印结果
            # for i, item in enumerate(consensus_features[:top_n]):
            #     print(f"{i + 1:2d}. {item['feature']:<30} 平均分: {item['avg_score']:>5.2f} 方法数: {item['method_count']}")
            #     # with open('/tmp/zfh_1/net_load_forecasting/PIC/zonghe.csv', 'a', newline='') as f:
            #     #     f.write(f"{i + 1:2d}. {item['feature']:<30} 平均分: {item['avg_score']:>5.2f} 方法数: {item['method_count']}\n")

    def save_all_results(self, top_n=10, consensus_n=4):
        """保存所有分析结果"""
        print(f"\n=== 保存分析结果 ===")

        # 保存前N个特征分析结果
        self.save_top_features_to_csv(top_n)

        # 保存综合排名结果
        self.save_consensus_features_to_csv(consensus_n)

        print(f"所有分析结果已保存到: {self.save_path}")

# 运行分析
selector = WeatherFeatureSelector(df, target_vars, feature_vars)
results = selector.run_analysis()
selector.print_top_features(top_n=10)
selector.get_consensus_features(top_n=4)
# 保存所有结果到CSV文件
selector.save_all_results(top_n=10, consensus_n=4)
# with open('/tmp/zfh_1/net_load_forecasting/PIC/zonghe.csv', 'a', newline='') as f:
#     f.write(str(zonghe)+'\n')
# print("\n=== 分析说明 ===")
# print("1. 皮尔逊相关系数: 衡量线性相关性，值越接近±1表示相关性越强")
# print("2. 互信息: 衡量非线性相关性，值越大表示包含的信息越多")
# print("3. F统计量: 衡量特征与目标变量的线性关系显著性，值越大越显著")
# print("4. p值: 统计显著性，通常p<0.05认为显著")
