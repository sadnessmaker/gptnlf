from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.My_Loss import MyLoss
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
import csv
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mutual_info_score


class FeatureVisualizer:
    def __init__(self, save_path='/tmp/zfh_1/net_load_forecasting/PIC/huitu/GE/'):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 设置学术化的绘图风格
        plt.style.use('seaborn-whitegrid')
        sns.set_palette("husl")

        # 存储特定epoch的特征状态，用于后续比较
        self.epoch_features_storage = {}

        # # 特征名称映射
        #AT
        # self.feature_names = {
        #     0: 'Net Load',
        #     1: 'AC Load',
        #     2: 'PV Load',
        #     3: 'Wind Load',
        #     4: 'air_density',
        #     5: 'humidity',
        #     6: 'ir_surface',
        #     7: 'ir_toa',
        #     8: 'snow_mass',
        #     9: 'snowfall',
        #     10: 'temp',
        #     11: 'wind_speed'
        # }
        #BE
        self.feature_names = {
            0: 'Net Load',
            1: 'AC Load',
            2: 'PV Load',
            3: 'Wind Load',
            4:'wind_speed',
            5: 'ir_surface',
            6: 'air_density',
            7: 'humidity',
            8: 'precipitat',
            9: 'ir_toa',
            10: 'temp',
            11: 'snow_mass'
        }
        # # GE
        # self.feature_names = {
        #     0: 'Net Load',
        #     1: 'AC Load',
        #     2: 'PV Load',
        #     3: 'Wind Load',
        #     4: 'wind_speed',
        #     5: 'air_density',
        #     6: 'precipitat',
        #     7: 'ir_surface',
        #     8: 'ir_toa',
        #     9: 'snow_mass',
        #     10: 'temp',
        #     11: 'humidity'
        # }

    def _discretize_features(self, data, n_bins=10, strategy='uniform'):
        """
        将连续特征离散化，用于计算互信息
        """
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        return discretizer.fit_transform(data)

    def _compute_mutual_information_matrix(self, data, method='regression', n_bins=10):
        """
        计算特征间的互信息矩阵
        """
        n_features = data.shape[1]
        mi_matrix = np.zeros((n_features, n_features))

        if method == 'regression':
            for i in range(n_features):
                X = np.delete(data, i, axis=1)
                y = data[:, i]
                mi_scores = mutual_info_regression(X, y, random_state=42)
                j_idx = 0
                for j in range(n_features):
                    if i == j:
                        mi_matrix[i, j] = 1.0
                    else:
                        mi_matrix[i, j] = mi_scores[j_idx]
                        j_idx += 1

        elif method == 'direct':
            data_discrete = self._discretize_features(data, n_bins=n_bins)
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        mi_matrix[i, j] = 1.0
                    else:
                        mi_score = mutual_info_score(data_discrete[:, i], data_discrete[:, j])
                        mi_matrix[i, j] = mi_score

            mi_matrix = (mi_matrix + mi_matrix.T) / 2
            mi_matrix = mi_matrix / np.max(mi_matrix)
            np.fill_diagonal(mi_matrix, 1.0)

        else:
            raise ValueError("Method must be 'regression' or 'direct'")

        return mi_matrix

    def _normalize_mi_matrix(self, mi_matrix):
        """
        归一化互信息矩阵
        """
        np.fill_diagonal(mi_matrix, 1.0)
        mi_matrix = (mi_matrix + mi_matrix.T) / 2
        mi_matrix = np.maximum(mi_matrix, 0)
        return mi_matrix

    def store_epoch_features(self, b_features, epoch, batch_idx=None):
        """
        存储特定epoch的b特征状态

        Args:
            b_features: b状态特征
            epoch: 当前epoch
            batch_idx: 批次索引
        """
        if torch.is_tensor(b_features):
            b_features = b_features.detach().cpu().numpy()

        key = f"epoch_{epoch}"
        if batch_idx is not None:
            key += f"_batch_{batch_idx}"

        # 计算时间维度上的平均值
        b_mean = np.mean(b_features, axis=1)  # [128, num_features]
        self.epoch_features_storage[key] = b_mean
        print(f"Stored features for {key}")

    def visualize_b_state_comparison(self, epoch_0_key="epoch_0_batch_0", epoch_23_key="epoch_26_batch_0",
                                     method='direct', n_bins=10, save=True):
        """
        生成第0次和第23次迭代b状态的互信息热力图变化图

        Args:
            epoch_0_key: 第0次epoch的存储键
            epoch_23_key: 第23次epoch的存储键
            method: 互信息计算方法
            n_bins: 离散化箱数
            save: 是否保存
        """
        if epoch_0_key not in self.epoch_features_storage or epoch_23_key not in self.epoch_features_storage:
            print(f"Warning: Required epoch data not found. Available keys: {list(self.epoch_features_storage.keys())}")
            return None

        b_epoch_0 = self.epoch_features_storage[epoch_0_key]
        b_epoch_23 = self.epoch_features_storage[epoch_23_key]

        print(f"Computing mutual information comparison between {epoch_0_key} and {epoch_23_key}")
        print(f"Data shapes - epoch_0: {b_epoch_0.shape}, epoch_26: {b_epoch_23.shape}")

        # 计算互信息矩阵
        try:
            mi_epoch_0 = self._compute_mutual_information_matrix(b_epoch_0, method=method, n_bins=n_bins)
            mi_epoch_23 = self._compute_mutual_information_matrix(b_epoch_23, method=method, n_bins=n_bins)

            mi_epoch_0 = self._normalize_mi_matrix(mi_epoch_0)
            mi_epoch_23 = self._normalize_mi_matrix(mi_epoch_23)

        except Exception as e:
            print(f"Error computing mutual information: {e}")
            print("Falling back to correlation analysis...")
            mi_epoch_0 = np.corrcoef(b_epoch_0.T)
            mi_epoch_23 = np.corrcoef(b_epoch_23.T)
            mi_epoch_0 = np.nan_to_num(np.abs(mi_epoch_0), nan=0.0)
            mi_epoch_23 = np.nan_to_num(np.abs(mi_epoch_23), nan=0.0)

        # 创建特征名称
        num_features = mi_epoch_0.shape[0]
        feature_labels = [self.feature_names[i] for i in range(num_features)]

        # 创建热力图
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # 第0次epoch的互信息热力图
        im1 = sns.heatmap(mi_epoch_0,
                          xticklabels=feature_labels,
                          yticklabels=feature_labels,
                          annot=True,
                          cmap='YlOrRd',
                          fmt=".3f",
                          vmin=0,
                          vmax=1,
                          square=True,
                          linewidths=0.5,
                          cbar_kws={"shrink": .8},
                          ax=axes[0])
        axes[0].set_title('(a)GE latent feature MI - Epoch 0', fontsize=17, pad=20)
        axes[0].tick_params(axis='x', rotation=45, labelsize=14)
        axes[0].tick_params(axis='y', rotation=0, labelsize=14)

        # 第23次epoch的互信息热力图
        im2 = sns.heatmap(mi_epoch_23,
                          xticklabels=feature_labels,
                          yticklabels=feature_labels,
                          annot=True,
                          cmap='YlOrRd',
                          fmt=".3f",
                          vmin=0,
                          vmax=1,
                          square=True,
                          linewidths=0.5,
                          cbar_kws={"shrink": .8},
                          ax=axes[1])
        axes[1].set_title('(b)BE latent feature MI - Epoch 26', fontsize=17, pad=20)
        axes[1].tick_params(axis='x', rotation=45, labelsize=14)
        axes[1].tick_params(axis='y', rotation=0, labelsize=14)

        # 互信息变化热力图
        mi_diff = mi_epoch_23 - mi_epoch_0
        im3 = sns.heatmap(mi_diff,
                          xticklabels=feature_labels,
                          yticklabels=feature_labels,
                          annot=True,
                          fmt=".3f",
                          cmap='RdBu_r',
                          center=0,
                          square=True,
                          linewidths=0.5,
                          cbar_kws={"shrink": .8},
                          ax=axes[2])
        axes[2].set_title('(c)BE latent feature MI Change(Epoch 26 - Epoch 0)', fontsize=17, pad=20)
        axes[2].tick_params(axis='x', rotation=45, labelsize=14)
        axes[2].tick_params(axis='y', rotation=0, labelsize=14)

        plt.tight_layout()

        if save:
            save_name = 'BE_epoch_0_vs_26'#b_state_mutual_information_comparison_epoch_0_vs_24
            plt.savefig(os.path.join(self.save_path, f'{save_name}.png'),
                        dpi=300, bbox_inches='tight', facecolor='white')
            print(f'B-state mutual information comparison saved: {save_name}.png')

        plt.close()
        return mi_epoch_0, mi_epoch_23, mi_diff



    def visualize_mutual_information_heatmap(self, a, b, epoch, batch_idx=None, save=True,
                                             method='direct', n_bins=10):
        """
        生成基于互信息的特征相关性热力图分析
        """
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        if torch.is_tensor(b):
            b = b.detach().cpu().numpy()

        a_mean = np.mean(a, axis=1)
        b_mean = np.mean(b, axis=1)

        num_features = a.shape[2]
        feature_labels = [self.feature_names[i] for i in range(num_features)]

        print(f"Computing mutual information matrices using method: {method}")
        print(f"Data shapes - a_mean: {a_mean.shape}, b_mean: {b_mean.shape}")

        try:
            mi_a = self._compute_mutual_information_matrix(a_mean, method=method, n_bins=n_bins)
            mi_b = self._compute_mutual_information_matrix(b_mean, method=method, n_bins=n_bins)

            mi_a = self._normalize_mi_matrix(mi_a)
            mi_b = self._normalize_mi_matrix(mi_b)

        except Exception as e:
            print(f"Error computing mutual information: {e}")
            print("Falling back to correlation analysis...")
            mi_a = np.corrcoef(a_mean.T)
            mi_b = np.corrcoef(b_mean.T)
            mi_a = np.nan_to_num(np.abs(mi_a), nan=0.0)
            mi_b = np.nan_to_num(np.abs(mi_b), nan=0.0)

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        im1 = sns.heatmap(mi_a,
                          xticklabels=feature_labels,
                          yticklabels=feature_labels,
                          annot=True,
                          cmap='YlOrRd',
                          fmt=".3f",
                          vmin=0,
                          vmax=1,
                          square=True,
                          linewidths=0.5,
                          cbar_kws={"shrink": .8},
                          ax=axes[0])
        axes[0].set_title('(a) Mutual Information Before Model', fontsize=14, pad=20)
        axes[0].tick_params(axis='x', rotation=45, labelsize=10)
        axes[0].tick_params(axis='y', rotation=0, labelsize=10)

        im2 = sns.heatmap(mi_b,
                          xticklabels=feature_labels,
                          yticklabels=feature_labels,
                          annot=True,
                          cmap='YlOrRd',
                          fmt=".3f",
                          vmin=0,
                          vmax=1,
                          square=True,
                          linewidths=0.5,
                          cbar_kws={"shrink": .8},
                          ax=axes[1])
        axes[1].set_title('(b) Mutual Information After Model', fontsize=14, pad=20)
        axes[1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1].tick_params(axis='y', rotation=0, labelsize=10)

        mi_diff = mi_b - mi_a
        im3 = sns.heatmap(mi_diff,
                          xticklabels=feature_labels,
                          yticklabels=feature_labels,
                          annot=True,
                          fmt=".3f",
                          cmap='RdBu_r',
                          center=0,
                          square=True,
                          linewidths=0.5,
                          cbar_kws={"shrink": .8},
                          ax=axes[2])
        axes[2].set_title('(c) Mutual Information Change (After - Before)', fontsize=14, pad=20)
        axes[2].tick_params(axis='x', rotation=45, labelsize=10)
        axes[2].tick_params(axis='y', rotation=0, labelsize=10)

        plt.tight_layout()

        if save:
            save_name = f'mutual_information_heatmap_epoch_{epoch}'
            if batch_idx is not None:
                save_name += f'_batch_{batch_idx}'
            plt.savefig(os.path.join(self.save_path, f'{save_name}.png'),
                        dpi=300, bbox_inches='tight', facecolor='white')
            print(f'Mutual information heatmap saved: {save_name}.png')

        plt.close()
        return mi_a, mi_b, mi_diff

    def generate_comprehensive_analysis(self, a, b, epoch, batch_idx=None, save=True):
        """
        生成综合分析报告，包含所有热力图
        """
        print(f"Generating comprehensive feature analysis for epoch {epoch}...")

        # 存储特定epoch的b特征状态
        if epoch == 0 or epoch == 26:
            self.store_epoch_features(b, epoch, batch_idx)


        self.visualize_mutual_information_heatmap(a, b, epoch, method='direct', n_bins=10)

        # 如果是第23次epoch，生成b状态比较图
        if epoch == 26 and 'epoch_0' in self.epoch_features_storage:
            print("Generating B-state comparison between epoch 0 and 26...")
            self.visualize_b_state_comparison(save=save)

        if save:
            print(f"Comprehensive analysis completed for epoch {epoch}")
        return

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):

        if self.args.model=='GPT4TS':
            model = self.model_dict[self.args.model].Model(self.args,self.device).float()
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader



    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim

    def _select_criterion(self):

        #criterion = nn.MSELoss()
        #criterion = nn.L1Loss()
        criterion=MyLoss()
        #criterion = nn.HuberLoss()
        # criterion = nn.SmoothL1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        b=[]
        feture = self.args.feture_loss  # 组件
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)#,ex_a,ex_b
                else:
                    outputs,ex_a,ex_b= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)#
                f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, 0:feture]#[256,1,4]
                # batch_y = batch_y[:, -self.args.pred_len:, 0:feture].to(self.device)#[256,1,4]
                outputs = outputs[:, -self.args.pred_len:, :]  # [256,1,19]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)  # [256,1,19]
                # 单
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)



                total_loss.append(loss.item())
                b.append(loss.item())
        total_loss = np.average(total_loss)

        plt.show()
        self.model.train()
        return total_loss,b

    def train(self, setting):
        train_start_time = time.time()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        feture = self.args.feture_loss  # 组件
        train_loss_list = []
        vali_loss_list = []
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        criterion = self._select_criterion()
        a = []
        b = []
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 初始化特征可视化器
        self.feature_visualizer = FeatureVisualizer(
            save_path='/tmp/zfh_1/net_load_forecasting/PIC/huitu/BE_1/'
        )
        epoch_times = []
        # 修改这里：定义需要分析和存储特征的epoch
        #epochs_to_analyze = [0, 23,24]  # 增加第23次epoch
        epochs_to_store_b_features = [0, 24]  # 专门存储b特征的epoch
        for epoch in range(self.args.train_epochs):
            epoch_start_time = time.time()  # 记录epoch开始时间
            iter_count = 0
            train_loss = []



            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, :]  # [256,1,19]
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)  # [256,1,19]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs, ex_a, ex_b= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)#

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]  # [256,1,19]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)  # [256,1,19]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    # #收集特征数据用于可视化（只在特定epoch和前几个batch收集）
                    # if epoch in epochs_to_analyze and i < 128:  # 只收集前3个batch的数据
                    #     epoch_features_a.append(ex_a.detach().cpu().numpy())
                    #     epoch_features_b.append(ex_b.detach().cpu().numpy())

                    #实时特征可视化（每个epoch的第一个batch和每200个batch）
                    # if (i == 0 or i % 200 == 0) and (epoch in [0, 26]):  # 修改这里，包含第0和第23次epoch
                    #     with torch.no_grad():
                    #         print(f"Performing real-time feature visualization at epoch {epoch}, batch {i}")
                    #         try:
                    #             self.feature_visualizer.generate_comprehensive_analysis(ex_a, ex_b, epoch, batch_idx=i,
                    #                                                                     save=True)
                    #         except Exception as e:
                    #             print(f"Real-time feature visualization error: {e}")
                    #
                    # # 特别针对第0次和第23次epoch的第一个batch存储b特征
                    # if epoch in epochs_to_store_b_features and i == 0:
                    #     print(f"Storing B features for epoch {epoch}")
                    #     self.feature_visualizer.store_epoch_features(ex_b, epoch, batch_idx=i)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    a.append(loss.item())
                    loss.backward()
                    model_optim.step()

            #在epoch结束后，如果是第23次epoch，生成b状态对比图
            # if epoch == 26:
            #     print("Generating final B-state comparison analysis...")
            #     try:
            #         self.feature_visualizer.visualize_b_state_comparison(save=True)
            #     except Exception as e:
            #         print(f"Error generating B-state comparison: {e}")

            # 记录epoch时间
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            train_loss = np.average(train_loss)
            vali_loss,b = self.vali(vali_data, vali_loader, criterion)
            train_loss_list.append(train_loss)
            vali_loss_list.append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | Time: {4:.2f}s".format(
                epoch + 1, train_steps, train_loss, vali_loss, epoch_time))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # ============ 训练结束，计算总时间 ============
            train_total_time = time.time() - train_start_time

        print("\n" + "=" * 80)
        print("训练时间统计")
        print("=" * 80)
        print(
                f"总训练时间:        {train_total_time:.2f}秒 ({train_total_time / 60:.2f}分钟 / {train_total_time / 3600:.2f}小时)")
        print(f"完成的epoch数:     {len(epoch_times)}")
        print(f"平均每epoch时间:   {np.mean(epoch_times):.2f}秒")
        print(f"最快epoch时间:     {np.min(epoch_times):.2f}秒")
        print(f"最慢epoch时间:     {np.max(epoch_times):.2f}秒")
        print("=" * 80 + "\n")
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # 绘制训练过程图
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 2, 1)
        plt.plot(a[:min(100, len(a))], color="#ef8a62", label="train", marker='o', linestyle='-', markersize=3)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss (First 100 iterations)")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(b[:min(len(b), len(a))], color="#67a9cf", label="vali", marker='o', linestyle='-', markersize=3)
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(train_loss_list, color="#ef8a62", label="train", marker='o', linestyle='-', markersize=3)
        plt.plot(vali_loss_list, color="#67a9cf", label="vali", marker='x', linestyle='--', markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss")
        plt.legend()
        # with open('/tmp/zfh_1/net_load_forecasting/PIC/AT/BE_mae.csv', 'w',newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["train_loss_list", "vali_loss_list"])
        #     for train_loss, vali_loss in zip(train_loss_list,vali_loss_list):
        #         writer.writerow([train_loss,vali_loss])
        plt.tight_layout()
        #plt.savefig("/tmp/zfh_1/net_load_forecasting/PIC/AT/BE_mae.png")
        print('训练图表已保存')
        plt.show()
        return self.model



    def test(self, setting, test=1):#Rye
        test_start_time = time.time()
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('/root/autodl-tmp/GPTNET/checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        inference_start = time.time()
        batch_times = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_start = time.time()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs ,ex_a,ex_b= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)#

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        inference_time = time.time() - inference_start
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1,  preds.shape[-1])
        trues = trues.reshape(-1,  trues.shape[-1])


        #反归一化后
        pred = test_data.inverse_transform(preds)
        print(pred)
        true = test_data.inverse_transform(trues)
        print(true)
        # 修正精度误差
        threshold = 1e-6
        pred = np.where(np.abs(pred) < threshold, 0, pred)
        true = np.where(np.abs(true) < threshold, 0, true)

        # #净负荷加入气象
        # pred_ac=pred[:,0]
        # pred_sun = pred[:,1]
        # pred_wind=pred[:,2]
        # pred_net_1=pred[:,3]
        # true_load=true[:,3]
        # true_ac=true[:,0]
        # true_sun = true[:,1]
        # true_wind = true[:,2]
        # # print(pred_ac)
        # # print(pred_sun)
        # # print(pred_wind)
        # pred_net_2=pred_ac-(pred_sun+pred_wind)

        pred_ac=pred[:,1]
        pred_sun = pred[:,2]
        pred_wind=pred[:,3]
        pred_net_1=pred[:,0]
        true_load=true[:,0]
        true_ac=true[:,1]
        true_sun = true[:, 2]
        true_wind = true[:, 3]
        # print(pred_ac)
        # print(pred_sun)
        # print(pred_wind)
        pred_net_2=pred_ac-(pred_sun+pred_wind)

        # print(pred_net_2)
        # print(true_load)
        mae_after, mse_after, rmse_after, smape_after,r2_after =metric(pred_net_2,true_load)#预测后计算的减去实际的
        print('mae_after: {},mse_after: {},rmse_after: {},smape_after: {},r2_after: {}'.format(mae_after,mse_after, rmse_after,smape_after,r2_after))
        mae_befor, mse_befor, rmse_befor, smape_befor,r2_befor = metric(pred_net_1, true_load)  # 预测前计算的减去实际的
        print('mae_befor: {},mse_befor: {},rmse_befor: {},mape_befor: {},r2_befor: {}'.format(mae_befor,mse_befor, rmse_befor,smape_befor,r2_befor))
        # with open('/tmp/zfh_1/net_load_forecasting/PIC/LLM_compsion/TEMPO/AT/zhibiao.csv', 'a',newline='') as file:
        #     file.write('GPTNET_1_ex_gpt_rmse_ex_gpt_TEMPO_AT:\nmae_after: {},\nmse_after: {},\nrmse_after: {},\nsmape_after: {},\nr2_after: {}\n'.format(mae_after,mse_after, rmse_after,smape_after,r2_after))

        # ============ 测试结束，计算总时间 ============
        test_total_time = time.time() - test_start_time

        print("\n" + "=" * 80)
        print("测试时间统计")
        print("=" * 80)
        print(f"总测试时间:        {test_total_time:.2f}秒 ({test_total_time / 60:.2f}分钟)")
        print(f"推理时间:          {inference_time:.2f}秒")
        print(f"后处理时间:        {test_total_time - inference_time:.2f}秒")
        print(f"测试样本数:        {len(preds)}")
        print(f"平均每样本时间:    {test_total_time / len(preds) * 1000:.2f}毫秒")
        print(f"测试batch数:       {len(batch_times)}")
        print(f"平均每batch时间:   {np.mean(batch_times):.4f}秒")
        print(f"最快batch时间:     {np.min(batch_times):.4f}秒")
        print(f"最慢batch时间:     {np.max(batch_times):.4f}秒")
        print(f"吞吐量:            {len(preds) / inference_time:.2f} 样本/秒")
        print("=" * 80 + "\n")

        plt.figure(figsize=(15, 8))
        K=30
        # 预测结果对比
        plt.subplot(2, 2, 1)
        kw_true = true_load[:K]
        kw_pred = pred_net_2[:K]
        # print(true_load)
        # print(pred_net_2)
        plt.plot(kw_true, color="#ef8a62", label="True", marker='o', linestyle='-', markersize=3)
        plt.plot(kw_pred, color="#67a9cf", label="Pred", marker='x', linestyle='--', markersize=3)
        plt.xlabel("Hour")
        plt.ylabel("Power (net_load)")
        plt.title("net_load Prediction Results")
        plt.legend()
        # with open('/tmp/zfh_1/net_load_forecasting/PIC/LLM_compsion/TEMPO/AT/GPTNET_1_ex_gpt_rmse_ex_gpt_TEMPO_AT.csv', 'w',newline='') as f:  # '/tmp/zfh_1/net_load_forecasting/PIC/XiaoRong/yuceandzhengshi_ex_gpt.csv'
        #     writer = csv.writer(f)
        #     writer.writerow(["true_load", "pred_net_2", "pred_net_1","pred_ac","true_ac","pred_sun","true_sun","pred_wind","true_wind"])
        #     for true_val, pred_val2,pred_val1,pre_ac,tru_ac,pre_sun,tru_sun,pre_wind,tru_wind in zip(true_load, pred_net_2,pred_net_1,pred_ac,true_ac,pred_sun,true_sun,pred_wind,true_wind):
        #         writer.writerow([true_val, pred_val2,pred_val1,pre_ac,tru_ac,pre_sun,tru_sun,pre_wind,tru_wind])

        plt.subplot(2, 2, 2)
        plt.axline((0, 0), (70, 70), color='r', linestyle='--', label='y=x')
        plt.scatter(true_load, pred_net_2)
        plt.xlabel('true')
        plt.ylabel('prue')

        # plt.subplot(2, 2, 3)

        plt.tight_layout()
        #plt.savefig("/tmp/zfh_1/net_load_forecasting/PIC/methods_pic/AT/result_net_load_ex_gpt_nofft.png")#/tmp/zfh_1/net_load_forecasting/PIC/result_net_load_rye.png
        print('已保存')
        plt.show()

        return rmse_after,mae_after


# def test(self, setting, test=0):#france_data
#     test_data, test_loader = self._get_data(flag='test')
#     if test:
#         print('loading model')
#         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
#
#     preds = []
#     trues = []
#     folder_path = './test_results/' + setting + '/'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#
#     self.model.eval()
#     with torch.no_grad():
#         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#             batch_x = batch_x.float().to(self.device)
#             batch_y = batch_y.float().to(self.device)
#
#             batch_x_mark = batch_x_mark.float().to(self.device)
#             batch_y_mark = batch_y_mark.float().to(self.device)
#
#             # decoder input
#             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#             # encoder - decoder
#             if self.args.use_amp:
#                 with torch.cuda.amp.autocast():
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#             else:
#                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#             f_dim = -1 if self.args.features == 'MS' else 0
#             outputs = outputs[:, -self.args.pred_len:, :]
#             batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
#             outputs = outputs.detach().cpu().numpy()
#             batch_y = batch_y.detach().cpu().numpy()
#             if test_data.scale and self.args.inverse:
#                 shape = outputs.shape
#                 outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                 batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
#
#             outputs = outputs[:, :, f_dim:]
#             batch_y = batch_y[:, :, f_dim:]
#
#             pred = outputs
#             true = batch_y
#
#             preds.append(pred)
#             trues.append(true)
#             if i % 20 == 0:
#                 input = batch_x.detach().cpu().numpy()
#                 if test_data.scale and self.args.inverse:
#                     shape = input.shape
#                     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
#
#     preds = np.concatenate(preds, axis=0)
#     trues = np.concatenate(trues, axis=0)
#     print('test shape:', preds.shape, trues.shape)
#     preds = preds.reshape(-1,  preds.shape[-1])
#     trues = trues.reshape(-1,  trues.shape[-1])
#
#     #反归一化后
#     pred = test_data.inverse_transform(preds)
#     print(pred)
#     true = test_data.inverse_transform(trues)
#     print(true)
#     pred_ac=pred[:,1]+pred[:,2]+pred[:,3]#总负荷1+2+3
#     pred_re = pred[:,4]#预测可再生能源
#     pred_net_1=pred[:,0]#直接预测净负荷
#     true_load=true[:,0]#真实净负荷
#     pred_net_2=pred_ac-pred_re#间接预测净负荷
#     pred_ac_1=pred[:,1]#预测房子1
#     pred_ac_2=pred[:,2]#预测房子2
#     pred_ac_3=pred[:,3]#预测房子3
#     true_ac_1 = pred[:, 1]  # 真实房子1
#     true_ac_2 = pred[:, 2]  # 真实房子2
#     true_ac_3 = pred[:, 3]  # 真实房子3
#     true_ac = true[:, 1] + true[:, 2] + true[:, 3]  # 总负荷1+2+3
#     true_re=pred[:,4]#真实可再生能源
#
#
#     mae_after, mse_after, rmse_after, smape_after,r2_after =metric(pred_net_2,true_load)#预测后计算的减去实际的
#     print('mae_after: {},mse_after: {},rmse_after: {},smape_after: {},r2_after: {}'.format(mae_after,mse_after, rmse_after,smape_after,r2_after))
#     mae_befor, mse_befor, rmse_befor, smape_befor,r2_befor = metric(pred_net_1, true_load)  # 预测前计算的减去实际的
#     print('mae_befor: {},mse_befor: {},rmse_befor: {},mape_befor: {},r2_befor: {}'.format(mae_befor,mse_befor, rmse_befor,smape_befor,r2_befor))
#     with open('/tmp/zfh_1/net_load_forecasting/PIC/LLM_size/Rye/zhibiao_methods.txt', 'a') as f:#'/tmp/zfh_1/net_load_forecasting/PIC/XiaoRong/zhibiao_xiaorong.txt'
#             f.write(f"\ngptnet间接_19_GPTNET_1_small_llm:\nRMSE: {rmse_after:.6f}\nMAE: {mae_after:.6f}\n")#gptnet_adapter
#             f.write(f"\ngptnet直接_19_GPTNET_1_small_llm:\nRMSE: {rmse_befor:.6f}\nMAE: {mae_befor:.6f}\n")
#     # plt.figure(figsize=(15, 8))
#     # K=100
#     # # 预测结果对比
#     # plt.subplot(1, 1, 1)
#     # kw_true = true_load[:K]
#     # kw_pred = pred_net_2[:K]
#     # plt.plot(kw_true, color="#ef8a62", label="True", marker='o', linestyle='-', markersize=3)
#     # plt.plot(kw_pred, color="#67a9cf", label="Pred", marker='x', linestyle='--', markersize=3)
#     # plt.xlabel("Hour")
#     # plt.ylabel("Power (net_load)")
#     # plt.title("net_load Prediction Results")
#     # plt.legend()
#     with open('/tmp/zfh_1/net_load_forecasting/PIC/LLM_size/Rye/GPTNET_1_ture_and_pre_small_llm.csv', 'w',newline='') as f:  # '/tmp/zfh_1/net_load_forecasting/PIC/XiaoRong/yuceandzhengshi_ex_gpt.csv'
#             writer = csv.writer(f)
#             writer.writerow(["true_load", "pred_net_2", "pred_net_1","pred_ac","true_ac","pred_sun","true_sun","pred_wind","true_wind"])
#             for true_val, pred_val2,pred_val1,tru_ac,pre_ac,tru_ac_1,pre_ac_1,tru_ac_2,pre_ac_2,tru_ac_3,pre_ac_3,tru_re,pre_re in zip(true_load, pred_net_2,pred_net_1,
#                                                                    true_ac,pred_ac,
#                                                                    true_ac_1,pred_ac_1,
#                                                                    true_ac_2,pred_ac_2,
#                                                                    true_ac_3,pred_ac_3,
#                                                                    true_re,pred_re):
#                 writer.writerow([true_val, pred_val2,pred_val1,tru_ac,pre_ac,tru_ac_1,pre_ac_1,tru_ac_2,pre_ac_2,tru_ac_3,pre_ac_3,tru_re,pre_re])
#     # plt.tight_layout()
#     # plt.savefig("/tmp/zfh_1/net_load_forecasting/PIC/france/result_net_load_france.png")#/tmp/zfh_1/net_load_forecasting/PIC/result_net_load_rye.png
#     # print('已保存')
#     # plt.show()
#
#     return

# def test(self, setting, test=0):#AT
#     test_data, test_loader = self._get_data(flag='test')
#
#     if test:
#         print('loading model')
#         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
#
#     preds = []
#     trues = []
#     folder_path = './test_results/' + setting + '/'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#
#     self.model.eval()
#     with torch.no_grad():
#         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#             batch_x = batch_x.float().to(self.device)
#             batch_y = batch_y.float().to(self.device)
#
#             batch_x_mark = batch_x_mark.float().to(self.device)
#             batch_y_mark = batch_y_mark.float().to(self.device)
#
#             # decoder input
#             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#             # encoder - decoder
#             if self.args.use_amp:
#                 with torch.cuda.amp.autocast():
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#             else:
#                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#             f_dim = -1 if self.args.features == 'MS' else 0
#             outputs = outputs[:, -self.args.pred_len:, f_dim:]
#             batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#             outputs = outputs.detach().cpu().numpy()
#             batch_y = batch_y.detach().cpu().numpy()
#             if test_data.scale and self.args.inverse:
#                 shape = outputs.shape
#                 outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                 batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
#
#             outputs = outputs[:, :, f_dim:]
#             batch_y = batch_y[:, :, f_dim:]
#
#             pred = outputs
#             true = batch_y
#
#             preds.append(pred)
#             trues.append(true)
#             if i % 20 == 0:
#                 input = batch_x.detach().cpu().numpy()
#                 if test_data.scale and self.args.inverse:
#                     shape = input.shape
#                     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
#
#     preds = np.concatenate(preds, axis=0)
#     trues = np.concatenate(trues, axis=0)
#     print('test shape:', preds.shape, trues.shape)
#     preds = preds.reshape(-1,  preds.shape[-1])
#     trues = trues.reshape(-1,  trues.shape[-1])
#     # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#     # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#     print('test shape:', preds.shape, trues.shape)
#     # pred=preds
#     # true=trues
#     pred = test_data.inverse_transform(preds)
#     print(pred)
#     true = test_data.inverse_transform(trues)
#     print(true)
#     pred_ac=pred[:,1]
#     pred_sun = pred[:, 2]
#     pred_wind=pred[:,3]
#     pred_net_1=pred[:,0]
#     true_load=true[:,0]
#     print(pred_ac)
#     print(pred_sun)
#     print(pred_wind)
#     pred_net_2=pred_ac-(pred_sun+pred_wind)
#     mae_after, mse_after, rmse_after, mape_after,r2_after =metric(pred_net_2,true_load)#预测后计算的减去实际的
#     print('mae_after: {},mse_after: {},rmse_after: {},mape_after: {},r2_after: {}'.format(mae_after,mse_after, rmse_after,mape_after,r2_after))
#     mae_befor, mse_befor, rmse_befor, mape_befor,r2_befor = metric(pred_net_1, true_load)  # 预测前计算的减去实际的
#     print('mae_befor: {},mse_befor: {},rmse_befor: {},mape_befor: {},r2_befor: {}'.format(mae_befor,mse_befor, rmse_befor,mape_befor,r2_befor))
#     with open('/tmp/zfh_1/net_load_forecasting/PIC/AT/zhibiao_methods.txt','a') as f:  # '/tmp/zfh_1/net_load_forecasting/PIC/XiaoRong/zhibiao_xiaorong.txt'
#             f.write(f"\ngptnet间接_7_GPTNET_1:\nRMSE: {rmse_after:.6f}\nMAE: {mae_after:.6f}\n")#gptnet_adapter
#             f.write(f"\ngptnet直接_7_GPTNET_1:\nRMSE: {rmse_befor:.6f}\nMAE: {mae_befor:.6f}\n")
#     with open('/tmp/zfh_1/net_load_forecasting/PIC/AT/GPTNET_1_ture_and_pre.csv', 'w',newline='') as f:  # '/tmp/zfh_1/net_load_forecasting/PIC/XiaoRong/yuceandzhengshi_ex_gpt.csv'
#             writer = csv.writer(f)
#             writer.writerow(["true_load", "pred_net_2", "pred_net_1"])
#             for true_val, pred_val2,pred_val1 in zip(true_load, pred_net_2,pred_net_1):
#                 writer.writerow([true_val, pred_val2,pred_val1])
#     # plt.figure(figsize=(15, 8))
#     # K=100
#     # # 预测结果对比
#     # plt.subplot(1, 1, 1)
#     # kw_true = true_load[:K]
#     # kw_pred = pred_net_2[:K]
#     # plt.plot(kw_true, color="#ef8a62", label="True", marker='o', linestyle='-', markersize=3)
#     # plt.plot(kw_pred, color="#67a9cf", label="Pred", marker='x', linestyle='--', markersize=3)
#     # plt.xlabel("Hour")
#     # plt.ylabel("Power (net_load)")
#     # plt.title("net_load Prediction Results")
#     # plt.legend()
#     #
#     # plt.tight_layout()
#     # plt.savefig("/tmp/zfh_1/net_load_forecasting/PIC/result_net_load.png")
#     # print('已保存')
#     # plt.show()
#
#     return

# def test(self, setting, test=0):#能源
#     test_data, test_loader = self._get_data(flag='test')
#     if test:
#         print('loading model')
#         self.model.load_state_dict(
#             torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
#
#     preds = []
#     trues = []
#     folder_path = './test_results/' + setting + '/'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#
#     self.model.eval()
#     with torch.no_grad():
#         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#             batch_x = batch_x.float().to(self.device)
#             batch_y = batch_y.float().to(self.device)
#
#             batch_x_mark = batch_x_mark.float().to(self.device)
#             batch_y_mark = batch_y_mark.float().to(self.device)
#
#             # decoder input
#             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#             # encoder - decoder
#             if self.args.use_amp:
#                 with torch.cuda.amp.autocast():
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#             else:
#                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#             f_dim = -1 if self.args.features == 'MS' else 0
#             outputs = outputs[:, -self.args.pred_len:, :]
#             batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
#             outputs = outputs.detach().cpu().numpy()
#             batch_y = batch_y.detach().cpu().numpy()
#             if test_data.scale and self.args.inverse:
#                 shape = outputs.shape
#                 outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                 batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
#
#             outputs = outputs[:, :, f_dim:]
#             batch_y = batch_y[:, :, f_dim:]
#
#             pred = outputs
#             true = batch_y
#
#             preds.append(pred)
#             trues.append(true)
#             if i % 20 == 0:
#                 input = batch_x.detach().cpu().numpy()
#                 if test_data.scale and self.args.inverse:
#                     shape = input.shape
#                     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
#
#     preds = np.concatenate(preds, axis=0)
#     trues = np.concatenate(trues, axis=0)
#     print('test shape:', preds.shape, trues.shape)
#     preds = preds.reshape(-1, preds.shape[-1])
#     trues = trues.reshape(-1, trues.shape[-1])
#     # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#     # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#     print('test shape:', preds.shape, trues.shape)
#
#     pred = test_data.inverse_transform(preds)
#     print(pred)
#     true = test_data.inverse_transform(trues)
#     print(true)
#     pred_ac = pred[:, 1]
#     pred_re = pred[:, 2]
#
#     pred_net_1 = pred[:, 0]
#     true_load = true[:, 0]
#
#     pred_net_2 = pred_ac - pred_re
#     mae_after, mse_after, rmse_after, mape_after = metric(pred_net_2, true_load)  # 预测后计算的减去实际的
#     print('mae_after: {},rmse_after: {},mape_after: {}'.format(mae_after, rmse_after, mape_after))
#     mae_befor, mse_befor, rmse_befor, mape_befor = metric(pred_net_1, true_load)  # 预测后计算的减去实际的
#     print('mae_befor: {},rmse_befor: {},mape_befor: {}'.format(mae_befor, rmse_befor, mape_befor))
#     with open('/tmp/zfh_1/net_load_forecasting/PIC/zhibiao.txt', 'a') as f:
#         f.write(
#             f"\nLR=0.0001_dr=0.1_PatchTST_adapter_13_energy_exen测试集结果 (原始尺度):\nRMSE: {rmse_after:.6f}\nMAE: {mae_after:.6f}\nMAPE: {mape_after:.6f}\n")
#     plt.figure(figsize=(15, 8))
#     K = 100
#     # 预测结果对比
#     plt.subplot(1, 1, 1)
#     kw_true = true_load[:K]
#     kw_pred = pred_net_2[:K]
#     plt.plot(kw_true, color="#ef8a62", label="True", marker='o', linestyle='-', markersize=3)
#     plt.plot(kw_pred, color="#67a9cf", label="Pred", marker='x', linestyle='--', markersize=3)
#     plt.xlabel("Hour")
#     plt.ylabel("Power (net_load)")
#     plt.title("energy net_load Prediction Results")
#     plt.legend()
#
#     plt.tight_layout()
#     plt.savefig("/tmp/zfh_1/net_load_forecasting/PIC/result_energy_net_load.png")
#     print('已保存')
#     plt.show()
#
#     return



        # test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(
        #         torch.load(os.path.join('/tmp/zfh_1/net_load_forecasting/checkpoints/' + setting, 'checkpoint.pth'),
        #                    map_location=self.device))
        # preds = []
        # trues = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        #
        # self.model.eval()
        # with torch.no_grad():
        #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        #         batch_x = batch_x.float().to(self.device)
        #         batch_y = batch_y.float().to(self.device)
        #
        #         batch_x_mark = batch_x_mark.float().to(self.device)
        #         batch_y_mark = batch_y_mark.float().to(self.device)
        #
        #         outputs = self.autoregressive_forecast(batch_x, batch_y, batch_x_mark, batch_y_mark)
        #         batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        #         outputs = outputs.detach().cpu().numpy()
        #         batch_y = batch_y.detach().cpu().numpy()
        #
        #
        #         preds.append(outputs)
        #         trues.append(batch_y)
        #
        #
        # preds = np.concatenate(preds, axis=0)
        # trues = np.concatenate(trues, axis=0)
        # print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-1])
        #
        #
        # # 未反归一化
        # # print('test shape:', preds.shape, trues.shape)
        # # pred_ac=preds[:,1]
        # # pred_sun = preds[:,2]
        # # pred_wind=preds[:,3]
        # # pred_net_1=preds[:,0]
        # # true_load=trues[:,0]
        # # pred_net_2=pred_ac-(pred_sun+pred_wind)
        # # mae_after, mse_after, rmse_after, mape_after=metric(pred_net_2,true_load)#预测后计算的减去实际的
        # # print('mae_after: {},rmse_after: {},mape_after: {}'.format(mae_after, rmse_after,mape_after))
        # # mae_befor, mse_befor, rmse_befor, mape_befor = metric(pred_net_1, true_load)  # 预测前计算的减去实际的
        # # print('mae_befor: {},rmse_befor: {},mape_befor: {}'.format(mae_befor, rmse_befor,mape_befor))
        #
        # # 反归一化后
        # pred = test_data.inverse_transform(preds)
        # print(pred)
        # true = test_data.inverse_transform(trues)
        # print(true)
        # # 修正精度误差
        # threshold = 1e-6
        # pred = np.where(np.abs(pred) < threshold, 0, pred)
        # true = np.where(np.abs(true) < threshold, 0, true)
        #
        # # #净负荷加入气象
        # # pred_ac=pred[:,0]
        # # pred_sun = pred[:,1]
        # # pred_wind=pred[:,2]
        # # pred_net_1=pred[:,3]
        # # true_load=true[:,3]
        # # true_ac=true[:,0]
        # # true_sun = true[:,1]
        # # true_wind = true[:,2]
        # # # print(pred_ac)
        # # # print(pred_sun)
        # # # print(pred_wind)
        # # pred_net_2=pred_ac-(pred_sun+pred_wind)
        #
        # pred_ac = pred[:, 1]
        # pred_sun = pred[:, 2]
        # pred_wind = pred[:, 3]
        # pred_net_1 = pred[:, 0]
        # true_load = true[:, 0]
        # true_ac = true[:, 1]
        # true_sun = true[:, 2]
        # true_wind = true[:, 3]
        # # print(pred_ac)
        # # print(pred_sun)
        # # print(pred_wind)
        # pred_net_2 = pred_ac - (pred_sun + pred_wind)
        #
        # # print(pred_net_2)
        # # print(true_load)
        # mae_after, mse_after, rmse_after, smape_after, r2_after = metric(pred_net_2, true_load)  # 预测后计算的减去实际的
        # print('mae_after: {},mse_after: {},rmse_after: {},smape_after: {},r2_after: {}'.format(mae_after, mse_after,
        #                                                                                        rmse_after, smape_after,
        #                                                                                        r2_after))
        # mae_befor, mse_befor, rmse_befor, smape_befor, r2_befor = metric(pred_net_1, true_load)  # 预测前计算的减去实际的
        # print('mae_befor: {},mse_befor: {},rmse_befor: {},mape_befor: {},r2_befor: {}'.format(mae_befor, mse_befor,
        #                                                                                       rmse_befor, smape_befor,
        #                                                                                       r2_befor))
        # # with open('/tmp/zfh_1/net_load_forecasting/PIC/methods_pic/Rye/zhibiao_methods.txt', 'a') as f:#'/tmp/zfh_1/net_load_forecasting/PIC/methods_pic/zhibiao_methods.txt'
        # #     f.write(f"\ngptnet间接_19_GPTNET_1:\nRMSE: {rmse_after:.6f}\nMAE: {mae_after:.6f}\n")#gptnet_adapter
        # #     f.write(f"\ngptnet直接_19_GPTNET_1:\nRMSE: {rmse_befor:.6f}\nMAE: {mae_befor:.6f}\n")
        # plt.figure(figsize=(15, 8))
        # K = 100
        # # 预测结果对比
        # plt.subplot(1, 2, 1)
        # kw_true = true_load[:K]
        # kw_pred = pred_net_2[:K]
        # # print(true_load)
        # # print(pred_net_2)
        # plt.plot(kw_true, color="#ef8a62", label="True", marker='o', linestyle='-', markersize=3)
        # plt.plot(kw_pred, color="#67a9cf", label="Pred", marker='x', linestyle='--', markersize=3)
        # plt.xlabel("Hour")
        # plt.ylabel("Power (net_load)")
        # plt.title("net_load Prediction Results")
        # plt.legend()
        # # with open('/tmp/zfh_1/net_load_forecasting/PIC/methods_pic/Rye/GPTNET_1_ture_and_pre.csv', 'w',newline='') as f:  # '/tmp/zfh_1/net_load_forecasting/PIC/XiaoRong/yuceandzhengshi_ex_gpt.csv'
        # #     writer = csv.writer(f)
        # #     writer.writerow(["true_load", "pred_net_2", "pred_net_1","pred_ac","true_ac","pred_sun","true_sun","pred_wind","true_wind"])
        # #     for true_val, pred_val2,pred_val1,pre_ac,tru_ac,pre_sun,tru_sun,pre_wind,tru_wind in zip(true_load, pred_net_2,pred_net_1,pred_ac,true_ac,pred_sun,true_sun,pred_wind,true_wind):
        # #         writer.writerow([true_val, pred_val2,pred_val1,pre_ac,tru_ac,pre_sun,tru_sun,pre_wind,tru_wind])
        #
        # plt.subplot(1, 2, 2)
        # # 获取坐标轴范围
        # # x_min, x_max = plt.xlim()
        # # y_min, y_max = plt.ylim()
        # plt.axline((0, 0), (70, 70), color='r', linestyle='--', label='y=x')
        # plt.scatter(true_load, pred_net_2)
        # plt.xlabel('true')
        # plt.ylabel('prue')
        # plt.tight_layout()
        # plt.savefig(
        #     "/tmp/zfh_1/net_load_forecasting/PIC/methods_pic/Rye/result_net_load_rye.png")  # /tmp/zfh_1/net_load_forecasting/PIC/result_net_load_rye.png
        # print('已保存')
        # plt.show()
        #
        # return rmse_after, mae_after


