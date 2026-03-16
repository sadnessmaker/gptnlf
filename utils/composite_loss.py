# utils/composite_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeLoss(nn.Module):
    """
    复合损失：
      - base: 'mse' or 'mae' (per-element 基损失)
      - alpha: 基损失权重
      - beta: diff loss 权重 (一阶差分)
      - gamma: spec loss 权重 (频谱幅值一致性)
      - delta: smooth loss 权重 (相邻时间步平滑约束)
      - task_weights: array-like length C 或 None，用来对各通道加权
      - feature_idx: None 或 slice/list，指定在通道维上对哪些通道计算损失
    pred / target expected shape: [B, T, C]
    """
    def __init__(self,
                 base='mse',
                 alpha=1.0,
                 beta=0.0,
                 gamma=0.0,
                 delta=0.0,
                 task_weights=None,
                 feature_idx=None,
                 eps=1e-8):
        super().__init__()
        assert base in ('mse', 'mae')
        self.base = base
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.eps = eps
        self.feature_idx = feature_idx

        if task_weights is not None:
            tw = torch.tensor(task_weights, dtype=torch.float32)
            # register buffer so it moves with .to(device)
            self.register_buffer('task_weights', tw)
        else:
            self.task_weights = None

        self.last_components = {}

    def _select_features(self, x):
        if self.feature_idx is None:
            return x
        # support slice, list, numpy array etc.
        return x[..., self.feature_idx]

    def forward(self, pred, target):
        # allow numpy arrays (but recommend tensors on device)
        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, dtype=torch.float32, device=next(self.parameters()).device)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32, device=pred.device)

        if pred.dim() != 3 or target.dim() != 3:
            raise ValueError(f"pred/target must be [B,T,C], got {pred.shape} and {target.shape}")

        pred = self._select_features(pred)
        target = self._select_features(target)

        B, T, C = pred.shape

        # base loss
        if self.base == 'mse':
            per_elem = (pred - target) ** 2
        else:
            per_elem = torch.abs(pred - target)

        per_channel = per_elem.mean(dim=(0,1))  # [C]

        if self.task_weights is not None:
            w = self.task_weights.to(pred.device)
            if w.numel() != C:
                # if mismatch, try to broadcast or truncate/extend to ones
                if w.numel() > C:
                    w = w[:C]
                else:
                    w = torch.cat([w, torch.ones(C - w.numel(), device=w.device)])
        else:
            w = torch.ones(C, device=pred.device)

        base_loss = (per_channel * w).sum() / (w.sum() + self.eps)

        # diff loss (一阶差分)
        diff_loss = torch.tensor(0.0, device=pred.device)
        if self.beta > 0 and T > 1:
            dp = pred[:, 1:, :] - pred[:, :-1, :]
            dt = target[:, 1:, :] - target[:, :-1, :]
            diff_loss = ((dp - dt) ** 2).mean()

        # spec loss (频谱幅值)
        spec_loss = torch.tensor(0.0, device=pred.device)
        if self.gamma > 0 and T > 1:
            sp = torch.fft.rfft(pred, dim=1)
            st = torch.fft.rfft(target, dim=1)
            mag_p = torch.abs(sp)
            mag_t = torch.abs(st)
            spec_loss = ((mag_p - mag_t) ** 2).mean()

        # smooth loss (邻差平方)
        smooth_loss = torch.tensor(0.0, device=pred.device)
        if self.delta > 0 and T > 1:
            smooth_loss = ((pred[:, 1:, :] - pred[:, :-1, :]) ** 2).mean()

        loss = self.alpha * base_loss + self.beta * diff_loss + self.gamma * spec_loss + self.delta * smooth_loss

        # 保存用于监控
        self.last_components = {
            'total': float(loss.detach().cpu().item()),
            'base': float(base_loss.detach().cpu().item()),
            'diff': float(diff_loss.detach().cpu().item()) if isinstance(diff_loss, torch.Tensor) else float(diff_loss),
            'spec': float(spec_loss.detach().cpu().item()) if isinstance(spec_loss, torch.Tensor) else float(spec_loss),
            'smooth': float(smooth_loss.detach().cpu().item()) if isinstance(smooth_loss, torch.Tensor) else float(smooth_loss),
        }

        return loss
