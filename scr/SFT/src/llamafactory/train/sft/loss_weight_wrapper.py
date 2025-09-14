import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = -100   # ↔ 你代码里用的那个常量

class LossWeightedWrapper(nn.Module):
    """
    把任意 CausalLM 模型包一层，使其支持 loss_weight 加权。
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.config = base_model.config   # 让 Trainer 能访问到

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        loss_weight=None,          # ← collator 会自动传进来
        **kwargs
    ):
        # 1) 先拿 logits（不让底层算 loss）
        base_out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
        )
        logits = base_out.logits

        #  自己算加权 loss
        loss = None
        if labels is not None:
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=IGNORE_INDEX,
            ).view(shift_labels.size())

            if loss_weight is not None:
                shift_weights = loss_weight[..., 1:].contiguous()  # 与 labels 对齐
                ce_loss = ce_loss * shift_weights
                # denom = shift_weights.gt(0).float().sum().clamp(min=1.0)
                denom = shift_weights.sum().clamp(min=1e-8)

                loss = ce_loss.sum() / denom
            else:
                denom = shift_labels.ne(IGNORE_INDEX).float().sum().clamp(min=1.0)
                loss = ce_loss.sum() / denom

        return {"loss": loss, "logits": logits}