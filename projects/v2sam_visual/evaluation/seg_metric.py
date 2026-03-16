from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.logging import MMLogger, print_log
import torch

@METRICS.register_module()
class SegMetric(BaseMetric):
    """计算 IoU 和 Dice 的简单 metric，兼容 MMEngine"""

    def __init__(self, iou_metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.iou_metrics = iou_metrics or ["IoU", "Dice"]
        self.results = []

    def process(self, data_batch, data_samples):

        gt_masks_all = data_batch["data"].get("masks", None)
        bs = len(gt_masks_all) # 1
        # print(f"bs{bs}")
        num_obj = gt_masks_all[0].shape[0]

        for i in range(bs):
            gt_masks = gt_masks_all[i]
            pred_masks = data_samples[i]["pred_masks"]
            for j in range(num_obj):
                gt_mask_i = gt_masks[j]
                pred_mask_i = pred_masks[j]
                pred = pred_mask_i.float().to(pred_mask_i.device)
                gt = gt_mask_i.float().to(pred_mask_i.device)
                inter = (pred * gt).sum()
                union = (pred + gt - pred * gt).sum()
                iou = inter / (union + 1e-6)
                dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-6)
                # print(iou)
                self.results.append(dict(IoU=iou.item(), Dice=dice.item()))
            
    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()

        if not results:
            print_log("⚠️ No results collected in SegMetric!", logger=logger)
            return {}

        ious = [r["IoU"] for r in results]
        dices = [r["Dice"] for r in results]
        mean_iou = sum(ious) / len(ious)
        mean_dice = sum(dices) / len(dices)

        # ✅ 使用 MMEngine 的日志系统
        print_log("=" * 40, logger=logger)
        print_log(f"✅ Mean IoU:  {mean_iou:.4f}", logger=logger)
        print_log(f"✅ Mean Dice: {mean_dice:.4f}", logger=logger)
        print_log("=" * 40, logger=logger)

        return dict(mean_IoU=mean_iou, mean_Dice=mean_dice)