from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.logging import MMLogger, print_log
import torch

@METRICS.register_module()
class SegMetric_DualDecoder(BaseMetric):
    """计算 IoU 和 Dice 的简单 metric, 兼容 MMEngine"""

    def __init__(self, iou_metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.iou_metrics = iou_metrics or ["IoU", "Dice"]
        self.results = []

    def process(self, data_batch, data_samples):
        """
        计算双decoder的IoU/Dice, 选择更好的结果
        """
        # 尝试从data_batch获取gt_masks（推理时传递）
        gt_masks_all = data_batch["data"].get("masks", None)
        
        if gt_masks_all is None:
            print("⚠️ No GT masks found in data_batch!")
            return
            
        bs = len(gt_masks_all)
        
        for i in range(bs):
            gt_masks = gt_masks_all[i]
            num_obj = gt_masks.shape[0] # debug: num_obj = gt_masks_all[0].shape[0]
            
            # 获取两个decoder的预测结果
            pred_masks_vp = data_samples[i].get("pred_masks_vp", None)
            pred_masks_sparse = data_samples[i].get("pred_masks_sparse", None)
            
            # 如果只有一个decoder结果（向后兼容）
            if pred_masks_vp is None and pred_masks_sparse is None:
                pred_masks_vp = data_samples[i].get("pred_masks", None)
            
            for j in range(num_obj):
                gt_mask_j = gt_masks[j].float()
                
                iou_vp, dice_vp = None, None
                iou_sparse, dice_sparse = None, None
                
                # 计算vp-decoder的IoU/Dice
                if pred_masks_vp is not None:
                    pred_vp = pred_masks_vp[j].float().to(gt_mask_j.device)
                    gt = gt_mask_j.to(pred_vp.device)
                    inter_vp = (pred_vp * gt).sum()
                    union_vp = (pred_vp + gt - pred_vp * gt).sum()
                    iou_vp = (inter_vp / (union_vp + 1e-6)).item()
                    dice_vp = ((2 * inter_vp) / (pred_vp.sum() + gt.sum() + 1e-6)).item()
                
                # 计算sparse-decoder的IoU/Dice
                if pred_masks_sparse is not None:
                    pred_sparse = pred_masks_sparse[j].float().to(gt_mask_j.device)
                    gt = gt_mask_j.to(pred_sparse.device)
                    inter_sparse = (pred_sparse * gt).sum()
                    union_sparse = (pred_sparse + gt - pred_sparse * gt).sum()
                    iou_sparse = (inter_sparse / (union_sparse + 1e-6)).item()
                    dice_sparse = ((2 * inter_sparse) / (pred_sparse.sum() + gt.sum() + 1e-6)).item()
                
                # 选择IoU更高的结果
                if iou_vp is not None and iou_sparse is not None:
                    if iou_vp >= iou_sparse:
                        best_iou, best_dice, best_decoder = iou_vp, dice_vp, "vp"
                    else:
                        best_iou, best_dice, best_decoder = iou_sparse, dice_sparse, "sparse"
                elif iou_vp is not None:
                    best_iou, best_dice, best_decoder = iou_vp, dice_vp, "vp"
                elif iou_sparse is not None:
                    best_iou, best_dice, best_decoder = iou_sparse, dice_sparse, "sparse"
                else:
                    continue
                
                # 记录结果，包含详细信息用于分析
                self.results.append(dict(
                    IoU=best_iou,
                    Dice=best_dice,
                    best_decoder=best_decoder,
                    iou_vp=iou_vp if iou_vp is not None else 0.0,
                    iou_sparse=iou_sparse if iou_sparse is not None else 0.0
                ))
            
    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()

        if not results:
            print_log("⚠️ No results collected in SegMetric!", logger=logger)
            return {}

        ious = [r["IoU"] for r in results]
        dices = [r["Dice"] for r in results]
        mean_iou = sum(ious) / len(ious)
        mean_dice = sum(dices) / len(dices)

        # 统计每个decoder被选中的次数
        vp_count = sum(1 for r in results if r.get("best_decoder") == "vp")
        sparse_count = sum(1 for r in results if r.get("best_decoder") == "sparse")
        
        # 分别计算两个decoder的平均IoU（用于对比分析）
        iou_vp_list = [r.get("iou_vp", 0.0) for r in results if r.get("iou_vp", 0.0) > 0]
        iou_sparse_list = [r.get("iou_sparse", 0.0) for r in results if r.get("iou_sparse", 0.0) > 0]
        
        avg_iou_vp = sum(iou_vp_list) / len(iou_vp_list) if iou_vp_list else 0.0
        avg_iou_sparse = sum(iou_sparse_list) / len(iou_sparse_list) if iou_sparse_list else 0.0

        # ✅ 使用 MMEngine 的日志系统
        print_log("=" * 60, logger=logger)
        print_log("📊 Dual Decoder Evaluation Results", logger=logger)
        print_log("=" * 60, logger=logger)
        print_log(f"✅ Best Mean IoU:  {mean_iou:.4f} (选择更好的decoder)", logger=logger)
        print_log(f"✅ Best Mean Dice: {mean_dice:.4f}", logger=logger)
        print_log("-" * 60, logger=logger)
        print_log(f"🔹 VP-Decoder选中次数: {vp_count}/{len(results)} ({vp_count/len(results)*100:.1f}%)", logger=logger)
        print_log(f"🔹 Sparse-Decoder选中次数: {sparse_count}/{len(results)} ({sparse_count/len(results)*100:.1f}%)", logger=logger)
        print_log("-" * 60, logger=logger)
        print_log(f"📈 VP-Decoder平均IoU: {avg_iou_vp:.4f}", logger=logger)
        print_log(f"📈 Sparse-Decoder平均IoU: {avg_iou_sparse:.4f}", logger=logger)
        print_log("=" * 60, logger=logger)

        return dict(
            mean_IoU=mean_iou, 
            mean_Dice=mean_dice,
            vp_decoder_selected=vp_count,
            sparse_decoder_selected=sparse_count,
            vp_decoder_avg_iou=avg_iou_vp,
            sparse_decoder_avg_iou=avg_iou_sparse
        )