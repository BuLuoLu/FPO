"""Evaluate FPO JSON predictions using point and directed-angle matching.

MLE and MAE are averaged over matched pairs per image, then over images
with valid errors. JF1 is averaged over images. Joint AP uses predictions
ranked across all images and 101-point interpolated precision; JmAP is the
mean over the specified distance/angle threshold combinations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _safe_float(x, default=0.0):
    """安全转换为浮点数"""
    try:
        return float(x)
    except Exception:
        return default


class HairFollicleEvaluator:
    """
    毛囊检测与方向回归评估器
    
    核心指标:
    - MLE (Mean Localization Error): 定位误差
    - MAE (Mean Angular Error): 角度误差  
    - JF1 (Joint F1-score): 联合F1分数
    - JmAP (Joint mean Average Precision): 联合平均精度
    """
    
    def __init__(
        self,
        distance_thresholds: Optional[List[float]] = None,
        angle_thresholds_deg: Optional[List[float]] = None,
        min_score: float = 0.0,
    ):
        """
        Args:
            distance_thresholds: 距离阈值列表 (像素)
            angle_thresholds_deg: 角度阈值列表 (度)
            min_score: 最小置信度阈值
        """
        if distance_thresholds is None:
            distance_thresholds = [5, 10, 15, 20, 25, 30]
        if angle_thresholds_deg is None:
            angle_thresholds_deg = [5, 10, 15, 20, 25, 30]
        self.distance_thresholds = [float(d) for d in distance_thresholds]
        self.angle_thresholds_deg = [float(a) for a in angle_thresholds_deg]
        self.min_score = float(min_score)
        if not self.distance_thresholds or any(
            not np.isfinite(d) or d <= 0 for d in self.distance_thresholds
        ):
            raise ValueError("Distance thresholds must be finite and positive.")
        if not self.angle_thresholds_deg or any(
            not np.isfinite(a) or not 0 <= a <= 180 for a in self.angle_thresholds_deg
        ):
            raise ValueError("Angle thresholds must lie between 0 and 180 degrees.")
        if not np.isfinite(self.min_score):
            raise ValueError("min_score must be finite.")
        
        # 用于JmAP计算的所有(距离,角度)组合
        self.joint_thresholds = [
            (d, a) for d in self.distance_thresholds for a in self.angle_thresholds_deg
        ]
    
    # ============================================================================
    # I/O 工具
    # ============================================================================
    
    @staticmethod
    def load_json(json_path: str) -> Dict:
        """加载JSON文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # ============================================================================
    # 几何计算
    # ============================================================================
    
    @staticmethod
    def compute_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
        """计算两点之间的欧氏距离"""
        return float(np.linalg.norm(pt1 - pt2))
    
    @staticmethod
    def compute_angle_difference(angle1_rad: float, angle2_rad: float) -> float:
        """
        计算两个角度之间的最小差值 (弧度)
        
        Returns:
            角度差 in [0, π]
        """
        diff = abs(angle1_rad - angle2_rad)
        diff = min(diff, 2 * np.pi - diff)
        return diff
    
    # ============================================================================
    # 数据解析
    # ============================================================================
    
    @staticmethod
    def _extract_score(det: Dict) -> float:
        """从检测结果中提取置信度分数"""
        for k in ("score", "confidence", "conf", "prob", "center_score", "heatmap_score"):
            if k in det:
                return _safe_float(det.get(k), 0.0)
        return 1.0
    
    @staticmethod
    def _extract_angle_rad(det: Dict) -> Optional[float]:
        """从检测结果中提取角度 (弧度)"""
        if isinstance(det.get("direction"), dict) and "angle_rad" in det["direction"]:
            return _safe_float(det["direction"]["angle_rad"], None)
        if "angle_rad" in det:
            return _safe_float(det["angle_rad"], None)
        if "angle" in det:
            return _safe_float(det["angle"], None)
        direction = det.get("direction", {})
        if isinstance(direction, dict) and "cos" in direction and "sin" in direction:
            x = _safe_float(direction["cos"], None)
            y = _safe_float(direction["sin"], None)
            if x is not None and y is not None and np.isfinite([x, y]).all() and (x != 0 or y != 0):
                return float(np.arctan2(y, x))
        return None
    
    def parse_detections(self, data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        解析检测结果
        
        Returns:
            points: [N, 2] array of (x, y)
            angles: [N] array of angles in radians (may contain NaN)
            scores: [N] array of confidence scores
        """
        dets = data.get("detections", [])
        
        points = np.array([det["center"] for det in dets], dtype=np.float32)
        
        angles = np.array([
            self._extract_angle_rad(det) if self._extract_angle_rad(det) is not None else np.nan
            for det in dets
        ], dtype=np.float32)
        
        scores = np.array([self._extract_score(det) for det in dets], dtype=np.float32)
        
        return points, angles, scores
    
    # ============================================================================
    # 匹配算法
    # ============================================================================
    
    def greedy_match(
        self,
        pred_points: np.ndarray,
        pred_scores: np.ndarray,
        gt_points: np.ndarray,
        dist_thresh: float,
        pred_angles: Optional[np.ndarray] = None,
        gt_angles: Optional[np.ndarray] = None,
        angle_thresh_deg: Optional[float] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        贪心匹配算法：按置信度从高到低匹配
        
        Args:
            pred_points: [N_pred, 2] 预测点
            pred_scores: [N_pred] 置信度
            gt_points: [N_gt, 2] 真值点
            dist_thresh: 距离阈值
            pred_angles: [N_pred] 预测角度 (可选)
            gt_angles: [N_gt] 真值角度 (可选)
            angle_thresh_deg: 角度阈值 (度, 可选)
        
        Returns:
            matches: [(pred_idx, gt_idx), ...]
            unmatched_pred: [pred_idx, ...]
            unmatched_gt: [gt_idx, ...]
        """
        n_pred = len(pred_points)
        n_gt = len(gt_points)
        
        if n_pred == 0 or n_gt == 0:
            return [], list(range(n_pred)), list(range(n_gt))
        
        # 按置信度降序排序
        order = np.argsort(-pred_scores)
        gt_used = np.zeros(n_gt, dtype=bool)
        
        matches = []
        
        for pi in order:
            pred_pt = pred_points[pi]
            
            # 计算与所有未匹配GT的距离
            distances = np.array([
                self.compute_distance(pred_pt, gt_points[gi])
                for gi in range(n_gt)
            ])
            
            # 筛选满足距离条件的候选
            candidates = np.where((distances <= dist_thresh) & (~gt_used))[0]
            
            if len(candidates) == 0:
                continue
            
            # 如果需要角度约束
            if angle_thresh_deg is not None and pred_angles is not None and gt_angles is not None:
                pred_ang = pred_angles[pi]
                
                # 跳过无效角度
                if np.isnan(pred_ang):
                    continue
                
                # 进一步筛选满足角度条件的候选
                valid_candidates = []
                for gi in candidates:
                    gt_ang = gt_angles[gi]
                    if np.isnan(gt_ang):
                        continue
                    
                    angle_diff_rad = self.compute_angle_difference(pred_ang, gt_ang)
                    angle_diff_deg = np.degrees(angle_diff_rad)
                    
                    if angle_diff_deg <= angle_thresh_deg:
                        valid_candidates.append(gi)
                
                if len(valid_candidates) == 0:
                    continue
                
                candidates = np.array(valid_candidates)
            
            # 选择距离最近的候选
            gi = candidates[np.argmin(distances[candidates])]
            gt_used[gi] = True
            matches.append((int(pi), int(gi)))
        
        # 统计未匹配的预测和真值
        matched_pred = {p for p, _ in matches}
        matched_gt = {g for _, g in matches}
        unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
        unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
        
        return matches, unmatched_pred, unmatched_gt
    
    # ============================================================================
    # 核心指标计算
    # ============================================================================
    
    def compute_mle_mae(
        self,
        pred_points: np.ndarray,
        pred_angles: np.ndarray,
        pred_scores: np.ndarray,
        gt_points: np.ndarray,
        gt_angles: np.ndarray,
    ) -> Dict[str, float]:
        """
        计算 MLE 和 MAE (基于最宽松的匹配)
        
        MLE: Mean Localization Error - 所有匹配对的平均距离误差
        MAE: Mean Angular Error - 所有匹配对的平均角度误差
        
        使用最大距离阈值进行匹配，不使用角度约束
        """
        # 应用置信度过滤
        valid_mask = pred_scores >= self.min_score
        pred_points_f = pred_points[valid_mask]
        pred_angles_f = pred_angles[valid_mask]
        pred_scores_f = pred_scores[valid_mask]
        
        if len(pred_points_f) == 0 or len(gt_points) == 0:
            return {
                "MLE": float('nan'),
                "MAE_deg": float('nan'),
                "n_matched": 0,
            }
        
        # 使用最大距离阈值进行匹配 (不使用角度约束)
        max_dist_thresh = max(self.distance_thresholds)
        matches, _, _ = self.greedy_match(
            pred_points_f,
            pred_scores_f,
            gt_points,
            max_dist_thresh,
        )
        
        if len(matches) == 0:
            return {
                "MLE": float('nan'),
                "MAE_deg": float('nan'),
                "n_matched": 0,
            }
        
        # 计算定位误差
        loc_errors = []
        for pi, gi in matches:
            dist = self.compute_distance(pred_points_f[pi], gt_points[gi])
            loc_errors.append(dist)
        
        mle = float(np.mean(loc_errors))
        
        # 计算角度误差 (仅针对有效角度)
        angle_errors = []
        for pi, gi in matches:
            pred_ang = pred_angles_f[pi]
            gt_ang = gt_angles[gi]
            
            if not np.isnan(pred_ang) and not np.isnan(gt_ang):
                ang_diff = self.compute_angle_difference(pred_ang, gt_ang)
                angle_errors.append(np.degrees(ang_diff))
        
        mae = float(np.mean(angle_errors)) if len(angle_errors) > 0 else float('nan')
        
        return {
            "MLE": mle,
            "MAE_deg": mae,
            "n_matched": len(matches),
            "n_angle_valid": len(angle_errors),
        }
    
    def compute_jf1(
        self,
        pred_points: np.ndarray,
        pred_angles: np.ndarray,
        pred_scores: np.ndarray,
        gt_points: np.ndarray,
        gt_angles: np.ndarray,
        dist_thresh: float,
        angle_thresh_deg: float,
    ) -> Dict[str, float]:
        """
        计算 Joint F1-score
        
        TP: 预测点同时满足距离阈值 d_k 和角度阈值 θ_k
        """
        # 应用置信度过滤
        valid_mask = pred_scores >= self.min_score
        pred_points_f = pred_points[valid_mask]
        pred_angles_f = pred_angles[valid_mask]
        pred_scores_f = pred_scores[valid_mask]
        
        n_pred = len(pred_points_f)
        n_gt = len(gt_points)
        
        if n_pred == 0 or n_gt == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp": 0,
                "fp": n_pred,
                "fn": n_gt,
            }
        
        # 联合匹配 (距离 + 角度)
        matches, unmatched_pred, unmatched_gt = self.greedy_match(
            pred_points_f,
            pred_scores_f,
            gt_points,
            dist_thresh,
            pred_angles_f,
            gt_angles,
            angle_thresh_deg,
        )
        
        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
    
    def compute_joint_ap(
        self,
        all_image_preds: List[Dict],
        all_image_gts: List[Dict],
        dist_thresh: float,
        angle_thresh_deg: float,
    ) -> float:
        """
        计算 Joint AP (在特定距离和角度阈值下)
        
        Args:
            all_image_preds: 所有图像的预测 [{"points": ..., "angles": ..., "scores": ...}, ...]
            all_image_gts: 所有图像的真值 [{"points": ..., "angles": ...}, ...]
            dist_thresh: 距离阈值
            angle_thresh_deg: 角度阈值
        
        Returns:
            AP值
        """
        # 收集所有预测 (image_idx, pred_idx, score)
        all_preds = []
        total_gt = 0
        
        for img_idx, (pred, gt) in enumerate(zip(all_image_preds, all_image_gts)):
            total_gt += len(gt["points"])
            
            for pred_idx, score in enumerate(pred["scores"]):
                if score >= self.min_score:
                    all_preds.append((img_idx, pred_idx, score))
        
        if total_gt == 0:
            return float('nan')
        
        if len(all_preds) == 0:
            return 0.0
        
        # 按置信度降序排序
        all_preds.sort(key=lambda x: -x[2])
        
        # 标记每个GT是否已被匹配
        gt_matched = [np.zeros(len(gt["points"]), dtype=bool) for gt in all_image_gts]
        
        tp = np.zeros(len(all_preds), dtype=np.float32)
        fp = np.zeros(len(all_preds), dtype=np.float32)
        
        for k, (img_idx, pred_idx, score) in enumerate(all_preds):
            pred = all_image_preds[img_idx]
            gt = all_image_gts[img_idx]
            
            if len(gt["points"]) == 0:
                fp[k] = 1.0
                continue
            
            pred_pt = pred["points"][pred_idx]
            pred_ang = pred["angles"][pred_idx]
            
            # 跳过无效角度
            if np.isnan(pred_ang):
                fp[k] = 1.0
                continue
            
            # 找到满足条件的最近GT
            best_gi = -1
            best_dist = float('inf')
            
            for gi in range(len(gt["points"])):
                if gt_matched[img_idx][gi]:
                    continue
                
                gt_pt = gt["points"][gi]
                gt_ang = gt["angles"][gi]
                
                # 跳过无效角度
                if np.isnan(gt_ang):
                    continue
                
                # 检查距离约束
                dist = self.compute_distance(
                    np.array(pred_pt, dtype=np.float32),
                    np.array(gt_pt, dtype=np.float32)
                )
                
                if dist > dist_thresh:
                    continue
                
                # 检查角度约束
                ang_diff = self.compute_angle_difference(pred_ang, gt_ang)
                ang_diff_deg = np.degrees(ang_diff)
                
                if ang_diff_deg > angle_thresh_deg:
                    continue
                
                # 记录最近的候选
                if dist < best_dist:
                    best_dist = dist
                    best_gi = gi
            
            if best_gi >= 0:
                gt_matched[img_idx][best_gi] = True
                tp[k] = 1.0
            else:
                fp[k] = 1.0
        
        # 计算precision-recall曲线
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
        
        # 计算AP (COCO-style 101点插值)
        ap = self._compute_coco_ap(recalls, precisions)
        
        return float(ap)
    
    @staticmethod
    def _compute_coco_ap(recall: np.ndarray, precision: np.ndarray) -> float:
        """COCO-style AP计算 (101点插值)"""
        # 添加边界值
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        
        # 计算precision包络线 (右侧最大值)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        
        # 101点插值
        recall_levels = np.linspace(0, 1, 101)
        ap_values = []
        
        for r_level in recall_levels:
            indices = np.where(mrec >= r_level)[0]
            if len(indices) > 0:
                ap_values.append(mpre[indices[0]])
            else:
                ap_values.append(0.0)
        
        return float(np.mean(ap_values))
    
    # ============================================================================
    # 单图像评估
    # ============================================================================
    
    def evaluate_single_image(
        self,
        pred_json_path: str,
        gt_json_path: str,
    ) -> Dict:
        """评估单张图像"""
        # 加载数据
        pred_data = self.load_json(pred_json_path)
        gt_data = self.load_json(gt_json_path)
        
        # 解析检测结果
        pred_points, pred_angles, pred_scores = self.parse_detections(pred_data)
        gt_points, gt_angles, _ = self.parse_detections(gt_data)
        
        image_name = Path(pred_json_path).stem.replace("_result", "")
        
        # 计算 MLE & MAE
        mle_mae = self.compute_mle_mae(pred_points, pred_angles, pred_scores, gt_points, gt_angles)
        
        # 计算所有(距离, 角度)组合下的JF1
        jf1_results = {}
        for dist_th, ang_th in self.joint_thresholds:
            jf1 = self.compute_jf1(
                pred_points, pred_angles, pred_scores,
                gt_points, gt_angles,
                dist_th, ang_th
            )
            key = f"JF1@{dist_th:g}px,{ang_th:g}deg"
            jf1_results[key] = jf1["f1"]
        
        return {
            "image_name": image_name,
            "MLE": mle_mae["MLE"],
            "MAE_deg": mle_mae["MAE_deg"],
            "n_matched": mle_mae["n_matched"],
            **jf1_results,
            # 保存原始数据用于后续JmAP计算
            "_pred_points": pred_points,
            "_pred_angles": pred_angles,
            "_pred_scores": pred_scores,
            "_gt_points": gt_points,
            "_gt_angles": gt_angles,
        }
    
    # ============================================================================
    # 数据集级评估
    # ============================================================================
    
    def evaluate_dataset(
        self,
        pred_gt_pairs: List[Tuple[str, str]],
    ) -> Tuple[List[Dict], Dict]:
        """
        评估整个数据集
        
        Returns:
            per_image_results: 每张图像的结果
            summary: 汇总结果
        """
        if not pred_gt_pairs:
            raise ValueError("At least one prediction/annotation pair is required.")
        
        per_image_results = []
        
        # 收集所有图像的预测和真值 (用于JmAP计算)
        all_image_preds = []
        all_image_gts = []
        
        for pred_path, gt_path in pred_gt_pairs:
            result = self.evaluate_single_image(pred_path, gt_path)
            
            # 保存图像级结果 (移除临时数据)
            cleaned_result = {k: v for k, v in result.items() if not k.startswith("_")}
            per_image_results.append(cleaned_result)
            
            # 收集数据用于JmAP
            all_image_preds.append({
                "points": result["_pred_points"],
                "angles": result["_pred_angles"],
                "scores": result["_pred_scores"],
            })
            all_image_gts.append({
                "points": result["_gt_points"],
                "angles": result["_gt_angles"],
            })
        
        # 计算汇总统计
        summary = self._compute_summary(per_image_results, all_image_preds, all_image_gts)
        
        return per_image_results, summary
    
    def _compute_summary(
        self,
        per_image_results: List[Dict],
        all_image_preds: List[Dict],
        all_image_gts: List[Dict],
    ) -> Dict:
        """计算数据集级汇总指标"""
        summary = {}
        
        # ========== MLE & MAE (数据集级平均) ==========
        mle_values = [r["MLE"] for r in per_image_results if not np.isnan(r["MLE"])]
        mae_values = [r["MAE_deg"] for r in per_image_results if not np.isnan(r["MAE_deg"])]
        
        summary["MLE"] = float(np.mean(mle_values)) if len(mle_values) > 0 else float('nan')
        summary["MAE_deg"] = float(np.mean(mae_values)) if len(mae_values) > 0 else float('nan')
        summary["n_images"] = len(per_image_results)
        
        # ========== JF1 (所有阈值组合的平均) ==========
        jf1_keys = [k for k in per_image_results[0].keys() if k.startswith("JF1@")]
        
        for key in jf1_keys:
            values = [r[key] for r in per_image_results]
            summary[key] = float(np.mean(values))
        
        # ========== JmAP (Joint mean AP) ==========
        jmap_values = []
        for dist_th, ang_th in self.joint_thresholds:
            ap = self.compute_joint_ap(all_image_preds, all_image_gts, dist_th, ang_th)
            key = f"JointAP@{dist_th:g}px,{ang_th:g}deg"
            summary[key] = ap
            
            if not np.isnan(ap):
                jmap_values.append(ap)
        
        summary["JmAP"] = float(np.mean(jmap_values)) if len(jmap_values) > 0 else float('nan')
        
        return summary

    def print_summary(self, summary: Dict):
        """Print the core metrics without experiment-specific formatting."""
        print(f"Images: {summary['n_images']}")
        print(f"MLE: {summary['MLE']:.4f} px")
        print(f"Directed MAE: {summary['MAE_deg']:.4f} deg")
        print(f"JmAP: {summary['JmAP']:.6f}")
        for key, value in summary.items():
            if key.startswith("JF1@"):
                print(f"{key}: {value:.6f}")

    def save_results(self, per_image_results: List[Dict], summary: Dict, output_path: str):
        """Save metrics only; undefined errors/AP are represented as JSON null."""
        def json_value(value):
            if isinstance(value, dict):
                return {key: json_value(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_value(item) for item in value]
            if isinstance(value, (float, np.floating)):
                return float(value) if np.isfinite(value) else None
            if isinstance(value, np.integer):
                return int(value)
            return value

        results = {
            "summary": summary,
            "per_image": per_image_results,
            "config": {
                "distance_thresholds": self.distance_thresholds,
                "angle_thresholds_deg": self.angle_thresholds_deg,
                "min_score": self.min_score,
            },
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(json_value(results), handle, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FPO JSON predictions (MLE, directed MAE, JF1, JmAP)."
    )
    parser.add_argument("--pred_dir", type=Path, required=True, help="Prediction JSON directory.")
    parser.add_argument("--gt_dir", type=Path, required=True, help="Annotation JSON directory.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--image_list", type=Path, help="Text file with one image ID per line.")
    selection.add_argument("--image_names", nargs="+", help="Image IDs, e.g. 4 sample_02.")
    parser.add_argument("--output_dir", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--distance_thresholds", type=float, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--angle_thresholds", type=float, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--min_score", type=float, default=0.0)
    args = parser.parse_args()

    if args.image_list is not None:
        if not args.image_list.is_file():
            parser.error(f"Image list not found: {args.image_list}")
        image_names = [
            line.strip() for line in args.image_list.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        image_names = args.image_names
    if not image_names:
        parser.error("The image list is empty.")

    pairs = []
    seen = set()
    for image_name in image_names:
        if "/" in image_name or "\\" in image_name or image_name in (".", ".."):
            parser.error(f"Expected an image ID, not a path: {image_name}")
        suffix = Path(image_name).suffix.lower()
        image_id = Path(image_name).stem if suffix in (".jpg", ".jpeg", ".png", ".json") else image_name
        if image_id in seen:
            parser.error(f"Duplicate image ID: {image_id}")
        seen.add(image_id)
        gt_path = args.gt_dir / f"{image_id}.json"
        pred_path = args.pred_dir / f"{image_id}_result.json"
        if not pred_path.is_file():
            pred_path = args.pred_dir / f"{image_id}.json"
        if not gt_path.is_file() or not pred_path.is_file():
            parser.error(
                f"Missing JSON pair for {image_id}: expected {gt_path} and "
                f"{args.pred_dir / (image_id + '_result.json')} (or {image_id}.json). "
                'For no detections, provide a prediction JSON with "detections": [].'
            )
        pairs.append((str(pred_path), str(gt_path)))

    try:
        evaluator = HairFollicleEvaluator(
            distance_thresholds=args.distance_thresholds,
            angle_thresholds_deg=args.angle_thresholds,
            min_score=args.min_score,
        )
        per_image, summary = evaluator.evaluate_dataset(pairs)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    evaluator.print_summary(summary)
    output_path = args.output_dir / "evaluation_results.json"
    evaluator.save_results(per_image, summary, output_path)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
