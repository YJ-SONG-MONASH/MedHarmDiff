# MedHarmDiff Camelyon17-WILDS 实验报告（一页版）

报告日期：2026-07-08
实验日期：2026-07-06

## 实验内容

1. 准备并验证 Camelyon17-WILDS 真实数据，共 455,954 个 patch，5 个 center。
2. 导出 metadata.csv，并确认 center_id、label、patient_or_group_id、split_group 可用于 group-safe benchmark。
3. 构建 paper-safe split：train=302,436，val=34,904，test=85,054；train/val/test group overlap 均为空。
4. 生成两类 feature representation：color_stats_v0（37 维）和 ResNet18 ImageNet embedding（512 维）。
5. 将 embeddings 转成 feature CSV，运行 group-safe Camelyon17 benchmark 和 embedding-family benchmark。
6. 评估方法包括 identity、source_standardize、center_mean、ridge_denoising、latent_diffusion_v0 和 diffusion_placeholder。

## 实验结果

| Embedding | Samples | Dim | Best statistical AUC | Ridge AUC | Latent diffusion AUC | Claim gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| color_stats_v0 | 455,954 | 37 | 0.8853 | 0.8991 | 0.8975 | baseline_not_beaten |
| resnet18_imagenet_v0 | 455,954 | 512 | 0.9685 | 0.9685 | 0.9665 | baseline_not_beaten |

安全检查结果：paper_safe_split=True，uses_target_labels=False，group_overlap_train_val=[]，group_overlap_train_test=[]，group_overlap_val_test=[]，confounding_status=pass。

## 结论

ResNet18 明显提升了整体任务性能，但提升主要被 identity/source_standardize/ridge 等非 diffusion baseline 吸收。全量 ResNet18 下 latent_diffusion_v0 target AUC=0.9665，低于最强非 diffusion baseline 0.9685，差值为 -0.0020，没有达到 claim gate 要求的 +0.01 target AUC margin。

因此，当前 Camelyon17 feature-level 设定下 diffusion 路线不可行：不是因为 pipeline 没跑通，而是因为强 baseline 已经能很好处理该表征空间中的 shift。当前不应继续投入 neural diffusion v1，除非换到更有 domain-shift headroom 的任务、表征或数据设定，并重新通过同样的 paper-safe claim gate。
