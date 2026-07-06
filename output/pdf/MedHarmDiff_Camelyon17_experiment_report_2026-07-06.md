# MedHarmDiff Camelyon17-WILDS 实验报告

报告日期：2026-07-06

分支：`codex/mvp-benchmark`

最新提交：`e68332e Cap latent diffusion training pairs for full benchmarks`

## 一句话结论

MedHarmDiff 的真实数据 benchmark pipeline 已经成立：Camelyon17-WILDS 数据、paper-safe split、target labels 不进训练、group overlap 为空、embedding-family benchmark 和 claim gate 都已跑通。

当前不能声称 diffusion 已经带来贡献：全量 ResNet18 下 `latent_diffusion_v0` 的 target AUC 为 `0.9665`，低于最强非 diffusion baseline `0.9685`，claim gate 仍为 `baseline_not_beaten`。

## 全量 embedding-family 结果

| embedding | samples | dim | best statistical | ridge | latent diffusion | claim gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| color_stats_v0 | 455954 | 37 | 0.8853 | 0.8991 | 0.8975 | baseline_not_beaten |
| resnet18_imagenet_v0 | 455954 | 512 | 0.9685 | 0.9685 | 0.9665 | baseline_not_beaten |

## 决策

继续暂停 neural diffusion v1。下一阶段应把项目定位为“安全、可复现、多中心 harmonization benchmark 已成立；diffusion 尚未超过强 baseline”。
