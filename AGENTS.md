# AGENTS.md

Guidelines for Codex / research agents working on this repository.

## Project Goal

This project investigates diffusion-based harmonization for multi-center medical data. The target claim is not simply "diffusion removes site shift". The target claim is:

> Diffusion harmonization improves held-out-center clinical generalization while reducing site/device/protocol shift and preserving clinically meaningful signal.

## Non-Negotiable Rules

1. Do not tune on the held-out center labels.
2. Do not use target-center labels during training unless the experiment is explicitly supervised target adaptation.
3. Always separate:
   - transductive target-unlabeled adaptation;
   - zero-shot unseen-center generalization.
4. Always compare against strong non-diffusion baselines:
   - no harmonization;
   - ComBat;
   - CORAL;
   - MMD;
   - adversarial domain adaptation;
   - VAE.
5. Always report both:
   - site/domain removal metrics;
   - clinical task preservation/generalization metrics.
6. Do not claim success if site classifier accuracy drops but clinical AUC/Dice/regression performance also drops.
7. Audit label-site confounding before making any clinical claim.
8. Keep data out of git. Use `data/` and `results/` only as local ignored folders.

## Required Reading Order

1. `README.md`
2. `docs/research_plan.md`
3. `docs/experiment_protocol.md`
4. `docs/method_design.md`
5. `docs/codex_initial_research_prompt.md`

## Required Test Command

After code changes, run:

```bash
pytest -q
```

Report the exact command and result.
