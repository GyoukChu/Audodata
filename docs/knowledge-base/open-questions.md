# Open questions (to revisit after the smoke test)
- Throughput of GLM-5.3-NVFP4 at reasoning_effort=max: if judge calls dominate wall-clock, consider lowering the
  judge's effort (config knob exists).
- Whether vLLM 0.30 loads nvidia/GLM-5.3-NVFP4 (ModelOpt) cleanly; fallback Inferact/GLM-5.3-NVFP4.
- Whether the co-located solvers get enough KV cache for 3 concurrent 32k-token attempts per paper at the target
  paper-level concurrency; tune max-num-seqs / memory fractions.
- Semantic Scholar rate limit behaviour under sustained use (429s observed even at 1.3 s spacing).
- Whether accepted-item statistics land near Table 1 (weak 0.458 / strong 0.772 / gap 0.314 / 6.59 rounds) given
  the model substitutions; a much stronger "weak" solver than the paper's (Qwen3.5-4B is the same, fine) but a much
  weaker "strong" solver (27B dense vs 397B MoE) may lower acceptance rates.
- Later phases: legal (App. C.2), scientific (App. C.3), meta-optimization (Sec 4), RL (GRPO).
