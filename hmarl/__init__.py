"""HMARL-GRF: Hierarchical Multi-Agent Reinforcement Learning for GRF.

Package structure:
    hmarl.env          – GRF environment wrapper + state extraction
    hmarl.policy       – Hierarchical policy networks (High/Mid/Low)
    hmarl.expert       – Rule-based expert policy
    hmarl.reward       – Reward shaping (FAI, PPR, RCI)
    hmarl.rci          – Role Coherence Index computation
    hmarl.metrics      – Evaluation metrics suite
    hmarl.ippo         – Independent PPO baseline
    hmarl.shppo        – Shared PPO baseline (pooled experience)
    hmarl.mappo        – MAPPO baseline (centralized critic)
"""
