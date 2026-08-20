# HMARL-GRF

Hierarchical Multi-Agent Reinforcement Learning untuk Google Research Football.

Implementasi arsitektur hierarkis 3-level sesuai tesis:
- **High-Level** (rule-based) → strategi makro (pressing, counter-attack, possession)
- **Mid-Level** (rule-based) → sub-goal per agen (press, mark, hold)
- **Low-Level** (PPO) → aksi per agen

## Struktur

```
hmarl-grf/
├── hmarl/                    # Package utama
│   ├── __init__.py
│   ├── env.py                # GRF env wrapper, state extraction
│   ├── policy.py             # Jaringan hierarkis (High/Mid/Low)
│   ├── expert.py             # Policy ahli rule-based (referensi RCI)
│   ├── reward.py             # Reward shaping (FAI, PPR, RCI)
│   ├── rci.py                # Role Coherence Index
│   ├── metrics.py            # Metrik evaluasi
│   └── ippo.py               # Baseline Independent PPO
├── scripts/
│   ├── train.py              # Training HMARL
│   └── eval.py               # Evaluasi lengkap
├── evaluation/               # Utilitas analisis (kerja sebelumnya)
│   ├── average_position.py
│   ├── coordination_metrics.py
│   └── baselines/            # Baseline lama (A2C, PPO, Random)
├── Dockerfile
├── docker-compose.yml
├── setup.py
└── README.md
```

## Setup

```bash
docker compose up -d
docker exec -it gfootball-dev bash
```

Working directory di dalam container: `/gfootball` (volume-mounted dari host).

## Training

```bash
# Train HMARL
python3 scripts/train.py \
    --timesteps 50000 \
    --eval-freq 5000 \
    --log-dir logs/ \
    --model-dir checkpoints/

# Resume dari checkpoint
python3 scripts/train.py --resume checkpoints/hmarl_model.pt

# Train IPPO baseline
python3 hmarl/ippo.py --timesteps 50000 --log-dir dumps/
```

> **Catatan:** Training GRF 11v11 lambat (~58 steps/s raw, lebih lambat dengan hierarki). Satu episode penuh butuh ~5-10 menit. Gunakan `--timesteps` kecil untuk testing, misal `100` atau `300`.

## Evaluasi

```bash
# Evaluasi model HMARL
python3 scripts/eval.py --checkpoint checkpoints/hmarl_model.pt --episodes 10

# Evaluasi dengan baseline random
python3 scripts/eval.py --checkpoint checkpoints/hmarl_model.pt --episodes 10 --baseline random

# Evaluasi semua baseline
python3 scripts/eval.py --checkpoint checkpoints/hmarl_model.pt --episodes 10 --baseline all
```

> **Catatan:** Satu episode evaluasi butuh ~5-10 menit (3000 GRF steps + inferensi hierarki). Gunakan `--episodes` kecil untuk testing.

### Output Evaluasi

File JSON disimpan di `evaluation_results/`:
- `hmarl_results.json` — Metrik HMARL
- `hmarl_episodes.json` — Detail per episode

Contoh output:
```
============================================================
  Evaluation Results: HMARL
============================================================

  --- Performance Metrics ---
  Win Rate:              0.0%
  Goal Difference:       0
  Goals For / Against:   0 / 0
  Cumulative Reward:     -48.96

  --- Coordination Metrics ---
  PSR:                   25.00% (10/40)
  PPR:                   0.00% (0/10)
  Positional Entropy:    2.1499 bits
  Team Compactness:      15.2605 ± 5.0603
  FAI:                   0.7183 ± 0.0593

  --- Role Coherence Index ---
  RCI_strict:            0.0053
  RCI_cat:               0.8860
============================================================
```

## Argumen

| Script | Argumen | Default | Keterangan |
|--------|---------|---------|------------|
| `train.py` | `--timesteps` | 3000 | Jumlah timestep training |
| | `--eval-freq` | 5000 | Frekuensi evaluasi |
| | `--log-dir` | `logs/` | Direktori log TensorBoard |
| | `--model-dir` | `checkpoints/` | Direktori simpan model |
| | `--resume` | - | Path checkpoint untuk lanjutkan |
| | `--render` | off | Tampilkan rendering |
| `eval.py` | `--checkpoint` | **required** | Path model `.pt` |
| | `--episodes` | 100 | Jumlah episode evaluasi |
| | `--output-dir` | `evaluation_results` | Direktori output |
| | `--baseline` | - | `random` atau `all` |
| | `--render` | off | Tampilkan rendering |
| `ippo.py` | `--timesteps` | 3000 | Jumlah timestep training |
| | `--log-dir` | `dumps/` | Direktori log |
| | `--render` | off | Tampilkan rendering |

## Metrik

| Metrik | Keterangan | Referensi |
|--------|------------|-----------|
| WR | Win Rate | BAB 4 |
| GD | Goal Difference | BAB 4 |
| PSR | Pass Success Ratio | BAB 3 |
| PPR | Pass Progression Ratio | BAB 3 |
| FAI | Formation Adherence Index | BAB 3 |
| H | Positional Entropy (bits) | BAB 3 |
| TC | Team Compactness | BAB 3 |
| RCI_strict | Role Coherence Index (exact match) | BAB 3 — metrik novel |
| RCI_cat | Role Coherence Index (category match) | BAB 3 — metrik novel |

## Dependensi

- Python 3.6+
- PyTorch (CUDA)
- NumPy
- Google Research Football (via Docker)
- TensorBoard (opsional)

## Referensi

Tesis: *Hierarchical Multi-Agent Reinforcement Learning untuk Koordinasi Taktik dalam Google Research Football*
Universitas Gadjah Mada, 2026.
