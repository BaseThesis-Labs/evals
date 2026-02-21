#!/usr/bin/env python3
"""Clean top-to-bottom flow diagram for the TTS benchmark."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── palette ──────────────────────────────────────────────────────────────────
BG      = '#111318'
PANEL   = '#1c1f2b'
BORDER  = '#2d3148'
TEXT    = '#e2e8f0'
MUTED   = '#8892a4'

BLUE    = '#3b82f6'
PURPLE  = '#8b5cf6'
GREEN   = '#10b981'
AMBER   = '#f59e0b'
PINK    = '#ec4899'
RED     = '#ef4444'
TEAL    = '#06b6d4'

fig, ax = plt.subplots(figsize=(16, 20))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 20)
ax.axis('off')

# ── helpers ───────────────────────────────────────────────────────────────────
def rbox(cx, cy, w, h, fc=PANEL, ec=BORDER, lw=1.5, radius=0.3, alpha=1.0):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f'round,pad=0,rounding_size={radius}',
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=2)
    ax.add_patch(p)

def label(x, y, s, size=9, color=TEXT, weight='normal', ha='center', va='center', style='normal'):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, fontstyle=style,
            fontfamily='DejaVu Sans')

def arr(x0, y0, x1, y1, color=BORDER, lw=1.8, head=0.25):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f'->', color=color,
                                lw=lw, mutation_scale=16,
                                connectionstyle='arc3,rad=0'))

def section_line(y, color=BORDER):
    ax.axhline(y=y, xmin=0.03, xmax=0.97, color=color, lw=0.6, alpha=0.4, zorder=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 0. TITLE
# ═══════════════════════════════════════════════════════════════════════════════
label(8, 19.4, 'TTS Benchmark — Data Flow & Pipeline', size=15, weight='bold', color=TEXT)
label(8, 19.0, '200 utterances  ·  9+ TTS models  ·  35 metrics  ·  5 use-case composites',
      size=9, color=MUTED)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
label(8, 18.5, '① DATASETS', size=8, color=MUTED, weight='bold')
section_line(18.3)

# Seed-TTS-Eval
rbox(4.5, 17.55, 5.0, 1.3, fc=PANEL, ec=BLUE, lw=2)
label(4.5, 18.05, 'Seed-TTS-Eval', size=10, weight='bold', color=BLUE)
label(4.5, 17.70, '200 sampled English utterances', size=8, color=MUTED)
label(4.5, 17.38, 'text  ·  reference audio  ·  speaker_id', size=7.5, color=MUTED, style='italic')

# LibriTTS-R
rbox(11.5, 17.55, 5.0, 1.3, fc=PANEL, ec=TEAL, lw=2)
label(11.5, 18.05, 'LibriTTS-R', size=10, weight='bold', color=TEAL)
label(11.5, 17.70, '50 / 200 sample splits', size=8, color=MUTED)
label(11.5, 17.38, 'clean read speech  ·  multiple speakers', size=7.5, color=MUTED, style='italic')

# arrow merge to manifest
arr(4.5, 16.89, 4.5, 16.25, color=BLUE)
arr(11.5, 16.89, 11.5, 16.25, color=TEAL)
# converge lines to center
ax.annotate('', xy=(8, 16.10), xytext=(4.5, 16.25),
            arrowprops=dict(arrowstyle='->', color=BORDER, lw=1.5, mutation_scale=14))
ax.annotate('', xy=(8, 16.10), xytext=(11.5, 16.25),
            arrowprops=dict(arrowstyle='->', color=BORDER, lw=1.5, mutation_scale=14))

# manifest file
rbox(8, 15.72, 5.2, 0.68, fc='#141929', ec=BORDER, lw=1.2, radius=0.2)
label(8, 15.85, 'datasets/manifest.json', size=8.5, weight='bold', color=TEXT)
label(8, 15.58, 'id  ·  text  ·  reference_audio_path  ·  speaker_id  ·  difficulty', size=7, color=MUTED)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TTS MODELS
# ═══════════════════════════════════════════════════════════════════════════════
arr(8, 15.37, 8, 14.90, color=BORDER)

label(8, 14.7, '② TTS MODELS  (configs/models.yaml)', size=8, color=MUTED, weight='bold')
section_line(14.55)

# two columns: Local and API
rbox(3.5, 13.55, 5.6, 1.65, fc=PANEL, ec=GREEN, lw=2)
label(3.5, 14.25, 'Local Models', size=9.5, weight='bold', color=GREEN)
local_models = [
    ('Kokoro ONNX',  'Fast · ONNX runtime · CPU/GPU'),
    ('Piper TTS',    'Lightweight · disabled (pkg conflict)'),
    ('Coqui XTTS v2','Voice cloning · disabled (py<3.12)'),
]
for i, (name, note) in enumerate(local_models):
    y = 13.90 - i * 0.38
    label(3.5, y,       f'• {name}', size=8, weight='bold', color=TEXT, ha='center')
    label(3.5, y - 0.17, note,       size=7,               color=MUTED, ha='center')

rbox(12.0, 13.55, 6.5, 1.65, fc=PANEL, ec=AMBER, lw=2)
label(12.0, 14.25, 'API Models', size=9.5, weight='bold', color=AMBER)
api_models = [
    'Deepgram Aura  ·  Cartesia Sonic  ·  ElevenLabs',
    'Google Cloud TTS  ·  Azure Speech  ·  AWS Polly',
    'LMNT  ·  Hume AI  ·  PlayHT (disabled)',
]
for i, line in enumerate(api_models):
    label(12.0, 13.95 - i * 0.38, line, size=7.8, color=MUTED, ha='center')

# arrows from manifest to both model boxes
arr(8, 15.37, 8, 14.90, color=BORDER)
ax.annotate('', xy=(3.5, 14.33), xytext=(6.5, 14.60),
            arrowprops=dict(arrowstyle='->', color=BORDER, lw=1.4, mutation_scale=12))
ax.annotate('', xy=(12.0, 14.33), xytext=(9.5, 14.60),
            arrowprops=dict(arrowstyle='->', color=BORDER, lw=1.4, mutation_scale=12))

# note about cloning
rbox(8, 12.74, 3.8, 0.42, fc='#1a1428', ec=PURPLE, lw=1, radius=0.15)
label(8, 12.74, '★  Cartesia & XTTS v2 use reference_audio for voice cloning',
      size=7, color=PURPLE)

arr(3.5, 12.62, 3.5, 12.40, color=GREEN)
arr(12.0, 12.62, 12.0, 12.40, color=AMBER)
ax.annotate('', xy=(8, 12.25), xytext=(3.5, 12.40),
            arrowprops=dict(arrowstyle='->', color=BORDER, lw=1.4, mutation_scale=12))
ax.annotate('', xy=(8, 12.25), xytext=(12.0, 12.40),
            arrowprops=dict(arrowstyle='->', color=BORDER, lw=1.4, mutation_scale=12))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GENERATE
# ═══════════════════════════════════════════════════════════════════════════════
label(8, 12.12, '③ GENERATE  (generate.py)', size=8, color=MUTED, weight='bold')
section_line(11.97)

rbox(8, 11.38, 9.5, 1.15, fc=PANEL, ec=BLUE, lw=2)
label(8, 11.76, 'For each model × utterance:', size=9, weight='bold', color=BLUE)
gen_steps = 'call client.generate(text, ref_audio, speaker_id)  →  resample to 24 kHz mono PCM-16  →  save .wav  +  log metadata'
label(8, 11.38, gen_steps, size=7.8, color=MUTED)
label(8, 11.03, 'Records:  inference_time_ms  ·  ttfa_ms (time-to-first-audio)  ·  duration  ·  rtf', size=7.5, color=MUTED)

arr(8, 10.80, 8, 10.38, color=BLUE)

# output file
rbox(8, 10.18, 6.5, 0.58, fc='#141929', ec=BORDER, lw=1.2, radius=0.2)
label(8, 10.30, 'generated_audio/{model}/{utt_id}.wav', size=8.5, weight='bold', color=TEXT)
label(8, 10.05, 'generated_audio/{model}/gen_meta.jsonl', size=7.5, color=MUTED)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
arr(8, 9.88, 8, 9.45, color=BORDER)

label(8, 9.30, '④ EVALUATE  (evaluate.py)  —  35 metrics per utterance', size=8, color=MUTED, weight='bold')
section_line(9.15)

# ASR strip
rbox(8, 8.78, 10.0, 0.62, fc='#17212e', ec=BLUE, lw=1.5, radius=0.2)
label(8, 8.90, 'ASR Transcription', size=8.5, weight='bold', color=BLUE)
label(8, 8.65, 'AssemblyAI  /  Deepgram  /  Whisper (local)  →  lowercase hypothesis text', size=7.5, color=MUTED)

arr(8, 8.46, 8, 8.04, color=BORDER)

# 5 metric columns
metric_cols = [
    (2.2,   'Intelligibility', BLUE,
     ['WER   (S+D+I)/N',
      'CER   char edit dist',
      'MER   1−match/ref',
      'word_skip_rate',
      'insertion_rate',
      'substitution_rate']),
    (5.1,   'Naturalness', PINK,
     ['UTMOS  [1–5]',
      '  torch.hub model',
      'DNSMOS  [1–5]',
      '  sig / bak / ovrl',
      '  ONNX @16 kHz',
      '  144 160 samples']),
    (8.0,   'Prosody', GREEN,
     ['f0_mean  (Hz)',
      'f0_std   (Hz)',
      'f0_range P95−P5',
      '  via Parselmouth',
      'pause_ratio',
      '  Silero VAD']),
    (10.9,  'Spk Similarity', PURPLE,
     ['Resemblyzer',
      '  256-dim embed',
      '  cosine sim [0,1]',
      '  ref vs generated',
      'ECAPA-TDNN',
      '  (disabled)']),
    (13.8,  'Robustness + Latency', AMBER,
     ['has_repetition',
      '  ngram ≥ 3×',
      'is_empty_or_short',
      '  dur < 0.1 s',
      'rtf = inf/(1000×dur)',
      'ttfa_ms  first chunk']),
]

box_top = 7.87
for cx, title, col, items in metric_cols:
    h = 2.20
    rbox(cx, box_top - h/2 + 0.12, 2.55, h, fc=PANEL, ec=col, lw=1.8, radius=0.25)
    # header bar
    rbox(cx, box_top, 2.55, 0.44, fc=col, ec=col, lw=0, radius=0.20, alpha=0.9)
    label(cx, box_top, title, size=7.5, weight='bold', color='#ffffff')
    for i, item in enumerate(items):
        label(cx, box_top - 0.52 - i * 0.30, item, size=6.8, color=MUTED)

arr(8, 5.66, 8, 5.22, color=BORDER)

# output file
rbox(8, 5.02, 6.2, 0.58, fc='#141929', ec=BORDER, lw=1.2, radius=0.2)
label(8, 5.14, 'results/{model}_metrics.json', size=8.5, weight='bold', color=TEXT)
label(8, 4.89, 'per-utterance · 35 fields each', size=7.5, color=MUTED)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. AGGREGATE
# ═══════════════════════════════════════════════════════════════════════════════
arr(8, 4.70, 8, 4.28, color=BORDER)

label(8, 4.12, '⑤ AGGREGATE  (aggregate.py)', size=8, color=MUTED, weight='bold')
section_line(3.98)

rbox(8, 3.35, 14.0, 1.10, fc=PANEL, ec=GREEN, lw=2)
label(8, 3.82, 'Normalise all metrics to [0,1]  →  Average into 6 Dimensions  →  Weighted sum into 5 Use-Case Composites',
      size=8.5, weight='bold', color=GREEN)

dims = [
    ('Naturalness',     PINK,   '0.35 / 0.20 / 0.20 / 0.15 / 0.20'),
    ('Intelligibility', BLUE,   '0.15 / 0.25 / 0.20 / 0.20 / 0.20'),
    ('Spk Similarity',  PURPLE, '0.10 / 0.00 / 0.40 / 0.00 / 0.15'),
    ('Prosody',         GREEN,  '0.25 / 0.10 / 0.05 / 0.05 / 0.10'),
    ('Robustness',      AMBER,  '0.15 / 0.15 / 0.10 / 0.15 / 0.15'),
    ('Latency',         RED,    '0.00 / 0.30 / 0.10 / 0.45 / 0.20'),
]
dim_xs = [1.6, 4.1, 6.4, 8.7, 11.0, 13.4]
for dx, (dname, col, weights) in zip(dim_xs, dims):
    label(dx, 3.40, dname,   size=7.5, weight='bold', color=col)
    label(dx, 3.12, weights, size=6.3, color=MUTED)

label(1.0, 2.88, 'Weights:', size=6.5, color=MUTED, ha='left')
uc_labels = 'Audiobook / Convo-AI / Voice Clone / Low-Latency / Balanced'
label(8, 2.88, uc_labels, size=6.5, color=MUTED)

arr(8, 2.78, 8, 2.38, color=GREEN)

# output
rbox(8, 2.18, 6.2, 0.58, fc='#141929', ec=BORDER, lw=1.2, radius=0.2)
label(8, 2.30, 'analysis/leaderboard.json', size=8.5, weight='bold', color=TEXT)
label(8, 2.05, 'dimension scores  ·  composites  ·  rankings  ·  Wilcoxon p-values', size=7.5, color=MUTED)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZE
# ═══════════════════════════════════════════════════════════════════════════════
arr(8, 1.87, 8, 1.48, color=BORDER)

label(8, 1.34, '⑥ VISUALIZE  (visualize.py)', size=8, color=MUTED, weight='bold')
section_line(1.20)

charts = [
    ('01 Leaderboard',   'Bar chart\nBalanced score'),
    ('02 Radar',         '6-axis polar\nper model'),
    ('03 Heatmap',       'Metrics grid\nnormalised'),
    ('04 Use Cases',     'Grouped bars\ncomposite scores'),
    ('05 WER Dist.',     'Box plot\nper utterance'),
    ('06 UTMOS Dist.',   'Violin plot\nper utterance'),
]
cxs = [1.2, 3.7, 6.0, 9.0, 11.4, 14.2]
for cx, (ctitle, cdesc) in zip(cxs, charts):
    rbox(cx, 0.70, 2.10, 0.84, fc=PANEL, ec=AMBER, lw=1.5, radius=0.2)
    label(cx, 0.95, ctitle, size=7.5, weight='bold', color=AMBER)
    for j, line in enumerate(cdesc.split('\n')):
        label(cx, 0.68 - j * 0.20, line, size=6.5, color=MUTED)

plt.tight_layout(pad=0)
plt.savefig('analysis/architecture.png', dpi=160, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print("Saved → analysis/architecture.png")
plt.show()
