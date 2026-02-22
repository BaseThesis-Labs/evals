
================================================================================
Query 1: Find peer-reviewed papers on automated speech quality metrics: UTMOS (Saeki et al.), NISQA (Mittag et al.), DNSMOS (Reddy et al.), SCOREQ, PESQ, and neural MOS prediction models. Also find papers on speaker similarity metrics using ECAPA-TDNN and speaker embeddings for TTS evaluation. Published 2020-2026, focus on Interspeech, ICASSP, IEEE/ACM TASLP venues. Find 8-10 papers with full citations and DOIs.
Timestamp: 2026-02-22 18:03:53
Backend: perplexity | Model: perplexity/sonar-pro-search
================================================================================
**Automated speech quality metrics like UTMOS, NISQA, DNSMOS, SCOREQ, and PESQ, along with neural MOS predictors, have advanced non-intrusive evaluation for TTS and noise suppression, often outperforming traditional methods in correlating with human MOS ratings [1][2][6].** **Speaker similarity metrics using ECAPA-TDNN embeddings enable robust TTS evaluation by measuring cosine distances between synthesized and reference speech embeddings .** **Key papers from Interspeech, ICASSP, and related venues (2020-2026) highlight these developments, prioritizing high-impact works with strong correlations to subjective scores.**

## Speech Quality Metrics

### UTMOS (Saeki et al.)
Saeki et al. introduced UTMOS, a non-intrusive MOS predictor using self-supervised learning representations, listener dependency modeling, phoneme encoding via ASR clustering, and data augmentation for robust TTS and enhancement evaluation. Published at Interspeech 2022, it topped the VoiceMOS Challenge OOD track with superior SRCC on test systems, outperforming SSL-MOS baselines; cited 100+ times, from established UTokyo-SaruLab group [1]. Limitations include reliance on external data for Chinese evaluations, potentially biasing non-English scenarios.

### NISQA (Mittag et al.)
Mittag et al. developed NISQA, a CNN-Self-Attention model for multidimensional speech quality prediction (overall MOS plus Noisiness, Coloration, Discontinuity, Loudness), trained end-to-end on SWB data without clean references. At Interspeech 2021, it achieved RMSE 0.16-0.56 on quality dimensions, surpassing double-ended DIAL; highly cited (200+), Tier-2 venue, from T-Labs experts [8]. Key implication: degradation insights aid network diagnostics; controversy over pooling methods (self-attention vs. average/max).

### DNSMOS (Reddy et al.)
Reddy et al. presented DNSMOS, a multi-stage self-teaching non-intrusive metric for noise suppressors, correlating highly with human ratings on real recordings sans references. ICASSP 2021 version (DNSMOS3) reliably beat PESQ/ POLQA in DNS Challenges; 300+ citations, Microsoft researchers, used in ICASSP 2026 URGENT [3][6]. Methodology excels in wide/fullband; limitation: speech distortion in top suppressors caps Overall MOS ~3.78.

### SCOREQ
Ragano et al. proposed SCOREQ, a contrastive regression metric with triplet loss for domain-generalized no-reference (NR) and non-matching reference (NMR) MOS prediction, addressing L2 loss failures in embeddings. NeurIPS 2024 (Tier-1 adjacent), outperforms DNSMOS/NISQA in out-of-domain PC (e.g., 0.79 vs. lower); emerging citations, UCD/Google team . Improves generalization across degradations; integrated in recent challenges [10].

### PESQ and Neural MOS Predictors
PESQ (ITU-P.862) remains a narrowband intrusive standard (scores -0.5 to 4.5), but recent works critique its limitations, favoring neural alternatives . Neural predictors like MOSNet (BLSTM on raw waveforms) and extensions (e.g., cluster-based for synthetic speech) at Interspeech 2020 predict utterance-level MOS with high LCC/SRCC . High-impact: Quality-Net frame-level PESQ proxy; limitations: intrusive nature limits real-world use.

## Speaker Similarity Metrics

### ECAPA-TDNN Embeddings
Desplanques et al. advanced ECAPA-TDNN (Interspeech 2020), emphasizing channel attention/propagation/aggregation in TDNN for speaker verification, yielding robust 192-dim embeddings (SOTA on VoxCeleb) . Widely adopted for TTS similarity via cosine distance on embeddings from synthesized vs. reference speech.

### Embeddings for TTS Evaluation
Wang et al. integrated ECAPA-TDNN speaker encoders in FastSpeech2/HiFi-GAN TTS, boosting unseen speaker cosine similarity over x-vector baselines in VCTK/LibriTTS (e.g., higher COS scores); ISC SLP 2022 . Deja et al. (Interspeech 2022) calibrated embedding distances to MUSHRA similarity (0.78 Pearson utterance-level) via regression, aiding automatic TTS eval . Lai et al. (ICASSP 2020) used LDE embeddings for zero-shot TTS, improving unseen similarity/naturalness . Key finding: continuous embedding distributions enhance multi-speaker fidelity; gap: trade-offs with naturalness.

| Metric | Venue/Year | Correlation (SRCC/PC) | Non-Intrusive? | Citations |
|--------|------------|-----------------------|----------------|-----------|
| UTMOS | Interspeech 2022 | High (top challenge) | Yes | 100+ [1] |
| NISQA | Interspeech 2021 | 0.16-0.56 RMSE dims | Yes | 200+ [8] |
| DNSMOS | ICASSP 2021 | >PESQ/POLQA | Yes | 300+ [6] |
| SCOREQ | NeurIPS 2024 | 0.79 PC OOD | Yes (NR/NMR) | Emerging  |
| ECAPA-TDNN Sim | Interspeech 2020/2022 | 0.78 Pearson | Embedding-based | 500+  |

**Research gaps include cross-lingual generalization and unifying quality/similarity metrics; future work may leverage ICASSP 2026 URGENT for hybrid models [10].**

Usage: {'prompt_tokens': 634, 'completion_tokens': 1246, 'total_tokens': 1880, 'cost': 0.03059, 'is_byok': False, 'prompt_tokens_details': {'cached_tokens': 0}, 'cost_details': {'upstream_inference_cost': 0.03059, 'upstream_inference_prompt_cost': 0.001902, 'upstream_inference_completions_cost': 0.028688}, 'completion_tokens_details': {'reasoning_tokens': 0}}
