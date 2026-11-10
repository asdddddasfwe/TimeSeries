# Vendored PyPOTS TimeMixer++

Source: https://github.com/WenjieDu/PyPOTS
Commit: 53b3eac34be9491ac3f28e65ee1993436e9318af
License: BSD-3-Clause (full notice in LICENSE).
UPSTREAM.json records the original paths and hashes, not local modified hashes.

The retained local forecast path uses average pooling and independent channel
encoding with coarsest-scale channel attention. FFT periods use the coarsest
length. RowAttention (within a period) feeds bottom-up seasonal mixing, while
ColAttention (across periods) feeds top-down trend mixing. Historical module names
are preserved for loading the best checkpoint. All prediction heads are averaged
before the finest-scale RevIN inverse.

Local edge-case fixes exclude the DC frequency, clamp top_k, promote half/bfloat16
FFT to float32, and preserve padding dtype. Alternative masking, channel,
routing and output-aggregation branches have been removed, along
with upstream task methods that were never exposed by the forecasting adapter.

This is a locally selected forecast model, not an exact upstream or paper
reproduction. See docs/TimeMixerPPPyPOTS.md for configuration and validation.

Shared-convolution downsampling and original-length period ablations were
removed on 2026-09-21. Only average pooling and coarsest-grid periods remain.
Historical experiment artifacts and the upstream license are preserved.
