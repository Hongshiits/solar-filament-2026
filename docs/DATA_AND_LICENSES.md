# Data, licenses and competition compliance

## Competition data

- Dataset shipped with the competition: MAGFiLO 1.0 / MLEcoFi (Manually Annotated GONG Filaments from H-alpha Observations), converted to 8-bit JPEG, 2048×2048 px.
- GONG H-alpha images: NSO Integrated Synoptic Program; see attribution below.
- License: **CC BY-NC 4.0** (Attribution-NonCommercial 4.0 International), as declared on the competition page. **Non-commercial / academic use only.**
- The data is **not** redistributed in this repository. Download it from the competition Data tab and unzip next to the repo (see README §5).

## Ground truth and rules

- Only the annotations provided by the competition are used for training/evaluation.
- The publicly released MAGFiLO dataset contains annotations that overlap the competition test set; **those were not used** in any form.
- Inference uses **only** the H-alpha test images provided by the competition. No neighbour/future frames, no extra observations, no metadata leaks (competition rules §6 and forum clarifications).
- External data used for training: None (empty if none).

## Code license

- Code: MIT (see ../LICENSE).
- Trained weights: provided for reproduction of this competition submission, non-commercial use only.

## Attribution

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar Observatory, which is operated by the Association of Universities for Research in Astronomy (AURA), Inc. under a cooperative agreement with the National Science Foundation and with contribution from the National Oceanic and Atmospheric Administration. The GONG network of instruments is hosted by the Big Bear Solar Observatory, High Altitude Observatory, Learmonth Solar Observatory, Udaipur Solar Observatory, Instituto de Astrofísica de Canarias, and Cerro Tololo Interamerican Observatory.

Ground truth: Ahmadzadeh et al., "A dataset of manually annotated filaments from H-alpha observations", Scientific Data 11, 1031 (2024), doi:10.1038/s41597-024-03876-y.

Competition: Solar Filament Segmentation Challenge 2026, Earth-Space AI Research (ESAIR) Lab, https://www.kaggle.com/competitions/filament-segmentation-2026.
