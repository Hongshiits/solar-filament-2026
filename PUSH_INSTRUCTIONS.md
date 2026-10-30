# Publish this repository (user action, ~10 minutes)

This folder is already a git repository with one commit containing code, report, notebook and weights.
The two weight files are under GitHub's 100 MB per-file limit, so plain git works (no LFS needed).

## Steps

1. Create an empty **public** GitHub repository, e.g. `<your-github-name>/solar-filament-2026` (no README auto-init).
2. Replace the owner placeholder in `README.md` (line with the `git clone` command):
   `TODO-GITHUB-OWNER` -> your GitHub user name.
3. Push:

       cd repo
       git remote add origin https://github.com/<your-github-name>/solar-filament-2026.git
       git branch -M main
       git push -u origin main

4. On GitHub: Settings -> General -> confirm the repository is **public**; check that README renders,
   `docs/report.pdf` opens, and `weights/` contains `best_det.pt` (20 MB) and `best_ref.pt` (57 MB).
5. Copy the repository URL into the official Google Form together with `docs/report.pdf`
   (see `docs/DELIVERABLES.md` for the exact form link and remaining checklist items).

## Optional checks before pushing

- `python src/validate_submission.py submissions/dariush_unet_full_v1.csv` (submission format sanity);
- open `notebooks/final_pipeline.ipynb` and confirm the CPU smoke block runs (it was verified on 2 images);
- keep the repository public until after the winners are announced.

## Notes

- The report (docs/report.pdf) is built from docs/main.tex with the official acmart template; it compiles with
  `pdflatex main.tex` (tested: 6 pages, 1.93 MB).
- If you prefer to drop the weights from git, move `repo/weights/*.pt` to a GitHub Release and remove them from
  the commit; the README already explains how to wire the download.
