# Evolutionary Stability of Ethics in Pharma

Code and data for "Cutting Corners or Saving Lives? The Evolutionary Stability of Ethics in
Pharma Medicine" (Polyak and Zdyrski).

The model is a three-population replicator system for ethical firms (*x*), active regulators
(*y*) and active public scrutiny (*z*), with bureaucratic friction *k*. Eight payoff constants
are fitted with bounded L-BFGS-B to biennial composite indices for 2010-2024. The notebook
regenerates every figure in the paper.

## Contents

```
egt_core.py                  model, calibration, fixed points, thresholds, conserved quantity,
                             Lyapunov exponents
egt_pharma_analysis.ipynb    makes every figure (executed copy, outputs included)
interactive_dashboard.ipynb  interactive version of the dashboard (Appendix D)
data/
  pharma_indices.csv         composite x, y, z, k indices (biennial, 2010-2024)
  firm_classification.csv    15 firms: zone, R&D intensity, penalties, 2019 revenue, sources
  SOURCES.md                 how the indices and the firm table were built
figures/                     generated figures (PNG, 300 dpi)
scripts/
  notebook_src.py            plain-Python source of the notebook
  build_notebook.py          builds and runs the notebook; copies figures into paper/figures
chaos/                       Lyapunov-exponent and trajectory scripts
paper/
  main.tex, references.bib   manuscript (biblatex + biber)
  figures/                   figures used by the LaTeX files
pyproject.toml, requirements.txt
```

## Running

```bash
pip install -r requirements.txt
jupyter lab                  # open egt_pharma_analysis.ipynb and run all cells
# or, without Jupyter:
python scripts/build_notebook.py scripts/notebook_src.py egt_pharma_analysis.ipynb
```

The notebook also runs on its own (for example in Google Colab): it contains a copy of 
`egt_core.py` and of both data files and uses them when the repository files are not present. 
After running the python command above, the `figures/` and `paper/figures/` directories will be populated with
the generated images, and `egt_pharma_analysis.ipynb` will be updated with the images. The interactive dashboard
of Appendix D is in a separate notebook, `interactive_dashboard.ipynb`, which needs `ipywidgets` (included in
`requirements.txt`). An interactive instance of the `interactive_dashboard.ipynb` notebook is available on mybinder at
URL.

## Building the paper

Upload the contents of `paper/` to Overleaf.

## Key quantities (printed by the notebook)

| quantity | value |
|---|---|
| $`x_z^* = 1 - C_r/B_r`$ (x-coordinate of the $`z = 0`$ center) | 0.900 |
| $`x_y^* = 1 - C_p/B_p`$ (x-coordinate of the $`y = 1`$ center) | 0.887 |
| $`dV/dt = x_z^* - x_y^*`$, with $`V = \ln[y/(1-y)]/B_r - \ln[z/(1-z)]/B_p`$ | 0.0133 |
| decay rate of $`z`$ near $`z = 0`$, $`B_p(x_y^* - x_z^*)`$ | -0.0119 |
| $`k^* = (S_{pol} + \alpha - C_{base})^{1/3}`$ | 0.754 |
| $`k^{**} = (S_{pol} + \alpha + \gamma - C_{base})^{1/3}`$, collapse threshold | 0.809 |
| period of small orbits at $`k = 0.60`$ / of the Fig. 1 orbit | 80.3 / 104.6 |
| Lyapunov exponents at $`k = 0.60`$, $`T = 6000`$ | (0.0007, -0.0017, -0.0126) |

## Figures

| Paper | File | Content |
|---|---|---|
| 1 | `fig1.png` | core dynamics, k = 0.60 |
| 2 | `fig2.png` | phase portraits |
| 3 | `fig3.png` | noise, path dependence, delay |
| 4 | `fig4.png` | policy experiments and sensitivity |
| 5 | `fig5.png` | two-parameter maps |
| 6 | `fig6.png` | spatial lattice |
| 7 | `fig7.png` | calibration and projection |
| 8 | `fig8.png` | the empirical record |
| 9 | `fig_basin.png` | grid of starting points |
| 10 | `fig_stability.png` | conserved quantity and face centers |
| 11 | `fig_bifurcation.png` | long-run behavior across k |
| 12 | `fig_zdecay.png` | decay of public scrutiny |
| 13 | `fig_lyapunov.png` | Lyapunov exponents |
| 14 | `fig_rnd_penalty.png` | R&D intensity vs penalties |
| 15 | `fig_rnd_approvals.png` | R&D spending and approvals |
| 16 | `fig_recalls.png` | recalls and inspections |
| 17 | `fig_prv.png` | priority review voucher prices |
| 18 | `fig_zones.png` | penalties by strategy zone |
| 19 | `fig_dashboard.png` | dashboard examples (Appendix D) |
