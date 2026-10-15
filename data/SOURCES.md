# Data origin

## Composite indices (`pharma_indices.csv`)
Biennial 2010-2024. Constructed before notebook execution from raw federal datasets; every
raw series is min-max normalized over 2010-2024 (paper Sec. 2.4, Eqs. 10-14):
- **x** (ethical firms) = 0.4(1 - recalls) + 0.3(1 - warning letters) + 0.3(audit pass rate)
- **y** (regulators) = 0.5(inspection activity) + 0.5(enforcement activity)
- **z** (public) = mean of litigation, media coverage, trust indicators
- **k** (friction) = mean of approval delay, compliance cost, administrative burden

Raw sources: FDA recall, warning-letter and inspection dashboards; DOJ settlement press releases;
Good Jobs First Violation Tracker; SEC 10-K filings; Public Citizen (2024) penalty report.

## Firm table (`firm_classification.csv`)
- **Inclusion rule.** Every company on Public Citizen's top-20 list of 1991-2021 penalty
  totals that existed as an independent company in fiscal 2019 and has all three numbers
  published: 15 firms.
  - Excluded: Purdue (private; no audited R&D or revenue), Reckitt Benckiser (no FY2019 R&D
    figure obtained), and Schering-Plough, TAP and Serono (no longer existed in 2019).
- `total_penalties_musd`: Public Citizen (Abrams and Wolfe, May 2024), Table 4, federal plus
  state totals.
- `rnd_pct_of_revenue`: FY2019 R&D expense divided by FY2019 revenue. The figures used and the
  document they come from are in `rnd_basis` and `rnd_source`.
- `revenue_2019_musd`: FY2019 revenue as reported, from the document named in
  `revenue_basis`. Pfizer is as originally reported in its FY2019 10-K; GSK is converted to USD.
- `strategy_zone`: fixed thresholds on R&D intensity (<10% S2, 10-21% Mixed, >21% S1).
