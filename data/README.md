# Data

Raw data is **not** committed. Download it yourself and unzip it into `data/raw/`; the loader finds the files in nested folders.

## Zhang et al. (2020) EIS and capacity data
- Source: https://zenodo.org/records/3633835 (DOI 10.5281/zenodo.3633835)
- Licence: CC BY 4.0
- Download `EIS data.zip` (about 27 MB) and `Capacity data.zip` (about 92 MB), then unzip both into `data/raw/`.
- Paper: Y. Zhang, Q. Tang, Y. Zhang, J. Wang, U. Stimming, A. A. Lee, "Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning", *Nature Communications* 11, 1706 (2020). https://doi.org/10.1038/s41467-020-15235-7

## Tutorial CSV (optional, used only for cross-checking)
- `data_eis_zhang2020.csv` and `freq_eis_zhang2020.csv` from https://github.com/battery-data-commons/mrs-sp22-tutorial (folder `predict_capacity_from_eis/data/`, MIT licence)
- Copy both into `data/raw/`.

## Expected layout
```
data/raw/
├─ EIS data/            EIS_state_<I..IX>_<25|35|45>C<01..08>.txt
├─ Capacity data/       Data_Capacity_<cell>.txt
├─ data_eis_zhang2020.csv   (optional)
└─ freq_eis_zhang2020.csv   (optional)
```
