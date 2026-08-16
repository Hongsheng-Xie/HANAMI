# DRKG validation plots

`plot_drkg_results.Rmd` contains the R workflow used to summarize the ten-seed DRKG benchmark results and prepare the cross-network validation plots. Set `filename` in the first data-loading chunk to the result workbook containing the `DRKG` worksheet, then knit the document in R.

Required R packages: `ggplot2`, `tidyr`, `dplyr`, `readr`, `stringr`, `readxl`, `patchwork`, `ggsignif`, and `grid`.
