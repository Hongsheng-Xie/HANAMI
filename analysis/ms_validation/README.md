# MS validation plots

`plot_ms_results.Rmd` contains the R workflow used to summarize the ten-seed MS benchmark results and prepare the MS validation plots. Set `filename` in the first data-loading chunk to the result workbook containing the `MS` worksheet, then knit the document in R.

Required R packages: `ggplot2`, `tidyr`, `dplyr`, `readr`, `stringr`, `readxl`, `patchwork`, `ggsignif`, and `grid`.
