# Transfer and cold-start validation plots

`plot_transfer_results.Rmd` contains the R workflow used to summarize the ten-seed transfer-learning results and prepare the cold-start validation plots. Set `filename` in the first data-loading chunk to the result workbook containing the `Transfer` worksheet, then knit the document in R.

Required R packages: `ggplot2`, `tidyr`, `dplyr`, `readr`, `stringr`, `readxl`, `patchwork`, `ggsignif`, and `grid`.
