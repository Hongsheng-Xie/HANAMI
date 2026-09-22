# Evaluate the canonical Rmd and save its figures/tables without Pandoc or LaTeX.
# Rscript tools/figure5/render_figure5.R [OUTPUT_DIRECTORY]
args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script <- normalizePath(sub("^--file=", "", script_arg[[1L]]), winslash = "/")
repo_root <- normalizePath(file.path(dirname(script), "../.."), winslash = "/")
analysis_dir <- file.path(repo_root, "analysis/clinical_concordance")
output <- if (length(args)) args[[1L]] else file.path(repo_root, "results/clinical_concordance/figure5")
dir.create(output, recursive = TRUE, showWarnings = FALSE)
output <- normalizePath(output, winslash = "/", mustWork = TRUE)
packages <- c("knitr", "yaml", "ggplot2", "tidyr", "dplyr", "patchwork", "scales")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Install required R packages: ", paste(missing, collapse = ", "))
rmd <- file.path(analysis_dir, "plot_figure5.Rmd")
spec <- knitr::knit_params(readLines(rmd, warn = FALSE))
params <- lapply(spec, function(x) x$value)
# Keep the Rmd's prepared input paths; only generated output paths are changed.
for (key in grep("^output_", names(params), value = TRUE)) {
  params[[key]] <- file.path(output, basename(params[[key]]))
}
env <- new.env(parent = globalenv())
env$params <- params
knitr::opts_chunk$set(error = FALSE, fig.path = paste0(output, "/.render/"))
knitr::knit(rmd, output = file.path(output, "figure5_render.md"), envir = env, quiet = TRUE)
versions <- setNames(lapply(packages, function(x) as.character(packageVersion(x))), packages)
writeLines(c(R.version.string, paste(names(versions), unlist(versions))),
           file.path(output, "r_package_versions.txt"))
