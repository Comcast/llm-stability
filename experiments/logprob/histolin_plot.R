#!/usr/bin/env Rscript

# Load required libraries
library("vioplot")

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript histolin_plot.R <answer_file.csv> <resp_file.csv> <output_plot.png> <plot_title>")
}

answer_file <- args[1]
resp_file <- args[2]
output_file <- args[3]
plot_title <- args[4]

# Read CSV files
answer_df <- read.csv(answer_file, header = TRUE, stringsAsFactors = TRUE)
resp_df <- read.csv(resp_file, header = TRUE, stringsAsFactors = TRUE)

# Set histogram bins
bins <- seq(0, 1.3, by = 0.05)

# Open PNG output
png(filename = output_file, width = 1000, height = 700)

# Plot histograms
histoplot(probability ~ MACr, data = answer_df,
          col = "violetred", plotCentre = "line", side = "right", breaks = bins)

histoplot(probability ~ MACr, data = resp_df,
          col = "blue", plotCentre = "line", side = "left", add = TRUE, breaks = bins)

# Add legend and title
legend("bottomleft", fill = c("blue", "violetred"), legend = c("answ", "resp"), title = "Model")
title(main = plot_title)

# Close file
dev.off()
