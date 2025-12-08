# ==================================================================================
# 01b - Clean Tourism Data
# ==================================================================================

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)

# Note: Working directory should be set by the master script (00_run_all.R)
# or manually before running this script

# Load data
tourism_raw <- fread("foot_traffic_data/raw/tourism/international-visitors-london-raw.csv")

# Fix column names
col_names <- c("year", "quarter", "market", "dur_stay", "mode", "purpose", 
               "area", "visits_k", "spend_m", "nights_k", "sample")
setnames(tourism_raw, col_names)

# Clean and aggregate to quarterly
tourism_quarterly <- tourism_raw %>%
  filter(
    !is.na(year),
    year != "Year",
    str_detect(year, "^[0-9]{4}$")
  ) %>%
  mutate(
    year = as.integer(year),
    quarter_num = case_when(
      str_detect(quarter, "January") ~ 1,
      str_detect(quarter, "April") ~ 2,
      str_detect(quarter, "July") ~ 3,
      str_detect(quarter, "October") ~ 4
    ),
    month = quarter_num * 3 - 1,
    date = ymd(paste(year, month, "01")),
    visits_k = as.numeric(visits_k),
    spend_m = as.numeric(spend_m),
    nights_k = as.numeric(nights_k)
  ) %>%
  filter(!is.na(year), !is.na(quarter_num), !is.na(visits_k), visits_k > 0) %>%
  filter(str_detect(area, "LONDON")) %>%
  group_by(year, quarter_num, date) %>%
  summarise(
    total_visits_k = sum(visits_k, na.rm = TRUE),
    total_spend_m = sum(spend_m, na.rm = TRUE),
    total_nights_k = sum(nights_k, na.rm = TRUE),
    avg_spend_per_visit = (total_spend_m * 1000) / total_visits_k,
    avg_nights_per_visit = total_nights_k / total_visits_k,
    .groups = "drop"
  ) %>%
  arrange(date)

# Save
fwrite(tourism_quarterly, "foot_traffic_data/cleaned/tourism_quarterly.csv")
