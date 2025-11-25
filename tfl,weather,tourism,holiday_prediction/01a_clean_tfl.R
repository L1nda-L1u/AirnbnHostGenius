# ==================================================================================
# 01a - Clean TfL Transport Data
# ==================================================================================

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)

setwd("/Users/xiongyi/Desktop/Airbnb/AirbnbHostGenius/foot_traffic_prediction")
dir.create("foot_traffic_data/cleaned", recursive = TRUE, showWarnings = FALSE)

# Load data
tfl_raw <- fread("foot_traffic_data/raw/tfl/tfl-journeys-type.csv")

# Clean and aggregate to monthly
tfl_monthly <- tfl_raw %>%
  rename_with(~str_replace_all(str_replace_all(tolower(.), " ", "_"), "[()]", "")) %>%
  mutate(
    start_date = dmy(period_beginning),
    year = year(start_date),
    month = month(start_date),
    date = ymd(paste(year, month, "01"))
  ) %>%
  mutate(across(contains("journeys"), ~replace_na(., 0))) %>%
  mutate(
    total_journeys_m = bus_journeys_m + underground_journeys_m + dlr_journeys_m + 
                       tram_journeys_m + overground_journeys_m + 
                       london_cable_car_journeys_m + tfl_rail_journeys_m,
    daily_avg_m = total_journeys_m / days_in_period
  ) %>%
  filter(!is.na(start_date), total_journeys_m > 0) %>%
  group_by(year, month, date) %>%
  summarise(
    avg_daily_journeys_m = mean(daily_avg_m, na.rm = TRUE),
    total_monthly_journeys_m = sum(total_journeys_m, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(date)

# Save
fwrite(tfl_monthly, "foot_traffic_data/cleaned/tfl_monthly.csv")
