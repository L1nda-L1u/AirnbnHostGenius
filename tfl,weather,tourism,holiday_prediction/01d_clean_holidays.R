# ==================================================================================
# 01d - Clean Holidays Data
# ==================================================================================

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)

# Note: Working directory should be set by the master script (00_run_all.R)
# or manually before running this script
dir.create("foot_traffic_data/cleaned", recursive = TRUE, showWarnings = FALSE)

# Load historical holidays data from CSV
holidays_raw <- fread("foot_traffic_data/raw/events/UK_holiday.csv") %>%
  mutate(date = as.Date(date))

# Check for API holiday data
api_cache_pattern <- "foot_traffic_data/raw/api_cache/uk_holidays_*.rds"
api_cache_files <- list.files("foot_traffic_data/raw/api_cache", 
                               pattern = "uk_holidays_.*\\.rds$", 
                               full.names = TRUE)

if (length(api_cache_files) > 0 && exists("USE_APIS") && USE_APIS) {
  # Use most recent cache file
  latest_cache <- api_cache_files[which.max(file.info(api_cache_files)$mtime)]
  
  tryCatch({
    api_holidays <- readRDS(latest_cache)
    
    # Only keep future holidays from API
    max_historical_date <- max(holidays_raw$date)
    api_holidays_new <- api_holidays %>%
      filter(date > max_historical_date)
    
    if (nrow(api_holidays_new) > 0) {
      holidays_raw <- bind_rows(holidays_raw, api_holidays_new) %>%
        distinct(date, title, .keep_all = TRUE) %>%
        arrange(date)
      message("Added ", nrow(api_holidays_new), " future holidays from API")
    }
  }, error = function(e) {
    message("Could not load API holidays, using historical only")
  })
}

# Clean and categorize
holidays_list <- holidays_raw %>%
  mutate(
    date = as.Date(date),
    year = year(date),
    month = month(date),
    holiday_type = case_when(
      str_detect(title, "Christmas|Boxing") ~ "Christmas",
      str_detect(title, "New Year") ~ "New Year",
      str_detect(title, "Easter|Good Friday") ~ "Easter",
      str_detect(title, "bank holiday") ~ "Bank Holiday",
      TRUE ~ "Other"
    ),
    holiday_weight = case_when(
      holiday_type == "Christmas" ~ 3.0,
      holiday_type == "New Year" ~ 2.5,
      holiday_type == "Easter" ~ 2.0,
      holiday_type == "Bank Holiday" ~ 1.5,
      TRUE ~ 1.0
    ),
    is_major_holiday = holiday_type %in% c("Christmas", "New Year", "Easter")
  ) %>%
  select(date, title, holiday_type, holiday_weight, is_major_holiday)

# Create daily calendar
date_range <- seq(min(holidays_list$date), max(holidays_list$date), by = "day")

holidays_daily <- tibble(date = date_range) %>%
  left_join(holidays_list, by = "date") %>%
  mutate(
    year = year(date),
    month = month(date),
    day_of_week = lubridate::wday(date, label = TRUE),
    is_weekend = day_of_week %in% c("Sat", "Sun"),
    is_holiday = !is.na(title),
    is_major_holiday = replace_na(is_major_holiday, FALSE),
    holiday_weight = replace_na(holiday_weight, 0),
    holiday_type = replace_na(holiday_type, "None"),
    prev_day_holiday = lag(is_holiday, default = FALSE),
    next_day_holiday = lead(is_holiday, default = FALSE),
    is_holiday_period = is_holiday | prev_day_holiday | next_day_holiday
  ) %>%
  select(date, year, month, day_of_week, is_weekend,
         is_holiday, is_major_holiday, holiday_type, holiday_weight,
         is_holiday_period, title) %>%
  arrange(date)

# Save
fwrite(holidays_daily, "foot_traffic_data/cleaned/holidays_daily.csv")
