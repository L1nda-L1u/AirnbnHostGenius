# ==================================================================================
# 02 - Merge All Data to Daily Framework
# ==================================================================================

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)

# Note: Working directory should be set by the master script (00_run_all.R)
# or manually before running this script

# Load all cleaned data
weather_daily <- fread("foot_traffic_data/cleaned/weather_daily.csv") %>%
  mutate(date = as.Date(date))

holidays_daily <- fread("foot_traffic_data/cleaned/holidays_daily.csv") %>%
  mutate(date = as.Date(date))

tfl_monthly <- fread("foot_traffic_data/cleaned/tfl_monthly.csv") %>%
  mutate(date = as.Date(date))

tourism_quarterly <- fread("foot_traffic_data/cleaned/tourism_quarterly.csv") %>%
  mutate(date = as.Date(date))

# Create master daily framework
date_range <- seq(min(weather_daily$date), max(weather_daily$date), by = "day")

foot_traffic_daily <- tibble(date = date_range) %>%
  mutate(
    year = year(date),
    month = month(date),
    quarter = quarter(date),
    day_of_week = lubridate::wday(date, label = TRUE),
    day_of_month = day(date),
    day_of_year = yday(date),
    is_weekend = day_of_week %in% c("Sat", "Sun"),
    year_month = ymd(paste(year, month, "01"))
  )

# Join all datasets
foot_traffic_daily <- foot_traffic_daily %>%
  left_join(
    weather_daily %>% 
      select(date, temp_c, wind_kmh, precip_mm, weather_quality, is_good_weather),
    by = "date"
  ) %>%
  left_join(
    holidays_daily %>%
      select(date, is_holiday, is_major_holiday, holiday_type, 
             holiday_weight, is_holiday_period, title),
    by = "date"
  )

# Join monthly data
foot_traffic_daily <- foot_traffic_daily %>%
  left_join(
    tfl_monthly %>%
      select(year, month, avg_daily_journeys_m, total_monthly_journeys_m) %>%
      rename(tfl_daily_avg_m = avg_daily_journeys_m,
             tfl_monthly_total_m = total_monthly_journeys_m),
    by = c("year", "month")
  )

# Join quarterly data
tourism_quarterly_prep <- tourism_quarterly %>%
  select(year, quarter_num, total_visits_k, total_spend_m, 
         avg_spend_per_visit, avg_nights_per_visit) %>%
  rename(tourism_quarterly_visits_k = total_visits_k,
         tourism_quarterly_spend_m = total_spend_m,
         tourism_avg_spend = avg_spend_per_visit,
         tourism_avg_nights = avg_nights_per_visit)

foot_traffic_daily <- foot_traffic_daily %>%
  left_join(tourism_quarterly_prep, by = c("year" = "year", "quarter" = "quarter_num"))

# Create normalized indices for each component (for potential future use)
# Note: These are NOT combined into a single score - each component is kept separate
foot_traffic_daily <- foot_traffic_daily %>%
  mutate(
    tfl_index = scales::rescale(tfl_daily_avg_m, to = c(0, 1), na.rm = TRUE),
    tourism_index = scales::rescale(tourism_quarterly_visits_k, to = c(0, 1), na.rm = TRUE),
    weather_index = weather_quality,
    holiday_index = case_when(
      is_major_holiday ~ 1.0,
      is_holiday ~ 0.7,
      is_holiday_period ~ 0.5,
      is_weekend ~ 0.3,
      TRUE ~ 0.0
    )
  )

# Final cleanup and save
foot_traffic_daily <- foot_traffic_daily %>%
  select(
    date, year, month, quarter, day_of_week, day_of_month, day_of_year,
    is_weekend,
    temp_c, wind_kmh, precip_mm, weather_quality, is_good_weather,
    is_holiday, is_major_holiday, holiday_type, holiday_weight, 
    is_holiday_period, title,
    tfl_daily_avg_m, tfl_monthly_total_m,
    tourism_quarterly_visits_k, tourism_quarterly_spend_m,
    tourism_avg_spend, tourism_avg_nights,
    tfl_index, tourism_index, weather_index, holiday_index
  ) %>%
  select(-any_of("year_month")) %>%
  arrange(date)

# Save
fwrite(foot_traffic_daily, "foot_traffic_data/cleaned/foot_traffic_daily.csv")

# Clean up intermediate variables
rm(weather_daily, holidays_daily, tfl_monthly, tourism_quarterly, 
   tourism_quarterly_prep, date_range)
