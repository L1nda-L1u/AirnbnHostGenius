# ==================================================================================
# 05 - Descriptive Analysis & Visualization
# ==================================================================================
# Exploratory data analysis of REAL components: TfL, Tourism, Weather, Holidays
# No synthetic foot_traffic_score - only analyze actual observed data

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)
library(corrplot)
library(gridExtra)
library(scales)

# Note: Working directory should be set by the master script (00_run_all.R)
# or manually before running this script

# Create output directory for plots
dir.create("outputs/plots", recursive = TRUE, showWarnings = FALSE)

# Load cleaned data
df <- fread("foot_traffic_data/cleaned/foot_traffic_daily.csv") %>%
  mutate(
    date = as.Date(date),
    day_of_week = factor(day_of_week, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")),
    month_name = lubridate::month(date, label = TRUE, abbr = FALSE),
    season = case_when(
      month %in% c(12, 1, 2) ~ "Winter",
      month %in% c(3, 4, 5) ~ "Spring",
      month %in% c(6, 7, 8) ~ "Summer",
      month %in% c(9, 10, 11) ~ "Autumn"
    ),
    season = factor(season, levels = c("Spring", "Summer", "Autumn", "Winter"))
  ) %>%
  # Ensure date is Date type and filter out invalid dates
  filter(!is.na(date), is.finite(as.numeric(date))) %>%
  mutate(date = as.Date(date))

message("\n==========================================================")
message("DESCRIPTIVE ANALYSIS - REAL COMPONENTS ONLY")
message("==========================================================\n")

message("Data period: ", min(df$date), " to ", max(df$date))
message("Total observations: ", nrow(df))
message("\nComponents analyzed:")
message("  - TfL Daily Journeys (millions)")
message("  - Tourism Quarterly Visits (thousands)")
message("  - Weather (temperature, precipitation, quality)")
message("  - Holidays (bank holidays, major holidays)")

# ==================================================================================
# 1. TFL TRANSPORT ANALYSIS
# ==================================================================================

message("\n[1/4] Analyzing TfL Transport Patterns...")

tfl_data <- df %>% filter(!is.na(tfl_daily_avg_m))

## 1.1 Time Series
tfl_data_plot <- tfl_data %>%
  filter(is.finite(tfl_daily_avg_m), !is.na(tfl_daily_avg_m))

p1 <- ggplot(tfl_data_plot, aes(x = as.Date(date), y = tfl_daily_avg_m)) +
  geom_line(color = "steelblue", alpha = 0.7) +
  geom_smooth(method = "loess", color = "red", se = TRUE, span = 0.1, na.rm = TRUE) +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
  labs(title = "TfL Daily Journeys Over Time",
       subtitle = "Daily average journeys (millions) with smoothed trend",
       x = "Date", y = "TfL Journeys (millions)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/plots/01_tfl_timeseries.png", p1, width = 12, height = 6, dpi = 300)

## 1.2 By Year
tfl_yearly <- tfl_data %>%
  group_by(year) %>%
  summarise(
    mean_journeys = mean(tfl_daily_avg_m, na.rm = TRUE),
    median_journeys = median(tfl_daily_avg_m, na.rm = TRUE),
    sd_journeys = sd(tfl_daily_avg_m, na.rm = TRUE),
    .groups = "drop"
  )

print(tfl_yearly)

tfl_data_box <- tfl_data %>%
  filter(is.finite(tfl_daily_avg_m), !is.na(tfl_daily_avg_m))

p2 <- ggplot(tfl_data_box, aes(x = factor(year), y = tfl_daily_avg_m, fill = factor(year))) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3, na.rm = TRUE) +
  labs(title = "TfL Journeys Distribution by Year",
       x = "Year", y = "Daily Journeys (millions)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/plots/02_tfl_by_year.png", p2, width = 10, height = 6, dpi = 300)

## 1.3 Seasonal Patterns
p3 <- ggplot(tfl_data_box, aes(x = season, y = tfl_daily_avg_m, fill = season)) +
  geom_boxplot(alpha = 0.7, na.rm = TRUE) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "TfL Journeys by Season",
       x = "Season", y = "Daily Journeys (millions)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/plots/03_tfl_by_season.png", p3, width = 10, height = 6, dpi = 300)

# ==================================================================================
# 2. TOURISM ANALYSIS
# ==================================================================================

message("\n[2/4] Analyzing Tourism Patterns...")

tourism_data <- df %>% filter(!is.na(tourism_quarterly_visits_k))

## 2.1 Time Series
tourism_data_plot <- tourism_data %>%
  filter(is.finite(tourism_quarterly_visits_k), !is.na(tourism_quarterly_visits_k))

p4 <- ggplot(tourism_data_plot, aes(x = as.Date(date), y = tourism_quarterly_visits_k)) +
  geom_line(color = "darkgreen", alpha = 0.7) +
  geom_smooth(method = "loess", color = "red", se = TRUE, span = 0.1, na.rm = TRUE) +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
  labs(title = "International Visitors Over Time",
       subtitle = "Quarterly visitor counts (thousands) with smoothed trend",
       x = "Date", y = "Quarterly Visitors (thousands)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/plots/04_tourism_timeseries.png", p4, width = 12, height = 6, dpi = 300)

## 2.2 By Year
tourism_yearly <- tourism_data %>%
  group_by(year) %>%
  summarise(
    mean_visits = mean(tourism_quarterly_visits_k, na.rm = TRUE),
    median_visits = median(tourism_quarterly_visits_k, na.rm = TRUE),
    sd_visits = sd(tourism_quarterly_visits_k, na.rm = TRUE),
    .groups = "drop"
  )

print(tourism_yearly)

tourism_data_box <- tourism_data %>%
  filter(is.finite(tourism_quarterly_visits_k), !is.na(tourism_quarterly_visits_k))

p5 <- ggplot(tourism_data_box, aes(x = factor(year), y = tourism_quarterly_visits_k, fill = factor(year))) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3, na.rm = TRUE) +
  labs(title = "Tourism Visits Distribution by Year",
       x = "Year", y = "Quarterly Visits (thousands)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/plots/05_tourism_by_year.png", p5, width = 10, height = 6, dpi = 300)

# ==================================================================================
# 3. WEATHER ANALYSIS
# ==================================================================================

message("\n[3/4] Analyzing Weather Patterns...")

## 3.1 Temperature Time Series
weather_data_plot <- df %>%
  filter(is.finite(temp_c), !is.na(temp_c))

p6 <- ggplot(weather_data_plot, aes(x = as.Date(date), y = temp_c)) +
  geom_line(color = "orange", alpha = 0.7) +
  geom_smooth(method = "loess", color = "red", se = TRUE, span = 0.1, na.rm = TRUE) +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
  labs(title = "Daily Temperature Over Time",
       x = "Date", y = "Temperature (°C)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/plots/06_temp_timeseries.png", p6, width = 12, height = 6, dpi = 300)

## 3.2 Temperature by Season
p7 <- ggplot(weather_data_plot, aes(x = season, y = temp_c, fill = season)) +
  geom_boxplot(alpha = 0.7, na.rm = TRUE) +
  scale_fill_brewer(palette = "YlOrRd") +
  labs(title = "Temperature Distribution by Season",
       x = "Season", y = "Temperature (°C)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/plots/07_temp_by_season.png", p7, width = 10, height = 6, dpi = 300)

## 3.3 Weather Quality Distribution
weather_quality_plot <- df %>%
  filter(is.finite(weather_quality), !is.na(weather_quality))

p8 <- ggplot(weather_quality_plot, aes(x = weather_quality)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, fill = "lightblue", alpha = 0.7, na.rm = TRUE) +
  geom_density(color = "darkblue", linewidth = 1, na.rm = TRUE) +
  labs(title = "Distribution of Weather Quality",
       x = "Weather Quality (0-1)", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/plots/08_weather_quality_dist.png", p8, width = 10, height = 6, dpi = 300)

# ==================================================================================
# 4. COMPONENT CORRELATIONS
# ==================================================================================

message("\n[4/4] Computing Component Correlations...")

# Correlation between real components
cor_features <- df %>%
  select(temp_c, wind_kmh, precip_mm, weather_quality,
         tfl_daily_avg_m, tourism_quarterly_visits_k,
         holiday_weight, is_weekend, is_holiday, is_major_holiday) %>%
  mutate(
    is_weekend = as.numeric(is_weekend),
    is_holiday = as.numeric(is_holiday),
    is_major_holiday = as.numeric(is_major_holiday)
  ) %>%
  na.omit()

cor_matrix <- cor(cor_features)

png("outputs/plots/09_component_correlation.png", 
    width = 12, height = 10, units = "in", res = 300)
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         title = "Component Correlation Matrix (Real Data Only)",
         mar = c(0, 0, 2, 0))
dev.off()

# Key correlations
message("\nKey Component Correlations:")
message("  TfL vs Tourism: ", round(cor(df$tfl_daily_avg_m, df$tourism_quarterly_visits_k, use = "complete.obs"), 3))
message("  TfL vs Temperature: ", round(cor(df$tfl_daily_avg_m, df$temp_c, use = "complete.obs"), 3))
message("  Tourism vs Temperature: ", round(cor(df$tourism_quarterly_visits_k, df$temp_c, use = "complete.obs"), 3))

# Create summary
sink("outputs/plots/00_ANALYSIS_SUMMARY.txt")
cat("==========================================================\n")
cat("COMPONENT DESCRIPTIVE ANALYSIS SUMMARY\n")
cat("==========================================================\n\n")
cat("Analysis Date:", as.character(Sys.Date()), "\n")
cat("Data Period:", min(df$date), "to", max(df$date), "\n")
cat("Total Days:", nrow(df), "\n\n")

cat("COMPONENTS ANALYZED:\n\n")
cat("1. TfL TRANSPORT\n")
cat("   - Mean daily journeys: ", round(mean(tfl_data$tfl_daily_avg_m, na.rm = TRUE), 2), " million\n")
cat("   - Median: ", round(median(tfl_data$tfl_daily_avg_m, na.rm = TRUE), 2), " million\n")
cat("   - Date range: ", min(tfl_data$date), " to ", max(tfl_data$date), "\n\n")

cat("2. TOURISM\n")
cat("   - Mean quarterly visits: ", round(mean(tourism_data$tourism_quarterly_visits_k, na.rm = TRUE), 0), " thousand\n")
cat("   - Median: ", round(median(tourism_data$tourism_quarterly_visits_k, na.rm = TRUE), 0), " thousand\n")
cat("   - Date range: ", min(tourism_data$date), " to ", max(tourism_data$date), "\n\n")

cat("3. WEATHER\n")
cat("   - Mean temperature: ", round(mean(df$temp_c, na.rm = TRUE), 1), "°C\n")
cat("   - Mean weather quality: ", round(mean(df$weather_quality, na.rm = TRUE), 3), "\n\n")

cat("KEY CORRELATIONS:\n")
cat("   - TfL vs Tourism: ", round(cor(df$tfl_daily_avg_m, df$tourism_quarterly_visits_k, use = "complete.obs"), 3), "\n")
cat("   - TfL vs Temperature: ", round(cor(df$tfl_daily_avg_m, df$temp_c, use = "complete.obs"), 3), "\n")
cat("   - Tourism vs Temperature: ", round(cor(df$tourism_quarterly_visits_k, df$temp_c, use = "complete.obs"), 3), "\n\n")

cat("==========================================================\n")
cat("All visualizations saved to: outputs/plots/\n")
cat("==========================================================\n")
sink()

message("\n==========================================================")
message("DESCRIPTIVE ANALYSIS COMPLETE!")
message("==========================================================")
message("\nGenerated 9 visualizations + 1 summary report")
message("Output directory: outputs/plots/\n")
