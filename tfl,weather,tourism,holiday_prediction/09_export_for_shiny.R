# ==================================================================================
# 08 - Export for Shiny Integration
# ==================================================================================
# Prepare models and prediction functions for Shiny app integration

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)

# Note: Working directory should be set by the master script (00_run_all.R)
# or manually before running this script

# Create Shiny export directory
dir.create("shiny_export", recursive = TRUE, showWarnings = FALSE)

message("\n==========================================================")
message("EXPORT FOR SHINY INTEGRATION")
message("==========================================================\n")

# ==================================================================================
# 1. COPY COMPONENT PREDICTION MODELS
# ==================================================================================

message("\n[1/4] Copying component prediction models...")

# List of component model files (TfL, Tourism, Weather)
component_models <- c(
  "outputs/models/tfl_arima_model.rds",
  "outputs/models/tourism_arima_model.rds",
  "outputs/models/tourism_recovery_params.rds"
)

for (file in component_models) {
  if (file.exists(file)) {
    file.copy(file, file.path("shiny_export", basename(file)), overwrite = TRUE)
    message("✓ Copied: ", basename(file))
  } else {
    message("✗ Missing: ", file)
  }
}

# ==================================================================================
# 2. EXPORT COMPONENT PREDICTIONS
# ==================================================================================

message("\n[2/4] Creating component prediction lookup table...")

# Load component predictions
component_data <- list()

# TfL predictions
if (file.exists("outputs/tfl_daily_forecast.csv")) {
  tfl_forecast <- fread("outputs/tfl_daily_forecast.csv") %>%
    mutate(date = as.Date(date)) %>%
    select(date, tfl_daily_avg_m)
  component_data$tfl <- tfl_forecast
  message("✓ Loaded TfL forecast: ", nrow(tfl_forecast), " days")
}

# Tourism predictions
if (file.exists("outputs/tourism_daily_forecast.csv")) {
  tourism_forecast <- fread("outputs/tourism_daily_forecast.csv") %>%
    mutate(date = as.Date(date)) %>%
    select(date, tourism_quarterly_visits_k, tourism_quarterly_spend_m)
  component_data$tourism <- tourism_forecast
  message("✓ Loaded Tourism forecast: ", nrow(tourism_forecast), " days")
}

# Weather predictions
if (file.exists("outputs/weather_predictions_complete.csv")) {
  weather_forecast <- fread("outputs/weather_predictions_complete.csv") %>%
    mutate(date = as.Date(date)) %>%
    select(date, temp_c, precip_mm, weather_quality, is_good_weather)
  component_data$weather <- weather_forecast
  message("✓ Loaded Weather forecast: ", nrow(weather_forecast), " days")
}

# Merge all components
if (length(component_data) > 0) {
  lookup_table <- component_data[[1]]
  for (i in 2:length(component_data)) {
    lookup_table <- lookup_table %>%
      full_join(component_data[[i]], by = "date")
  }
  
  lookup_table <- lookup_table %>%
    arrange(date) %>%
    filter(date >= as.Date("2025-01-01"))
  
  fwrite(lookup_table, "shiny_export/component_predictions_lookup.csv")
  saveRDS(lookup_table, "shiny_export/component_predictions_lookup.rds")
  
  message("✓ Created component lookup table: ", nrow(lookup_table), " days")
} else {
  message("⚠️  No component predictions found. Run prediction scripts first.")
}

# ==================================================================================
# 3. CREATE HELPER FUNCTIONS FOR SHINY
# ==================================================================================

message("\n[3/4] Creating helper functions...")

# Save helper functions as R script
helper_code <- '
# ==================================================================================
# Component Prediction Helper Functions for Shiny
# ==================================================================================
# Source this file in your Shiny app
# Provides functions to access TfL, Tourism, and Weather predictions

library(data.table)
library(lubridate)

# Load component predictions lookup table (call once in server)
load_component_predictions <- function() {
  readRDS("foot_traffic_prediction/shiny_export/component_predictions_lookup.rds")
}

# Get component predictions for a specific date
get_components <- function(date, lookup_table) {
  date <- as.Date(date)
  
  result <- lookup_table %>%
    filter(date == date)
  
  if (nrow(result) == 0) {
    # Date not in forecast - return averages
    return(list(
      tfl = mean(lookup_table$tfl_daily_avg_m, na.rm = TRUE),
      tourism = mean(lookup_table$tourism_quarterly_visits_k, na.rm = TRUE),
      temp = mean(lookup_table$temp_c, na.rm = TRUE),
      weather_quality = mean(lookup_table$weather_quality, na.rm = TRUE),
      available = FALSE
    ))
  }
  
  return(list(
    tfl = result$tfl_daily_avg_m,
    tourism = result$tourism_quarterly_visits_k,
    temp = result$temp_c,
    weather_quality = result$weather_quality,
    is_good_weather = result$is_good_weather,
    available = TRUE
  ))
}

# Get components for date range (e.g., booking period)
get_components_range <- function(start_date, end_date, lookup_table) {
  start_date <- as.Date(start_date)
  end_date <- as.Date(end_date)
  
  result <- lookup_table %>%
    filter(date >= start_date & date <= end_date)
  
  if (nrow(result) == 0) {
    return(list(
      mean_tfl = mean(lookup_table$tfl_daily_avg_m, na.rm = TRUE),
      mean_tourism = mean(lookup_table$tourism_quarterly_visits_k, na.rm = TRUE),
      mean_temp = mean(lookup_table$temp_c, na.rm = TRUE),
      dates_available = 0
    ))
  }
  
  return(list(
    mean_tfl = mean(result$tfl_daily_avg_m, na.rm = TRUE),
    mean_tourism = mean(result$tourism_quarterly_visits_k, na.rm = TRUE),
    mean_temp = mean(result$temp_c, na.rm = TRUE),
    mean_weather_quality = mean(result$weather_quality, na.rm = TRUE),
    good_weather_days = sum(result$is_good_weather, na.rm = TRUE),
    dates_available = nrow(result)
  ))
}

# Example usage in Shiny:
#
# server <- function(input, output, session) {
#   
#   # Load data once
#   component_lookup <- load_component_predictions()
#   
#   # Get predictions for checkin date
#   checkin_date <- input$checkin_date
#   components <- get_components(checkin_date, component_lookup)
#   
#   # Use components for pricing logic
#   # (You can create your own adjustment function based on these real values)
# }
'

writeLines(helper_code, "shiny_export/foot_traffic_helpers.R")
message("✓ Created helper functions: shiny_export/foot_traffic_helpers.R")

# ==================================================================================
# 4. CREATE INTEGRATION GUIDE
# ==================================================================================

message("\n[4/4] Creating integration guide...")

integration_guide <- '
# Component Predictions Integration Guide for Shiny App

## Quick Start

### 1. Copy Files to Shiny App Directory

Copy the entire `shiny_export/` folder to your Shiny app directory:

```
your_shiny_app/
├── app.R
├── foot_traffic_prediction/
│   └── shiny_export/
│       ├── component_predictions_lookup.rds
│       ├── component_predictions_lookup.csv
│       ├── foot_traffic_helpers.R
│       └── *.rds (component model files)
```

### 2. Source Helper Functions

In your Shiny app.R or server.R:

```r
source("foot_traffic_prediction/shiny_export/foot_traffic_helpers.R")
```

### 3. Load Data (Once in Server)

```r
server <- function(input, output, session) {
  
  # Load component predictions lookup table
  component_data <- load_component_predictions()
  
  # ... rest of your server code
}
```

### 4. Use Components in Pricing Logic

```r
# In your price calculation reactive:
adjusted_price <- reactive({
  
  checkin <- input$checkin_date
  checkout <- input$checkout_date
  base_price <- input$base_price  # or calculated base price
  
  # Get component predictions for booking period
  components <- get_components_range(checkin, checkout, component_data)
  
  # Create your own pricing adjustment based on REAL components
  # Example: Higher TfL + Tourism + Good weather = Higher demand
  tfl_factor <- (components$mean_tfl / 10) - 0.9  # Normalize around 10M
  tourism_factor <- (components$mean_tourism / 5000) - 0.9  # Normalize around 5000k
  weather_factor <- components$mean_weather_quality - 0.6  # Normalize around 0.6
  
  # Simple weighted adjustment (you can refine based on your data)
  adjustment <- 0.3 * tfl_factor + 0.3 * tourism_factor + 0.2 * weather_factor
  multiplier <- 1 + adjustment * 0.2  # Max ±20% adjustment
  
  # Apply to base price
  final_price <- base_price * multiplier
  
  return(list(
    price = final_price,
    tfl = components$mean_tfl,
    tourism = components$mean_tourism,
    weather = components$mean_weather_quality,
    multiplier = multiplier
  ))
})

# Display
output$price_display <- renderText({
  result <- adjusted_price()
  paste0("£", round(result$price, 2), " per night")
})

output$component_info <- renderUI({
  result <- adjusted_price()
  tagList(
    p(strong("TfL Journeys: "), round(result$tfl, 2), "M/day"),
    p(strong("Tourism: "), round(result$tourism, 0), "k/quarter"),
    p(strong("Weather Quality: "), round(result$weather, 2)),
    p(strong("Price Adjustment: "), 
      scales::percent(result$multiplier - 1, accuracy = 0.1))
  )
})
```

## Component-Based Pricing Strategies

### Simple Rule-Based Approach

```r
# Example: Higher TfL + Tourism + Good weather = Higher demand
calculate_price_adjustment <- function(components) {
  # Normalize components (adjust thresholds based on your data)
  tfl_norm <- (components$mean_tfl - 9) / 2  # Center around 9M
  tourism_norm <- (components$mean_tourism - 5000) / 1000  # Center around 5000k
  weather_norm <- components$mean_weather_quality - 0.6  # Center around 0.6
  
  # Weighted combination (you can adjust weights based on your analysis)
  adjustment <- 0.3 * tfl_norm + 0.3 * tourism_norm + 0.2 * weather_norm
  
  # Convert to multiplier (max ±20% adjustment)
  multiplier <- 1 + adjustment * 0.2
  multiplier <- max(0.8, min(1.2, multiplier))  # Clamp to reasonable range
  
  return(multiplier)
}
```

### Data-Driven Approach (Recommended)

If you have historical Airbnb price/occupancy data, you can:

1. **Regress occupancy/price on components:**
   ```r
   # occupancy ~ TfL + Tourism + Weather + Holiday
   # Use regression coefficients as weights
   ```

2. **Use correlation analysis:**
   ```r
   # Find which components correlate most with demand
   # Weight accordingly
   ```

## Displaying Component Insights

### Example UI Elements

```r
# In your UI:
wellPanel(
  h4("Demand Indicators"),
  uiOutput("component_gauge"),
  plotOutput("component_forecast", height = "200px")
)

# In server:
output$component_gauge <- renderUI({
  comps <- get_components(input$date, component_data)
  
  tagList(
    p(strong("TfL Journeys: "), round(comps$tfl, 2), "M/day"),
    p(strong("Tourism: "), round(comps$tourism, 0), "k/quarter"),
    p(strong("Weather: "), round(comps$weather_quality, 2))
  )
})

output$component_forecast <- renderPlot({
  date <- input$date
  date_range <- component_data %>%
    filter(date >= (date - 15) & date <= (date + 15))
  
  ggplot(date_range, aes(x = date)) +
    geom_line(aes(y = tfl_daily_avg_m, color = "TfL")) +
    geom_line(aes(y = tourism_quarterly_visits_k / 100, color = "Tourism")) +
    theme_minimal() +
    labs(x = "", y = "Normalized Values", color = "Component")
})
```

## Troubleshooting

### Issue: File not found errors
- Check file paths are correct relative to app.R
- Ensure shiny_export folder is copied completely

### Issue: Dates outside prediction range
- Predictions cover 2025-2026
- For other dates, function returns historical average
- Consider re-running component prediction scripts for extended range

### Issue: Performance concerns
- Lookup table loading is fast (~1MB)
- Consider caching in global.R for multi-user apps
- Pre-filter lookup table if only specific date ranges needed

## Updating Predictions

To update component predictions:

1. Navigate to foot_traffic_prediction/
2. Run component prediction scripts:
   - `source("03a_predict_tfl.R")`
   - `source("03b_predict_tourism.R")`
   - `source("03c_predict_weather.R")`
3. Run: `source("09_export_for_shiny.R")`
4. Restart Shiny app

---

**Note**: Component predictions are REAL data (TfL, Tourism, Weather), not synthetic scores.
You should create your own pricing adjustment logic based on these components and your Airbnb data.

For questions or issues, refer to README_MODEL.md
'

writeLines(integration_guide, "shiny_export/INTEGRATION_GUIDE.md")
message("✓ Created integration guide: shiny_export/INTEGRATION_GUIDE.md")

# ==================================================================================
# SUMMARY
# ==================================================================================

message("\n==========================================================")
message("SHINY EXPORT COMPLETE!")
message("==========================================================")
message("\nFiles exported to: shiny_export/")
message("\nContents:")
if (exists("lookup_table")) {
  message("  - component_predictions_lookup.rds (", nrow(lookup_table), " days)")
} else {
  message("  - component_predictions_lookup.rds (N/A - run component predictions first)")
}
message("  - foot_traffic_helpers.R (helper functions)")
message("  - INTEGRATION_GUIDE.md (documentation)")
message("  - Component model files (*.rds)")

message("\nNext Steps:")
message("  1. Copy shiny_export/ folder to your Shiny app directory")
message("  2. Review INTEGRATION_GUIDE.md")
message("  3. Source foot_traffic_helpers.R in your app")
message("  4. Use get_components() function to access TfL, Tourism, Weather predictions")
message("  5. Create your own pricing adjustment logic based on these REAL components")

message("\nSee README_MODEL.md for full documentation\n")

