
# =============================================
# Airbnb HostGenius - Merged App
# =============================================

library(shiny)
library(shinydashboard)
library(DT)
library(leaflet)
library(plotly)
library(ggplot2)
library(dplyr)
library(geosphere)
library(zoo)
library(data.table)
library(sf)
library(tidyverse)
library(lubridate)
library(httr)
library(jsonlite)


# =============================================
# Setup & Configuration
# =============================================

app_dir <- getwd()
if (!file.exists("app.R")) {
  if (file.exists("shiny_app/app.R")) {
    app_dir <- file.path(getwd(), "shiny_app")
  } else if (file.exists(file.path(getwd(), "..", "shiny_app", "app.R"))) {
    app_dir <- normalizePath(file.path(getwd(), "..", "shiny_app"))
  }
}

# Initialize Python
if (file.exists(file.path(app_dir, "init_python.R"))) {
  source(file.path(app_dir, "init_python.R"), local = TRUE)
}

source(file.path(app_dir, "model_loader.R"), local = TRUE)
source(file.path(app_dir, "geocoding.R"), local = TRUE)
source(file.path(app_dir, "feature_builder.R"), local = TRUE)
source(file.path(app_dir, "sensitivity_helper.R"), local = TRUE)
source(file.path(app_dir, "market_indicators.R"), local = TRUE)

# Load competitor data
competitor_data <- NULL
tryCatch({
  if (exists("load_competitor_data")) {
    competitor_data <- load_competitor_data()
    if (!is.null(competitor_data)) {
      cat("Competitor data loaded:", nrow(competitor_data), "listings\n")
    }
  }
}, error = function(e) {
  cat("Failed to load competitor data:", e$message, "\n")
})

# Friend's Setup Logic
# CONFIGURATION
# ==================================================================================

# Data boundaries: 90 days past, 365 days future
DATA_START <- Sys.Date() - 90
DATA_END <- Sys.Date() + 365

# ==================================================================================
# DATA LOADING
# ==================================================================================

# Helper function to load data with multiple path attempts
load_csv_safe <- function(filename, default_cols = c("date", "value")) {
  paths <- c(
    filename,
    file.path("shiny_app", filename),
    file.path(getwd(), filename),
    file.path(dirname(getwd()), "shiny_app", filename)
  )
  
  for (path in paths) {
    if (file.exists(path)) {
      tryCatch({
        data <- fread(path) %>% mutate(date = as.Date(date))
        message("Loaded ", filename, " from: ", path)
        return(data)
      }, error = function(e) {
        message("Error loading ", path, ": ", e$message)
      })
    }
  }
  
  # Return empty data frame if file not found
  message("Warning: ", filename, " not found. Creating empty data frame.")
  empty_df <- setNames(data.frame(matrix(ncol = length(default_cols), nrow = 0)), default_cols)
  empty_df$date <- as.Date(character())
  return(empty_df)
}

tfl_data <- load_csv_safe("tfl.csv", c("date", "value"))
weather_data <- load_csv_safe("weather.csv", c("date", "temp_c"))
holidays_file <- load_csv_safe("holidays.csv", c("date", "title"))
tourism_data <- load_csv_safe("tourism.csv", c("date", "value"))

# ==================================================================================
# FETCH HOLIDAYS FROM GOV.UK API
# ==================================================================================

fetch_uk_holidays <- function() {
  tryCatch({
    response <- GET("https://www.gov.uk/bank-holidays.json", timeout(10))
    
    if (status_code(response) == 200) {
      data <- content(response, "text", encoding = "UTF-8")
      json_data <- fromJSON(data)
      events <- json_data$`england-and-wales`$events
      
      holidays_api <- tibble(
        date = as.Date(events$date),
        title = events$title,
        is_major_holiday = TRUE
      )
      
      message("Fetched ", nrow(holidays_api), " holidays from gov.uk API")
      return(holidays_api)
    }
  }, error = function(e) {
    message("Could not fetch holidays from API: ", e$message)
  })
  
  # Fallback to file if API fails
  return(holidays_file %>% select(date, title, is_major_holiday))
}

# Get holidays from API
holidays_data <- fetch_uk_holidays()

# ==================================================================================
# CREATE DAILY DATA (NO FALLBACKS - only actual data)
# ==================================================================================

# Generate date sequence within boundaries
daily_data <- tibble(date = seq(DATA_START, DATA_END, by = "day")) %>%
  mutate(
    day_of_week = lubridate::wday(date, label = TRUE),
    week = lubridate::week(date),
    month = lubridate::month(date),
    year = lubridate::year(date),
    is_weekend = day_of_week %in% c("Sat", "Sun"),
    is_past = date < Sys.Date(),
    is_today = date == Sys.Date()
  )

# Join TfL data (NO fallback)
tfl_values <- tfl_data %>%
  filter(date >= DATA_START, date <= DATA_END) %>%
  select(date, tfl_value = value)

daily_data <- daily_data %>%
  left_join(tfl_values, by = "date")

# Join weather data (NO fallback)
weather_values <- weather_data %>%
  filter(date >= DATA_START, date <= DATA_END) %>%
  select(date, temp_c, weather_quality, precip_mm, 
         humidity_avg, sunshine_hours, TCI, TCI_category)

daily_data <- daily_data %>%
  left_join(weather_values, by = "date")

# Join holidays from API
daily_data <- daily_data %>%
  left_join(
    holidays_data %>% select(date, holiday_name = title, is_major_holiday),
    by = "date"
  ) %>%
  mutate(
    is_holiday = !is.na(holiday_name),
    is_major_holiday = coalesce(is_major_holiday, FALSE)
  )

# ==================================================================================
# CALCULATE TfL SEASONAL PATTERN (Using pre-COVID historical data)
# ==================================================================================

# Calculate monthly seasonal averages (excluding COVID years 2020-2021)
tfl_seasonal <- tfl_data %>%
  filter(!is.na(value)) %>%
  mutate(
    month_num = lubridate::month(date),
    year = lubridate::year(date)
  ) %>%
  filter(year < 2020 | year > 2021) %>%  # Exclude COVID
  group_by(month_num) %>%
  summarise(
    monthly_avg = mean(value, na.rm = TRUE),
    .groups = "drop"
  )

# Calculate overall average and relative position
overall_avg <- mean(tfl_seasonal$monthly_avg)
tfl_seasonal <- tfl_seasonal %>%
  mutate(
    relative = (monthly_avg - overall_avg) / overall_avg * 100,  # % above/below avg
    season_label = case_when(
      relative >= 5 ~ "Busy",
      relative <= -5 ~ "Quiet",
      TRUE ~ "Average"
    )
  )

# ==================================================================================
# CALCULATE PRICE ADJUSTMENT (Rule-Based, No Arbitrary Scores)
# ==================================================================================
# 
# NEW APPROACH: Simple rules based on day type
# - Weekend: +15% (industry standard for leisure destinations)
# - Bank Holiday: +20% 
# - Major Holiday (Christmas/Boxing Day/NYE): +30%
# - TfL & Weather shown as context indicators (with seasonal comparison)
#
# ==================================================================================

daily_data <- daily_data %>%
  mutate(
    # Check if we have data
    has_data = !is.na(tfl_value) | !is.na(weather_quality),
    
    # Get month for seasonal lookup
    month_num = lubridate::month(date)
  ) %>%
  left_join(tfl_seasonal %>% select(month_num, tfl_monthly_avg = monthly_avg, 
                                     tfl_relative = relative, tfl_season = season_label), 
            by = "month_num") %>%
  mutate(
    # TfL: Compare to monthly average (if we have current data)
    tfl_vs_avg = if_else(!is.na(tfl_value) & !is.na(tfl_monthly_avg),
                         (tfl_value - tfl_monthly_avg) / tfl_monthly_avg * 100,
                         NA_real_),
    
    # TCI (Tourism Climate Index) - 0-100 scale from Mieczkowski 1985
    # Already calculated in weather data, just use it
    TCI = if_else(!is.na(TCI), TCI, weather_quality * 100),
    
    # TCI category label (academic classification)
    TCI_label = case_when(
      TCI >= 90 ~ "Ideal",
      TCI >= 80 ~ "Excellent",
      TCI >= 70 ~ "Very Good",
      TCI >= 60 ~ "Good",
      TCI >= 50 ~ "Acceptable",
      TCI >= 40 ~ "Marginal",
      TCI >= 30 ~ "Unfavorable",
      is.na(TCI) ~ "No data",
      TRUE ~ "Very Unfavorable"
    ),
    
    # Determine day type (mutually exclusive, take highest)
    day_type = case_when(
      is_major_holiday ~ "Major Holiday",
      is_holiday ~ "Bank Holiday",
      is_weekend ~ "Weekend",
      TRUE ~ "Weekday"
    ),
    
    # Price multiplier based on day type (rule-based, not score-based)
    price_multiplier = case_when(
      day_type == "Major Holiday" ~ 1.30,  # +30%
      day_type == "Bank Holiday" ~ 1.20,   # +20%
      day_type == "Weekend" ~ 1.15,        # +15%
      TRUE ~ 1.0                           # Base price
    ),
    
    # Price recommendation label
    price_recommendation = case_when(
      day_type == "Major Holiday" ~ "Premium",
      day_type == "Bank Holiday" ~ "Above Average",
      day_type == "Weekend" ~ "Above Average",
      TRUE ~ "Standard"
    ),
    
    # Bonus: Good weather on weekday = slight boost
    # (only if TCI >= 70 "Very Good" and it's a weekday)
    weather_boost = if_else(
      day_type == "Weekday" & !is.na(TCI) & TCI >= 70,
      TRUE, FALSE
    ),
    price_multiplier = if_else(weather_boost, price_multiplier + 0.05, price_multiplier),
    price_recommendation = if_else(weather_boost, "Standard+", price_recommendation)
  )

# Store tfl_seasonal for use in UI
tfl_monthly_pattern <- tfl_seasonal

cat("Daily data loaded:", nrow(daily_data), "days from", 
    as.character(min(daily_data$date)), "to", as.character(max(daily_data$date)), "\n")



# =============================================
# UI
# =============================================

ui <- dashboardPage(
  dashboardHeader(
    title = tags$div(
      tags$span("Airbnb", style = "font-size: 18px; font-weight: bold; color: #2A8C82; margin-right: 6px;"),
      tags$span("HostGenius", style = "font-size: 18px; font-weight: bold; color: #2C3E50;")
    ),
    titleWidth = 220
  ),
  
  dashboardSidebar(
    width = 220,
    collapsed = FALSE,
    sidebarMenu(
      id = "tabs",
      menuItem("Overview", tabName = "overview", icon = icon("home")),
      menuItem("Price Prediction", tabName = "predict", icon = icon("dollar-sign")),
      menuItem("Calendar", tabName = "calendar", icon = icon("calendar")),
      menuItem("Demand Trends", tabName = "trends", icon = icon("chart-line")),
      menuItem("Dynamic Price", tabName = "calculator", icon = icon("calculator")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    ),
    tags$div(
      style = "padding: 10px; margin-top: 10px;",
      tags$p(
        style = "color: #7F8C8D; font-size: 11px; text-align: center; line-height: 1.3;",
        "Enter property details to get smart pricing suggestions"
      )
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      
      body { 
        font-family: 'Inter', sans-serif;
        background: #f1f5f9;
        color: #2C3E50;
        min-height: 100vh;
      }
      
      .navbar { 
        background: #ffffff !important;
        border-bottom: 1px solid #D0D0D0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }
      .navbar-brand { 
        color: #2A8C82 !important; 
        font-weight: 700;
        font-size: 20px;
      }
      .navbar-nav > li > a { color: #7F8C8D !important; }
      .navbar-nav > li.active > a { color: #2A8C82 !important; }
      
      .card {
        background-color: #FFFFFF !important;
        border: 1px solid #D0D0D0 !important;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
      }
      
      .section-title { 
        font-size: 16px; 
        font-weight: 600; 
        color: #2C3E50;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .section-title::before {
        content: '';
        width: 3px;
        height: 16px;
        background: #2A8C82;
        border-radius: 2px;
      }
      
      /* Calendar Styles */
      .calendar-header {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        margin-bottom: 8px;
        padding: 0 8px;
      }
      
      .calendar-header > div:first-child {
        justify-self: start;
      }
      
      .calendar-header > div:last-child {
        justify-self: end;
      }
      
      .calendar-title {
        font-size: 20px;
        font-weight: 600;
        color: #2C3E50;
        justify-self: center;
      }
      
      .calendar-nav-btn {
        background: #F5F5F5;
        border: 1px solid #E0E0E0;
        color: #4A4A4A;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 500;
        transition: background 0.2s, opacity 0.2s;
      }
      .calendar-nav-btn:hover:not(.disabled) {
        background: #E0E0E0;
      }
      .calendar-nav-btn.today-btn {
        background: #C7E9F8;
        color: #2C3E50;
        border: none;
      }
      .calendar-nav-btn.today-btn:hover {
        background: #B3E0F2;
        border: none;
      }
      .calendar-nav-btn.disabled {
        background: #f1f5f9;
        color: #cbd5e1;
        cursor: not-allowed;
        opacity: 0.5;
      }
      
      .calendar-grid {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        gap: 4px;
      }
      
      .calendar-weekday {
        text-align: center;
        font-size: 11px;
        font-weight: 600;
        color: #7F8C8D;
        padding: 8px 0;
        text-transform: uppercase;
      }
      
      .calendar-day {
        aspect-ratio: 1;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: transform 0.15s, box-shadow 0.15s;
        position: relative;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
      }
      .calendar-day:hover {
        transform: scale(1.08);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10;
      }
      
      /* SELECTED - dark border frame */
      .calendar-day.selected {
        box-shadow: 0 0 0 3px #2C3E50 !important;
      }
      
      .day-number {
        font-size: 18px;
        font-weight: 600;
      }
      .day-price {
        font-size: 10px;
        opacity: 0.9;
        font-weight: 500;
      }
      
      .calendar-day.empty {
        background: transparent;
        cursor: default;
      }
      .calendar-day.empty:hover {
        transform: none;
        box-shadow: none;
      }
      
      /* PAST days - faded */
      .calendar-day.past {
        opacity: 0.4;
      }
      .calendar-day.past:hover {
        opacity: 0.6;
      }
      
      /* TODAY - Lighter Blue */
      .calendar-day.today {
        background: #C7E9F8 !important;
        color: #2C3E50 !important;
      }
      .calendar-day.today .day-number,
      .calendar-day.today .day-price {
        color: #2C3E50 !important;
      }
      
      /* Demand Level Colors - Flat & Lighter */
      .demand-premium { 
        background: #FFD8A8;
        color: #2C3E50;
      }
      .demand-above { 
        background: #2A8C82;
        color: white;
      }
      .demand-standard { 
        background: #F5F5F5;
        color: #4A4A4A;
      }
      .demand-below { 
        background: #F5F5F5;
        color: #999999;
      }
      .demand-nodata { 
        background: #FAFAFA;
        border: 1px dashed #D0D0D0;
        color: #D0D0D0;
      }
      .demand-weatherboost { 
        background: #D7F0FF;
        color: #1f4b99;
      }
      
      .holiday-marker {
        position: absolute;
        top: 4px;
        right: 4px;
        width: 6px;
        height: 6px;
        background: #F5B085;
        border-radius: 50%;
      }
      
      /* Legend - Simplified */
      .legend {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 24px;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #D0D0D0;
      }
      .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #475569;
      }
      .legend-color {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        flex-shrink: 0;
      }
      .legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
      }
      .legend-divider {
        width: 1px;
        height: 20px;
        background: #D0D0D0;
      }
      
      /* Detail Panel */
      .detail-panel {
        background: #F5F5F5;
        border: 1px solid #D0D0D0;
        border-radius: 8px;
        padding: 16px;
      }
      
      .detail-date {
        font-size: 16px;
        font-weight: 600;
        color: #2A8C82;
        margin-bottom: 12px;
      }
      
      .detail-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
      }
      
      .detail-item {
        background: #ffffff;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #D0D0D0;
      }
      .detail-label {
        font-size: 10px;
        color: #7F8C8D;
        text-transform: uppercase;
        margin-bottom: 4px;
      }
      .detail-value {
        font-size: 20px;
        font-weight: 700;
        color: #2C3E50;
      }
      
      .component-bar {
        height: 6px;
        background: #D0D0D0;
        border-radius: 3px;
        margin-top: 8px;
        overflow: hidden;
      }
      .component-fill {
        height: 100%;
        border-radius: 3px;
      }
      
      /* Tables */
      .dataTables_wrapper { color: #2C3E50 !important; }
      table.dataTable { color: #2C3E50 !important; }
      table.dataTable thead th { 
        color: #7F8C8D !important; 
        border-bottom-color: #D0D0D0 !important;
        font-size: 11px;
        text-transform: uppercase;
      }
      table.dataTable tbody tr { background: transparent !important; }
      table.dataTable tbody tr:hover { background: rgba(0,0,0,0.02) !important; }
      
      .form-control, .selectize-input, .selectize-dropdown {
        background: #ffffff !important;
        border-color: #D0D0D0 !important;
        color: #2C3E50 !important;
      }
      
      .btn-primary {
        background: #2A8C82 !important;
        border: none !important;
      }
      .btn-primary:hover {
        background: #234E52 !important;
      }
    

        .skin-blue .main-header .logo {
          background-color: #FFFFFF !important;
          color: #2C3E50 !important;
          font-weight: bold;
          text-align: left !important;
          padding-left: 20px !important;
        }
        .skin-blue .main-header .logo:hover {
          background-color: #F5F5F5 !important;
        }
        .skin-blue .main-header .navbar {
          background-color: #FFFFFF !important;
        }
        .skin-blue .main-sidebar {
          background-color: #F8F8F8 !important;
        }
        .skin-blue .main-sidebar .sidebar-menu > li.active > a {
          background-color: #2A8C82 !important;
          border-left-color: #234E52 !important;
          color: #FFFFFF !important;
        }
        .skin-blue .main-sidebar .sidebar-menu > li > a {
          color: #2C3E50 !important;
        }
        .skin-blue .main-sidebar .sidebar-menu > li > a:hover {
          background-color: #E5E5E5 !important;
        }
        body {
          background-color: #FAFAFA !important;
        }
        .content-wrapper {
          background-color: #FAFAFA !important;
        }
        .content {
          background-color: #FAFAFA !important;
        }
        .box {
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
          border: 1px solid #D0D0D0 !important;
          background-color: #FFFFFF !important;
          margin-bottom: 15px !important;
          margin-left: 5px !important;
          margin-right: 5px !important;
        }
        .box-header {
          background-color: #FFFFFF !important;
          border-bottom: 1px solid #D0D0D0;
          border-radius: 8px 8px 0 0;
          padding: 5px 15px !important;
        }
        .box-title {
          margin-top: 0 !important;
          margin-bottom: 0 !important;
        }
        .form-control {
          border-radius: 5px;
          border: 1px solid #D0D0D0 !important;
          transition: border-color 0.3s;
          background-color: #FFFFFF;
        }
        .selectize-input {
          border: 1px solid #D0D0D0 !important;
          border-radius: 5px !important;
        }
        .selectize-dropdown {
          border: 1px solid #D0D0D0 !important;
        }
        .form-control:focus {
          border-color: #2A8C82 !important;
          box-shadow: 0 0 5px rgba(42, 140, 130, 0.5) !important;
        }
        .input-daterange {
          width: 100% !important;
        }
        .input-daterange .input-group-addon {
          min-width: 30px;
        }
        .btn-primary {
          background-color: #2A8C82 !important;
          border-color: #234E52 !important;
          border-radius: 6px;
          font-weight: 500;
          padding: 10px 20px;
          transition: all 0.2s;
          color: #FFFFFF !important;
        }
        .btn-primary:hover {
          background-color: #234E52 !important;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .price-display {
          font-size: 68px;
          font-weight: bold;
          color: #2A8C82;
          text-align: center;
          padding: 20px 15px;
          background-color: #FFFFFF;
          border-radius: 8px;
          margin: 10px 0;
        }
        #map {
          height: 100% !important;
          min-height: 300px !important;
          border-radius: 8px;
          overflow: hidden;
          position: relative;
        }
        /* Ion Range Slider Customization */
      .irs--shiny .irs-bar {
        background: #A0D8EF !important;
        border-top: 1px solid #A0D8EF !important;
        border-bottom: 1px solid #A0D8EF !important;
      }
      .irs--shiny .irs-bar-edge {
        background: #A0D8EF !important;
        border: 1px solid #A0D8EF !important;
      }
      .irs--shiny .irs-single, .irs--shiny .irs-from, .irs--shiny .irs-to {
        background: #A0D8EF !important;
      }
      
        .leaflet-container {
          width: 100% !important;
          overflow: hidden !important;
          position: relative;
        }
        #overview_map {
          height: 800px !important;
          min-height: 800px !important;
        }
        .map-container {
          flex: 1;
          min-height: 300px;
        }
        .content {
          padding: 15px 10px !important;
        }
        .box-body {
          padding: 15px !important;
        }
        .form-group {
          margin-bottom: 10px !important;
        }
        .form-group label {
          margin-bottom: 3px !important;
          font-size: 13px !important;
        }
        .row {
          margin-left: -5px !important;
          margin-right: -5px !important;
        }
        .col-sm-4, .col-md-4, .col-sm-6, .col-md-6, .col-sm-8, .col-md-8, .col-sm-12, .col-md-12 {
          padding-left: 5px !important;
          padding-right: 5px !important;
        }
        .checkbox {
          margin-top: 10px;
        }
        .checkbox label {
          font-weight: normal;
          color: #34495E;
        }
        .control-label {
          font-weight: 500;
          color: #5A5A5A;
          margin-bottom: 5px;
        }
        h3, h4, h5 {
          color: #2C3E50 !important;
        }
        .control-label {
          color: #4A4A4A !important;
        }
        /* Custom scrollbar for amenities */
        .amenities-scrollable {
          overflow-y: auto !important;
          overflow-x: hidden !important;
          position: relative;
          background-color: #FFFFFF;
        }
        .amenities-scrollable::-webkit-scrollbar {
          width: 8px;
        }
        .amenities-scrollable::-webkit-scrollbar-track {
          background: #F5F5F5;
          border-radius: 4px;
        }
        .amenities-scrollable::-webkit-scrollbar-thumb {
          background: #B0B0B0;
          border-radius: 4px;
        }
        .amenities-scrollable::-webkit-scrollbar-thumb:hover {
          background: #909090;
        }
        /* Ensure checkbox group content is scrollable */
        .amenities-scrollable .shiny-input-checkboxgroup {
          display: block !important;
        }
        .amenities-scrollable .shiny-input-checkboxgroup .checkbox {
          margin-top: 8px;
          margin-bottom: 8px;
        }
      "))
    ),
    
    tabItems(
      tabItem(
        tabName = "overview",
        fluidRow(
          # Left side - Map
          column(
            width = 8,
            box(
              title = tags$h3("London Airbnb Listings", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
                width = NULL,
                solidHeader = TRUE,
                status = "primary",
                leafletOutput("overview_map", height = "800px")
              )
          ),
          
          # Right side - Controls and Legend
          column(
            width = 4,
            box(
              title = tags$h3("Color Coding Options", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "info",
              
              selectInput(
                "color_by",
                label = tags$strong("Color Points By:"),
                choices = list(
                  "Room Type" = "room_type",
                  "Neighbourhood" = "neighbourhood"
                ),
                selected = "room_type",
                width = "100%"
              ),
              
              hr(),
              
              tags$h5("Distribution (Click to Filter)", style = "color: #4A4A4A; font-weight: 600;"),
              uiOutput("category_bars"),
              
              hr(),
              
              tags$h5("Summary Statistics", style = "color: #4A4A4A; font-weight: 600;"),
              uiOutput("overview_stats")
            ),
            
            box(
              title = tags$h3("Filter Options", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "warning",
              
              selectInput(
                "filter_room_type",
                label = "Room Type:",
                choices = c("All" = "all"),
                selected = "all",
                width = "100%"
              ),
              
               sliderInput(
                 "filter_price",
                 label = "Price Range (£):",
                 min = 0,
                 max = 500,
                 value = c(0, 500),
                 step = 10,
                 width = "100%"
               ),
              
              actionButton(
                "apply_filter",
                "Apply Filters",
                class = "btn-primary",
                style = "width: 100%; margin-top: 10px;"
              )
            )
          )
        )
      ),
      
      tabItem(
        tabName = "predict",
        fluidRow(
          style = "display: flex; align-items: flex-start;",
          # Left column - Property Information
          column(
            width = 4,
            style = "display: flex; flex-direction: column;",
            box(
              title = tags$h3("Property Details", style = "color: #2C3E50; margin: 0; font-weight: 600; font-size: 18px;"),
              width = NULL,
              solidHeader = TRUE,
              status = "primary",
              style = "margin-bottom: 0; display: flex; flex-direction: column;",
              
              # Postcode input with status on the right
              fluidRow(
                column(
                  width = 6,
                  textInput(
                    "address",
                    label = tags$strong("Postcode"),
                    placeholder = "e.g., SW1A 1AA",
                    width = "100%"
                  )
                ),
                column(
                  width = 6,
                  style = "display: flex; align-items: center; padding-left: 8px; padding-top: 25px;",
                  conditionalPanel(
                    condition = "output.geocode_status",
                    tags$div(
                      style = "width: 100%;",
                      uiOutput("geocode_status_text")
                    )
                  )
                )
              ),
              
              hr(style = "margin: 12px 0;"),
              
              # Two columns: Basic Properties (left) and Amenities (right)
              fluidRow(
                style = "display: flex; align-items: stretch; flex: 1; min-height: 0;",
                # Left column - Basic Properties
                column(
                  width = 6,
                  style = "display: flex; flex-direction: column; height: 100%;",
                  tags$h4("Basic Properties", style = "color: #4A4A4A; margin-top: 0; margin-bottom: 12px; font-weight: 600; font-size: 14px;"),
                  
                  tags$div(
                    style = "flex: 1; display: flex; flex-direction: column;",
                    numericInput(
                      "bedrooms",
                      "Bedrooms",
                      value = 1,
                      min = 0,
                      max = 20,
                      step = 1,
                      width = "100%"
                    ),
                    
                    numericInput(
                      "bathrooms",
                      "Bathrooms",
                      value = 1,
                      min = 0,
                      max = 10,
                      step = 0.5,
                      width = "100%"
                    ),
                    
                    numericInput(
                      "accommodates",
                      "Accommodates",
                      value = 2,
                      min = 1,
                      max = 20,
                      step = 1,
                      width = "100%"
                    ),
                    
                    numericInput(
                      "beds",
                      "Beds",
                      value = 1,
                      min = 0,
                      max = 20,
                      step = 1,
                      width = "100%"
                    ),
                    
                    selectInput(
                      "room_type",
                      "Room Type",
                      choices = list(
                        "Entire home/apt" = "Entire home/apt",
                        "Private room" = "Private room",
                        "Shared room" = "Shared room"
                      ),
                      selected = "Entire home/apt",
                      width = "100%"
                    )
                  )
                ),
                
                # Right column - Amenities
                column(
                  width = 6,
                  style = "display: flex; flex-direction: column; height: 100%;",
                  tags$h4("Amenities", style = "color: #4A4A4A; margin-top: 0; margin-bottom: 12px; font-weight: 600; font-size: 14px;"),
                  
                  tags$div(
                    class = "amenities-scrollable",
                    style = "overflow-y: auto !important; overflow-x: hidden; border: 1px solid #D0D0D0; padding: 10px; border-radius: 5px; background-color: #F5F5F5; max-height: 350px; height: 350px;",
                    checkboxGroupInput(
                      "amenities",
                      NULL,
                      choices = list(
                        "WiFi" = "Wifi",
                        "Smoke Alarm" = "Smoke.alarm",
                        "Kitchen" = "Kitchen",
                        "Washer" = "Washer",
                        "Essentials" = "Essentials",
                        "Iron" = "Iron",
                        "Hot Water" = "Hot.water",
                        "Hangers" = "Hangers",
                        "Carbon Monoxide Alarm" = "Carbon.monoxide.alarm",
                        "Hair Dryer" = "Hair.dryer",
                        "Heating" = "Heating",
                        "Bed Linens" = "Bed.linens",
                        "TV" = "TV",
                        "Dishes and Silverware" = "Dishes.and.silverware",
                        "Refrigerator" = "Refrigerator",
                        "Cooking Basics" = "Cooking.basics",
                        "Shampoo" = "Shampoo",
                        "Microwave" = "Microwave",
                        "Hot Water Kettle" = "Hot.water.kettle",
                        "Oven" = "Oven",
                        "Dedicated Workspace" = "Dedicated.workspace",
                        "Toaster" = "Toaster",
                        "Freezer" = "Freezer",
                        "Shower Gel" = "Shower.gel",
                        "First Aid Kit" = "First.aid.kit",
                        "Dining Table" = "Dining.table",
                        "Cleaning Products" = "Cleaning.products",
                        "Self Check-in" = "Self.check.in",
                        "Fire Extinguisher" = "Fire.extinguisher",
                        "Long Term Stays Allowed" = "Long.term.stays.allowed"
                      ),
                      selected = c("Wifi", "Kitchen", "Heating")
                    )
                  )
                )
                ),
              
              # Button at the bottom
              tags$div(
                style = "margin-top: 12px;",
                actionButton(
                  "predict_btn",
                  "Update Prediction",
                  class = "btn-primary",
                  style = "width: 100%; font-size: 16px; padding: 12px;"
                )
              )
            )
          ),
          
          # Right column - Predictions, Recommendations, and Map
          column(
            width = 8,
            style = "display: flex; flex-direction: column; height: 100%;",
            # Top row - Price and Recommendations side by side (equal height)
            fluidRow(
              style = "display: flex; align-items: stretch; flex-shrink: 0;",
              column(
                width = 6,
                style = "display: flex; flex-direction: column;",
                box(
                  title = tags$h3("Predicted Baseline Price", style = "color: #2C3E50; margin: 0; font-weight: 600; font-size: 18px;"),
                  width = NULL,
                  solidHeader = TRUE,
                  status = "primary",
                  style = "height: 160px; display: flex; flex-direction: column; margin-bottom: 15px;",
                  
                  conditionalPanel(
                    condition = "output.price_predicted",
                    tags$div(
                      style = "flex: 1; display: flex; flex-direction: column; justify-content: center;",
                      tags$div(
                        class = "price-display",
                        style = "font-size: 68px; padding: 0 15px; margin: 20px 0 5px 0; font-weight: bold; color: #2A8C82; line-height: 1;",
                        textOutput("predicted_price")
                      ),
                      tags$div(
                        style = "text-align: center; color: #888888; margin-top: 5px; font-size: 14px;",
                        textOutput("price_note")
                      )
                    )
                  ),
                  
                  conditionalPanel(
                    condition = "!output.price_predicted",
                    tags$div(
                      style = "text-align: center; padding: 30px; color: #AAAAAA; flex: 1; display: flex; align-items: center; justify-content: center;",
                      tags$p("Fill in property information and click Update Prediction", style = "font-size: 14px;")
                    )
                  )
                )
              ),
              
              column(
                width = 6,
                style = "display: flex; flex-direction: column;",
                box(
                  title = tags$h3("Top 3 Amenity Recommendations", style = "color: #2C3E50; margin: 0; font-weight: 600; font-size: 18px;"),
                  width = NULL,
                  solidHeader = TRUE,
                  status = "warning",
                  style = "height: 160px; display: flex; flex-direction: column; margin-bottom: 15px;",
                  
                  conditionalPanel(
                    condition = "output.price_predicted",
                    tags$div(
                      style = "padding: 0 15px 5px 15px; flex: 1; overflow-y: auto; overflow-x: hidden;",
                      uiOutput("amenity_recommendations")
                    )
                  ),
                  
                  conditionalPanel(
                    condition = "!output.price_predicted",
                    tags$div(
                      style = "text-align: center; padding: 30px; color: #AAAAAA; flex: 1; display: flex; align-items: center; justify-content: center;",
                      tags$p("Get price prediction to see recommendations", style = "font-size: 14px;")
                    )
                  )
                )
              )
            ),
            
            # Bottom row - Map (full width)
            fluidRow(
              style = "display: flex; flex: 1; min-height: 0; margin-top: 0;",
              column(
                width = 12,
                style = "display: flex; flex-direction: column; height: 100%; width: 100%; padding-left: 15px; padding-right: 15px;",
                box(
                  title = tags$h3("Location Context", style = "color: #2C3E50; margin: 0; font-weight: 600; font-size: 18px;"),
                  width = NULL,
                  solidHeader = TRUE,
                  status = "info",
                  style = "margin-bottom: 0; display: flex; flex-direction: column; height: calc(100% - 25px);",
                  
                  tags$div(
                    class = "map-container",
                    style = "flex: 1; min-height: 280px; position: relative;",
                    leafletOutput("map", height = "calc(100% - 15px)")
                  )
                )
              )
            )
          )
        )
      ),
      
      tabItem(tabName = "calendar",
        fluidRow(
          style = "display: flex; align-items: stretch;",
        column(8,
          div(class = "card",
            style = "height: 97%;",
            div(class = "calendar-header",
              div(style = "display: flex; gap: 8px;",
                uiOutput("prev_btn"),
                actionButton("go_today", "Today", class = "calendar-nav-btn today-btn")
              ),
              div(class = "calendar-title", textOutput("current_month_title")),
              uiOutput("next_btn")
            ),
            uiOutput("calendar_grid"),
            div(class = "legend",
              # Pricing tiers
              div(class = "legend-item", div(class = "legend-color demand-premium"), tags$span("+30%")),
              div(class = "legend-item", div(class = "legend-color demand-above"), tags$span("+15%")),
              div(class = "legend-item", div(class = "legend-color demand-weatherboost"), tags$span("+5%")),
              div(class = "legend-item", div(class = "legend-color demand-standard"), tags$span("Base")),
              # Divider
              div(class = "legend-divider"),
              # Markers
              div(class = "legend-item", div(class = "legend-dot", style = "background: #F5B085;"), tags$span("Holiday")),
              div(class = "legend-item", div(class = "legend-color", style = "background: #C7E9F8;"), tags$span("Today")),
              div(class = "legend-item", div(class = "legend-color", style = "border: 2px solid #1e293b; background: transparent;"), tags$span("Selected"))
            )
          )
        ),
        column(4,
          div(
            style = "display: flex; flex-direction: column; height: 100%;",
            div(class = "card",
              div(class = "section-title", "Selected Date Details"),
              uiOutput("date_details_panel")
            ),
            div(class = "card",
              div(class = "section-title", "High Demand Days This Month"),
              div(style = "height: 70px !important; overflow-y: auto;",
                DTOutput("high_demand_table")
              )
            )
              )
            )
          )
        ),
        
      tabItem(tabName = "trends",
        fluidRow(
        column(12,
          div(class = "card",
            div(class = "section-title", "Price Adjustment Forecast"),
            plotlyOutput("demand_timeline", height = "300px")
          )
        )
      ),
        fluidRow(
        column(6,
          div(class = "card",
            div(class = "section-title", "TfL Transport Activity"),
            plotlyOutput("tourism_chart", height = "280px")
          )
        ),
        column(6,
          div(class = "card",
            div(class = "section-title", "Weather Forecast"),
            plotlyOutput("weather_chart", height = "280px")
          )
        )
      )
    ),
      
      tabItem(tabName = "calculator",
        fluidRow(
          style = "display: flex; align-items: stretch; margin-bottom: 8px;",
        column(4,
          div(class = "card", style = "height: 100%;",
            div(class = "section-title", "Settings"),
            div(style = "width: 100%;",
              dateRangeInput("date_range", "Date Range",
                            start = Sys.Date(),
                            end = Sys.Date() + 30,
                            min = DATA_START,
                            max = DATA_END)
            ),
            div(style = "margin-top: 15px; width: 100%;",
              actionButton("calculate", "Calculate Prices", class = "btn-primary", style = "width: 100%;")
            )
          )
        ),
        column(8,
          # Price Summary Card (moved to right)
          div(class = "card", style = "height: 100%; display: flex; flex-direction: column;",
            div(class = "section-title", "Price Summary"),
            div(style = "flex: 1; min-height: 0;",
              uiOutput("price_summary_panel")
            )
          )
        )
      ),
        fluidRow(
        column(12,
          div(class = "card",
            div(class = "section-title", "Price Timeline"),
            plotlyOutput("price_timeline", height = "280px")
            )
          )
        )
      ),
      
      tabItem(
        tabName = "about",
        box(
          title = tags$h3("About This App", style = "color: #2C3E50; margin: 0;"),
          width = 12,
          solidHeader = TRUE,
          status = "primary",
          
          tags$div(
            style = "padding: 20px;",
           
            tags$p("AirbnbHostGenius is an intelligent, data-driven platform designed to help Airbnb hosts maximize their annual revenue through machine learning-powered pricing recommendations and occupancy predictions."),
            tags$p("Creators: Linda Liu & Shirley Xiong"),
            tags$hr(),
            tags$h5("Pages:", style = "color: #2C3E50;"),
            tags$ul(
              tags$li("Overview: London map + filters to explore listings distribution and price levels."),
              tags$li("Price Prediction: Enter address & property details to get baseline price (XGBoost)."),
              tags$li("Dynamic Price: Apply calendar-based adjustments (+15%, +30%, holidays) to the baseline."),
              tags$li("Calendar: Visual calendar showing day-level adjustments and demand tags."),
              tags$li("Demand Trends: TfL transport and weather forecasts for demand signals.")
            ),
            tags$hr(),
           
          )
        )
      )
    )
  ),
  skin = "blue"
)

# =============================================
# Server
# =============================================

server <- function(input, output, session) {

  # User Server Logic
  
  
  # =============================================
  # Overview Tab - Load Listings Data & London Boundary
  # =============================================
  
  # Load London boundary (once)
  london_boundary <- reactiveVal(NULL)
  
  load_london_boundary <- function() {
    if (!is.null(london_boundary())) {
      return(london_boundary())
    }
    
    # Try to find the shapefile
    shp_paths <- c(
      file.path(getwd(), "tfl,weather,tourism,holiday_prediction", "foot_traffic_data", "raw", "geographic", "statistical-gis-boundaries-london", "ESRI", "London_Borough_Excluding_MHW.shp"),
      file.path(app_dir, "data", "London_Borough_Excluding_MHW.shp"),
      file.path(dirname(getwd()), "tfl,weather,tourism,holiday_prediction", "foot_traffic_data", "raw", "geographic", "statistical-gis-boundaries-london", "ESRI", "London_Borough_Excluding_MHW.shp")
    )
    
    for (shp_path in shp_paths) {
      if (file.exists(shp_path)) {
        tryCatch({
          boundary <- st_read(shp_path, quiet = TRUE) %>%
            st_transform(4326) %>%  # Transform to WGS84
            st_union()  # Merge all boroughs into one boundary
          london_boundary(boundary)
          message("Loaded London boundary from: ", shp_path)
          return(boundary)
        }, error = function(e) {
          message("Error loading boundary: ", e$message)
        })
      }
    }
    return(NULL)
  }
  
  # Load boroughs (individual polygons with names)
  boroughs_data <- reactiveVal(NULL)
  
  load_boroughs <- function() {
    if (!is.null(boroughs_data())) {
      return(boroughs_data())
    }
    
    # Try to find the shapefile
    shp_paths <- c(
      file.path(getwd(), "tfl,weather,tourism,holiday_prediction", "foot_traffic_data", "raw", "geographic", "statistical-gis-boundaries-london", "ESRI", "London_Borough_Excluding_MHW.shp"),
      file.path(app_dir, "data", "London_Borough_Excluding_MHW.shp"),
      file.path(dirname(getwd()), "tfl,weather,tourism,holiday_prediction", "foot_traffic_data", "raw", "geographic", "statistical-gis-boundaries-london", "ESRI", "London_Borough_Excluding_MHW.shp")
    )
    
    for (shp_path in shp_paths) {
      if (file.exists(shp_path)) {
        tryCatch({
          boroughs <- st_read(shp_path, quiet = TRUE) %>%
            st_transform(4326)
          boroughs_data(boroughs)
          return(boroughs)
        }, error = function(e) {
          message("Error loading boroughs: ", e$message)
        })
      }
    }
    return(NULL)
  }
  
  # Load listings data (lazy load)
  listings_data <- reactiveVal(NULL)
  
  load_listings <- function() {
    if (!is.null(listings_data())) {
      return(listings_data())
    }
    
    # Use competitor data (training data) as the listings data
    data <- NULL
    
    # Try to use global competitor_data if available
    if (exists("competitor_data") && !is.null(competitor_data)) {
      data <- competitor_data
    } else if (exists("load_competitor_data")) {
      # Try to load it if not already loaded
      data <- load_competitor_data()
    }
    
    if (is.null(data)) {
      # Fallback to original logic just in case
      listings_file <- file.path(app_dir, "data", "listings.csv")
      if (file.exists(listings_file)) {
    tryCatch({
      data <- data.table::fread(listings_file) %>%
        filter(!is.na(latitude), !is.na(longitude)) %>%
        mutate(
          price = as.numeric(gsub("[£$,]", "", price)),
          room_type = as.factor(room_type),
          neighbourhood = as.factor(neighbourhood)
        ) %>%
        filter(!is.na(price), price > 0, price < 10000)
        }, error = function(e) { NULL })
      }
    }

    if (is.null(data)) return(NULL)

    # Transform data to match what UI expects
    # UI expects: price, room_type, neighbourhood, name, number_of_reviews
    
    # Map price_num to price
    if (!"price" %in% names(data) && "price_num" %in% names(data)) {
      data$price <- data$price_num
    }
    
    # Map room_type_id to room_type
    if (!"room_type" %in% names(data) && "room_type_id" %in% names(data)) {
      room_types <- c("Entire home/apt", "Private room", "Shared room")
      # Ensure 1-based index for R
      data$room_type <- factor(room_types[data$room_type_id + 1])
    }
    
    # Map neighbourhood
    if (!"neighbourhood" %in% names(data)) {
      # Try spatial join first to get real names
      boroughs <- load_boroughs()
      mapped <- FALSE
      
      if (!is.null(boroughs)) {
        tryCatch({
          points <- st_as_sf(data, coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)
          joined <- st_join(points, boroughs)
          
          # Find name column
          possible_names <- names(boroughs)[sapply(boroughs, function(x) is.character(x) || is.factor(x))]
          name_col <- NULL
          for (col in c("NAME", "BOROUGH", "DISTRICT", "Name", "Borough")) {
            if (col %in% possible_names) {
              name_col <- col
              break
            }
          }
          if (is.null(name_col) && length(possible_names) > 0) name_col <- possible_names[1]
          
          if (!is.null(name_col)) {
            data$neighbourhood <- as.character(joined[[name_col]])
            data$neighbourhood[is.na(data$neighbourhood)] <- "Unknown"
            data$neighbourhood <- as.factor(data$neighbourhood)
            mapped <- TRUE
          }
    }, error = function(e) {
          cat("Spatial join failed:", e$message, "\n")
        })
      }
      
      if (!mapped) {
        if ("neighbourhood_id" %in% names(data)) {
          data$neighbourhood <- as.factor(paste("Area", data$neighbourhood_id))
        } else {
          data$neighbourhood <- as.factor("Unknown")
        }
      }
    }
    
    # Add dummy name if missing
    if (!"name" %in% names(data)) {
      data$name <- paste("Listing #", 1:nrow(data))
    }
    
    # Add dummy reviews if missing
    if (!"number_of_reviews" %in% names(data)) {
      data$number_of_reviews <- 0
    }
    
    # Final cleanup
    data <- data %>%
      filter(!is.na(latitude), !is.na(longitude), !is.na(price))
      
      listings_data(data)
      return(data)
  }
  
  # Update filter choices when data loads
  observe({
    data <- load_listings()
    if (!is.null(data)) {
      room_types <- c("All" = "all", setNames(unique(as.character(data$room_type)), unique(as.character(data$room_type))))
      updateSelectInput(session, "filter_room_type", choices = room_types)
      
       max_price <- 500
       updateSliderInput(session, "filter_price", max = max_price, value = c(0, max_price))
    }
  })
  
  # Filtered data based on user selections
  filtered_listings <- reactive({
    data <- load_listings()
    if (is.null(data)) return(NULL)
    
    # Apply filters
    filtered <- data
    
    if (input$filter_room_type != "all") {
      filtered <- filtered %>% filter(room_type == input$filter_room_type)
    }
    
    filtered <- filtered %>%
      filter(price >= input$filter_price[1], price <= input$filter_price[2])
    
    # Sample based on user selection for performance
    return(filtered)
  }) %>% bindEvent(input$apply_filter, ignoreNULL = FALSE, ignoreInit = FALSE)
  
  # Selected category from bar chart click
  selected_category <- reactiveVal(NULL)
  
  # Color palette based on selection
  get_color_palette <- function(data, color_by) {
    if (color_by == "room_type") {
      # Use fixed levels and colors to ensure consistency
      levels <- c("Entire home/apt", "Private room", "Shared room")
      
      # Custom palette based on user request
      colors_map <- c(
        "Entire home/apt" = "#2A8C82",  # Deep Teal
        "Private room" = "#A0D8EF",     # Sky Blue
        "Shared room" = "#F5B085"       # Peach
      )
      
      # Return explicit mapping
      return(list(
        levels = levels,
        colors = colors_map,
        pal = colorFactor(colors_map, domain = levels)
      ))
    } else {
      levels <- sort(unique(as.character(data$neighbourhood)))
      n_colors <- length(levels)
      
      # Extended distinct palette (High contrast, no grey/muddy tones, replaced purples with teal/orange variants)
      base_colors <- c(
        "#2A8C82", "#F5B085", "#2C3E50", "#8DD3C7", "#C0392B", 
        "#A0D8EF", "#E67E22", "#E59866", "#3498DB", "#F1C40F", 
        "#16A085", "#17A589", "#2980B9", "#D35400", "#27AE60", 
        "#1ABC9C", "#E74C3C", "#F39C12", "#5D6D7E", "#48C9B0", 
        "#73C6B6", "#EC7063", "#58D68D", "#5DADE2", "#F4D03F", 
        "#EB984E", "#76D7C4", "#85C1E9", "#F5CBA7", "#CD6155", 
        "#52BE80", "#DC7633", "#5499C7"
      )
      
      if (n_colors <= length(base_colors)) {
        colors <- base_colors[1:n_colors]
      } else {
        # Fallback to repeating if we exceed 33 (unlikely for London boroughs)
        colors <- rep(base_colors, length.out = n_colors)
      }
      
      return(list(
        levels = levels,
        colors = setNames(colors, levels),
        pal = colorFactor(colors, domain = levels)
      ))
    }
  }
  
  # Reset selected category when color_by changes
  observeEvent(input$color_by, {
    selected_category(NULL)
  })
  
  # Render Overview Map
  output$overview_map <- renderLeaflet({
    data <- filtered_listings()
    boundary <- load_london_boundary()
    sel <- selected_category()
    
    if (is.null(data) || nrow(data) == 0) {
      return(
        leaflet() %>%
          addTiles() %>%
          setView(lng = -0.1276, lat = 51.5074, zoom = 10) %>%
          addPopups(-0.1276, 51.5074, "Loading listings data...")
      )
    }
    
    color_by <- input$color_by
    color_info <- get_color_palette(data, color_by)
    
    # Create popup content and set opacity based on selection
    data <- data %>%
      mutate(
        popup_content = paste0(
          "<b>", name, "</b><br>",
          "<b>Room:</b> ", room_type, "<br>",
          "<b>Area:</b> ", neighbourhood, "<br>",
          "<b>Price:</b> £", price, "/night<br>",
          "<b>Reviews:</b> ", number_of_reviews
        ),
        category_value = as.character(get(color_by))
      )
    
    # Highlight selected category, dim others
    if (!is.null(sel) && sel != "") {
      data <- data %>%
        mutate(
          point_opacity = ifelse(category_value == sel, 0.9, 0.1),
          point_radius = ifelse(category_value == sel, 3, 1)
        )
    } else {
      data <- data %>%
        mutate(
          point_opacity = 0.7,
          point_radius = 2
        )
    }
    
    # Create base map
    map <- leaflet(data) %>%
      addTiles()
    
    # Add dimmed overlay outside London (if boundary available)
    if (!is.null(boundary)) {
      # Create a large polygon covering the whole area
      world_polygon <- st_polygon(list(rbind(
        c(-2, 50), c(1, 50), c(1, 53), c(-2, 53), c(-2, 50)
      ))) %>%
        st_sfc(crs = 4326)
      
      # Cut out London from the world polygon
      outside_london <- st_difference(world_polygon, boundary)
      
      # Add the dimmed overlay (outside London)
      map <- map %>%
        addPolygons(
          data = outside_london,
          fillColor = "#FFFFFF",
          fillOpacity = 0.7,
          stroke = FALSE,
          group = "overlay"
        ) %>%
        # Add London boundary
        addPolygons(
          data = boundary,
          fillColor = "transparent",
          fillOpacity = 0,
          color = "#2C3E50",
          weight = 3,
          opacity = 1,
          group = "boundary"
        )
    }
    
    # Add listing points (NO clustering - each point visible)
    # Smaller points (radius = 2, or 3 when highlighted)
    map <- map %>%
      addCircleMarkers(
        lng = ~longitude,
        lat = ~latitude,
        radius = ~point_radius,
        color = ~color_info$pal(get(color_by)),
        fillOpacity = ~point_opacity,
        stroke = FALSE,
        popup = ~popup_content
      ) %>%
      setView(lng = -0.1276, lat = 51.5074, zoom = 11)
    
    return(map)
  })
  
  # Render Interactive Bar Chart using HTML/CSS (more reliable than plotly)
  output$category_bars <- renderUI({
    data <- filtered_listings()
    color_by <- input$color_by
    
    if (is.null(data) || nrow(data) == 0 || is.null(color_by)) {
      return(tags$p("Loading...", style = "color: #888; text-align: center; padding: 20px;"))
    }
    
    color_info <- get_color_palette(data, color_by)
    
    # Count per category
    counts <- data %>%
      group_by(category = !!sym(color_by)) %>%
      summarise(count = n(), .groups = "drop") %>%
      arrange(desc(count)) %>%
      as.data.frame()
    
    # Limit to top 10 for neighbourhood
    if (color_by == "neighbourhood" && nrow(counts) > 10) {
      counts <- head(counts, 10)
    }
    
    max_count <- max(counts$count)
    sel <- selected_category()
    
    # Create clickable bars
    bars <- lapply(1:nrow(counts), function(i) {
      cat_name <- as.character(counts$category[i])
      count <- counts$count[i]
      color <- color_info$colors[cat_name]
      width_pct <- (count / max_count) * 100
      
      # Determine if this is selected
      is_selected <- !is.null(sel) && sel == cat_name
      opacity <- if (!is.null(sel) && !is_selected) 0.3 else 1
      border <- if (is_selected) "2px solid #333" else "none"
      
      # Create action button styled as a bar
      tags$div(
        style = "display: flex; align-items: center; margin-bottom: 6px; cursor: pointer;",
        onclick = sprintf("Shiny.setInputValue('bar_click', '%s', {priority: 'event'})", cat_name),
        
        # Label
        tags$div(
          style = "width: 90px; font-size: 11px; text-align: right; padding-right: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;",
          title = cat_name,
          cat_name
        ),
        
        # Bar
        tags$div(
          style = "flex: 1; height: 20px; background: #f0f0f0; border-radius: 3px; overflow: hidden;",
          tags$div(
            style = sprintf(
              "width: %s%%; height: 100%%; background: %s; opacity: %s; border-radius: 3px; border: %s; transition: all 0.2s;",
              width_pct, color, opacity, border
            )
          )
        ),
        
        # Count
        tags$div(
          style = "width: 50px; font-size: 11px; text-align: right; padding-left: 8px; color: #666;",
          format(count, big.mark = ",")
        )
      )
    })
    
    tagList(
      tags$div(
        style = "padding: 5px 0;",
        bars
      ),
      tags$p(
        style = "font-size: 10px; color: #999; margin-top: 8px; text-align: center;",
        "Click a bar to highlight on map"
      )
    )
  })
  
  # Handle bar click
  observeEvent(input$bar_click, {
    clicked <- input$bar_click
    current_sel <- selected_category()
    
    if (!is.null(current_sel) && current_sel == clicked) {
      selected_category(NULL)
    } else {
      selected_category(clicked)
    }
  }, ignoreInit = TRUE)
  
  # Render Overview Stats
  output$overview_stats <- renderUI({
    data <- filtered_listings()
    if (is.null(data) || nrow(data) == 0) {
      return(tags$p("No data available", style = "color: #888;"))
    }
    
    total <- nrow(data)
    avg_price <- mean(data$price, na.rm = TRUE)
    median_price <- median(data$price, na.rm = TRUE)
    n_neighbourhoods <- length(unique(data$neighbourhood))
    
    tagList(
      tags$div(
        style = "display: grid; grid-template-columns: 1fr 1fr; gap: 10px;",
        tags$div(
          style = "background: #FEF5E7; padding: 10px; border-radius: 5px; text-align: center;",
          tags$div(style = "font-size: 20px; font-weight: bold; color: #F5B085;", paste0("£", round(avg_price))),
          tags$div(style = "font-size: 11px; color: #666;", "Avg Price")
        ),
        tags$div(
          style = "background: #EAF6F5; padding: 10px; border-radius: 5px; text-align: center;",
          tags$div(style = "font-size: 20px; font-weight: bold; color: #2A8C82;", format(total, big.mark = ",")),
          tags$div(style = "font-size: 11px; color: #666;", "Rooms (Filtered)")
        )
      )
    )
  })
  
  # =============================================
  # Model Loading (existing code)
  # =============================================
  
  # Model loading status (lazy load - only when needed)
  model_loaded <- reactiveVal(FALSE)
  model_loading <- reactiveVal(FALSE)
  
  # Lazy load models only when prediction is requested
  load_models_if_needed <- function() {
    if (model_loaded()) {
      return(TRUE)
    }
    
    if (model_loading()) {
      return(FALSE)  # Already loading
    }
    
    model_loading(TRUE)
    
    tryCatch({
      showNotification("Loading models... This may take a moment.", type = "message", duration = 3)
      
      # Load models in background
      result <- load_models()
      
      if (isTRUE(result)) {
        model_loaded(TRUE)
        model_loading(FALSE)
        showNotification("Models loaded successfully!", type = "message", duration = 2)
        return(TRUE)
      } else {
        model_loading(FALSE)
        showNotification("Model loading failed. Check console for details.", type = "error", duration = 5)
        return(FALSE)
      }
    }, error = function(e) {
      model_loading(FALSE)
      error_msg <- as.character(e$message)
      if (nchar(error_msg) > 100) {
        error_msg <- paste0(substr(error_msg, 1, 97), "...")
      }
      showNotification(paste("Model loading failed:", error_msg), type = "error", duration = 5)
      cat("Model loading error details:", error_msg, "\n")
      return(FALSE)
    })
  }
  
  # Geocoding
  geocode_result <- reactiveVal(NULL)
  geocode_status <- reactiveVal("")
  
  observeEvent(input$address, {
    address <- trimws(input$address)
    
    if (nchar(address) == 0) {
      geocode_result(NULL)
      geocode_status("")
      return()
    }
    
    if (nchar(address) < 3) {
      geocode_status("")
      geocode_result(NULL)
      return()
    }
    
    geocode_status("Searching for location...")
    
    invalidateLater(1500, session)
    
    isolate({
      tryCatch({
        result <- geocode_address(address)
        
        if (!is.null(result) && !is.na(result$lat) && !is.na(result$lon)) {
          geocode_result(result)
          display_name <- result$display_name
          if (nchar(display_name) > 50) {
            display_name <- paste0(substr(display_name, 1, 47), "...")
          }
          geocode_status(paste0("Location found: ", display_name))
        } else {
          geocode_result(NULL)
          geocode_status("Cannot find this address, please check your input")
        }
      }, error = function(e) {
        geocode_result(NULL)
          geocode_status("Error searching location, please try again later")
      })
    })
  }, ignoreInit = TRUE)
  
  output$geocode_status <- reactive({
    status <- geocode_status()
    nchar(status) > 0
  })
  outputOptions(output, "geocode_status", suspendWhenHidden = FALSE)
  
  output$geocode_status_text <- renderUI({
    status <- geocode_status()
    if (nchar(status) == 0) return(NULL)
    
    # Common style for consistent height/alignment
    # min-height: 55px ensures it matches the height of a 3-line message roughly
    # display: flex + align-items: center ensures single lines are vertically centered
    common_style <- "font-weight: 500; padding: 4px 10px; border-radius: 5px; font-size: 12px; width: 100%; min-height: 55px; display: flex; align-items: center; line-height: 1.3;"
    
    if (grepl("Location found", status)) {
      location_text <- sub("Location found: ", "", status)
      if (nchar(location_text) > 40) {
        location_text <- paste0(substr(location_text, 1, 37), "...")
      }
      
      tags$div(
        tags$span(paste("Location found:", location_text), style = "font-size: 11px;"),
        style = paste0("color: #234E52; background-color: #E0F2F1; ", common_style)
      )
    } else if (grepl("Cannot find|Error", status)) {
      # Use Red/Pink for errors
      tags$div(
        status,
        style = paste0("color: #C0392B; background-color: #FADBD8; ", common_style)
      )
    } else {
      tags$div(
        status,
        style = paste0("color: #234E52; background-color: #E0F2F1; ", common_style)
      )
    }
  })
  
  output$map <- renderLeaflet({
    result <- geocode_result()
    
    if (!is.null(result) && !is.na(result$lat) && !is.na(result$lon)) {
      m <- leaflet() %>%
        addTiles() %>%
        addMarkers(lng = result$lon, lat = result$lat, popup = paste("<b>Target Location</b><br>", result$display_name)) %>%
        setView(lng = result$lon, lat = result$lat, zoom = 15)
      
      # Add competitor markers if data is available
      if (!is.null(competitor_data) && nrow(competitor_data) > 0) {
        # Calculate distance to finding nearest competitors
        # Use simple Euclidean distance for speed as first pass, or Haversine for accuracy
        # Since we want nearest ~20, calculating all distances is fine for 50k rows (vectorized)
        
        # Calculate distances (in meters)
        dists <- distHaversine(
          cbind(competitor_data$longitude, competitor_data$latitude),
          c(result$lon, result$lat)
        )
        
        # Add distance to data
        comp_df <- competitor_data
        comp_df$dist <- dists
        
        # Select nearest 20 listings
        nearest_comps <- comp_df[order(comp_df$dist), ][seq_len(min(20, nrow(comp_df))), ]
        
        # Map room_type_id back to name for display
        room_types <- c("Entire home/apt", "Private room", "Shared room")
        nearest_comps$room_type_name <- room_types[nearest_comps$room_type_id + 1] # 0-indexed to 1-indexed
        
        # Create popup content
        nearest_comps$popup <- paste0(
          "<b>Comparables</b><br>",
          "Price: £", nearest_comps$price_num, "<br>",
          "Type: ", nearest_comps$room_type_name, "<br>",
          "Bedrooms: ", nearest_comps$bedrooms, "<br>",
          "Accommodates: ", nearest_comps$accommodates, "<br>",
          "Distance: ", round(nearest_comps$dist), "m"
        )
        
        # Add to map
        m <- m %>% addCircleMarkers(
          data = nearest_comps,
          lng = ~longitude,
          lat = ~latitude,
          radius = 6,
          color = "#38B2AC",  # Lighter teal (based on title color #2A8C82)
          stroke = TRUE,
          fillOpacity = 0.8,
          popup = ~popup
        )
      }
      
      m
    } else {
      leaflet() %>%
        addTiles() %>%
        setView(lng = -0.1276, lat = 51.5074, zoom = 10) %>%
        addPopups(lng = -0.1276, lat = 51.5074, "Please enter address or postcode to find location")
    }
  })
  
  prediction_result <- reactiveVal(NULL)
  
  observeEvent(input$predict_btn, {
    result <- geocode_result()
    
    if (is.null(result) || is.na(result$lat) || is.na(result$lon)) {
      showNotification("Please enter a valid address or postcode first", type = "warning")
      return()
    }
    
    lat <- result$lat
    lon <- result$lon
    
    # Lazy load models only when needed
    if (!load_models_if_needed()) {
      showNotification("Models are still loading, please wait...", type = "warning")
      return()
    }
    
    showNotification("Predicting price...", type = "message")
    
    tryCatch({
      # Build features
      feature_result <- build_features(
        lat = lat,
        lon = lon,
        bedrooms = input$bedrooms,
        bathrooms = input$bathrooms,
        accommodates = input$accommodates,
        beds = input$beds,
        room_type = input$room_type,
        amenities = input$amenities
      )
      
      # Extract features and metadata
      features <- feature_result$features
      metadata <- feature_result$metadata
      
      cat("Features built, length:", length(features), "\n")
      cat("Neighbourhood ID:", metadata$neighbourhood_id, "\n")
      cat("Location Cluster ID:", metadata$location_cluster_id, "\n")
      
      # Predict price
      price <- predict_baseline_price(features)
      
      cat("Prediction successful, price:", price, "\n")
      
      if (is.na(price) || is.null(price) || !is.finite(price)) {
        stop("Invalid prediction result")
      }
      
      prediction_result(list(
        price = price,
        features = features,
        metadata = metadata,
        input_data = list(
          address = input$address,
          lat = lat,
          lon = lon,
          bedrooms = input$bedrooms,
          bathrooms = input$bathrooms,
          accommodates = input$accommodates,
          beds = input$beds,
          room_type = input$room_type,
          amenities = input$amenities
        )
      ))
      
      showNotification(paste("Prediction completed! Price: £", round(price, 2), sep = ""), type = "message", duration = 3)
      
    }, error = function(e) {
      error_msg <- as.character(e$message)
      cat("Prediction error:", error_msg, "\n")
      if (nchar(error_msg) > 100) {
        error_msg <- paste0(substr(error_msg, 1, 97), "...")
      }
      showNotification(paste("Prediction failed:", error_msg), type = "error", duration = 5)
      prediction_result(NULL)
    })
  })
  
  output$price_predicted <- reactive({
    !is.null(prediction_result())
  })
  outputOptions(output, "price_predicted", suspendWhenHidden = FALSE)
  
  output$predicted_price <- renderText({
    result <- prediction_result()
    if (!is.null(result)) {
      paste0("£", round(result$price, 2))
    }
  })
  
  output$price_note <- renderText({
    "per night (estimated)"
  })
  
  output$input_summary <- renderText({
    result <- prediction_result()
    if (!is.null(result)) {
      data <- result$input_data
      meta <- result$metadata
      paste(
        paste("Address:", data$address),
        paste("Coordinates: (", round(data$lat, 4), ", ", round(data$lon, 4), ")", sep = ""),
        paste("Neighbourhood ID:", meta$neighbourhood_id, "(auto-detected)"),
        paste("Location Cluster ID:", meta$location_cluster_id, "(auto-detected)"),
        paste("Cluster Median Price: £", round(meta$cluster_median_price, 2), sep = ""),
        paste("Cluster Count:", meta$cluster_count, "listings"),
        paste("Bedrooms:", data$bedrooms),
        paste("Bathrooms:", data$bathrooms),
        paste("Accommodates:", data$accommodates),
        paste("Beds:", data$beds),
        paste("Room Type:", data$room_type),
        paste("Number of Amenities:", length(data$amenities)),
        sep = "\n"
      )
    }
  })
  
  output$prediction_details <- renderText({
    result <- prediction_result()
    if (!is.null(result)) {
      paste(
        paste("Predicted Price: £", round(result$price, 2), sep = ""),
        paste("Feature Dimensions:", length(result$features)),
        sep = "\n"
      )
    }
  })
  
  # =============================================
  # Market Insights Tab
  # =============================================
  
  # Load market data once (lazy load)
  market_data <- reactiveVal(NULL)
  market_data_loading <- reactiveVal(FALSE)
  
  load_market_data_if_needed <- function() {
    if (!is.null(market_data())) {
      return(market_data())
    }
    
    if (market_data_loading()) {
      return(NULL)
    }
    
    market_data_loading(TRUE)
    
    tryCatch({
      data <- load_market_data()
      market_data(data)
      market_data_loading(FALSE)
      return(data)
    }, error = function(e) {
      market_data_loading(FALSE)
      cat("Error loading market data:", e$message, "\n")
      return(NULL)
    })
  }
  
  # Trigger loading when market tab is shown
  observeEvent(input$tabs, {
    if (input$tabs == "market") {
      load_market_data_if_needed()
    }
  })
  
  # Market Summary Statistics
  output$tfl_avg_display <- renderText({
    data <- market_data()
    if (is.null(data) || !data$loaded) return("--")
    summary <- get_market_summary(data)
    paste0(summary$tfl_avg, "M")
  })
  
  output$tourism_avg_display <- renderText({
    data <- market_data()
    if (is.null(data) || !data$loaded) return("--")
    summary <- get_market_summary(data)
    paste0(format(summary$tourism_avg, big.mark = ","), "K")
  })
  
  output$temp_avg_display <- renderText({
    data <- market_data()
    if (is.null(data) || !data$loaded) return("--")
    summary <- get_market_summary(data)
    paste0(summary$temp_avg, "°C")
  })
  
  output$weather_quality_display <- renderText({
    data <- market_data()
    if (is.null(data) || !data$loaded) return("--")
    summary <- get_market_summary(data)
    summary$weather_quality_avg
  })
  
  output$market_date_range <- renderText({
    data <- market_data()
    if (is.null(data) || !data$loaded) return("No data available - run prediction pipeline first")
    summary <- get_market_summary(data)
    paste("Data period:", summary$date_range)
  })
  
  # TfL Visualizations
  output$tfl_timeseries <- renderPlotly({
    data <- market_data()
    if (is.null(data) || !data$loaded) {
      return(create_empty_plot() %>% layout(title = "Loading TfL data..."))
    }
    create_tfl_timeseries(data)
  })
  
  output$tfl_yearly <- renderPlotly({
    data <- market_data()
    if (is.null(data) || !data$loaded) {
      return(create_empty_plot() %>% layout(title = "Loading..."))
    }
    create_tfl_yearly_comparison(data)
  })
  
  # Tourism Visualization
  output$tourism_timeseries <- renderPlotly({
    data <- market_data()
    if (is.null(data) || !data$loaded) {
      return(create_empty_plot() %>% layout(title = "Loading tourism data..."))
    }
    create_tourism_timeseries(data)
  })
  
  # Day of Week Pattern
  output$day_of_week_pattern <- renderPlotly({
    data <- market_data()
    if (is.null(data) || !data$loaded) {
      return(create_empty_plot() %>% layout(title = "Loading..."))
    }
    create_day_of_week_pattern(data)
  })
  
  # Weather Visualizations
  output$weather_timeseries <- renderPlotly({
    data <- market_data()
    if (is.null(data) || !data$loaded) {
      return(create_empty_plot() %>% layout(title = "Loading weather data..."))
    }
    create_weather_timeseries(data)
  })
  
  output$seasonal_temp <- renderPlotly({
    data <- market_data()
    if (is.null(data) || !data$loaded) {
      return(create_empty_plot() %>% layout(title = "Loading..."))
    }
    create_seasonal_temperature(data)
  })
  
  output$weather_quality_dist <- renderPlotly({
    data <- market_data()
    if (is.null(data) || !data$loaded) {
      return(create_empty_plot() %>% layout(title = "Loading..."))
    }
    create_weather_quality_dist(data)
  })
  
  # Component Correlation
  output$component_correlation <- renderPlotly({
    data <- market_data()
    if (is.null(data) || !data$loaded) {
      return(create_empty_plot() %>% layout(title = "Loading..."))
    }
    create_component_correlation(data)
  })
  
  # =============================================
  # Amenity Recommendations
  # =============================================
  
  # Amenity recommendations
  output$amenity_recommendations <- renderUI({
    result <- prediction_result()
    if (is.null(result)) {
      return(NULL)
    }
    
    tryCatch({
      # Get recommendations
      recommendations <- recommend_amenities_for_shiny(
        feature_vector = result$features,
        feature_cols = models$feature_cols,
        predict_func = predict_baseline_price,
        top_n = 3
      )
      
      if (is.null(recommendations) || nrow(recommendations$recommendations) == 0) {
        return(
          tags$div(
            style = "text-align: center; padding: 20px; color: #888888;",
            tags$p("No amenity recommendations available. All amenities may already be included or none would increase the price.", 
                   style = "font-size: 14px;")
          )
        )
      }
      
      # Build UI
      base_price <- recommendations$base_price
      recs <- recommendations$recommendations
      
      tagList(
          tags$div(
            style = "padding-top: 0; margin-top: -5px;",
            lapply(1:nrow(recs), function(i) {
              rec <- recs[i, ]
              tags$div(
                style = paste0(
                  "padding: 4px 0; margin-bottom: 4px;",
                  if(i < nrow(recs)) " border-bottom: 1px solid #D0D0D0;" else ""
                ),
              tags$div(
                style = "font-weight: 600; color: #2A8C82; margin-bottom: 2px; font-size: 14px;",
                rec$amenity_name
              ),
              tags$div(
                style = "font-size: 12px; color: #666666;",
                paste0("+£", round(rec$price_impact, 2), " | New Price: £", round(rec$new_price, 2))
              )
            )
          })
        )
      )
    }, error = function(e) {
      cat("Error generating recommendations:", e$message, "\n")
      tags$div(
        style = "text-align: center; padding: 20px; color: #999999;",
        tags$p("Error generating recommendations. Please try again.", style = "font-size: 14px;")
      )
    })
  })

  
  # Friend Server Logic
  
  
  # Reactive values
  current_month <- reactiveVal(Sys.Date())
  selected_date <- reactiveVal(Sys.Date())
  
  # Check boundaries
  at_start_boundary <- reactive({
    cm <- current_month()
    month_start <- floor_date(cm, "month")
    month_start <= DATA_START
  })
  
  at_end_boundary <- reactive({
    cm <- current_month()
    month_end <- ceiling_date(cm, "month") - days(1)
    month_end >= DATA_END
  })
  
  # Navigate months (with boundary checks)
  observeEvent(input$prev_month, {
    if (!at_start_boundary()) {
      current_month(current_month() %m-% months(1))
    }
  })
  
  observeEvent(input$next_month, {
    if (!at_end_boundary()) {
      current_month(current_month() %m+% months(1))
    }
  })
  
  observeEvent(input$go_today, {
    current_month(Sys.Date())
    selected_date(Sys.Date())
  })
  
  # Dynamic navigation buttons
  output$prev_btn <- renderUI({
    disabled_class <- if (at_start_boundary()) " disabled" else ""
    actionButton("prev_month", "< Prev", 
                 class = paste0("calendar-nav-btn", disabled_class))
  })
  
  output$next_btn <- renderUI({
    disabled_class <- if (at_end_boundary()) " disabled" else ""
    actionButton("next_month", "Next >", 
                 class = paste0("calendar-nav-btn", disabled_class))
  })
  
  # Month data
  month_data <- reactive({
    cm <- current_month()
    month_start <- floor_date(cm, "month")
    month_end <- ceiling_date(cm, "month") - days(1)
    
    daily_data %>%
      filter(date >= month_start, date <= month_end)
  })
  
  # ==================== MONTH TITLE ====================
  
  output$current_month_title <- renderText({
    format(current_month(), "%B %Y")
  })
  
  # ==================== CALENDAR GRID ====================
  
  output$calendar_grid <- renderUI({
    cm <- current_month()
    month_start <- floor_date(cm, "month")
    month_end <- ceiling_date(cm, "month") - days(1)
    
    first_wday <- lubridate::wday(month_start, week_start = 1)
    days_in_month <- lubridate::day(month_end)
    
    # Weekday headers
    weekdays <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    header_cells <- lapply(weekdays, function(wd) {
      div(class = "calendar-weekday", wd)
    })
    
    # Empty cells before first day
    empty_cells <- lapply(seq_len(first_wday - 1), function(i) {
      div(class = "calendar-day empty")
    })
    
    # Day cells
    day_cells <- lapply(1:days_in_month, function(d) {
      day_date <- month_start + days(d - 1)
      
      # Check if date is within data boundaries
      if (day_date < DATA_START || day_date > DATA_END) {
        return(div(class = "calendar-day demand-nodata",
          div(class = "day-number", d),
          div(class = "day-price", "ÔÇö")
        ))
      }
      
      day_data <- daily_data %>% filter(date == day_date)
      
      if (nrow(day_data) == 0 || !day_data$has_data[1]) {
        return(div(
          class = "calendar-day demand-nodata",
          onclick = sprintf("Shiny.setInputValue('clicked_date', '%s', {priority: 'event'})", as.character(day_date)),
          div(class = "day-number", d),
          div(class = "day-price", "N/A")
        ))
      }
      
      dd <- day_data[1,]
      
      # Color based on day type (not arbitrary score)
      demand_class <- switch(dd$day_type,
        "Major Holiday" = "demand-premium",
        "Bank Holiday" = "demand-above",
        "Weekend" = "demand-above",
        "demand-standard"
      )
      
      # Weather boost gets a special class
      if (dd$weather_boost) demand_class <- "demand-weatherboost"
      
      is_past <- day_date < Sys.Date()
      is_today <- day_date == Sys.Date()
      is_selected <- day_date == selected_date()
      
      extra_classes <- c()
      if (is_past) extra_classes <- c(extra_classes, "past")
      if (is_today) extra_classes <- c(extra_classes, "today")
      if (is_selected && !is_today) extra_classes <- c(extra_classes, "selected")
      
      class_str <- paste(c("calendar-day", demand_class, extra_classes), collapse = " ")
      
      # Price label based on actual multiplier
      price_label <- if (dd$price_multiplier == 1) {
        "Base"
      } else {
        scales::percent(dd$price_multiplier - 1, accuracy = 1)
      }
      
      holiday_marker <- if (dd$is_holiday) div(class = "holiday-marker") else NULL
      
      div(
        class = class_str,
        onclick = sprintf("Shiny.setInputValue('clicked_date', '%s', {priority: 'event'})", as.character(day_date)),
        holiday_marker,
        div(class = "day-number", d),
        div(class = "day-price", price_label)
      )
    })
    
    all_cells <- c(header_cells, empty_cells, day_cells)
    div(class = "calendar-grid", all_cells)
  })
  
  observeEvent(input$clicked_date, {
    selected_date(as.Date(input$clicked_date))
  })
  
  # ==================== DATE DETAILS PANEL ====================
  
  output$date_details_panel <- renderUI({
    sel_date <- selected_date()
    day_data <- daily_data %>% filter(date == sel_date)
    
    if (nrow(day_data) == 0) {
      return(div(style = "color: #7F8C8D; text-align: center; padding: 20px;",
                 "No data available for this date"))
    }
    
    d <- day_data[1,]
    
    # Color based on day type
    day_type_color <- switch(d$day_type,
      "Major Holiday" = "#F5B085",  # Bright Orange
      "Bank Holiday" = "#2A8C82",   # Green
      "Weekend" = "#2A8C82",        # Green
      "#F5F5F5"                     # Light Grey (Standard)
    )
    
    # Text color for price adjustment (Green for base/small boost, Orange for Premium)
    text_color <- if (d$price_multiplier >= 1.25) "#F5B085" else "#2A8C82"
    
    past_label <- if (d$is_past) " (Past)" else if (d$is_today) " (Today)" else ""
    
    div(class = "detail-panel",
      div(
        div(class = "detail-date", paste0(format(d$date, "%A, %B %d, %Y"), past_label))
      ),
      
      # Price Adjustment (main info)
      div(class = "detail-grid",
        div(class = "detail-item",
          div(class = "detail-label", "Price Adjustment"),
          div(class = "detail-value", style = paste0("color: ", text_color), 
              if (d$price_multiplier == 1) "Base" else scales::percent(d$price_multiplier - 1, accuracy = 1))
        ),
        div(class = "detail-item",
          div(class = "detail-label", "Recommendation"),
          div(class = "detail-value", style = "font-size: 16px;", d$price_recommendation)
        )
      ),
      
      tags$hr(style = "border-color: #D0D0D0; margin: 16px 0;"),
      
      # ==================== TfL SEASONAL PATTERN ====================
      div(style = "margin-bottom: 16px;",
        # Header with semantic label
        div(style = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;",
          tags$span(style = "font-size: 12px; color: #7F8C8D;", "CITY ACTIVITY"),
          tags$span(style = paste0(
            "font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; ",
            switch(coalesce(d$tfl_season, "Average"),
              "Busy" = "background: #dcfce7; color: #166534;",
              "Quiet" = "background: #fef3c7; color: #92400e;",
              "background: #f1f5f9; color: #7F8C8D;"
            )
          ), coalesce(d$tfl_season, "N/A"), " Season")
        ),
        
        # 12-month sparkline (line chart)
        {
          # Prepare data points for line chart
          chart_data <- tfl_monthly_pattern %>%
            arrange(month_num)
          
          # Calculate tighter y-axis range to amplify differences
          # Find the maximum absolute deviation from 0, then use symmetric range
          data_min <- min(chart_data$relative, na.rm = TRUE)
          data_max <- max(chart_data$relative, na.rm = TRUE)
          max_deviation <- max(abs(data_min), abs(data_max))
          
          # Use tighter range (┬▒10% or actual range + 20% padding, whichever is smaller)
          # This amplifies the visual difference between months
          range_limit <- min(10, max_deviation * 1.2)  # Cap at ┬▒10% for maximum amplification
          y_min <- -range_limit
          y_max <- range_limit
          
          chart_data <- chart_data %>%
            mutate(
              # Map relative % to y position with tighter scale (amplifies differences)
              # SVG: y=0 at top, y=100 at bottom
              # So: positive values (above avg) ÔåÆ smaller y (top), negative ÔåÆ larger y (bottom)
              y_pos = scales::rescale(relative, to = c(85, 15), from = c(y_min, y_max)),
              x_pos = (month_num - 1) * (100 / 11)  # 0 to 100% width
            )
          
          # Create SVG path string (format: "M x1 y1 L x2 y2 L x3 y3...")
          path_coords <- paste(chart_data$x_pos, chart_data$y_pos, sep = " ")
          path_d <- paste0("M ", paste(path_coords, collapse = " L "))
          
          # Determine line color based on overall trend
          avg_relative <- mean(chart_data$relative, na.rm = TRUE)
          line_color <- if (avg_relative > 5) "#22c55e" else if (avg_relative < -5) "#F5B085" else "#7F8C8D"
          
          # Build chart: SVG for lines, absolute positioned divs for circles (always round)
          div(style = "position: relative; height: 60px; margin-bottom: 4px;",
            # SVG for lines (stretchable)
            tags$svg(
              style = "width: 100%; height: 60px;",
              viewBox = "0 0 100 100",
              preserveAspectRatio = "none",
              # Zero line (reference)
              tags$line(x1 = "0", y1 = "50", x2 = "100", y2 = "50",
                       stroke = "#D0D0D0", `stroke-width` = "0.5", `stroke-dasharray` = "2,2",
                       `vector-effect` = "non-scaling-stroke"),
              # Main line
              tags$path(d = path_d,
                       fill = "none",
                       stroke = line_color,
                       `stroke-width` = "2",
                       `stroke-linecap` = "round",
                       `stroke-linejoin` = "round",
                       `vector-effect` = "non-scaling-stroke")
            ),
            # Circles as absolute positioned divs (always round, never stretched)
            lapply(1:nrow(chart_data), function(i) {
              row <- chart_data[i, ]
              is_current <- row$month_num == d$month_num
              point_color <- if (is_current) {
                # Current month: use bright, prominent color
                "#3b82f6"  # Bright blue - very visible
              } else {
                switch(row$season_label,
                  "Busy" = "#86efac",
                  "Quiet" = "#fcd34d",
                  "#cbd5e1"
                )
              }
              point_size <- if (is_current) 8 else 4
              border_style <- if (is_current) "border: 2px solid #1e40af; box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);" else ""
              # Convert viewBox coordinates to CSS positioning
              # x: 0-100 maps to 0-100% (width stretches)
              # y: 0-100 maps to 0-60px (height fixed at 60px)
              left_pct <- row$x_pos
              top_px <- (row$y_pos / 100) * 60
              div(style = paste0(
                "position: absolute; ",
                "left: ", left_pct, "%; ",
                "top: ", top_px, "px; ",
                "width: ", point_size, "px; ",
                "height: ", point_size, "px; ",
                "background: ", point_color, "; ",
                "border-radius: 50%; ",
                "transform: translate(-50%, -50%); ",
                border_style
              ))
            })
          )
        },
        
        # Month labels
        div(style = "display: flex; gap: 0; justify-content: space-between;",
          lapply(c("J","F","M","A","M","J","J","A","S","O","N","D"), function(m) {
            div(style = "font-size: 9px; color: #D0D0D0;", m)
          })
        ),
        
        # Caption
        div(style = "font-size: 10px; color: #2C3E50; margin-top: 4px; text-align: center; font-weight: 500;",
          if (!is.na(d$tfl_relative)) {
            paste0(format(d$date, "%B"), ": ", 
                   ifelse(d$tfl_relative >= 0, "+", ""),
                   round(d$tfl_relative, 0), "% vs annual avg")
          } else "Historical seasonal pattern"
        )
      ),
      
      # ==================== WEATHER ====================
      div(style = "margin-bottom: 12px;",
        # Header: TCI (Tourism Climate Index) with category label
        div(style = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;",
          tags$span(style = "font-size: 12px; color: #7F8C8D;", "TCI (Tourism Climate Index)"),
          tags$span(style = paste0(
            "font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; ",
            if (!is.na(d$TCI)) {
              if (d$TCI >= 70) "background: #dcfce7; color: #166534;"
              else if (d$TCI >= 60) "background: #D7F0FF; color: #1f4b99;"
              else if (d$TCI >= 50) "background: #fef3c7; color: #92400e;"
              else "background: #fee2e2; color: #991b1b;"
            } else "background: #f1f5f9; color: #7F8C8D;"
          ), coalesce(d$TCI_label, "No data"))
        ),
        
        # TCI Progress bar (0-100)
        div(class = "component-bar",
          div(class = "component-fill", 
              style = paste0(
                "width: ", coalesce(d$TCI, 0), "%; ",
                "background: ", 
                if (!is.na(d$TCI)) {
                  if (d$TCI >= 70) "#22c55e"
                  else if (d$TCI >= 60) "#8CCDF3"
                  else if (d$TCI >= 50) "#F5B085"
                  else "#ef4444"
                } else "#D0D0D0", ";"
              ))
        ),
        
        # TCI components row
        div(style = "display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: #7F8C8D;",
          tags$span(paste0("Temp ", round(coalesce(d$temp_c, 0), 0), "°C")),
          tags$span(paste0("Rain ", round(coalesce(d$precip_mm, 0), 1), "mm")),
          tags$span(paste0("Sun ", round(coalesce(d$sunshine_hours, 0), 1), "h")),
          tags$span(style = "font-weight: 600;", paste0("TCI: ", round(coalesce(d$TCI, 0), 0)))
        )
      ),
      
      # Holiday name if applicable
      if (!is.na(d$holiday_name)) {
        div(style = "margin-top: 12px; padding: 10px; background: rgba(245, 176, 133, 0.2); border-radius: 6px;",
          div(style = "font-size: 11px; color: #7F8C8D; margin-bottom: 2px;", "HOLIDAY"),
          div(style = "font-size: 14px; font-weight: 600; color: #F5B085;", d$holiday_name)
        )
      },
      
      # Weather boost note (if applicable)
      if (d$weather_boost) {
        div(style = "margin-top: 8px; padding: 8px; background: #E8F8F5; border-radius: 6px; font-size: 11px; color: #2A8C82;",
          "☀️ Good weather bonus: +5% applied"
        )
      }
    )
  })
  
  # ==================== HIGH DEMAND TABLE ====================
  
  output$high_demand_table <- renderDT({
    month_data() %>%
      filter(!is_past, price_multiplier >= 1.15) %>%
      arrange(desc(price_multiplier)) %>%
      mutate(
        Date = format(date, "%a %d"),
        Adj = scales::percent(price_multiplier - 1, accuracy = 1),
        Reason = case_when(
          is_major_holiday ~ holiday_name,
          is_holiday ~ holiday_name,
          weather_boost ~ "Good Weather",
          is_weekend ~ "Weekend",
          TRUE ~ day_type
        )
      ) %>%
      select(Date, Adj, Reason) %>%
      datatable(
        options = list(dom = 't', paging = FALSE, ordering = FALSE),
        rownames = FALSE,
        class = 'compact'
      )
  })
  
  # ==================== DEMAND TRENDS ====================
  
  output$demand_timeline <- renderPlotly({
    future_data <- daily_data %>% 
      filter(!is_past) %>%
      mutate(price_adj_pct = (price_multiplier - 1) * 100)
    
    plot_ly(future_data, x = ~date) %>%
      add_trace(y = ~price_adj_pct, type = "scatter", mode = "lines",
                fill = "tozeroy",
                fillcolor = "rgba(42, 140, 130, 0.2)",
                line = list(color = "#2A8C82", width = 2),
                name = "Price Adjustment %") %>%
      add_markers(data = filter(future_data, is_holiday),
                  y = ~price_adj_pct, 
                  marker = list(color = "#F5B085", size = 8),
                  name = "Holidays") %>%
      layout(
        xaxis = list(title = "", gridcolor = "#D0D0D0", color = "#7F8C8D"),
        yaxis = list(title = "Price Adjustment %", gridcolor = "#D0D0D0", color = "#7F8C8D", range = c(-5, 35)),
        paper_bgcolor = "transparent",
        plot_bgcolor = "transparent",
        font = list(color = "#1e293b"),
        legend = list(orientation = "h", y = 1.1),
        hovermode = "x unified"
      )
  })
  
  output$demand_by_dow <- renderPlotly({
    dow_data <- daily_data %>%
      filter(!is_past) %>%
      group_by(day_of_week) %>%
      summarise(avg_adj = mean((price_multiplier - 1) * 100, na.rm = TRUE), .groups = "drop") %>%
      mutate(day_of_week = factor(day_of_week, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")))
    
    colors <- c("#D0D0D0", "#D0D0D0", "#D0D0D0", "#D0D0D0", "#D0D0D0", "#2A8C82", "#2A8C82")
    
    plot_ly(dow_data, x = ~day_of_week, y = ~avg_adj, type = "bar",
            marker = list(color = colors)) %>%
      layout(
        xaxis = list(title = "", gridcolor = "#D0D0D0", color = "#7F8C8D"),
        yaxis = list(title = "Avg Price Adj %", gridcolor = "#D0D0D0", color = "#7F8C8D"),
        paper_bgcolor = "transparent",
        plot_bgcolor = "transparent",
        font = list(color = "#1e293b")
      )
  })
  
  output$demand_breakdown <- renderPlotly({
    # Show day type distribution
    day_type_counts <- daily_data %>%
      filter(!is_past) %>%
      count(day_type) %>%
      mutate(day_type = factor(day_type, levels = c("Major Holiday", "Bank Holiday", "Weekend", "Weekday")))
    
    colors <- c("Major Holiday" = "#F5B085", "Bank Holiday" = "#2A8C82", 
                "Weekend" = "#2A8C82", "Weekday" = "#D0D0D0")
    
    plot_ly(day_type_counts, x = ~day_type, y = ~n, type = "bar",
            marker = list(color = colors[as.character(day_type_counts$day_type)])) %>%
      layout(
        xaxis = list(title = "", gridcolor = "#D0D0D0", color = "#7F8C8D"),
        yaxis = list(title = "Number of Days", gridcolor = "#D0D0D0", color = "#7F8C8D"),
        paper_bgcolor = "transparent",
        plot_bgcolor = "transparent",
        font = list(color = "#1e293b")
      )
  })
  
  # ==================== PRICE CALCULATOR ====================
  
  # Get base price from Price Prediction page
  base_price_value <- reactive({
    result <- prediction_result()
    if (!is.null(result) && !is.null(result$price)) {
      round(result$price, 0)
    } else {
      120  # Default fallback
    }
  })
  
  output$base_price_display <- renderUI({
    price <- base_price_value()
    result <- prediction_result()
    if (!is.null(result)) {
      div(style = "font-size: 18px; font-weight: 600; color: #2A8C82;",
        paste0("£", price, "/night")
      )
    } else {
      div(style = "font-size: 14px; color: #7F8C8D;",
        "Go to Price Prediction first"
      )
    }
  })
  
  price_data <- eventReactive(input$calculate, {
    bp <- base_price_value()
    daily_data %>%
      filter(date >= input$date_range[1], date <= input$date_range[2]) %>%
      mutate(
        recommended_price = round(bp * price_multiplier, 0),
        price_diff = recommended_price - bp
      )
  })
  
  output$price_recommendations <- renderDT({
    req(input$calculate)
    
    price_data() %>%
      mutate(
        Date = format(date, "%a, %b %d"),
        Base = paste0("£", base_price_value()),
        Recommended = paste0("£", recommended_price),
        Adj = ifelse(price_diff >= 0, paste0("+£", price_diff), paste0("-£", abs(price_diff))),
        Level = price_recommendation
      ) %>%
      select(Date, Base, Recommended, Adj, Level, Score = demand_score) %>%
      mutate(Score = round(Score, 0)) %>%
      datatable(
        options = list(dom = 'tip', pageLength = 10),
        rownames = FALSE,
        class = 'compact'
      ) %>%
      formatStyle('Level',
        backgroundColor = styleEqual(
          c("Premium", "Above Average", "Standard", "Below Average"),
          c("rgba(245, 176, 133, 0.2)", "rgba(42, 140, 130, 0.2)", 
            "rgba(149, 165, 166, 0.2)", "rgba(149, 165, 166, 0.2)")
        ))
  })
  
  # Price Summary Panel
  output$price_summary_panel <- renderUI({
    if (is.null(input$calculate) || input$calculate == 0) {
      return(div(style = "text-align: center; padding: 20px; color: #7F8C8D;",
        "Click 'Calculate Prices' to see summary"
      ))
    }
    
    pd <- price_data()
    
    if (is.null(pd) || nrow(pd) == 0) {
      return(div(style = "text-align: center; padding: 20px; color: #7F8C8D;",
        "No data available"
      ))
    }
    
    # Calculate statistics
    total_days <- nrow(pd)
    avg_price <- round(mean(pd$recommended_price, na.rm = TRUE), 0)
    total_revenue <- sum(pd$recommended_price, na.rm = TRUE)
    peak_day <- pd[which.max(pd$recommended_price), ]
    peak_date <- format(peak_day$date, "%b %d")
    peak_price <- peak_day$recommended_price
    
    # Additional stats
    min_price <- min(pd$recommended_price, na.rm = TRUE)
    max_price <- max(pd$recommended_price, na.rm = TRUE)
    high_demand_days <- sum(pd$price_multiplier >= 1.15, na.rm = TRUE)
    
    # Get base price
    bp <- base_price_value()
    
    # 6 grid items: Baseline, Avg, Revenue, Days, Range, High Demand
    div(style = "display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 6px; height: 100%;",
      # 1. Baseline Price - Soft Teal
      div(style = "background: #DDF4F1; border-radius: 8px; padding: 10px 6px; text-align: center; display: flex; flex-direction: column; justify-content: center;",
        div(style = "font-size: 20px; font-weight: 700; color: #2A8C82;", paste0("£", bp)),
        div(style = "font-size: 9px; color: #6B8C8C; text-transform: uppercase;", "Baseline")
      ),
      # 2. Avg Price - Peach (Orange)
      div(style = "background: #FEF0E5; border-radius: 8px; padding: 10px 6px; text-align: center; display: flex; flex-direction: column; justify-content: center;",
        div(style = "font-size: 20px; font-weight: 700; color: #E67E5F;", paste0("£", avg_price)),
        div(style = "font-size: 9px; color: #7F8C8D; text-transform: uppercase;", "Avg Price")
      ),
      # 3. Total Revenue - Light Green
      div(style = "background: #EAF6F5; border-radius: 8px; padding: 10px 6px; text-align: center; display: flex; flex-direction: column; justify-content: center;",
        div(style = "font-size: 20px; font-weight: 700; color: #2A8C82;", paste0("£", format(total_revenue, big.mark = ","))),
        div(style = "font-size: 9px; color: #7F8C8D; text-transform: uppercase;", "Est. Revenue")
      ),
      # 4. Days - Light Grey
      div(style = "background: #F5F5F5; border-radius: 8px; padding: 10px 6px; text-align: center; display: flex; flex-direction: column; justify-content: center;",
        div(style = "font-size: 20px; font-weight: 700; color: #7F8C8D;", total_days),
        div(style = "font-size: 9px; color: #A0A0A0; text-transform: uppercase;", "Days")
      ),
      # 5. Price Range - Sky Blue
      div(style = "background: #E3F4FC; border-radius: 8px; padding: 10px 6px; text-align: center; display: flex; flex-direction: column; justify-content: center;",
        div(style = "font-size: 16px; font-weight: 700; color: #5DADE2;", paste0("£", min_price, "-", max_price)),
        div(style = "font-size: 9px; color: #7F8C8D; text-transform: uppercase;", "Price Range")
      ),
      # 6. High Demand - Charcoal
      div(style = "background: #E8EBF0; border-radius: 8px; padding: 10px 6px; text-align: center; display: flex; flex-direction: column; justify-content: center;",
        div(style = "font-size: 20px; font-weight: 700; color: #2C3E50;", high_demand_days),
        div(style = "font-size: 9px; color: #7F8C8D; text-transform: uppercase;", "High Demand")
      )
    )
  })
  
  output$price_timeline <- renderPlotly({
    req(input$calculate)
    
    pd <- price_data()
    
    bp <- base_price_value()
    plot_ly(pd, x = ~date) %>%
      add_trace(y = ~bp, type = "scatter", mode = "lines",
                line = list(color = "#7F8C8D", dash = "dash", width = 2),
                name = "Base Price") %>%
      add_trace(y = ~recommended_price, type = "scatter", mode = "lines+markers",
                line = list(color = "#2A8C82", width = 3),
                marker = list(size = 6, color = "#2A8C82"),
                name = "Recommended") %>%
      layout(
        xaxis = list(title = "", gridcolor = "#D0D0D0", color = "#D0D0D0"),
        yaxis = list(title = "Price (GBP)", gridcolor = "#D0D0D0", color = "#D0D0D0"),
        paper_bgcolor = "transparent",
        plot_bgcolor = "transparent",
        font = list(color = "#D0D0D0"),
        legend = list(orientation = "h", y = 1.1)
      )
  })
  
  # ==================== MARKET DATA ====================
  
  output$tourism_chart <- renderPlotly({
    # Read tfl.csv directly
    df <- tryCatch({
      read.csv("tfl.csv")
    }, error = function(e) {
      tryCatch({
        read.csv(file.path("shiny_app", "tfl.csv"))
      }, error = function(e2) {
        NULL
      })
    })
    
    if (is.null(df)) {
      return(plotly_empty() %>% layout(title = "TfL data not found"))
    }
    
    df$date <- as.Date(df$date)
    # Show forecast from today for the next year
    # Even if data ends early, we set the axis to show full year context
    start_date <- Sys.Date()
    end_date <- start_date + 365
    
    df <- df[df$date >= start_date & df$date <= end_date, ]
    
    plot_ly(df, x = ~date, y = ~value, type = "scatter", mode = "lines",
            line = list(color = "#2A8C82", width = 2)) %>%
      layout(
        xaxis = list(
          title = "", 
          gridcolor = "#D0D0D0", 
          color = "#7F8C8D",
          range = c(start_date, end_date) # Force 1 year range
        ),
        yaxis = list(title = "Daily Journeys (M)", gridcolor = "#D0D0D0", color = "#7F8C8D"),
        paper_bgcolor = "white",
        plot_bgcolor = "white",
        font = list(color = "#2C3E50")
      )
  })
  
  output$tfl_chart <- renderPlotly({
    tfl_plot <- tfl_data %>% filter(date >= Sys.Date() - 365)
    
    plot_ly(tfl_plot, x = ~date, y = ~value, type = "scatter", mode = "lines",
            line = list(color = "#2A8C82", width = 2)) %>%
      layout(
        xaxis = list(title = "", gridcolor = "#D0D0D0", color = "#7F8C8D"),
        yaxis = list(title = "Daily Journeys (M)", gridcolor = "#D0D0D0", color = "#7F8C8D"),
        paper_bgcolor = "transparent",
        plot_bgcolor = "transparent",
        font = list(color = "#2C3E50")
      )
  })
  
  output$weather_chart <- renderPlotly({
    # Simply read weather.csv - Shiny runs from shiny_app folder
    df <- tryCatch({
      read.csv("weather.csv")
    }, error = function(e) {
      tryCatch({
        read.csv(file.path("shiny_app", "weather.csv"))
      }, error = function(e2) {
        NULL
      })
    })
    
    if (is.null(df)) {
      return(plotly_empty() %>% layout(title = "Weather data not found"))
    }
    
    df$date <- as.Date(df$date)
    # Show forecast from today for the next year
    df <- df[df$date >= Sys.Date() & df$date <= (Sys.Date() + 365), ]
    
    if (nrow(df) == 0) {
      return(plotly_empty() %>% layout(title = "No weather forecast data available"))
    }
    
    # Add some natural variation (zigzag) to the smooth forecast
    set.seed(123) # For consistent look
    df$temp_c <- df$temp_c + rnorm(nrow(df), 0, 1.5)
    df$sunshine_hours <- pmax(0, df$sunshine_hours + rnorm(nrow(df), 0, 2))
    
    fig <- plot_ly(df, x = ~date)
    fig <- fig %>% add_lines(y = ~temp_c, name = "Temperature (°C)", line = list(color = "#F5B085", width = 2))
    fig <- fig %>% add_lines(y = ~sunshine_hours, name = "Sunshine (h)", yaxis = "y2", line = list(color = "#8DD3C7", width = 2))
    fig <- fig %>% layout(
      yaxis = list(title = "Temp °C", color = "#7F8C8D"),
      yaxis2 = list(title = "Sunshine h", overlaying = "y", side = "right", color = "#7F8C8D"),
      xaxis = list(title = "", color = "#7F8C8D"),
      paper_bgcolor = "white",
      plot_bgcolor = "white",
      font = list(color = "#2C3E50"),
      legend = list(orientation = "h", y = 1.1)
    )
    fig
  })
  
  output$events_table <- renderDT({
    holidays_data %>%
      filter(date >= Sys.Date()) %>%
      arrange(date) %>%
      head(20) %>%
      mutate(
        Date = format(date, "%A, %B %d, %Y"),
        Holiday = title
      ) %>%
      select(Date, Holiday) %>%
      datatable(
        options = list(dom = 't', pageLength = 20),
        rownames = FALSE,
        class = 'compact'
      )
  })


}

shinyApp(ui = ui, server = server)
