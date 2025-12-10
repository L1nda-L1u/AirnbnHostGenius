# ==================================================================================
# Market Indicators Visualization Helper
# ==================================================================================
# Provides TfL, Tourism, and Weather visualizations for the Shiny app
# Data sourced from the tfl,weather,tourism,holiday_prediction module

library(data.table)
library(lubridate)
library(dplyr)
library(plotly)
library(scales)

# Helper function to create a styled empty plot (avoids plotly warnings)
create_empty_plot <- function(message = "No data available") {
  plot_ly() %>%
    add_annotations(
      x = 0.5, y = 0.5,
      text = message,
      showarrow = FALSE,
      font = list(size = 14, color = "#888888")
    ) %>%
    layout(
      xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE, title = ""),
      yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE, title = ""),
      plot_bgcolor = "#F8F8F8",
      paper_bgcolor = "#F8F8F8"
    )
}

# Path to component data folder - try multiple possible locations
find_component_data_path <- function() {
  possible_paths <- c(
    # From shiny_app directory
    normalizePath(file.path(getwd(), "..", "..", "tfl,weather,tourism,holiday_prediction"), mustWork = FALSE),
    # From baseline_price_predict directory
    normalizePath(file.path(getwd(), "..", "tfl,weather,tourism,holiday_prediction"), mustWork = FALSE),
    # From project root
    normalizePath(file.path(getwd(), "tfl,weather,tourism,holiday_prediction"), mustWork = FALSE)
  )
  
  for (path in possible_paths) {
    if (dir.exists(path)) {
      message("Found component data path: ", path)
      return(path)
    }
  }
  
  message("Component data path not found. Please ensure the tfl,weather,tourism,holiday_prediction folder is in the project.")
  return(possible_paths[1])
}

COMPONENT_DATA_PATH <- find_component_data_path()

# ==================================================================================
# DATA LOADING FUNCTIONS
# ==================================================================================

#' Load Component Predictions Data
#' @return A list with tfl, weather data frames
load_market_data <- function() {
  result <- list(
    tfl = NULL,
    weather = NULL,
    daily = NULL,
    loaded = FALSE
  )
  
  # Try to load component predictions lookup
  lookup_path <- file.path(COMPONENT_DATA_PATH, "shiny_export", "component_predictions_lookup.rds")
  if (file.exists(lookup_path)) {
    result$daily <- readRDS(lookup_path)
    result$loaded <- TRUE
    message("✓ Loaded component predictions from: ", lookup_path)
    return(result)
  }
  
  # Alternative: Load from cleaned daily data
  daily_path <- file.path(COMPONENT_DATA_PATH, "foot_traffic_data", "cleaned", "foot_traffic_daily.csv")
  if (file.exists(daily_path)) {
    result$daily <- fread(daily_path) %>%
      mutate(
        date = as.Date(date),
        day_of_week = factor(day_of_week, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")),
        month_name = lubridate::month(date, label = TRUE, abbr = TRUE),
        season = case_when(
          month %in% c(12, 1, 2) ~ "Winter",
          month %in% c(3, 4, 5) ~ "Spring",
          month %in% c(6, 7, 8) ~ "Summer",
          month %in% c(9, 10, 11) ~ "Autumn"
        ),
        season = factor(season, levels = c("Spring", "Summer", "Autumn", "Winter"))
      ) %>%
      filter(!is.na(date))
    result$loaded <- TRUE
    message("✓ Loaded daily data from: ", daily_path)
    return(result)
  }
  
  message("⚠ No component data found. Please run the prediction pipeline first.")
  return(result)
}

#' Get component values for a specific date
#' @param date The date to query
#' @param data The loaded market data
#' @return A list with tfl, weather values
get_components_for_date <- function(date, data) {
  if (is.null(data$daily) || !data$loaded) {
    return(list(
      tfl = NA,
      temp = NA,
      weather_quality = NA,
      is_holiday = FALSE,
      available = FALSE
    ))
  }
  
  date <- as.Date(date)
  
  result <- data$daily %>%
    filter(date == !!date)
  
  if (nrow(result) == 0) {
    # Date not in data - return averages
    return(list(
      tfl = mean(data$daily$tfl_daily_avg_m, na.rm = TRUE),
      temp = mean(data$daily$temp_c, na.rm = TRUE),
      weather_quality = mean(data$daily$weather_quality, na.rm = TRUE),
      is_holiday = FALSE,
      available = FALSE
    ))
  }
  
  return(list(
    tfl = result$tfl_daily_avg_m[1],
    temp = result$temp_c[1],
    weather_quality = result$weather_quality[1],
    is_holiday = ifelse(!is.null(result$is_holiday), result$is_holiday[1], FALSE),
    available = TRUE
  ))
}

# ==================================================================================
# VISUALIZATION FUNCTIONS
# ==================================================================================

#' Create TfL Time Series Plot
#' @param data The loaded market data
#' @return A plotly object
create_tfl_timeseries <- function(data) {
  if (is.null(data$daily) || !data$loaded) {
    return(create_empty_plot() %>% layout(title = "TfL Data Not Available"))
  }
  
  tfl_data <- data$daily %>%
    filter(!is.na(tfl_daily_avg_m), is.finite(tfl_daily_avg_m)) %>%
    arrange(date)
  
  if (nrow(tfl_data) == 0) {
    return(create_empty_plot() %>% layout(title = "No TfL Data"))
  }
  
  # Calculate 30-day rolling average
  tfl_data <- tfl_data %>%
    mutate(tfl_rolling = zoo::rollmean(tfl_daily_avg_m, k = 30, fill = NA, align = "right"))
  
  plot_ly(tfl_data) %>%
    add_lines(
      x = ~date, y = ~tfl_daily_avg_m,
      name = "Daily",
      line = list(color = "#3498DB", width = 1),
      opacity = 0.4
    ) %>%
    add_lines(
      x = ~date, y = ~tfl_rolling,
      name = "30-Day Avg",
      line = list(color = "#E74C3C", width = 2)
    ) %>%
    layout(
      title = list(text = "TfL Daily Journeys (Millions)", font = list(size = 14)),
      xaxis = list(title = "", tickformat = "%Y-%m"),
      yaxis = list(title = "Journeys (M)"),
      legend = list(orientation = "h", y = -0.15),
      hovermode = "x unified",
      margin = list(t = 40, b = 50)
    )
}

#' Create TfL by Year Comparison
#' @param data The loaded market data
#' @return A plotly object
create_tfl_yearly_comparison <- function(data) {
  if (is.null(data$daily) || !data$loaded) {
    return(create_empty_plot() %>% layout(title = "TfL Data Not Available"))
  }
  
  tfl_yearly <- data$daily %>%
    filter(!is.na(tfl_daily_avg_m)) %>%
    group_by(year) %>%
    summarise(
      mean = mean(tfl_daily_avg_m, na.rm = TRUE),
      median = median(tfl_daily_avg_m, na.rm = TRUE),
      min = min(tfl_daily_avg_m, na.rm = TRUE),
      max = max(tfl_daily_avg_m, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    filter(!is.na(year))
  
  plot_ly(tfl_yearly) %>%
    add_bars(
      x = ~factor(year), y = ~mean,
      name = "Mean",
      marker = list(color = "#3498DB"),
      text = ~paste0(round(mean, 1), "M"),
      textposition = "outside"
    ) %>%
    layout(
      title = list(text = "TfL Journeys by Year", font = list(size = 14)),
      xaxis = list(title = "Year"),
      yaxis = list(title = "Mean Daily Journeys (M)"),
      margin = list(t = 40)
    )
}

#' Create Tourism Time Series Plot
#' @param data The loaded market data
#' @return A plotly object
create_tourism_timeseries <- function(data) {
  if (is.null(data$daily) || !data$loaded) {
    return(create_empty_plot() %>% layout(title = "Tourism Data Not Available"))
  }
  
  tourism_data <- data$daily %>%
    filter(!is.na(tourism_quarterly_visits_k), is.finite(tourism_quarterly_visits_k)) %>%
    arrange(date)
  
  if (nrow(tourism_data) == 0) {
    return(create_empty_plot() %>% layout(title = "No Tourism Data"))
  }
  
  # Get quarterly summary for cleaner visualization
  tourism_quarterly <- tourism_data %>%
    mutate(quarter = paste0(year, "-Q", quarter(date))) %>%
    group_by(quarter) %>%
    summarise(
      date = min(date),
      tourism = mean(tourism_quarterly_visits_k, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(date)
  
  plot_ly(tourism_quarterly) %>%
    add_lines(
      x = ~date, y = ~tourism,
      name = "Quarterly Visits",
      line = list(color = "#27AE60", width = 2),
      fill = "tozeroy",
      fillcolor = "rgba(39, 174, 96, 0.2)"
    ) %>%
    layout(
      title = list(text = "International Tourism (Quarterly Visits, Thousands)", font = list(size = 14)),
      xaxis = list(title = "", tickformat = "%Y-%m"),
      yaxis = list(title = "Visitors (K)"),
      margin = list(t = 40, b = 50)
    )
}

#' Create Weather Time Series Plot
#' @param data The loaded market data
#' @return A plotly object
create_weather_timeseries <- function(data) {
  if (is.null(data$daily) || !data$loaded) {
    return(create_empty_plot() %>% layout(title = "Weather Data Not Available"))
  }
  
  weather_data <- data$daily %>%
    filter(!is.na(temp_c), is.finite(temp_c)) %>%
    arrange(date)
  
  if (nrow(weather_data) == 0) {
    return(create_empty_plot() %>% layout(title = "No Weather Data"))
  }
  
  # Calculate 30-day rolling average
  weather_data <- weather_data %>%
    mutate(temp_rolling = zoo::rollmean(temp_c, k = 30, fill = NA, align = "right"))
  
  plot_ly(weather_data) %>%
    add_lines(
      x = ~date, y = ~temp_c,
      name = "Daily",
      line = list(color = "#F39C12", width = 1),
      opacity = 0.4
    ) %>%
    add_lines(
      x = ~date, y = ~temp_rolling,
      name = "30-Day Avg",
      line = list(color = "#E74C3C", width = 2)
    ) %>%
    layout(
      title = list(text = "Daily Temperature (°C)", font = list(size = 14)),
      xaxis = list(title = "", tickformat = "%Y-%m"),
      yaxis = list(title = "Temperature (°C)"),
      legend = list(orientation = "h", y = -0.15),
      hovermode = "x unified",
      margin = list(t = 40, b = 50)
    )
}

#' Create Seasonal Temperature Box Plot
#' @param data The loaded market data
#' @return A plotly object
create_seasonal_temperature <- function(data) {
  if (is.null(data$daily) || !data$loaded) {
    return(create_empty_plot() %>% layout(title = "Weather Data Not Available"))
  }
  
  weather_data <- data$daily %>%
    filter(!is.na(temp_c), !is.na(season))
  
  if (nrow(weather_data) == 0) {
    return(create_empty_plot() %>% layout(title = "No Weather Data"))
  }
  
  # Define colors for seasons
  season_colors <- c(
    "Spring" = "#2ECC71",
    "Summer" = "#F39C12",
    "Autumn" = "#E67E22",
    "Winter" = "#3498DB"
  )
  
  plot_ly(weather_data, x = ~season, y = ~temp_c, color = ~season,
          colors = season_colors, type = "box") %>%
    layout(
      title = list(text = "Temperature by Season", font = list(size = 14)),
      xaxis = list(title = ""),
      yaxis = list(title = "Temperature (°C)"),
      showlegend = FALSE,
      margin = list(t = 40)
    )
}

#' Create Weather Quality Distribution
#' @param data The loaded market data
#' @return A plotly object
create_weather_quality_dist <- function(data) {
  if (is.null(data$daily) || !data$loaded) {
    return(create_empty_plot() %>% layout(title = "Weather Data Not Available"))
  }
  
  weather_data <- data$daily %>%
    filter(!is.na(weather_quality), is.finite(weather_quality))
  
  if (nrow(weather_data) == 0) {
    return(create_empty_plot() %>% layout(title = "No Weather Quality Data"))
  }
  
  plot_ly(weather_data, x = ~weather_quality, type = "histogram",
          marker = list(color = "#3498DB", line = list(color = "white", width = 1)),
          nbinsx = 30) %>%
    layout(
      title = list(text = "Weather Quality Distribution (0-1)", font = list(size = 14)),
      xaxis = list(title = "Weather Quality"),
      yaxis = list(title = "Count"),
      margin = list(t = 40)
    )
}

#' Create Component Correlation Heatmap
#' @param data The loaded market data
#' @return A plotly object
create_component_correlation <- function(data) {
  if (is.null(data$daily) || !data$loaded) {
    return(create_empty_plot() %>% layout(title = "Data Not Available"))
  }
  
  cor_data <- data$daily %>%
    select(any_of(c("temp_c", "tfl_daily_avg_m", 
                    "weather_quality", "is_weekend", "is_holiday"))) %>%
    mutate(across(c(is_weekend, is_holiday), as.numeric)) %>%
    na.omit()
  
  if (nrow(cor_data) < 10) {
    return(create_empty_plot() %>% layout(title = "Insufficient Data for Correlation"))
  }
  
  # Rename columns for display
  names(cor_data) <- c("Temp", "TfL", "Weather", "Weekend", "Holiday")[1:ncol(cor_data)]
  
  cor_matrix <- cor(cor_data, use = "complete.obs")
  
  plot_ly(
    x = colnames(cor_matrix),
    y = rownames(cor_matrix),
    z = cor_matrix,
    type = "heatmap",
    colorscale = "RdBu",
    zmin = -1, zmax = 1,
    text = round(cor_matrix, 2),
    texttemplate = "%{text}",
    showscale = TRUE
  ) %>%
    layout(
      title = list(text = "Component Correlations", font = list(size = 14)),
      xaxis = list(title = ""),
      yaxis = list(title = ""),
      margin = list(t = 40, l = 80)
    )
}

#' Create Market Summary Info Boxes
#' @param data The loaded market data
#' @return A list of summary statistics
get_market_summary <- function(data) {
  if (is.null(data$daily) || !data$loaded) {
    return(list(
      tfl_avg = "N/A",
      temp_avg = "N/A",
      weather_quality_avg = "N/A",
      date_range = "No data available"
    ))
  }
  
  daily <- data$daily
  
  list(
    tfl_avg = round(mean(daily$tfl_daily_avg_m, na.rm = TRUE), 2),
    temp_avg = round(mean(daily$temp_c, na.rm = TRUE), 1),
    weather_quality_avg = round(mean(daily$weather_quality, na.rm = TRUE), 2),
    date_range = paste(min(daily$date, na.rm = TRUE), "to", max(daily$date, na.rm = TRUE))
  )
}

#' Create Day of Week Pattern
#' @param data The loaded market data
#' @return A plotly object
create_day_of_week_pattern <- function(data) {
  if (is.null(data$daily) || !data$loaded) {
    return(create_empty_plot() %>% layout(title = "Data Not Available"))
  }
  
  dow_data <- data$daily %>%
    filter(!is.na(tfl_daily_avg_m), !is.na(day_of_week)) %>%
    group_by(day_of_week) %>%
    summarise(
      mean_tfl = mean(tfl_daily_avg_m, na.rm = TRUE),
      .groups = "drop"
    )
  
  if (nrow(dow_data) == 0) {
    return(create_empty_plot() %>% layout(title = "No Pattern Data"))
  }
  
  # Weekday vs weekend colors
  colors <- c(rep("#3498DB", 5), rep("#E74C3C", 2))
  
  plot_ly(dow_data) %>%
    add_bars(
      x = ~day_of_week, y = ~mean_tfl,
      marker = list(color = colors),
      text = ~paste0(round(mean_tfl, 1), "M"),
      textposition = "outside"
    ) %>%
    layout(
      title = list(text = "TfL Journeys by Day of Week", font = list(size = 14)),
      xaxis = list(title = ""),
      yaxis = list(title = "Mean Journeys (M)"),
      margin = list(t = 40)
    )
}

message("✓ Market indicators helper loaded")

