
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

