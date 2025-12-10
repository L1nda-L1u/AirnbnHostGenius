# ==================================================================================
# Airbnb Pricing Intelligence Dashboard
# ==================================================================================

library(shiny)
library(tidyverse)
library(data.table)
library(lubridate)
library(plotly)
library(DT)
library(httr)
library(jsonlite)

# ==================================================================================
# CONFIGURATION
# ==================================================================================

# Data boundaries: 90 days past, 365 days future
DATA_START <- Sys.Date() - 90
DATA_END <- Sys.Date() + 365

# ==================================================================================
# DATA LOADING
# ==================================================================================

tfl_data <- fread("data/tfl.csv") %>% mutate(date = as.Date(date))
weather_data <- fread("data/weather.csv") %>% mutate(date = as.Date(date))
holidays_file <- fread("data/holidays.csv") %>% mutate(date = as.Date(date))
tourism_data <- fread("data/tourism.csv") %>% mutate(date = as.Date(date))

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

# ==================================================================================
# UI
# ==================================================================================

ui <- fluidPage(
  
  tags$head(
    tags$style(HTML("
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      
      body { 
        font-family: 'Inter', sans-serif;
        background: #f1f5f9;
        color: #1e293b;
        min-height: 100vh;
      }
      
      .navbar { 
        background: #ffffff !important;
        border-bottom: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }
      .navbar-brand { 
        color: #10b981 !important; 
        font-weight: 700;
        font-size: 20px;
      }
      .navbar-nav > li > a { color: #64748b !important; }
      .navbar-nav > li.active > a { color: #10b981 !important; }
      
      .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }
      
      .section-title { 
        font-size: 16px; 
        font-weight: 600; 
        color: #1e293b;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .section-title::before {
        content: '';
        width: 3px;
        height: 16px;
        background: #10b981;
        border-radius: 2px;
      }
      
      /* Calendar Styles */
      .calendar-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding: 0 8px;
      }
      
      .calendar-title {
        font-size: 20px;
        font-weight: 600;
        color: #1e293b;
      }
      
      .calendar-nav-btn {
        background: #e2e8f0;
        border: none;
        color: #475569;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 500;
        transition: background 0.2s, opacity 0.2s;
      }
      .calendar-nav-btn:hover:not(.disabled) {
        background: #cbd5e1;
      }
      .calendar-nav-btn.today-btn {
        background: #10b981;
        color: white;
      }
      .calendar-nav-btn.today-btn:hover {
        background: #059669;
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
        color: #64748b;
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
      }
      .calendar-day:hover {
        transform: scale(1.08);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10;
      }
      
      /* SELECTED - dark border frame */
      .calendar-day.selected {
        box-shadow: 0 0 0 3px #1e293b !important;
      }
      
      .day-number {
        font-size: 14px;
        font-weight: 600;
      }
      .day-price {
        font-size: 9px;
        opacity: 0.8;
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
      
      /* TODAY - filled with pink */
      .calendar-day.today {
        background: #fda4af !important;
        color: #881337 !important;
      }
      .calendar-day.today .day-number,
      .calendar-day.today .day-price {
        color: #881337 !important;
      }
      
      /* Demand Level Colors */
      .demand-premium { 
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
      }
      .demand-above { 
        background: linear-gradient(135deg, #0891b2 0%, #22d3ee 100%);
        color: white;
      }
      .demand-standard { 
        background: linear-gradient(135deg, #d97706 0%, #fbbf24 100%);
        color: #1e293b;
      }
      .demand-below { 
        background: linear-gradient(135deg, #94a3b8 0%, #cbd5e1 100%);
        color: #1e293b;
      }
      .demand-nodata { 
        background: #f1f5f9;
        border: 1px dashed #cbd5e1;
        color: #94a3b8;
      }
      .demand-weatherboost { 
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #1e293b;
      }
      
      .holiday-marker {
        position: absolute;
        top: 4px;
        right: 4px;
        width: 6px;
        height: 6px;
        background: #f43f5e;
        border-radius: 50%;
      }
      
      /* Legend - Simplified */
      .legend {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 24px;
        margin-top: 20px;
        padding-top: 16px;
        border-top: 1px solid #e2e8f0;
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
        background: #e2e8f0;
      }
      
      /* Detail Panel */
      .detail-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
      }
      
      .detail-date {
        font-size: 16px;
        font-weight: 600;
        color: #10b981;
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
        border-radius: 6px;
        border: 1px solid #e2e8f0;
      }
      .detail-label {
        font-size: 10px;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 4px;
      }
      .detail-value {
        font-size: 20px;
        font-weight: 700;
        color: #1e293b;
      }
      
      .component-bar {
        height: 6px;
        background: #e2e8f0;
        border-radius: 3px;
        margin-top: 8px;
        overflow: hidden;
      }
      .component-fill {
        height: 100%;
        border-radius: 3px;
      }
      
      /* Tables */
      .dataTables_wrapper { color: #1e293b !important; }
      table.dataTable { color: #1e293b !important; }
      table.dataTable thead th { 
        color: #64748b !important; 
        border-bottom-color: #e2e8f0 !important;
        font-size: 11px;
        text-transform: uppercase;
      }
      table.dataTable tbody tr { background: transparent !important; }
      table.dataTable tbody tr:hover { background: rgba(0,0,0,0.02) !important; }
      
      .form-control, .selectize-input, .selectize-dropdown {
        background: #ffffff !important;
        border-color: #e2e8f0 !important;
        color: #1e293b !important;
      }
      
      .btn-primary {
        background: #10b981 !important;
        border: none !important;
      }
      .btn-primary:hover {
        background: #059669 !important;
      }
    "))
  ),
  
  navbarPage(
    title = "Airbnb Pricing Intelligence",
    
    # ==================== TAB 1: Calendar View ====================
    tabPanel("Calendar",
      fluidRow(
        column(8,
          div(class = "card",
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
              div(class = "legend-item", div(class = "legend-color demand-premium"), span("+30%")),
              div(class = "legend-item", div(class = "legend-color demand-above"), span("+15%")),
              div(class = "legend-item", div(class = "legend-color demand-weatherboost"), span("+5%")),
              div(class = "legend-item", div(class = "legend-color demand-standard"), span("Base")),
              # Divider
              div(class = "legend-divider"),
              # Markers
              div(class = "legend-item", div(class = "legend-dot", style = "background: #f43f5e;"), span("Holiday")),
              div(class = "legend-item", div(class = "legend-color", style = "background: #fda4af;"), span("Today")),
              div(class = "legend-item", div(class = "legend-color", style = "border: 2px solid #1e293b; background: transparent;"), span("Selected"))
            )
          )
        ),
        column(4,
          div(class = "card",
            div(class = "section-title", "Selected Date Details"),
            uiOutput("date_details_panel")
          ),
          div(class = "card",
            div(class = "section-title", "High Demand Days This Month"),
            DTOutput("high_demand_table")
          )
        )
      )
    ),
    
    # ==================== TAB 2: Demand Trends ====================
    tabPanel("Demand Trends",
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
            div(class = "section-title", "Price Adjustment by Day of Week"),
            plotlyOutput("demand_by_dow", height = "280px")
          )
        ),
        column(6,
          div(class = "card",
            div(class = "section-title", "Day Type Distribution"),
            plotlyOutput("demand_breakdown", height = "280px")
          )
        )
      )
    ),
    
    # ==================== TAB 3: Price Calculator ====================
    tabPanel("Price Calculator",
      fluidRow(
        column(4,
          div(class = "card",
            div(class = "section-title", "Settings"),
            numericInput("base_price", "Base Nightly Rate (GBP)", value = 120, min = 20, max = 1000),
            br(),
            dateRangeInput("date_range", "Date Range",
                          start = Sys.Date(),
                          end = Sys.Date() + 30,
                          min = DATA_START,
                          max = DATA_END),
            br(),
            actionButton("calculate", "Calculate Prices", class = "btn-primary", style = "width: 100%;")
          )
        ),
        column(8,
          div(class = "card",
            div(class = "section-title", "Recommended Pricing"),
            DTOutput("price_recommendations")
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
    
    # ==================== TAB 4: Market Data ====================
    tabPanel("Market Data",
      fluidRow(
        column(6,
          div(class = "card",
            div(class = "section-title", "TfL Transport Activity"),
            plotlyOutput("tfl_chart", height = "280px")
          )
        ),
        column(6,
          div(class = "card",
            div(class = "section-title", "Weather Forecast"),
            plotlyOutput("weather_chart", height = "280px")
          )
        )
      ),
      fluidRow(
        column(12,
          div(class = "card",
            div(class = "section-title", "Upcoming Holidays (gov.uk API)"),
            DTOutput("events_table")
          )
        )
      )
    )
  )
)

# ==================================================================================
# SERVER
# ==================================================================================

server <- function(input, output, session) {
  
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
          div(class = "day-price", "—")
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
      return(div(style = "color: #64748b; text-align: center; padding: 20px;",
                 "No data available for this date"))
    }
    
    d <- day_data[1,]
    
    # Color based on day type
    day_type_color <- switch(d$day_type,
      "Major Holiday" = "#10b981",
      "Bank Holiday" = "#22d3ee",
      "Weekend" = "#f59e0b",
      "#64748b"
    )
    
    past_label <- if (d$is_past) " (Past)" else if (d$is_today) " (Today)" else ""
    
    div(class = "detail-panel",
      div(class = "detail-date", paste0(format(d$date, "%A, %B %d, %Y"), past_label)),
      
      # Day Type Badge
      div(style = paste0(
        "display: inline-block; padding: 6px 12px; border-radius: 20px; ",
        "font-size: 12px; font-weight: 600; margin-bottom: 16px; ",
        "background: ", day_type_color, "; color: white;"
      ), d$day_type),
      
      # Price Adjustment (main info)
      div(class = "detail-grid",
        div(class = "detail-item",
          div(class = "detail-label", "Price Adjustment"),
          div(class = "detail-value", style = paste0("color: ", day_type_color), 
              if (d$price_multiplier == 1) "Base" else scales::percent(d$price_multiplier - 1, accuracy = 1))
        ),
        div(class = "detail-item",
          div(class = "detail-label", "Recommendation"),
          div(class = "detail-value", style = "font-size: 16px;", d$price_recommendation)
        )
      ),
      
      tags$hr(style = "border-color: #e2e8f0; margin: 16px 0;"),
      
      # ==================== TfL SEASONAL PATTERN ====================
      div(style = "margin-bottom: 16px;",
        # Header with semantic label
        div(style = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;",
          span(style = "font-size: 12px; color: #64748b;", "🚇 CITY ACTIVITY"),
          span(style = paste0(
            "font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; ",
            switch(coalesce(d$tfl_season, "Average"),
              "Busy" = "background: #dcfce7; color: #166534;",
              "Quiet" = "background: #fef3c7; color: #92400e;",
              "background: #f1f5f9; color: #64748b;"
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
          
          # Use tighter range (±10% or actual range + 20% padding, whichever is smaller)
          # This amplifies the visual difference between months
          range_limit <- min(10, max_deviation * 1.2)  # Cap at ±10% for maximum amplification
          y_min <- -range_limit
          y_max <- range_limit
          
          chart_data <- chart_data %>%
            mutate(
              # Map relative % to y position with tighter scale (amplifies differences)
              # SVG: y=0 at top, y=100 at bottom
              # So: positive values (above avg) → smaller y (top), negative → larger y (bottom)
              y_pos = scales::rescale(relative, to = c(85, 15), from = c(y_min, y_max)),
              x_pos = (month_num - 1) * (100 / 11)  # 0 to 100% width
            )
          
          # Create SVG path string (format: "M x1 y1 L x2 y2 L x3 y3...")
          path_coords <- paste(chart_data$x_pos, chart_data$y_pos, sep = " ")
          path_d <- paste0("M ", paste(path_coords, collapse = " L "))
          
          # Determine line color based on overall trend
          avg_relative <- mean(chart_data$relative, na.rm = TRUE)
          line_color <- if (avg_relative > 5) "#22c55e" else if (avg_relative < -5) "#f59e0b" else "#64748b"
          
          # Build chart: SVG for lines, absolute positioned divs for circles (always round)
          div(style = "position: relative; height: 60px; margin-bottom: 4px;",
            # SVG for lines (stretchable)
            tags$svg(
              style = "width: 100%; height: 60px;",
              viewBox = "0 0 100 100",
              preserveAspectRatio = "none",
              # Zero line (reference)
              tags$line(x1 = "0", y1 = "50", x2 = "100", y2 = "50",
                       stroke = "#e2e8f0", `stroke-width` = "0.5", `stroke-dasharray` = "2,2",
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
            div(style = "font-size: 9px; color: #94a3b8;", m)
          })
        ),
        
        # Caption
        div(style = "font-size: 10px; color: #94a3b8; margin-top: 4px; text-align: center;",
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
          span(style = "font-size: 12px; color: #64748b;", "🌤️ TCI (Tourism Climate Index)"),
          span(style = paste0(
            "font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; ",
            if (!is.na(d$TCI)) {
              if (d$TCI >= 70) "background: #dcfce7; color: #166534;"
              else if (d$TCI >= 50) "background: #fef3c7; color: #92400e;"
              else "background: #fee2e2; color: #991b1b;"
            } else "background: #f1f5f9; color: #64748b;"
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
                  else if (d$TCI >= 50) "#f59e0b"
                  else "#ef4444"
                } else "#e2e8f0", ";"
              ))
        ),
        
        # TCI components row
        div(style = "display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: #64748b;",
          span(paste0("🌡️ ", round(coalesce(d$temp_c, 0), 0), "°C")),
          span(paste0("💧 ", round(coalesce(d$precip_mm, 0), 1), "mm")),
          span(paste0("☀️ ", round(coalesce(d$sunshine_hours, 0), 1), "h")),
          span(style = "font-weight: 600;", paste0("TCI: ", round(coalesce(d$TCI, 0), 0)))
        )
      ),
      
      # Holiday name if applicable
      if (!is.na(d$holiday_name)) {
        div(style = "margin-top: 12px; padding: 10px; background: rgba(244, 63, 94, 0.1); border-radius: 6px;",
          div(style = "font-size: 11px; color: #64748b; margin-bottom: 2px;", "HOLIDAY"),
          div(style = "font-size: 14px; font-weight: 600; color: #f43f5e;", d$holiday_name)
        )
      },
      
      # Weather boost note (if applicable)
      if (d$weather_boost) {
        div(style = "margin-top: 8px; padding: 8px; background: rgba(16, 185, 129, 0.1); border-radius: 6px; font-size: 11px; color: #10b981;",
          "☀️ Good weather bonus: +5% applied"
        )
      }
    )
  })
  
  # ==================== HIGH DEMAND TABLE ====================
  
  output$high_demand_table <- renderDT({
    month_data() %>%
      filter(!is_past, price_recommendation %in% c("Premium", "Above Average", "Standard+")) %>%
      arrange(desc(price_multiplier)) %>%
      head(8) %>%
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
        options = list(dom = 't', pageLength = 8, ordering = FALSE),
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
                fillcolor = "rgba(16, 185, 129, 0.2)",
                line = list(color = "#10b981", width = 2),
                name = "Price Adjustment %") %>%
      add_markers(data = filter(future_data, is_holiday),
                  y = ~price_adj_pct, 
                  marker = list(color = "#f43f5e", size = 8),
                  name = "Holidays") %>%
      layout(
        xaxis = list(title = "", gridcolor = "#e2e8f0", color = "#64748b"),
        yaxis = list(title = "Price Adjustment %", gridcolor = "#e2e8f0", color = "#64748b", range = c(-5, 35)),
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
    
    colors <- c("#94a3b8", "#94a3b8", "#94a3b8", "#94a3b8", "#94a3b8", "#10b981", "#10b981")
    
    plot_ly(dow_data, x = ~day_of_week, y = ~avg_adj, type = "bar",
            marker = list(color = colors)) %>%
      layout(
        xaxis = list(title = "", gridcolor = "#e2e8f0", color = "#64748b"),
        yaxis = list(title = "Avg Price Adj %", gridcolor = "#e2e8f0", color = "#64748b"),
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
    
    colors <- c("Major Holiday" = "#10b981", "Bank Holiday" = "#22d3ee", 
                "Weekend" = "#f59e0b", "Weekday" = "#94a3b8")
    
    plot_ly(day_type_counts, x = ~day_type, y = ~n, type = "bar",
            marker = list(color = colors[as.character(day_type_counts$day_type)])) %>%
      layout(
        xaxis = list(title = "", gridcolor = "#e2e8f0", color = "#64748b"),
        yaxis = list(title = "Number of Days", gridcolor = "#e2e8f0", color = "#64748b"),
        paper_bgcolor = "transparent",
        plot_bgcolor = "transparent",
        font = list(color = "#1e293b")
      )
  })
  
  # ==================== PRICE CALCULATOR ====================
  
  price_data <- eventReactive(input$calculate, {
    daily_data %>%
      filter(date >= input$date_range[1], date <= input$date_range[2]) %>%
      mutate(
        recommended_price = round(input$base_price * price_multiplier, 0),
        price_diff = recommended_price - input$base_price
      )
  })
  
  output$price_recommendations <- renderDT({
    req(input$calculate)
    
    price_data() %>%
      mutate(
        Date = format(date, "%a, %b %d"),
        Base = paste0("£", input$base_price),
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
          c("rgba(16, 185, 129, 0.2)", "rgba(34, 211, 238, 0.2)", 
            "rgba(251, 191, 36, 0.2)", "rgba(100, 116, 139, 0.2)")
        ))
  })
  
  output$price_timeline <- renderPlotly({
    req(input$calculate)
    
    pd <- price_data()
    
    plot_ly(pd, x = ~date) %>%
      add_trace(y = ~input$base_price, type = "scatter", mode = "lines",
                line = list(color = "#64748b", dash = "dash", width = 2),
                name = "Base Price") %>%
      add_trace(y = ~recommended_price, type = "scatter", mode = "lines+markers",
                line = list(color = "#10b981", width = 3),
                marker = list(size = 6, color = "#10b981"),
                name = "Recommended") %>%
      layout(
        xaxis = list(title = "", gridcolor = "#334155", color = "#94a3b8"),
        yaxis = list(title = "Price (GBP)", gridcolor = "#334155", color = "#94a3b8"),
        paper_bgcolor = "transparent",
        plot_bgcolor = "transparent",
        font = list(color = "#e2e8f0"),
        legend = list(orientation = "h", y = 1.1)
      )
  })
  
  # ==================== MARKET DATA ====================
  
  output$tfl_chart <- renderPlotly({
    tfl_plot <- tfl_data %>% filter(date >= Sys.Date() - 365)
    
    plot_ly(tfl_plot, x = ~date, y = ~value, type = "scatter", mode = "lines",
            line = list(color = "#3b82f6", width = 2)) %>%
      layout(
        xaxis = list(title = "", gridcolor = "#334155", color = "#94a3b8"),
        yaxis = list(title = "Daily Journeys (M)", gridcolor = "#334155", color = "#94a3b8"),
        paper_bgcolor = "transparent",
        plot_bgcolor = "transparent",
        font = list(color = "#e2e8f0")
      )
  })
  
  output$weather_chart <- renderPlotly({
    weather_plot <- weather_data %>%
      filter(date >= Sys.Date() - 30, date <= Sys.Date() + 30)
    
    plot_ly(weather_plot, x = ~date, y = ~temp_c, type = "scatter", mode = "lines",
            line = list(color = "#f97316", width = 2)) %>%
      layout(
        xaxis = list(title = "", gridcolor = "#334155", color = "#94a3b8"),
        yaxis = list(title = "Temperature (°C)", gridcolor = "#334155", color = "#94a3b8"),
        paper_bgcolor = "transparent",
        plot_bgcolor = "transparent",
        font = list(color = "#e2e8f0")
      )
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

# ==================================================================================
# RUN APP
# ==================================================================================

shinyApp(ui = ui, server = server)
