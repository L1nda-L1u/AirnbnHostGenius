# =============================================
# Airbnb Baseline Price Predictor - Shiny App
# English Version - Optimized Model Loading
# =============================================

library(shiny)
library(shinydashboard)
library(DT)
library(leaflet)
library(plotly)
library(ggplot2)  # For bar chart
library(dplyr)
library(geosphere)
library(zoo)  # For rolling averages
library(data.table)  # For fast data loading
library(sf)  # For spatial data (London boundary)

# Load helper functions
app_dir <- getwd()
if (!file.exists("app.R")) {
  if (file.exists("shiny_app/app.R")) {
    app_dir <- file.path(getwd(), "shiny_app")
  } else if (file.exists(file.path(getwd(), "..", "shiny_app", "app.R"))) {
    app_dir <- normalizePath(file.path(getwd(), "..", "shiny_app"))
  }
}

# Initialize Python first (if available)
if (file.exists(file.path(app_dir, "init_python.R"))) {
  source(file.path(app_dir, "init_python.R"), local = TRUE)
}

source(file.path(app_dir, "model_loader.R"), local = TRUE)
source(file.path(app_dir, "geocoding.R"), local = TRUE)
source(file.path(app_dir, "feature_builder.R"), local = TRUE)
source(file.path(app_dir, "sensitivity_helper.R"), local = TRUE)
source(file.path(app_dir, "market_indicators.R"), local = TRUE)

# =============================================
# UI - Cyan/Gray Theme (Low Saturation, Fresh)
# =============================================

ui <- dashboardPage(
  dashboardHeader(
    title = tags$div(
      tags$span("Airbnb", style = "font-size: 20px; font-weight: bold; color: #FF5A5F; margin-right: 8px;"),
      tags$span("Baseline Pricing Tool", 
                style = "font-size: 20px; font-weight: bold; color: #2C3E50;")
    ),
    titleWidth = 350
  ),
  
  dashboardSidebar(
    width = 300,
    collapsed = TRUE,  # Default to collapsed
    sidebarMenu(
      id = "tabs",
      menuItem("Overview", tabName = "overview", icon = icon("map")),
      menuItem("Price Prediction", tabName = "predict", icon = icon("calculator")),
      menuItem("Market Insights", tabName = "market", icon = icon("chart-line")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    ),
    tags$div(
      style = "padding: 20px; margin-top: 20px;",
      tags$p(
        style = "color: #7F8C8D; font-size: 12px; text-align: center;",
        "Enter property details to get smart pricing suggestions"
      )
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .skin-blue .main-header .logo {
          background-color: #FFFFFF !important;
          color: #2C3E50 !important;
          font-weight: bold;
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
          background-color: #14B8A6 !important;
          border-left-color: #0D9488 !important;
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
          background: transparent !important;
        }
        .box {
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
          border: 1px solid #E0E0E0 !important;
          background-color: #FFFFFF !important;
          margin-bottom: 15px !important;
        }
        .box-header {
          background-color: #FFFFFF !important;
          border-bottom: 1px solid #E0E0E0;
          border-radius: 8px 8px 0 0;
        }
        .form-control {
          border-radius: 5px;
          border: 1px solid #D0D0D0;
          transition: border-color 0.3s;
          background-color: #FFFFFF;
        }
        .form-control:focus {
          border-color: #14B8A6;
          box-shadow: 0 0 5px rgba(20, 184, 166, 0.2);
        }
        .btn-primary {
          background-color: #14B8A6 !important;
          border-color: #0D9488 !important;
          border-radius: 6px;
          font-weight: 500;
          padding: 10px 20px;
          transition: all 0.2s;
          color: #FFFFFF !important;
        }
        .btn-primary:hover {
          background-color: #0D9488 !important;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .price-display {
          font-size: 48px;
          font-weight: bold;
          color: #2C3E50;
          text-align: center;
          padding: 30px 15px;
          background-color: #FFFFFF;
          border-radius: 8px;
          margin: 15px 0;
        }
        #map {
          height: 400px;
          border-radius: 8px;
        }
        .content {
          padding: 10px 15px !important;
        }
        .row {
          margin-left: -7.5px !important;
          margin-right: -7.5px !important;
        }
        .col-sm-4, .col-md-4 {
          padding-left: 7.5px !important;
          padding-right: 7.5px !important;
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
      "))
    ),
    
    tabItems(
      # =============================================
      # Overview Tab - Map with Color Coding
      # =============================================
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
              leafletOutput("overview_map", height = "700px")
            )
          ),
          
          # Right side - Controls and Legend
          column(
            width = 4,
            box(
              title = tags$h4("Color Coding Options", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
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
              title = tags$h4("Filter Options", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
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
                max = 1000,
                value = c(0, 500),
                step = 10,
                width = "100%"
              ),
              
              sliderInput(
                "sample_size",
                label = "Number of Points to Display:",
                min = 1000,
                max = 50000,
                value = 10000,
                step = 1000,
                width = "100%"
              ),
              
              tags$p(
                style = "font-size: 11px; color: #888; margin-top: 5px;",
                "Reduce for better performance"
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
          # Left column - Property Information
          column(
            width = 4,
            box(
              title = tags$h3("Property Details", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "primary",
              
              textInput(
                "address",
                label = tags$strong("Address or Postcode"),
                placeholder = "e.g., London, UK or SW1A 1AA",
                width = "100%"
              ),
              
              conditionalPanel(
                condition = "output.geocode_status",
                tags$div(
                  style = "margin-bottom: 15px;",
                  uiOutput("geocode_status_text")
                )
              ),
              
              hr(),
              
              tags$h4("Basic Properties", style = "color: #4A4A4A; margin-top: 20px; font-weight: 500;"),
              
              fluidRow(
                column(6,
                  numericInput(
                    "bedrooms",
                    "Bedrooms",
                    value = 1,
                    min = 0,
                    max = 20,
                    step = 1,
                    width = "100%"
                  )
                ),
                column(6,
                  numericInput(
                    "bathrooms",
                    "Bathrooms",
                    value = 1,
                    min = 0,
                    max = 10,
                    step = 0.5,
                    width = "100%"
                  )
                )
              ),
              
              fluidRow(
                column(6,
                  numericInput(
                    "accommodates",
                    "Accommodates",
                    value = 2,
                    min = 1,
                    max = 20,
                    step = 1,
                    width = "100%"
                  )
                ),
                column(6,
                  numericInput(
                    "beds",
                    "Beds",
                    value = 1,
                    min = 0,
                    max = 20,
                    step = 1,
                    width = "100%"
                  )
                )
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
              ),
              
              hr(),
              
              tags$h4("Amenities", style = "color: #4A4A4A; margin-top: 20px; font-weight: 500;"),
              
              tags$div(
                style = "max-height: 300px; overflow-y: auto; border: 1px solid #E0E0E0; padding: 15px; border-radius: 5px; background-color: #F5F5F5;",
                checkboxGroupInput(
                  "amenities",
                  NULL,
                  choices = list(
                    "WiFi" = "Wifi",
                    "Kitchen" = "Kitchen",
                    "Washer" = "Washer",
                    "TV" = "TV",
                    "Heating" = "Heating",
                    "Air Conditioning" = "Air conditioning",
                    "Free Parking" = "Free parking",
                    "Breakfast" = "Breakfast",
                    "Dedicated Workspace" = "Dedicated workspace",
                    "Pets Allowed" = "Pets allowed",
                    "Smoking Allowed" = "Smoking allowed",
                    "Elevator" = "Elevator",
                    "Gym" = "Gym",
                    "Pool" = "Pool",
                    "Hot Tub" = "Hot tub"
                  ),
                  selected = c("Wifi", "Kitchen", "Heating")
                )
              ),
              
              hr(),
              
              actionButton(
                "predict_btn",
                "Update Prediction",
                class = "btn-primary",
                style = "width: 100%; font-size: 18px; padding: 15px; margin-top: 20px;"
              )
            )
          ),
          
          # Middle column - Predictions
          column(
            width = 4,
            box(
              title = tags$h3("Predicted Baseline Price", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "primary",
              
              conditionalPanel(
                condition = "output.price_predicted",
                tags$div(
                  class = "price-display",
                  textOutput("predicted_price")
                ),
                tags$div(
                  style = "text-align: center; color: #888888; margin-top: 10px; font-size: 14px;",
                  textOutput("price_note")
                )
              ),
              
              conditionalPanel(
                condition = "!output.price_predicted",
                tags$div(
                  style = "text-align: center; padding: 50px; color: #AAAAAA;",
                  tags$p("Fill in property information and click Update Prediction", style = "font-size: 14px;")
                )
              )
            ),
            
            box(
              title = tags$h3("Occupancy Prediction", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "info",
              
              conditionalPanel(
                condition = "output.price_predicted",
                tags$div(
                  style = "text-align: center; padding: 30px;",
                  tags$p("Model integration pending", style = "color: #888888; font-size: 14px; font-style: italic;")
                )
              ),
              
              conditionalPanel(
                condition = "!output.price_predicted",
                tags$div(
                  style = "text-align: center; padding: 30px; color: #AAAAAA;",
                  tags$p("Will show after price prediction", style = "font-size: 14px;")
                )
              )
            ),
            
            box(
              title = tags$h3("Annual Revenue", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "success",
              
              conditionalPanel(
                condition = "output.price_predicted",
                tags$div(
                  style = "text-align: center; padding: 30px;",
                  tags$p("Model integration pending", style = "color: #888888; font-size: 14px; font-style: italic;")
                )
              ),
              
              conditionalPanel(
                condition = "!output.price_predicted",
                tags$div(
                  style = "text-align: center; padding: 30px; color: #AAAAAA;",
                  tags$p("Will show after price prediction", style = "font-size: 14px;")
                )
              )
            )
          ),
          
          # Right column - Recommendations and Map
          column(
            width = 4,
            box(
              title = tags$h3("Amenity Recommendations", style = "color: #2C3E50; margin: 0; font-weight: 600; font-size: 18px;"),
              width = NULL,
              solidHeader = TRUE,
              status = "warning",
              
              conditionalPanel(
                condition = "output.price_predicted",
                tags$div(
                  style = "padding: 15px;",
                  uiOutput("amenity_recommendations")
                )
              ),
              
              conditionalPanel(
                condition = "!output.price_predicted",
                tags$div(
                  style = "text-align: center; padding: 30px; color: #AAAAAA;",
                  tags$p("Get price prediction to see recommendations", style = "font-size: 14px;")
                )
              )
            ),
            
            box(
              title = tags$h3("Location Context", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "info",
              
              leafletOutput("map", height = "400px")
            )
          )
        )
      ),
      
      tabItem(
        tabName = "market",
        # Market Summary Row
        fluidRow(
          column(
            width = 12,
            box(
              title = tags$h3("London Market Indicators", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "primary",
              
              fluidRow(
                column(3,
                  tags$div(
                    style = "text-align: center; padding: 15px; background: linear-gradient(135deg, #3498DB15 0%, #3498DB30 100%); border-radius: 8px;",
                    tags$h5("TfL Daily Journeys", style = "color: #4A4A4A; margin-bottom: 5px;"),
                    tags$div(style = "font-size: 24px; font-weight: bold; color: #3498DB;", textOutput("tfl_avg_display")),
                    tags$small("million/day", style = "color: #888;")
                  )
                ),
                column(3,
                  tags$div(
                    style = "text-align: center; padding: 15px; background: linear-gradient(135deg, #27AE6015 0%, #27AE6030 100%); border-radius: 8px;",
                    tags$h5("International Tourism", style = "color: #4A4A4A; margin-bottom: 5px;"),
                    tags$div(style = "font-size: 24px; font-weight: bold; color: #27AE60;", textOutput("tourism_avg_display")),
                    tags$small("thousand/quarter", style = "color: #888;")
                  )
                ),
                column(3,
                  tags$div(
                    style = "text-align: center; padding: 15px; background: linear-gradient(135deg, #F39C1215 0%, #F39C1230 100%); border-radius: 8px;",
                    tags$h5("Average Temperature", style = "color: #4A4A4A; margin-bottom: 5px;"),
                    tags$div(style = "font-size: 24px; font-weight: bold; color: #F39C12;", textOutput("temp_avg_display")),
                    tags$small("°C", style = "color: #888;")
                  )
                ),
                column(3,
                  tags$div(
                    style = "text-align: center; padding: 15px; background: linear-gradient(135deg, #9B59B615 0%, #9B59B630 100%); border-radius: 8px;",
                    tags$h5("Weather Quality", style = "color: #4A4A4A; margin-bottom: 5px;"),
                    tags$div(style = "font-size: 24px; font-weight: bold; color: #9B59B6;", textOutput("weather_quality_display")),
                    tags$small("(0-1 scale)", style = "color: #888;")
                  )
                )
              ),
              
              tags$div(
                style = "margin-top: 15px; text-align: center; color: #888; font-size: 12px;",
                textOutput("market_date_range")
              )
            )
          )
        ),
        
        # TfL Transport Row
        fluidRow(
          column(
            width = 8,
            box(
              title = tags$h4("TfL Transport Trends", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "info",
              plotlyOutput("tfl_timeseries", height = "300px")
            )
          ),
          column(
            width = 4,
            box(
              title = tags$h4("TfL by Year", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "info",
              plotlyOutput("tfl_yearly", height = "300px")
            )
          )
        ),
        
        # Tourism Row
        fluidRow(
          column(
            width = 6,
            box(
              title = tags$h4("International Tourism", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "success",
              plotlyOutput("tourism_timeseries", height = "280px")
            )
          ),
          column(
            width = 6,
            box(
              title = tags$h4("Weekly Patterns", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "success",
              plotlyOutput("day_of_week_pattern", height = "280px")
            )
          )
        ),
        
        # Weather Row
        fluidRow(
          column(
            width = 6,
            box(
              title = tags$h4("Temperature Trends", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "warning",
              plotlyOutput("weather_timeseries", height = "280px")
            )
          ),
          column(
            width = 6,
            box(
              title = tags$h4("Seasonal Temperature", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "warning",
              plotlyOutput("seasonal_temp", height = "280px")
            )
          )
        ),
        
        # Correlation Row
        fluidRow(
          column(
            width = 6,
            box(
              title = tags$h4("Weather Quality", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              plotlyOutput("weather_quality_dist", height = "280px")
            )
          ),
          column(
            width = 6,
            box(
              title = tags$h4("Component Correlations", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              plotlyOutput("component_correlation", height = "280px")
            )
          )
        ),
        
        # Insight Box
        fluidRow(
          column(
            width = 12,
            box(
              title = tags$h4("Market Insights", style = "color: #2C3E50; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "primary",
              
              tags$div(
                style = "padding: 15px;",
                tags$div(
                  style = "display: flex; gap: 20px; flex-wrap: wrap;",
                  
                  tags$div(
                    style = "flex: 1; min-width: 250px; padding: 15px; background-color: #F8F8F8; border-radius: 8px; border-left: 4px solid #3498DB;",
                    tags$h5("Transport Impact", style = "color: #3498DB; margin-top: 0;"),
                    tags$p("Higher TfL journey volumes indicate more potential visitors in London. Weekdays typically see higher commuter traffic.", style = "color: #666; font-size: 13px; margin-bottom: 0;")
                  ),
                  
                  tags$div(
                    style = "flex: 1; min-width: 250px; padding: 15px; background-color: #F8F8F8; border-radius: 8px; border-left: 4px solid #27AE60;",
                    tags$h5("Tourism Seasons", style = "color: #27AE60; margin-top: 0;"),
                    tags$p("International tourism peaks in summer (Q2-Q3). Properties near tourist attractions benefit most during these periods.", style = "color: #666; font-size: 13px; margin-bottom: 0;")
                  ),
                  
                  tags$div(
                    style = "flex: 1; min-width: 250px; padding: 15px; background-color: #F8F8F8; border-radius: 8px; border-left: 4px solid #F39C12;",
                    tags$h5("Weather Effect", style = "color: #F39C12; margin-top: 0;"),
                    tags$p("Good weather increases outdoor activity and tourism. Consider dynamic pricing during favorable weather periods.", style = "color: #666; font-size: 13px; margin-bottom: 0;")
                  )
                )
              )
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
            tags$h4("Airbnb Baseline Price Predictor", style = "color: #1ABC9C;"),
            tags$p("A machine learning-based tool for predicting Airbnb baseline prices."),
            tags$p("Uses Stacking model (XGBoost + Neural Network) for price prediction."),
            tags$hr(),
            tags$h5("Features:", style = "color: #2C3E50;"),
            tags$ul(
              tags$li("Automatic address/postcode to coordinates conversion"),
              tags$li("Support for various property attributes"),
              tags$li("Rich amenities selection"),
              tags$li("Smart price prediction"),
              tags$li("Location visualization"),
              tags$li("Market insights with TfL, Tourism, and Weather data")
            ),
            tags$hr(),
            tags$h5("Model Information:", style = "color: #2C3E50;"),
            tags$p("Model trained on historical Airbnb data using:"),
            tags$ul(
              tags$li("Geographic location (coordinates, area clusters)"),
              tags$li("Property attributes (bedrooms, bathrooms, accommodates, etc.)"),
              tags$li("Amenities (WiFi, kitchen, washer, etc.)")
            ),
            tags$hr(),
            tags$h5("Market Data Sources:", style = "color: #2C3E50;"),
            tags$ul(
              tags$li("Transport for London (TfL) journey statistics"),
              tags$li("International tourism visitor data"),
              tags$li("Weather data (temperature, precipitation, quality)"),
              tags$li("UK bank holidays and major events")
            )
          )
        )
      )
    )
  ),
  
  skin = "blue"
)

# =============================================
# Server - Optimized Model Loading
# =============================================

server <- function(input, output, session) {
  
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
  
  # Load listings data (lazy load)
  listings_data <- reactiveVal(NULL)
  
  load_listings <- function() {
    if (!is.null(listings_data())) {
      return(listings_data())
    }
    
    listings_file <- file.path(app_dir, "data", "listings.csv")
    if (!file.exists(listings_file)) {
      return(NULL)
    }
    
    tryCatch({
      data <- data.table::fread(listings_file) %>%
        filter(!is.na(latitude), !is.na(longitude)) %>%
        mutate(
          price = as.numeric(gsub("[£$,]", "", price)),
          room_type = as.factor(room_type),
          neighbourhood = as.factor(neighbourhood)
        ) %>%
        filter(!is.na(price), price > 0, price < 10000)
      
      listings_data(data)
      return(data)
    }, error = function(e) {
      cat("Error loading listings:", e$message, "\n")
      return(NULL)
    })
  }
  
  # Update filter choices when data loads
  observe({
    data <- load_listings()
    if (!is.null(data)) {
      room_types <- c("All" = "all", setNames(unique(as.character(data$room_type)), unique(as.character(data$room_type))))
      updateSelectInput(session, "filter_room_type", choices = room_types)
      
      max_price <- min(quantile(data$price, 0.95, na.rm = TRUE), 1000)
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
    sample_size <- min(input$sample_size, nrow(filtered))
    if (nrow(filtered) > sample_size) {
      filtered <- filtered %>% sample_n(sample_size)
    }
    
    return(filtered)
  }) %>% bindEvent(input$apply_filter, ignoreNULL = FALSE, ignoreInit = FALSE)
  
  # Selected category from bar chart click
  selected_category <- reactiveVal(NULL)
  
  # Color palette based on selection
  get_color_palette <- function(data, color_by) {
    if (color_by == "room_type") {
      levels <- unique(as.character(data$room_type))
      colors <- c("#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6")[1:length(levels)]
      return(list(
        levels = levels,
        colors = setNames(colors, levels),
        pal = colorFactor(colors, domain = levels)
      ))
    } else {
      levels <- unique(as.character(data$neighbourhood))
      n_colors <- length(levels)
      colors <- colorRampPalette(c("#E74C3C", "#F39C12", "#2ECC71", "#3498DB", "#9B59B6"))(n_colors)
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
          style = "background: #F0F8FF; padding: 10px; border-radius: 5px; text-align: center;",
          tags$div(style = "font-size: 20px; font-weight: bold; color: #3498DB;", format(total, big.mark = ",")),
          tags$div(style = "font-size: 11px; color: #666;", "Listings")
        ),
        tags$div(
          style = "background: #F0FFF0; padding: 10px; border-radius: 5px; text-align: center;",
          tags$div(style = "font-size: 20px; font-weight: bold; color: #27AE60;", paste0("£", round(avg_price))),
          tags$div(style = "font-size: 11px; color: #666;", "Avg Price")
        ),
        tags$div(
          style = "background: #FFF8F0; padding: 10px; border-radius: 5px; text-align: center;",
          tags$div(style = "font-size: 20px; font-weight: bold; color: #E67E22;", paste0("£", round(median_price))),
          tags$div(style = "font-size: 11px; color: #666;", "Median Price")
        ),
        tags$div(
          style = "background: #F8F0FF; padding: 10px; border-radius: 5px; text-align: center;",
          tags$div(style = "font-size: 20px; font-weight: bold; color: #9B59B6;", n_neighbourhoods),
          tags$div(style = "font-size: 11px; color: #666;", "Areas")
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
    
    if (grepl("Location found", status)) {
      tags$div(
        status,
        style = "color: #27AE60; font-weight: bold; padding: 10px; background-color: #D5F4E6; border-radius: 5px;"
      )
    } else if (grepl("Cannot find|Error", status)) {
      tags$div(
        status,
        style = "color: #E74C3C; font-weight: bold; padding: 10px; background-color: #FADBD8; border-radius: 5px;"
      )
    } else {
      tags$div(
        status,
        style = "color: #3498DB; font-weight: bold; padding: 10px; background-color: #EBF5FB; border-radius: 5px;"
      )
    }
  })
  
  output$map <- renderLeaflet({
    result <- geocode_result()
    
    if (!is.null(result) && !is.na(result$lat) && !is.na(result$lon)) {
      leaflet() %>%
        addTiles() %>%
        addMarkers(lng = result$lon, lat = result$lat, popup = result$display_name) %>%
        setView(lng = result$lon, lat = result$lat, zoom = 15)
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
          style = "margin-bottom: 15px;",
          tags$h4("Top Recommended Amenities", style = "color: #2C3E50; margin-top: 0; margin-bottom: 10px; font-weight: 600; font-size: 16px;"),
          tags$p("(Only showing amenities that can increase price)", style = "color: #888888; font-size: 11px; margin-top: 5px; font-style: italic;")
        ),
        
        tags$div(
          lapply(1:nrow(recs), function(i) {
            rec <- recs[i, ]
            tags$button(
              paste0("+ ", rec$amenity_name, " (£", round(rec$price_impact, 0), "/night)"),
              style = "background-color: #14B8A6; color: #FFFFFF; padding: 10px 16px; border-radius: 6px; font-size: 13px; font-weight: 500; border: none; cursor: pointer; width: 100%; text-align: left; margin-bottom: 8px; transition: background-color 0.2s;",
              onmouseover = "this.style.backgroundColor='#0D9488'",
              onmouseout = "this.style.backgroundColor='#14B8A6'"
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
}

shinyApp(ui = ui, server = server)

