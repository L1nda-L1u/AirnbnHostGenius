# =============================================
# Airbnb Baseline Price Predictor - Shiny App
# English Version - Optimized Model Loading
# =============================================

library(shiny)
library(shinydashboard)
library(DT)
library(leaflet)
library(plotly)
library(dplyr)
library(geosphere)

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

# =============================================
# UI - Cyan/Gray Theme (Low Saturation, Fresh)
# =============================================

ui <- dashboardPage(
  dashboardHeader(
    title = tags$div(
      tags$span("🏠", style = "font-size: 24px; margin-right: 10px;"),
      tags$span("Airbnb Baseline Price Predictor", 
                style = "font-size: 20px; font-weight: bold; color: #2C3E50;")
    ),
    titleWidth = 350
  ),
  
  dashboardSidebar(
    width = 300,
    collapsed = TRUE,  # Default to collapsed
    sidebarMenu(
      id = "tabs",
      menuItem("Price Prediction", tabName = "predict", icon = icon("calculator")),
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
          background-color: #6BC4C4 !important;
          color: #fff !important;
          font-weight: bold;
        }
        .skin-blue .main-header .logo:hover {
          background-color: #5AB3B3 !important;
        }
        .skin-blue .main-header .navbar {
          background-color: #7BC4C4 !important;
        }
        .skin-blue .main-sidebar {
          background-color: #E8E8E8 !important;
        }
        .skin-blue .main-sidebar .sidebar-menu > li.active > a {
          background-color: #6BC4C4 !important;
          border-left-color: #5AB3B3 !important;
        }
        .skin-blue .main-sidebar .sidebar-menu > li > a:hover {
          background-color: #D8D8D8 !important;
        }
        body {
          background: linear-gradient(135deg, #E8F5F3 0%, #F0F8F7 25%, #F5F9F8 50%, #E8F0ED 75%, #E8F5F3 100%) !important;
          background-attachment: fixed;
        }
        .content-wrapper {
          background: transparent !important;
        }
        .box {
          border-radius: 10px;
          box-shadow: 0 3px 12px rgba(0,0,0,0.12) !important;
          border-top: 3px solid #88D8C0 !important;
          background-color: #FFFFFF !important;
          margin-bottom: 15px !important;
        }
        .box-header {
          background-color: #F8F8F8 !important;
          border-bottom: 1px solid #E8E8E8;
          border-radius: 10px 10px 0 0;
        }
        .form-control {
          border-radius: 5px;
          border: 1px solid #D0D0D0;
          transition: border-color 0.3s;
          background-color: #FFFFFF;
        }
        .form-control:focus {
          border-color: #88D8C0;
          box-shadow: 0 0 5px rgba(136, 216, 192, 0.2);
        }
        .btn-primary {
          background-color: #6BC4C4 !important;
          border-color: #5AB3B3 !important;
          border-radius: 5px;
          font-weight: bold;
          padding: 10px 20px;
          transition: all 0.3s;
        }
        .btn-primary:hover {
          background-color: #5AB3B3 !important;
          transform: translateY(-2px);
          box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .price-display {
          font-size: 42px;
          font-weight: bold;
          color: #4ECDC4;
          text-align: center;
          padding: 25px 15px;
          background: linear-gradient(135deg, #F0F8F7 0%, #E8F5F3 100%);
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
          color: #4A4A4A !important;
        }
      "))
    ),
    
    tabItems(
      tabItem(
        tabName = "predict",
        fluidRow(
          # Left column - Property Information
          column(
            width = 4,
            box(
              title = tags$h3("📍 Property Information", style = "color: #4A4A4A; margin: 0; font-weight: 600;"),
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
              
              tags$h4("🏠 Basic Properties", style = "color: #5A5A5A; margin-top: 20px; font-weight: 500;"),
              
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
              
              tags$h4("✨ Amenities", style = "color: #5A5A5A; margin-top: 20px; font-weight: 500;"),
              
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
                "🚀 Predict Price",
                class = "btn-primary",
                style = "width: 100%; font-size: 18px; padding: 15px; margin-top: 20px;"
              )
            )
          ),
          
          # Middle column - Predictions
          column(
            width = 4,
            box(
              title = tags$h3("💰 Baseline Price", style = "color: #4A4A4A; margin: 0; font-weight: 600;"),
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
                  tags$p("👆 Fill in property information and click Predict", style = "font-size: 14px;")
                )
              )
            ),
            
            box(
              title = tags$h3("📈 Occupancy Prediction", style = "color: #4A4A4A; margin: 0; font-weight: 600;"),
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
              title = tags$h3("💵 Annual Revenue", style = "color: #4A4A4A; margin: 0; font-weight: 600;"),
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
              title = tags$h3("💡 Amenity Recommendations", style = "color: #4A4A4A; margin: 0; font-weight: 600; font-size: 18px;"),
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
              title = tags$h3("🗺️ Location Map", style = "color: #4A4A4A; margin: 0; font-weight: 600;"),
              width = NULL,
              solidHeader = TRUE,
              status = "info",
              
              leafletOutput("map", height = "400px")
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
              tags$li("📍 Automatic address/postcode to coordinates conversion"),
              tags$li("🏠 Support for various property attributes"),
              tags$li("✨ Rich amenities selection"),
              tags$li("💰 Smart price prediction"),
              tags$li("🗺️ Location visualization")
            ),
            tags$hr(),
            tags$h5("Model Information:", style = "color: #2C3E50;"),
            tags$p("Model trained on historical Airbnb data using:"),
            tags$ul(
              tags$li("Geographic location (coordinates, area clusters)"),
              tags$li("Property attributes (bedrooms, bathrooms, accommodates, etc.)"),
              tags$li("Amenities (WiFi, kitchen, washer, etc.)")
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
          geocode_status(paste0("✓ Location found: ", display_name))
        } else {
          geocode_result(NULL)
          geocode_status("⚠ Cannot find this address, please check your input")
        }
      }, error = function(e) {
        geocode_result(NULL)
        geocode_status("⚠ Error searching location, please try again later")
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
    
    if (grepl("✓", status)) {
      tags$div(
        status,
        style = "color: #27AE60; font-weight: bold; padding: 10px; background-color: #D5F4E6; border-radius: 5px;"
      )
    } else if (grepl("⚠", status)) {
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
    "Baseline price per night (GBP)"
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
          tags$h4("Top Recommended Amenities", style = "color: #4A4A4A; margin-top: 0; margin-bottom: 10px; font-weight: 600; font-size: 16px;"),
          tags$p("(Only showing amenities that can increase price)", style = "color: #888888; font-size: 11px; margin-top: 5px; font-style: italic;")
        ),
        
        tags$div(
          lapply(1:nrow(recs), function(i) {
            rec <- recs[i, ]
            tags$div(
              style = "background-color: #F8F8F8; padding: 12px; border-left: 3px solid #88D8C0; margin-bottom: 10px; border-radius: 5px;",
              tags$h5(
                paste0(i, ". ", rec$amenity_name),
                style = "color: #4A4A4A; margin-top: 0; margin-bottom: 8px; font-weight: 600; font-size: 14px;"
              ),
              tags$p(
                tags$strong("Price Increase: ", style = "color: #666666; font-size: 12px;"),
                tags$span(paste0("£", round(rec$price_impact, 2), " (", round(rec$price_impact_pct, 2), "%)"), 
                         style = "color: #4ECDC4; font-weight: bold; font-size: 13px;")
              ),
              tags$p(
                tags$strong("New Price: ", style = "color: #666666; font-size: 12px;"),
                tags$span(paste0("£", round(rec$new_price, 2)), style = "color: #6BC4C4; font-weight: bold; font-size: 13px;")
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
}

shinyApp(ui = ui, server = server)

