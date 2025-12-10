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

source(file.path(app_dir, "model_loader.R"), local = TRUE)
source(file.path(app_dir, "geocoding.R"), local = TRUE)
source(file.path(app_dir, "feature_builder.R"), local = TRUE)

# =============================================
# UI - Teal/Blue Theme
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
          background-color: #1ABC9C !important;
          color: #fff !important;
          font-weight: bold;
        }
        .skin-blue .main-header .logo:hover {
          background-color: #16A085 !important;
        }
        .skin-blue .main-header .navbar {
          background-color: #3498DB !important;
        }
        .skin-blue .main-sidebar {
          background-color: #2C3E50 !important;
        }
        .skin-blue .main-sidebar .sidebar-menu > li.active > a {
          background-color: #1ABC9C !important;
          border-left-color: #16A085 !important;
        }
        .skin-blue .main-sidebar .sidebar-menu > li > a:hover {
          background-color: #34495E !important;
        }
        .box {
          border-radius: 10px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1);
          border-top: 3px solid #1ABC9C !important;
        }
        .box-header {
          background-color: #ECF0F1 !important;
          border-bottom: 1px solid #BDC3C7;
          border-radius: 10px 10px 0 0;
        }
        .form-control {
          border-radius: 5px;
          border: 2px solid #BDC3C7;
          transition: border-color 0.3s;
        }
        .form-control:focus {
          border-color: #1ABC9C;
          box-shadow: 0 0 5px rgba(26, 188, 156, 0.3);
        }
        .btn-primary {
          background-color: #3498DB !important;
          border-color: #2980B9 !important;
          border-radius: 5px;
          font-weight: bold;
          padding: 10px 20px;
          transition: all 0.3s;
        }
        .btn-primary:hover {
          background-color: #2980B9 !important;
          transform: translateY(-2px);
          box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .price-display {
          font-size: 48px;
          font-weight: bold;
          color: #1ABC9C;
          text-align: center;
          padding: 20px;
          background: linear-gradient(135deg, #E8F8F5 0%, #D5F4E6 100%);
          border-radius: 10px;
          margin: 20px 0;
        }
        #map {
          height: 400px;
          border-radius: 10px;
        }
        .checkbox {
          margin-top: 10px;
        }
        .checkbox label {
          font-weight: normal;
          color: #34495E;
        }
        .control-label {
          font-weight: bold;
          color: #2C3E50;
          margin-bottom: 5px;
        }
      "))
    ),
    
    tabItems(
      tabItem(
        tabName = "predict",
        fluidRow(
          column(
            width = 6,
            box(
              title = tags$h3("📍 Property Information", style = "color: #2C3E50; margin: 0;"),
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
              
              tags$h4("🏠 Basic Properties", style = "color: #2C3E50; margin-top: 20px;"),
              
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
              
              tags$h4("✨ Amenities", style = "color: #2C3E50; margin-top: 20px;"),
              
              tags$div(
                style = "max-height: 300px; overflow-y: auto; border: 1px solid #BDC3C7; padding: 15px; border-radius: 5px; background-color: #F8F9FA;",
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
          
          column(
            width = 6,
            box(
              title = tags$h3("💰 Prediction Result", style = "color: #2C3E50; margin: 0;"),
              width = NULL,
              solidHeader = TRUE,
              status = "success",
              
              conditionalPanel(
                condition = "output.price_predicted",
                tags$div(
                  class = "price-display",
                  textOutput("predicted_price")
                ),
                tags$div(
                  style = "text-align: center; color: #7F8C8D; margin-top: 10px;",
                  textOutput("price_note")
                )
              ),
              
              conditionalPanel(
                condition = "!output.price_predicted",
                tags$div(
                  style = "text-align: center; padding: 50px; color: #95A5A6;",
                  tags$p("👆 Please fill in the information on the left and click Predict", style = "font-size: 16px;")
                )
              )
            ),
            
            box(
              title = tags$h3("🗺️ Location Map", style = "color: #2C3E50; margin: 0;"),
              width = NULL,
              solidHeader = TRUE,
              status = "info",
              
              leafletOutput("map", height = "400px")
            ),
            
            box(
              title = tags$h3("📊 Details", style = "color: #2C3E50; margin: 0;"),
              width = NULL,
              solidHeader = TRUE,
              status = "info",
              
              conditionalPanel(
                condition = "output.price_predicted",
                tags$div(
                  style = "padding: 15px;",
                  tags$p(tags$strong("Input Information:"), style = "margin-bottom: 10px;"),
                  verbatimTextOutput("input_summary"),
                  tags$hr(),
                  tags$p(tags$strong("Model Prediction:"), style = "margin-bottom: 10px;"),
                  verbatimTextOutput("prediction_details")
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
      load_models()
      
      model_loaded(TRUE)
      model_loading(FALSE)
      showNotification("Models loaded successfully!", type = "success", duration = 2)
      return(TRUE)
    }, error = function(e) {
      model_loading(FALSE)
      showNotification(paste("Model loading failed:", e$message), type = "error", duration = 5)
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
      features <- build_features(
        lat = lat,
        lon = lon,
        bedrooms = input$bedrooms,
        bathrooms = input$bathrooms,
        accommodates = input$accommodates,
        beds = input$beds,
        room_type = input$room_type,
        amenities = input$amenities
      )
      
      price <- predict_baseline_price(features)
      
      prediction_result(list(
        price = price,
        features = features,
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
      
      showNotification("Prediction completed!", type = "success")
      
    }, error = function(e) {
      showNotification(paste("Prediction failed:", e$message), type = "error")
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
      paste(
        paste("Address:", data$address),
        paste("Coordinates: (", round(data$lat, 4), ", ", round(data$lon, 4), ")", sep = ""),
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
}

shinyApp(ui = ui, server = server)

