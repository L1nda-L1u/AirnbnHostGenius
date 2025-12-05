# =============================================
# Airbnb Baseline Price Predictor - Shiny App
# 蓝绿色调精美UI
# =============================================

library(shiny)
library(shinydashboard)
library(DT)
library(leaflet)
library(plotly)
library(dplyr)
library(geosphere)

# 加载模型和辅助函数
# 自动查找文件路径
app_dir <- getwd()
# 如果当前目录没有app.R，尝试查找
if (!file.exists("app.R")) {
  # 尝试在shiny_app子目录
  if (file.exists("shiny_app/app.R")) {
    app_dir <- file.path(getwd(), "shiny_app")
  } else if (file.exists(file.path(getwd(), "..", "shiny_app", "app.R"))) {
    app_dir <- normalizePath(file.path(getwd(), "..", "shiny_app"))
  }
}

# 加载辅助文件
source(file.path(app_dir, "model_loader.R"), local = TRUE)
source(file.path(app_dir, "geocoding.R"), local = TRUE)
source(file.path(app_dir, "feature_builder.R"), local = TRUE)

# =============================================
# UI - 蓝绿色调设计
# =============================================

ui <- dashboardPage(
  # Header
  dashboardHeader(
    title = tags$div(
      tags$span("🏠", style = "font-size: 24px; margin-right: 10px;"),
      tags$span("Airbnb Baseline Price Predictor", 
                style = "font-size: 20px; font-weight: bold; color: #2C3E50;")
    ),
    titleWidth = 350
  ),
  
  # Sidebar
  dashboardSidebar(
    width = 300,
    sidebarMenu(
      id = "tabs",
      menuItem("价格预测", tabName = "predict", icon = icon("calculator")),
      menuItem("关于", tabName = "about", icon = icon("info-circle"))
    ),
    tags$div(
      style = "padding: 20px; margin-top: 20px;",
      tags$p(
        style = "color: #7F8C8D; font-size: 12px; text-align: center;",
        "输入房源信息，获取智能定价建议"
      )
    )
  ),
  
  # Body
  dashboardBody(
    # 自定义CSS - 蓝绿色调
    tags$head(
      tags$style(HTML("
        /* 主色调 - 蓝绿色 */
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
        
        /* 卡片样式 */
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
        
        /* 输入框样式 */
        .form-control {
          border-radius: 5px;
          border: 2px solid #BDC3C7;
          transition: border-color 0.3s;
        }
        .form-control:focus {
          border-color: #1ABC9C;
          box-shadow: 0 0 5px rgba(26, 188, 156, 0.3);
        }
        
        /* 按钮样式 */
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
        
        /* 价格显示 */
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
        
        /* 地图容器 */
        #map {
          height: 400px;
          border-radius: 10px;
        }
        
        /* 复选框样式 */
        .checkbox {
          margin-top: 10px;
        }
        .checkbox label {
          font-weight: normal;
          color: #34495E;
        }
        
        /* 标签样式 */
        .control-label {
          font-weight: bold;
          color: #2C3E50;
          margin-bottom: 5px;
        }
      "))
    ),
    
    tabItems(
      # ==========================================
      # 价格预测标签页
      # ==========================================
      tabItem(
        tabName = "predict",
        fluidRow(
          # 左侧输入面板
          column(
            width = 6,
            box(
              title = tags$h3("📍 房源信息", style = "color: #2C3E50; margin: 0;"),
              width = NULL,
              solidHeader = TRUE,
              status = "primary",
              
              # 地址输入
              textInput(
                "address",
                label = tags$strong("地址或邮编"),
                placeholder = "例如: London, UK 或 SW1A 1AA",
                width = "100%"
              ),
              
              # 地址状态显示
              conditionalPanel(
                condition = "output.geocode_status",
                tags$div(
                  style = "margin-bottom: 15px;",
                  uiOutput("geocode_status_text")
                )
              ),
              
              hr(),
              
              # 基本属性
              tags$h4("🏠 基本属性", style = "color: #2C3E50; margin-top: 20px;"),
              
              fluidRow(
                column(6,
                  numericInput(
                    "bedrooms",
                    "卧室数",
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
                    "卫生间数",
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
                    "可住人数",
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
                    "床数",
                    value = 1,
                    min = 0,
                    max = 20,
                    step = 1,
                    width = "100%"
                  )
                )
              ),
              
              # 房型选择
              selectInput(
                "room_type",
                "房型",
                choices = list(
                  "Entire home/apt" = "Entire home/apt",
                  "Private room" = "Private room",
                  "Shared room" = "Shared room"
                ),
                selected = "Entire home/apt",
                width = "100%"
              ),
              
              hr(),
              
              # 评分
              tags$h4("⭐ 评分", style = "color: #2C3E50; margin-top: 20px;"),
              
              fluidRow(
                column(6,
                  numericInput(
                    "review_cleanliness",
                    "清洁度评分",
                    value = 4.5,
                    min = 0,
                    max = 5,
                    step = 0.1,
                    width = "100%"
                  )
                ),
                column(6,
                  numericInput(
                    "review_location",
                    "位置评分",
                    value = 4.5,
                    min = 0,
                    max = 5,
                    step = 0.1,
                    width = "100%"
                  )
                )
              ),
              
              hr(),
              
              # 设施选择
              tags$h4("✨ 设施 (Amenities)", style = "color: #2C3E50; margin-top: 20px;"),
              
              tags$div(
                style = "max-height: 300px; overflow-y: auto; border: 1px solid #BDC3C7; padding: 15px; border-radius: 5px; background-color: #F8F9FA;",
                checkboxGroupInput(
                  "amenities",
                  NULL,
                  choices = list(
                    "WiFi" = "Wifi",
                    "厨房" = "Kitchen",
                    "洗衣机" = "Washer",
                    "电视" = "TV",
                    "暖气" = "Heating",
                    "空调" = "Air conditioning",
                    "停车位" = "Free parking",
                    "早餐" = "Breakfast",
                    "工作区" = "Dedicated workspace",
                    "允许宠物" = "Pets allowed",
                    "允许吸烟" = "Smoking allowed",
                    "电梯" = "Elevator",
                    "健身房" = "Gym",
                    "游泳池" = "Pool",
                    "热水浴缸" = "Hot tub"
                  ),
                  selected = c("Wifi", "Kitchen", "Heating")
                )
              ),
              
              hr(),
              
              # 预测按钮
              actionButton(
                "predict_btn",
                "🚀 预测价格",
                class = "btn-primary",
                style = "width: 100%; font-size: 18px; padding: 15px; margin-top: 20px;"
              )
            )
          ),
          
          # 右侧结果面板
          column(
            width = 6,
            # 价格显示
            box(
              title = tags$h3("💰 预测结果", style = "color: #2C3E50; margin: 0;"),
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
                  tags$p("👆 请填写左侧信息并点击预测按钮", style = "font-size: 16px;")
                )
              )
            ),
            
            # 地图显示
            box(
              title = tags$h3("🗺️ 位置地图", style = "color: #2C3E50; margin: 0;"),
              width = NULL,
              solidHeader = TRUE,
              status = "info",
              
              leafletOutput("map", height = "400px")
            ),
            
            # 详细信息
            box(
              title = tags$h3("📊 详细信息", style = "color: #2C3E50; margin: 0;"),
              width = NULL,
              solidHeader = TRUE,
              status = "info",
              
              conditionalPanel(
                condition = "output.price_predicted",
                tags$div(
                  style = "padding: 15px;",
                  tags$p(tags$strong("输入信息:"), style = "margin-bottom: 10px;"),
                  verbatimTextOutput("input_summary"),
                  tags$hr(),
                  tags$p(tags$strong("模型预测:"), style = "margin-bottom: 10px;"),
                  verbatimTextOutput("prediction_details")
                )
              )
            )
          )
        )
      ),
      
      # ==========================================
      # 关于标签页
      # ==========================================
      tabItem(
        tabName = "about",
        box(
          title = tags$h3("关于此应用", style = "color: #2C3E50; margin: 0;"),
          width = 12,
          solidHeader = TRUE,
          status = "primary",
          
          tags$div(
            style = "padding: 20px;",
            tags$h4("Airbnb Baseline Price Predictor", style = "color: #1ABC9C;"),
            tags$p("这是一个基于机器学习的Airbnb房源基准价格预测工具。"),
            tags$p("使用Stacking模型（XGBoost + Neural Network）进行价格预测。"),
            tags$hr(),
            tags$h5("功能特点:", style = "color: #2C3E50;"),
            tags$ul(
              tags$li("📍 地址/邮编自动转换为经纬度"),
              tags$li("🏠 支持多种房源属性输入"),
              tags$li("✨ 丰富的设施选择"),
              tags$li("💰 智能价格预测"),
              tags$li("🗺️ 地理位置可视化")
            ),
            tags$hr(),
            tags$h5("模型信息:", style = "color: #2C3E50;"),
            tags$p("模型基于历史Airbnb数据训练，使用以下特征:"),
            tags$ul(
              tags$li("地理位置（经纬度、区域聚类）"),
              tags$li("房源属性（卧室、卫生间、可住人数等）"),
              tags$li("设施（WiFi、厨房、洗衣机等）"),
              tags$li("评分（清洁度、位置评分）")
            )
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
  
  # 初始化模型（全局加载一次）
  model_loaded <- reactiveVal(FALSE)
  
  observe({
    if (!model_loaded()) {
      showNotification("正在加载模型...", type = "message", duration = 2)
      tryCatch({
        load_models()
        model_loaded(TRUE)
        showNotification("模型加载成功！", type = "success", duration = 2)
      }, error = function(e) {
        showNotification(paste("模型加载失败:", e$message), type = "error", duration = 5)
      })
    }
  })
  
  # 地址转经纬度（自动处理，不显示在UI）
  geocode_result <- reactiveVal(NULL)
  geocode_status <- reactiveVal("")
  
  # 防抖处理：延迟执行地址转换，避免频繁请求导致卡顿
  observeEvent(input$address, {
    address <- trimws(input$address)
    
    if (nchar(address) == 0) {
      geocode_result(NULL)
      geocode_status("")
      return()
    }
    
    # 至少3个字符才查询
    if (nchar(address) < 3) {
      geocode_status("")
      geocode_result(NULL)
      return()
    }
    
    # 显示加载状态
    geocode_status("正在查找位置...")
    
    # 延迟1.5秒后执行查询（防抖，避免每次输入都查询）
    invalidateLater(1500, session)
    
    isolate({
      # 在后台执行，避免阻塞UI
      tryCatch({
        result <- geocode_address(address)
        
        if (!is.null(result) && !is.na(result$lat) && !is.na(result$lon)) {
          geocode_result(result)
          # 截断过长的地址显示
          display_name <- result$display_name
          if (nchar(display_name) > 50) {
            display_name <- paste0(substr(display_name, 1, 47), "...")
          }
          geocode_status(paste0("✓ 位置已找到: ", display_name))
        } else {
          geocode_result(NULL)
          geocode_status("⚠ 无法找到该地址，请检查输入")
        }
      }, error = function(e) {
        geocode_result(NULL)
        geocode_status("⚠ 查找位置时出错，请稍后重试")
      })
    })
  }, ignoreInit = TRUE)
  
  # 输出地址状态
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
  
  # 地图显示（使用geocode结果）
  output$map <- renderLeaflet({
    result <- geocode_result()
    
    if (!is.null(result) && !is.na(result$lat) && !is.na(result$lon)) {
      leaflet() %>%
        addTiles() %>%
        addMarkers(lng = result$lon, lat = result$lat, popup = result$display_name) %>%
        setView(lng = result$lon, lat = result$lat, zoom = 15)
    } else {
      # 默认显示伦敦
      leaflet() %>%
        addTiles() %>%
        setView(lng = -0.1276, lat = 51.5074, zoom = 10) %>%
        addPopups(lng = -0.1276, lat = 51.5074, "请输入地址或邮编查找位置")
    }
  })
  
  # 价格预测
  prediction_result <- reactiveVal(NULL)
  
  observeEvent(input$predict_btn, {
    # 获取地址转换结果
    result <- geocode_result()
    
    if (is.null(result) || is.na(result$lat) || is.na(result$lon)) {
      showNotification("请先输入有效的地址或邮编", type = "warning")
      return()
    }
    
    lat <- result$lat
    lon <- result$lon
    
    if (!model_loaded()) {
      showNotification("模型尚未加载完成，请稍候...", type = "warning")
      return()
    }
    
    showNotification("正在预测价格...", type = "message")
    
    tryCatch({
      # 构建特征向量
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
      
      # 预测价格
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
      
      showNotification("预测完成！", type = "success")
      
    }, error = function(e) {
      showNotification(paste("预测失败:", e$message), type = "error")
      prediction_result(NULL)
    })
  })
  
  # 价格显示
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
    "每晚基准价格（英镑）"
  })
  
  # 输入摘要
  output$input_summary <- renderText({
    result <- prediction_result()
    if (!is.null(result)) {
      data <- result$input_data
      paste(
        paste("地址:", data$address),
        paste("经纬度: (", round(data$lat, 4), ", ", round(data$lon, 4), ")", sep = ""),
        paste("卧室数:", data$bedrooms),
        paste("卫生间数:", data$bathrooms),
        paste("可住人数:", data$accommodates),
        paste("床数:", data$beds),
        paste("房型:", data$room_type),
        paste("设施数量:", length(data$amenities)),
        sep = "\n"
      )
    }
  })
  
  # 预测详情
  output$prediction_details <- renderText({
    result <- prediction_result()
    if (!is.null(result)) {
      paste(
        paste("预测价格: £", round(result$price, 2), sep = ""),
        paste("特征维度:", length(result$features)),
        sep = "\n"
      )
    }
  })
}

# 运行应用
shinyApp(ui = ui, server = server)

