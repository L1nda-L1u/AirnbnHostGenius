# =============================================
# Geocoding - 地址转经纬度
# =============================================

library(httr)
library(jsonlite)

# 使用Nominatim (OpenStreetMap) 免费地理编码服务
geocode_address <- function(address) {
  if (is.null(address) || nchar(trimws(address)) == 0) {
    return(NULL)
  }
  
  # 清理地址
  address_clean <- trimws(address)
  
  # Nominatim API endpoint
  base_url <- "https://nominatim.openstreetmap.org/search"
  
  # 构建请求
  params <- list(
    q = address_clean,
    format = "json",
    limit = 1,
    addressdetails = 1
  )
  
  # 添加User-Agent（Nominatim要求）
  headers <- add_headers(
    "User-Agent" = "AirbnbPricePredictor/1.0"
  )
  
  tryCatch({
    # 设置超时时间（5秒）
    response <- GET(base_url, query = params, headers, timeout(5))
    
    if (status_code(response) == 200) {
      content <- content(response, as = "text", encoding = "UTF-8")
      data <- fromJSON(content)
      
      if (length(data) > 0 && nrow(data) > 0) {
        result <- list(
          lat = as.numeric(data$lat[1]),
          lon = as.numeric(data$lon[1]),
          display_name = data$display_name[1],
          address = address_clean
        )
        return(result)
      } else {
        return(NULL)
      }
    } else {
      return(NULL)
    }
  }, error = function(e) {
    # 静默处理错误，不打印到控制台
    return(NULL)
  })
}

# 备用方法：使用Google Geocoding API（需要API key）
geocode_address_google <- function(address, api_key = NULL) {
  if (is.null(api_key)) {
    # 如果没有API key，使用Nominatim
    return(geocode_address(address))
  }
  
  base_url <- "https://maps.googleapis.com/maps/api/geocode/json"
  
  params <- list(
    address = address,
    key = api_key
  )
  
  tryCatch({
    response <- GET(base_url, query = params)
    
    if (status_code(response) == 200) {
      content <- content(response, as = "parsed")
      
      if (content$status == "OK" && length(content$results) > 0) {
        location <- content$results[[1]]$geometry$location
        result <- list(
          lat = location$lat,
          lon = location$lng,
          formatted_address = content$results[[1]]$formatted_address,
          address = address
        )
        return(result)
      }
    }
    
    return(NULL)
  }, error = function(e) {
    cat("Google Geocoding error:", e$message, "\n")
    return(NULL)
  })
}

