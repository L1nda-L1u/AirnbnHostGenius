# =============================================
# Geocoding - Address to Latitude/Longitude
# =============================================

library(httr)
library(jsonlite)

# Use Nominatim (OpenStreetMap) free geocoding service
geocode_address <- function(address) {
  if (is.null(address) || nchar(trimws(address)) == 0) {
    return(NULL)
  }
  
  # Clean address
  address_clean <- trimws(address)
  
  # Nominatim API endpoint
  base_url <- "https://nominatim.openstreetmap.org/search"
  
  # Build request
  params <- list(
    q = address_clean,
    format = "json",
    limit = 1,
    addressdetails = 1
  )
  
  # Add User-Agent (required by Nominatim)
  headers <- add_headers(
    "User-Agent" = "AirbnbPricePredictor/1.0"
  )
  
  tryCatch({
    # Set timeout (5 seconds)
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
    # Silently handle errors, don't print to console
    return(NULL)
  })
}

# Alternative method: Use Google Geocoding API (requires API key)
geocode_address_google <- function(address, api_key = NULL) {
  if (is.null(api_key)) {
    # If no API key, use Nominatim
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

