# ==================================================================================
# Holidays API Integration
# ==================================================================================
# Fetches UK bank holidays from gov.uk API and merges with historical CSV data

library(httr)
library(jsonlite)
library(tidyverse)
library(data.table)
library(lubridate)

# ==================================================================================
# Configuration
# ==================================================================================

GOV_UK_API_URL <- "https://www.gov.uk/bank-holidays.json"
CACHE_DIR <- "foot_traffic_data/raw/api_cache"
HISTORICAL_CSV <- "foot_traffic_data/raw/events/UK_holiday.csv"

# ==================================================================================
# Helper Functions
# ==================================================================================

create_cache_dir <- function() {
  if (!dir.exists(CACHE_DIR)) {
    dir.create(CACHE_DIR, recursive = TRUE)
  }
}

get_cache_file <- function(api_name) {
  file.path(CACHE_DIR, paste0(api_name, "_", Sys.Date(), ".rds"))
}

load_from_cache <- function(api_name, max_age_days = 7) {
  cache_file <- get_cache_file(api_name)
  
  if (file.exists(cache_file)) {
    cache_time <- file.info(cache_file)$mtime
    age_days <- as.numeric(difftime(Sys.time(), cache_time, units = "days"))
    
    if (age_days < max_age_days) {
      message("Loading holidays from cache (", round(age_days, 1), " days old)")
      return(readRDS(cache_file))
    }
  }
  return(NULL)
}

save_to_cache <- function(data, api_name) {
  create_cache_dir()
  cache_file <- get_cache_file(api_name)
  saveRDS(data, cache_file)
  message("Saved holidays to cache: ", cache_file)
}

# ==================================================================================
# API Functions
# ==================================================================================

#' Fetch UK bank holidays from gov.uk API
#' @return tibble with title, date, notes, bunting
fetch_uk_holidays_api <- function() {
  
  # Check cache first
  cached <- load_from_cache("uk_holidays")
  if (!is.null(cached)) return(cached)
  
  tryCatch({
    response <- GET(GOV_UK_API_URL, timeout(10))
    
    if (status_code(response) != 200) {
      warning("Gov.uk holidays API returned status ", status_code(response))
      return(NULL)
    }
    
    data <- content(response, "parsed")
    
    # Extract England and Wales holidays
    events <- data$`england-and-wales`$events
    
    if (is.null(events) || length(events) == 0) {
      warning("No holidays found in API response")
      return(NULL)
    }
    
    # Convert to tibble
    holidays_df <- map_df(events, function(event) {
      tibble(
        title = event$title,
        date = as.Date(event$date),
        notes = if (!is.null(event$notes)) event$notes else "",
        bunting = event$bunting
      )
    })
    
    # Save to cache
    save_to_cache(holidays_df, "uk_holidays")
    
    message("Fetched ", nrow(holidays_df), " holidays from gov.uk API")
    message("  Date range: ", min(holidays_df$date), " to ", max(holidays_df$date))
    
    return(holidays_df)
    
  }, error = function(e) {
    warning("Error fetching holidays from API: ", e$message)
    return(NULL)
  })
}

#' Load historical holidays from CSV
load_historical_holidays <- function() {
  
  if (!file.exists(HISTORICAL_CSV)) {
    warning("Historical holidays CSV not found: ", HISTORICAL_CSV)
    return(tibble(title = character(), date = Date(), notes = character(), bunting = logical()))
  }
  
  historical <- fread(HISTORICAL_CSV) %>%
    mutate(
      date = as.Date(date),
      bunting = as.logical(bunting)
    )
  
  message("Loaded ", nrow(historical), " historical holidays from CSV")
  message("  Date range: ", min(historical$date), " to ", max(historical$date))
  
  return(historical)
}

#' Merge historical and API holidays
merge_holidays <- function(historical, api_holidays) {
  
  if (is.null(api_holidays) || nrow(api_holidays) == 0) {
    message("No API holidays, using historical only")
    return(historical)
  }
  
  # Find the latest date in historical data
  max_historical_date <- max(historical$date)
  
  # Keep only future holidays from API
  api_future <- api_holidays %>%
    filter(date > max_historical_date)
  
  message("Adding ", nrow(api_future), " future holidays from API (after ", max_historical_date, ")")
  
  # Combine
  combined <- bind_rows(historical, api_future) %>%
    distinct(date, title, .keep_all = TRUE) %>%
    arrange(date)
  
  return(combined)
}

# ==================================================================================
# Main Function
# ==================================================================================

#' Get complete holidays: historical + future from API
#' @param use_api Attempt to fetch from API (TRUE) or use historical only (FALSE)
get_complete_holidays <- function(use_api = TRUE) {
  
  # Load historical data
  historical <- load_historical_holidays()
  
  if (!use_api) {
    message("API disabled, using historical holidays only")
    return(historical)
  }
  
  # Fetch from API
  api_holidays <- fetch_uk_holidays_api()
  
  # Merge
  combined <- merge_holidays(historical, api_holidays)
  
  message("\nTotal holidays: ", nrow(combined), " (", 
          min(combined$date), " to ", max(combined$date), ")")
  
  # Show future holidays
  future_holidays <- combined %>%
    filter(date > Sys.Date()) %>%
    arrange(date)
  
  if (nrow(future_holidays) > 0) {
    message("\nUpcoming holidays:")
    print(head(future_holidays, 10))
  }
  
  return(combined)
}

#' Save combined holidays to CSV (for updating the historical file)
save_combined_holidays <- function(holidays, output_file = NULL) {
  
  if (is.null(output_file)) {
    output_file <- file.path(CACHE_DIR, paste0("uk_holidays_combined_", Sys.Date(), ".csv"))
  }
  
  fwrite(holidays, output_file)
  message("Saved combined holidays to: ", output_file)
}

# ==================================================================================
# Testing
# ==================================================================================

if (interactive() || !exists("skip_test")) {
  message("\n=== Testing Holidays API ===\n")
  
  # Test with API
  holidays_data <- get_complete_holidays(use_api = TRUE)
  
  # Show summary
  holidays_summary <- holidays_data %>%
    mutate(year = year(date)) %>%
    count(year) %>%
    arrange(year)
  
  message("\nHolidays by year:")
  print(holidays_summary)
}

