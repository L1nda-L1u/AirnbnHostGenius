app_colors <- list(
  primary = "#2A8C82",
  accent = "#F5B085",
  dark = "#2C3E50",
  muted = "#7F8C8D",
  border = "#D0D0D0",
  light = "#F5F5F5",
  fill_primary = "rgba(42, 140, 130, 0.2)",
  fill_accent = "rgba(245, 176, 133, 0.2)"
)

resolve_data_paths <- function(app_dir, filename) {
  c(
    file.path(app_dir, "data", "preprocessed", filename),
    file.path(app_dir, "data", filename),
    file.path(getwd(), filename),
    file.path(dirname(getwd()), "shiny_app", filename)
  )
}

load_csv_safe <- function(filename, app_dir, default_cols = c("date", "value")) {
  for (path in resolve_data_paths(app_dir, filename)) {
    if (file.exists(path)) {
      tryCatch({
        data <- data.table::fread(path)
        if ("date" %in% names(data)) {
          data$date <- as.Date(data$date)
        }
        message("Loaded ", filename, " from: ", path)
        return(data)
      }, error = function(e) {
        message("Error loading ", path, ": ", e$message)
      })
    }
  }

  message("Warning: ", filename, " not found. Creating empty data frame.")
  empty_df <- setNames(data.frame(matrix(ncol = length(default_cols), nrow = 0)), default_cols)
  empty_df$date <- as.Date(character())
  empty_df
}

fetch_uk_holidays <- function(holidays_file) {
  api_data <- tryCatch({
    response <- httr::GET("https://www.gov.uk/bank-holidays.json", httr::timeout(10))
    if (httr::status_code(response) == 200) {
      data <- httr::content(response, "text", encoding = "UTF-8")
      json_data <- jsonlite::fromJSON(data)
      events <- json_data$`england-and-wales`$events

      tibble::tibble(
        date = as.Date(events$date),
        title = events$title,
        is_major_holiday = TRUE
      )
    } else {
      NULL
    }
  }, error = function(e) {
    message("Could not fetch holidays from API: ", e$message)
    NULL
  })

  if (!is.null(api_data)) {
    return(api_data)
  }

  holidays_file |>
    dplyr::select(date, title, is_major_holiday)
}

build_daily_data <- function(app_dir, data_start, data_end) {
  tfl_data <- load_csv_safe("tfl.csv", app_dir, c("date", "value"))
  weather_data <- load_csv_safe("weather.csv", app_dir, c("date", "temp_c"))
  holidays_file <- load_csv_safe("holidays.csv", app_dir, c("date", "title"))
  tourism_data <- load_csv_safe("tourism.csv", app_dir, c("date", "value"))

  if (!"is_major_holiday" %in% names(holidays_file)) {
    holidays_file$is_major_holiday <- FALSE
  }

  holidays_data <- fetch_uk_holidays(holidays_file)

  daily_data <- tibble::tibble(date = seq(data_start, data_end, by = "day")) |>
    dplyr::mutate(
      day_of_week = lubridate::wday(date, label = TRUE),
      week = lubridate::week(date),
      month = lubridate::month(date),
      year = lubridate::year(date),
      is_weekend = day_of_week %in% c("Sat", "Sun"),
      is_past = date < Sys.Date(),
      is_today = date == Sys.Date()
    ) |>
    dplyr::left_join(
      tfl_data |>
        dplyr::filter(date >= data_start, date <= data_end) |>
        dplyr::select(date, tfl_value = value),
      by = "date"
    ) |>
    dplyr::left_join(
      weather_data |>
        dplyr::filter(date >= data_start, date <= data_end) |>
        dplyr::select(date, temp_c, weather_quality, precip_mm, humidity_avg, sunshine_hours, TCI, TCI_category),
      by = "date"
    ) |>
    dplyr::left_join(
      tourism_data |>
        dplyr::filter(date >= data_start, date <= data_end) |>
        dplyr::select(date, tourism_value = value),
      by = "date"
    ) |>
    dplyr::left_join(
      holidays_data |>
        dplyr::select(date, holiday_name = title, is_major_holiday),
      by = "date"
    ) |>
    dplyr::mutate(
      is_holiday = !is.na(holiday_name),
      is_major_holiday = dplyr::coalesce(is_major_holiday, FALSE)
    )

  tfl_seasonal <- tfl_data |>
    dplyr::filter(!is.na(value)) |>
    dplyr::mutate(
      month_num = lubridate::month(date),
      year = lubridate::year(date)
    ) |>
    dplyr::filter(year < 2020 | year > 2021) |>
    dplyr::group_by(month_num) |>
    dplyr::summarise(monthly_avg = mean(value, na.rm = TRUE), .groups = "drop")

  overall_avg <- mean(tfl_seasonal$monthly_avg)
  tfl_seasonal <- tfl_seasonal |>
    dplyr::mutate(
      relative = (monthly_avg - overall_avg) / overall_avg * 100,
      season_label = dplyr::case_when(
        relative >= 5 ~ "Busy",
        relative <= -5 ~ "Quiet",
        TRUE ~ "Average"
      )
    )

  daily_data <- daily_data |>
    dplyr::mutate(
      has_data = !is.na(tfl_value) | !is.na(weather_quality),
      month_num = lubridate::month(date)
    ) |>
    dplyr::left_join(
      tfl_seasonal |>
        dplyr::select(month_num, tfl_monthly_avg = monthly_avg, tfl_relative = relative, tfl_season = season_label),
      by = "month_num"
    ) |>
    dplyr::mutate(
      tfl_vs_avg = dplyr::if_else(
        !is.na(tfl_value) & !is.na(tfl_monthly_avg),
        (tfl_value - tfl_monthly_avg) / tfl_monthly_avg * 100,
        NA_real_
      ),
      TCI = dplyr::if_else(!is.na(TCI), TCI, weather_quality * 100),
      TCI_label = dplyr::case_when(
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
      day_type = dplyr::case_when(
        is_major_holiday ~ "Major Holiday",
        is_holiday ~ "Bank Holiday",
        is_weekend ~ "Weekend",
        TRUE ~ "Weekday"
      ),
      price_multiplier = dplyr::case_when(
        day_type == "Major Holiday" ~ 1.30,
        day_type == "Bank Holiday" ~ 1.20,
        day_type == "Weekend" ~ 1.15,
        TRUE ~ 1.0
      ),
      price_recommendation = dplyr::case_when(
        day_type == "Major Holiday" ~ "Premium",
        day_type == "Bank Holiday" ~ "Above Average",
        day_type == "Weekend" ~ "Above Average",
        TRUE ~ "Standard"
      ),
      weather_boost = dplyr::if_else(day_type == "Weekday" & !is.na(TCI) & TCI >= 70, TRUE, FALSE),
      price_multiplier = dplyr::if_else(weather_boost, price_multiplier + 0.05, price_multiplier),
      price_recommendation = dplyr::if_else(weather_boost, "Standard+", price_recommendation)
    )

  post_dir <- file.path(app_dir, "data", "postprocessed")
  if (!dir.exists(post_dir)) dir.create(post_dir, recursive = TRUE, showWarnings = FALSE)
  saveRDS(daily_data, file.path(post_dir, "daily_data.rds"))

  message("Daily data loaded: ", nrow(daily_data), " days from ", as.character(min(daily_data$date)), " to ", as.character(max(daily_data$date)))

  list(
    daily_data = daily_data,
    tfl_monthly_pattern = tfl_seasonal,
    holidays_data = holidays_data
  )
}

