# =============================================
# Feature Builder - Build Model Input Feature Vector
# =============================================

library(dplyr)
library(geosphere)

# Global variables to store training data statistics (avoid reloading)
training_data_cache <- NULL
training_stats_cache <- NULL
cluster_centers_cache <- NULL  # Cache cluster centers
neighbourhood_map_cache <- NULL  # Cache neighbourhood mapping

# Load training data to get feature columns and default values
load_training_sample <- function() {
  # If already loaded, return directly
  if (!is.null(training_data_cache)) {
    return(training_data_cache)
  }
  
  # Try multiple possible paths (from root directory)
  possible_paths <- c(
    file.path(getwd(), "baseprice_model", "nn_price_training_v4.csv"),
    file.path(getwd(), "shiny_app", "baseprice_model", "nn_price_training_v4.csv"),
    file.path(dirname(getwd()), "shiny_app", "baseprice_model", "nn_price_training_v4.csv")
  )
  
  for (training_file in possible_paths) {
    if (file.exists(training_file)) {
      tryCatch({
        # Read more data for more accurate statistics (at least 1000 rows)
        data <- read.csv(training_file, nrows = 1000)
        training_data_cache <<- data  # Cache data
        cat("Loaded", nrow(data), "rows of training data for statistics\n")
        return(data)
      }, error = function(e) {
        cat("Error reading training file:", e$message, "\n")
      })
    }
  }
  
  cat("Warning: Training data file not found. Using default features.\n")
  return(NULL)
}

# Get nearest listing features for filling missing features
get_nearest_listing_features <- function(lat, lon, training_data) {
  if (is.null(training_data) || nrow(training_data) == 0) {
    return(NULL)
  }
  
  # Calculate distance
  distances <- distHaversine(
    cbind(training_data$longitude, training_data$latitude),
    c(lon, lat)
  ) / 1000  # Convert to kilometers
  
  nearest_idx <- which.min(distances)
  return(training_data[nearest_idx, ])
}

# Calculate cluster centers (for finding corresponding cluster by lat/lon)
calculate_cluster_centers <- function(training_data) {
  if (is.null(cluster_centers_cache) && !is.null(training_data) && nrow(training_data) > 0) {
    if ("location_cluster_id" %in% colnames(training_data) &&
        "latitude" %in% colnames(training_data) &&
        "longitude" %in% colnames(training_data)) {
      
      cluster_centers_cache <<- training_data %>%
        group_by(location_cluster_id) %>%
        summarise(
          center_lat = mean(latitude, na.rm = TRUE),
          center_lon = mean(longitude, na.rm = TRUE),
          .groups = "drop"
        )
      cat("Calculated cluster centers for", nrow(cluster_centers_cache), "clusters\n")
    }
  }
  return(cluster_centers_cache)
}

# Find corresponding cluster_id by lat/lon (via nearest cluster center)
find_cluster_by_location <- function(lat, lon, training_data) {
  cluster_centers <- calculate_cluster_centers(training_data)
  
  if (is.null(cluster_centers) || nrow(cluster_centers) == 0) {
    # If no cluster centers, use nearest listing
    nearest <- get_nearest_listing_features(lat, lon, training_data)
    if (!is.null(nearest) && "location_cluster_id" %in% colnames(nearest)) {
      return(as.integer(nearest$location_cluster_id))
    }
    return(0)
  }
  
  # Calculate distance to each cluster center
  distances <- distHaversine(
    cbind(cluster_centers$center_lon, cluster_centers$center_lat),
    c(lon, lat)
  ) / 1000  # Convert to kilometers
  
  # Find nearest cluster
  nearest_cluster_idx <- which.min(distances)
  cluster_id <- cluster_centers$location_cluster_id[nearest_cluster_idx]
  
  cat(sprintf("Location (%.4f, %.4f) assigned to cluster_id: %d (distance: %.2f km)\n", 
              lat, lon, cluster_id, distances[nearest_cluster_idx]))
  
  return(as.integer(cluster_id))
}

# Calculate location cluster features
calculate_cluster_features <- function(lat, lon, training_data) {
  if (is.null(training_data) || nrow(training_data) == 0) {
    # Return default values
    return(list(
      location_cluster_id = 0,
      cluster_median_price = 150,
      cluster_mean_price = 150,
      cluster_p25_price = 120,
      cluster_p75_price = 180,
      cluster_count = 100
    ))
  }
  
  # Find corresponding cluster_id by lat/lon
  cluster_id <- find_cluster_by_location(lat, lon, training_data)
  
  # Calculate price statistics for this cluster
  if ("location_cluster_id" %in% colnames(training_data) && 
      "price_num" %in% colnames(training_data)) {
    cluster_data <- training_data %>%
      filter(location_cluster_id == cluster_id)
    
    if (nrow(cluster_data) > 0) {
      return(list(
        location_cluster_id = cluster_id,
        cluster_median_price = median(cluster_data$price_num, na.rm = TRUE),
        cluster_mean_price = mean(cluster_data$price_num, na.rm = TRUE),
        cluster_p25_price = quantile(cluster_data$price_num, 0.25, na.rm = TRUE),
        cluster_p75_price = quantile(cluster_data$price_num, 0.75, na.rm = TRUE),
        cluster_count = nrow(cluster_data)
      ))
    }
  }
  
  # Use global statistics
  return(list(
    location_cluster_id = cluster_id,
    cluster_median_price = median(training_data$price_num, na.rm = TRUE),
    cluster_mean_price = mean(training_data$price_num, na.rm = TRUE),
    cluster_p25_price = quantile(training_data$price_num, 0.25, na.rm = TRUE),
    cluster_p75_price = quantile(training_data$price_num, 0.75, na.rm = TRUE),
    cluster_count = nrow(training_data)
  ))
}

# Build feature vector
build_features <- function(lat, lon, bedrooms, bathrooms, accommodates, beds,
                          room_type, amenities = c()) {
  
  # Load training data sample
  training_data <- load_training_sample()
  
  # Get feature column order
  if (!is.null(training_data)) {
    feature_cols <- setdiff(colnames(training_data), "price_num")
  } else {
    # Use default feature columns (from previous analysis)
    feature_cols <- c(
      "latitude", "longitude", "accommodates", "bathrooms", "bedrooms", "beds",
      "review_scores_cleanliness", "review_scores_location", "bath_num", 
      "bath_shared", "bath_type_unknown",
      paste0("amenity_", c("Wifi", "Smoke.alarm", "Kitchen", "Washer", "Essentials",
                          "Iron", "Hot.water", "Hangers", "Carbon.monoxide.alarm",
                          "Hair.dryer", "Heating", "Bed.linens", "TV", 
                          "Dishes.and.silverware", "Refrigerator", "Cooking.basics",
                          "Shampoo", "Microwave", "Hot.water.kettle", "Oven",
                          "Dedicated.workspace", "Toaster", "Freezer", "Shower.gel",
                          "First.aid.kit", "Dining.table", "Cleaning.products",
                          "Self.check.in", "Fire.extinguisher", "Long.term.stays.allowed")),
      "neighbourhood_id", "room_type_id",
      "cluster_median_price", "cluster_mean_price", "cluster_p25_price",
      "cluster_p75_price", "cluster_count", "location_cluster_id"
    )
  }
  
  # Calculate bathroom-related features
  bath_num <- ifelse(bathrooms > 0, round(bathrooms), 1)
  bath_shared <- 0  # Default: not shared
  bath_type_unknown <- ifelse(bathrooms == 0, 1, 0)
  
  # Room type ID mapping
  room_type_map <- c(
    "Entire home/apt" = 0,
    "Private room" = 1,
    "Shared room" = 2
  )
  room_type_id <- ifelse(room_type %in% names(room_type_map), 
                        room_type_map[room_type], 0)
  
  # Calculate location cluster features
  cluster_features <- calculate_cluster_features(lat, lon, training_data)
  
  # Get neighbourhood_id (using nearest listing)
  nearest <- get_nearest_listing_features(lat, lon, training_data)
  neighbourhood_id <- if (!is.null(nearest) && "neighbourhood_id" %in% colnames(nearest)) {
    nid <- as.integer(nearest$neighbourhood_id)
    cat(sprintf("Location (%.4f, %.4f) assigned to neighbourhood_id: %d (from nearest listing)\n", 
                lat, lon, nid))
    nid
  } else {
    # If no training data, use most common neighbourhood_id (usually 0)
    if (!is.null(training_data) && "neighbourhood_id" %in% colnames(training_data)) {
      # Use most common neighbourhood_id from training data
      most_common_id <- as.integer(names(sort(table(training_data$neighbourhood_id), decreasing = TRUE)[1]))
      if (is.na(most_common_id)) {
        cat("Warning: Could not determine neighbourhood_id, using default: 0\n")
        0
      } else {
        cat(sprintf("Using most common neighbourhood_id: %d\n", most_common_id))
        most_common_id
      }
    } else {
      cat("Warning: No training data available, using default neighbourhood_id: 0\n")
      0
    }
  }
  
  # Build amenity features (one-hot encoding)
  amenity_features <- list()
  amenity_list <- c(
    "Wifi", "Smoke.alarm", "Kitchen", "Washer", "Essentials", "Iron", 
    "Hot.water", "Hangers", "Carbon.monoxide.alarm", "Hair.dryer", 
    "Heating", "Bed.linens", "TV", "Dishes.and.silverware", "Refrigerator",
    "Cooking.basics", "Shampoo", "Microwave", "Hot.water.kettle", "Oven",
    "Dedicated.workspace", "Toaster", "Freezer", "Shower.gel", 
    "First.aid.kit", "Dining.table", "Cleaning.products", "Self.check.in",
    "Fire.extinguisher", "Long.term.stays.allowed"
  )
  
  for (amenity in amenity_list) {
    amenity_key <- paste0("amenity_", amenity)
    # Check user-selected amenities
    if (amenity %in% amenities || 
        (amenity == "Wifi" && "Wifi" %in% amenities) ||
        (amenity == "Kitchen" && "Kitchen" %in% amenities) ||
        (amenity == "Heating" && "Heating" %in% amenities)) {
      amenity_features[[amenity_key]] <- 1
    } else {
      amenity_features[[amenity_key]] <- 0
    }
  }
  
  # Build complete feature vector
  # Use default values for review scores (median from training data)
  review_cleanliness_default <- 4.5
  review_location_default <- 4.5
  
  features <- list(
    latitude = lat,
    longitude = lon,
    accommodates = accommodates,
    bathrooms = bathrooms,
    bedrooms = bedrooms,
    beds = beds,
    review_scores_cleanliness = review_cleanliness_default,
    review_scores_location = review_location_default,
    bath_num = bath_num,
    bath_shared = bath_shared,
    bath_type_unknown = bath_type_unknown
  )
  
  # Add amenity features
  features <- c(features, amenity_features)
  
  # Add ID and cluster features
  features <- c(features, list(
    neighbourhood_id = neighbourhood_id,
    room_type_id = room_type_id,
    cluster_median_price = cluster_features$cluster_median_price,
    cluster_mean_price = cluster_features$cluster_mean_price,
    cluster_p25_price = cluster_features$cluster_p25_price,
    cluster_p75_price = cluster_features$cluster_p75_price,
    cluster_count = cluster_features$cluster_count,
    location_cluster_id = cluster_features$location_cluster_id
  ))
  
  # Convert to vector, ensure correct order
  feature_vector <- numeric(length(feature_cols))
  names(feature_vector) <- feature_cols
  
  # Calculate training data statistics (for filling missing features)
  # Use cache to avoid repeated calculations
  if (is.null(training_stats_cache) && !is.null(training_data) && nrow(training_data) > 0) {
    training_stats_cache <<- list()
    for (col in feature_cols) {
      if (col %in% colnames(training_data)) {
        # Use median (more robust)
        val <- median(training_data[[col]], na.rm = TRUE)
        if (is.na(val)) {
          val <- mean(training_data[[col]], na.rm = TRUE)
        }
        if (is.na(val)) {
          val <- 0
        }
        training_stats_cache[[col]] <<- val
      } else {
        training_stats_cache[[col]] <<- 0
      }
    }
  }
  
  # Use cached statistics or calculate new
  if (!is.null(training_stats_cache)) {
    training_stats <- training_stats_cache
  } else {
    # If no training data, use default values
    training_stats <- list()
    for (col in feature_cols) {
      training_stats[[col]] <- 0
    }
  }
  
  # Fill feature vector
  for (col in feature_cols) {
    if (col %in% names(features)) {
      feature_vector[col] <- as.numeric(features[[col]])
    } else {
      # Use mean/median from training data
      feature_vector[col] <- training_stats[[col]]
    }
  }
  
  # Return feature vector and metadata (for display)
  return(list(
    features = as.numeric(feature_vector),
    metadata = list(
      neighbourhood_id = neighbourhood_id,
      location_cluster_id = cluster_features$location_cluster_id,
      cluster_median_price = cluster_features$cluster_median_price,
      cluster_mean_price = cluster_features$cluster_mean_price,
      cluster_count = cluster_features$cluster_count
    )
  ))
}

