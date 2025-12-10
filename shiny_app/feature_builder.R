# =============================================
# Feature Builder - 构建模型输入特征向量
# =============================================

library(dplyr)
library(geosphere)

# 全局变量存储训练数据统计（避免重复加载）
training_data_cache <- NULL
training_stats_cache <- NULL
cluster_centers_cache <- NULL  # 缓存cluster中心点
neighbourhood_map_cache <- NULL  # 缓存neighbourhood映射

# 加载训练数据以获取特征列和默认值
load_training_sample <- function() {
  # 如果已经加载过，直接返回
  if (!is.null(training_data_cache)) {
    return(training_data_cache)
  }
  
  # Try multiple possible paths (from root directory)
  possible_paths <- c(
    file.path(getwd(), "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv"),
    file.path(dirname(getwd()), "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv"),
    file.path(getwd(), "..", "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv"),
    file.path(getwd(), "baseprice_model", "nn_price_training_v4.csv")  # Fallback
  )
  
  for (training_file in possible_paths) {
    if (file.exists(training_file)) {
      tryCatch({
        # 读取更多数据以获得更准确的统计值（至少1000行）
        data <- read.csv(training_file, nrows = 1000)
        training_data_cache <<- data  # 缓存数据
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

# 获取最近的房源用于填充缺失特征
get_nearest_listing_features <- function(lat, lon, training_data) {
  if (is.null(training_data) || nrow(training_data) == 0) {
    return(NULL)
  }
  
  # 计算距离
  distances <- distHaversine(
    cbind(training_data$longitude, training_data$latitude),
    c(lon, lat)
  ) / 1000  # 转换为公里
  
  nearest_idx <- which.min(distances)
  return(training_data[nearest_idx, ])
}

# 计算cluster中心点（用于根据经纬度找到对应的cluster）
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

# 根据经纬度找到对应的cluster_id（通过最近的cluster中心）
find_cluster_by_location <- function(lat, lon, training_data) {
  cluster_centers <- calculate_cluster_centers(training_data)
  
  if (is.null(cluster_centers) || nrow(cluster_centers) == 0) {
    # 如果没有cluster中心，使用最近的房源
    nearest <- get_nearest_listing_features(lat, lon, training_data)
    if (!is.null(nearest) && "location_cluster_id" %in% colnames(nearest)) {
      return(as.integer(nearest$location_cluster_id))
    }
    return(0)
  }
  
  # 计算到每个cluster中心的距离
  distances <- distHaversine(
    cbind(cluster_centers$center_lon, cluster_centers$center_lat),
    c(lon, lat)
  ) / 1000  # 转换为公里
  
  # 找到最近的cluster
  nearest_cluster_idx <- which.min(distances)
  cluster_id <- cluster_centers$location_cluster_id[nearest_cluster_idx]
  
  cat(sprintf("Location (%.4f, %.4f) assigned to cluster_id: %d (distance: %.2f km)\n", 
              lat, lon, cluster_id, distances[nearest_cluster_idx]))
  
  return(as.integer(cluster_id))
}

# 计算位置聚类特征
calculate_cluster_features <- function(lat, lon, training_data) {
  if (is.null(training_data) || nrow(training_data) == 0) {
    # 返回默认值
    return(list(
      location_cluster_id = 0,
      cluster_median_price = 150,
      cluster_mean_price = 150,
      cluster_p25_price = 120,
      cluster_p75_price = 180,
      cluster_count = 100
    ))
  }
  
  # 根据经纬度找到对应的cluster_id
  cluster_id <- find_cluster_by_location(lat, lon, training_data)
  
  # 计算该聚类的价格统计
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
  
  # 使用全局统计
  return(list(
    location_cluster_id = cluster_id,
    cluster_median_price = median(training_data$price_num, na.rm = TRUE),
    cluster_mean_price = mean(training_data$price_num, na.rm = TRUE),
    cluster_p25_price = quantile(training_data$price_num, 0.25, na.rm = TRUE),
    cluster_p75_price = quantile(training_data$price_num, 0.75, na.rm = TRUE),
    cluster_count = nrow(training_data)
  ))
}

# 构建特征向量
build_features <- function(lat, lon, bedrooms, bathrooms, accommodates, beds,
                          room_type, amenities = c()) {
  
  # 加载训练数据样本
  training_data <- load_training_sample()
  
  # 获取特征列顺序
  if (!is.null(training_data)) {
    feature_cols <- setdiff(colnames(training_data), "price_num")
  } else {
    # 使用默认特征列（从之前的分析）
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
  
  # 计算卫生间相关特征
  bath_num <- ifelse(bathrooms > 0, round(bathrooms), 1)
  bath_shared <- 0  # 默认不共享
  bath_type_unknown <- ifelse(bathrooms == 0, 1, 0)
  
  # 房型ID映射
  room_type_map <- c(
    "Entire home/apt" = 0,
    "Private room" = 1,
    "Shared room" = 2
  )
  room_type_id <- ifelse(room_type %in% names(room_type_map), 
                        room_type_map[room_type], 0)
  
  # 计算位置聚类特征
  cluster_features <- calculate_cluster_features(lat, lon, training_data)
  
  # 获取neighbourhood_id（使用最近的房源）
  nearest <- get_nearest_listing_features(lat, lon, training_data)
  neighbourhood_id <- if (!is.null(nearest) && "neighbourhood_id" %in% colnames(nearest)) {
    nid <- as.integer(nearest$neighbourhood_id)
    cat(sprintf("Location (%.4f, %.4f) assigned to neighbourhood_id: %d (from nearest listing)\n", 
                lat, lon, nid))
    nid
  } else {
    # 如果没有训练数据，使用最常见的neighbourhood_id（通常是0）
    if (!is.null(training_data) && "neighbourhood_id" %in% colnames(training_data)) {
      # 使用训练数据中最常见的neighbourhood_id
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
  
  # 构建amenity特征（one-hot编码）
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
    # 检查用户选择的设施
    if (amenity %in% amenities || 
        (amenity == "Wifi" && "Wifi" %in% amenities) ||
        (amenity == "Kitchen" && "Kitchen" %in% amenities) ||
        (amenity == "Heating" && "Heating" %in% amenities)) {
      amenity_features[[amenity_key]] <- 1
    } else {
      amenity_features[[amenity_key]] <- 0
    }
  }
  
  # 构建完整特征向量
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
  
  # 添加amenity特征
  features <- c(features, amenity_features)
  
  # 添加ID和聚类特征
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
  
  # 转换为向量，确保顺序正确
  feature_vector <- numeric(length(feature_cols))
  names(feature_vector) <- feature_cols
  
  # 计算训练数据的统计值（用于填充缺失特征）
  # 使用缓存避免重复计算
  if (is.null(training_stats_cache) && !is.null(training_data) && nrow(training_data) > 0) {
    training_stats_cache <<- list()
    for (col in feature_cols) {
      if (col %in% colnames(training_data)) {
        # 使用中位数（更稳健）
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
  
  # 使用缓存的统计值或计算新的
  if (!is.null(training_stats_cache)) {
    training_stats <- training_stats_cache
  } else {
    # 如果没有训练数据，使用默认值
    training_stats <- list()
    for (col in feature_cols) {
      training_stats[[col]] <- 0
    }
  }
  
  # 填充特征向量
  for (col in feature_cols) {
    if (col %in% names(features)) {
      feature_vector[col] <- as.numeric(features[[col]])
    } else {
      # 使用训练数据的平均值/中位数
      feature_vector[col] <- training_stats[[col]]
    }
  }
  
  # 返回特征向量和元数据（用于显示）
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

