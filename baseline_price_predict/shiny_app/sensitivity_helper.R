# =============================================
# Sensitivity Analysis Helper for Shiny App
# =============================================

# Get amenity feature names
get_amenity_features <- function(feature_cols) {
  amenity_cols <- grep("^amenity_", feature_cols, value = TRUE)
  return(amenity_cols)
}

# Recommend amenities based on current features
recommend_amenities_for_shiny <- function(feature_vector, feature_cols, predict_func, top_n = 3) {
  
  # Convert feature vector to named list
  features_list <- as.list(feature_vector)
  names(features_list) <- feature_cols
  
  # Get amenity features
  amenity_cols <- get_amenity_features(feature_cols)
  
  # Find missing amenities (value = 0)
  missing_amenities <- c()
  for (amenity in amenity_cols) {
    if (amenity %in% names(features_list)) {
      if (as.numeric(features_list[[amenity]]) == 0) {
        missing_amenities <- c(missing_amenities, amenity)
      }
    }
  }
  
  if (length(missing_amenities) == 0) {
    return(data.frame())
  }
  
  # Get base price
  base_price <- predict_func(feature_vector)
  
  # Test each missing amenity
  amenity_impacts <- data.frame()
  
  for (amenity in missing_amenities) {
    # Create test feature vector with this amenity set to 1
    test_features <- feature_vector
    amenity_idx <- which(feature_cols == amenity)
    if (length(amenity_idx) > 0) {
      test_features[amenity_idx] <- 1
      
      # Predict price with this amenity
      new_price <- predict_func(test_features)
      
      # Calculate impact
      impact <- new_price - base_price
      impact_pct <- (impact / base_price) * 100
      
      if (impact > 0) {  # Only keep positive impacts
        amenity_name <- gsub("amenity_", "", amenity)
        amenity_name <- gsub("\\.", " ", amenity_name)
        amenity_name <- tools::toTitleCase(amenity_name)
        
        amenity_impacts <- rbind(amenity_impacts, data.frame(
          amenity = amenity,
          amenity_name = amenity_name,
          price_impact = impact,
          price_impact_pct = impact_pct,
          new_price = new_price,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  
  if (nrow(amenity_impacts) == 0) {
    return(data.frame())
  }
  
  # Sort by impact (descending) and return top N
  amenity_impacts <- amenity_impacts[order(-amenity_impacts$price_impact), ]
  top_recommendations <- head(amenity_impacts, min(top_n, nrow(amenity_impacts)))
  
  return(list(
    base_price = base_price,
    recommendations = top_recommendations
  ))
}

