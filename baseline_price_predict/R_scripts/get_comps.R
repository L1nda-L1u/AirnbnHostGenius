library(dplyr)
library(geosphere)

get_comps <- function(df, 
                      lat, lon, 
                      room_type, 
                      accommodates, 
                      bedrooms, 
                      bath_num, 
                      radius_km = 1) {
  
  df$dist_m <- distHaversine(
    cbind(df$longitude, df$latitude),
    c(lon, lat)
  )
  df$dist_km <- df$dist_m / 1000
  
  df_filtered <- df %>%
    filter(dist_km <= radius_km) %>%
    filter(room_type == !!room_type) %>%
    filter(accommodates >= accommodates - 1,
           accommodates <= accommodates + 1) %>%
    filter(bedrooms >= bedrooms - 1,
           bedrooms <= bedrooms + 1) %>%
    filter(bath_num >= bath_num - 1,
           bath_num <= bath_num + 1)
  
  n_comps <- nrow(df_filtered)
  median_p <- median(df_filtered$price_num, na.rm = TRUE)
  p25 <- quantile(df_filtered$price_num, 0.25, na.rm = TRUE)
  p75 <- quantile(df_filtered$price_num, 0.75, na.rm = TRUE)
  
  return(list(
    comps_table = df_filtered,
    n_comps = n_comps,
    local_median_price = median_p,
    local_p25 = p25,
    local_p75 = p75
  ))
}
