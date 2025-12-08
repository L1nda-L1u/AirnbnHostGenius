# 📊 Component Prediction Pipeline

## 🎯 Purpose

Clean and predict REAL components for Airbnb pricing:
- **TfL Transport** (daily journeys in millions) - REAL data
- **Tourism** (quarterly visitor counts in thousands) - REAL data
- **Weather** (temperature, precipitation, quality) - REAL data
- **UK Holidays** (bank holidays, major events) - REAL data

**Note**: No synthetic `foot_traffic_score` - only real, observable components.

---

## 🚀 Quick Start

### ⭐ Recommended: Run Complete Pipeline
```r
# Navigate to the project folder first, then run:
source("00_run_all.R")
```

**This will**:
1. Clean all 4 data sources
2. Merge into daily framework
3. Predict future components (TfL, Tourism, Weather)
4. Create `foot_traffic_daily.csv` with REAL components only

---

### Alternative: Run Step by Step

```r
# Step 1: Clean individual sources
source("01a_clean_tfl.R")       # TfL transport
source("01b_clean_tourism.R")   # Tourism  
source("01c_clean_weather.R")   # Weather
source("01d_clean_holidays.R")  # Holidays

# Step 2: Merge to daily framework
source("02_merge_to_daily.R")
```

---

## 📁 Structure

```
foot_traffic_prediction/
├── 00_run_cleaning.R           ← Run all scripts
├── 01a_clean_tfl.R            ← TfL cleaning
├── 01b_clean_tourism.R        ← Tourism cleaning
├── 01c_clean_weather.R        ← Weather cleaning
├── 01d_clean_holidays.R       ← Holidays cleaning
│
└── foot_traffic_data/
    ├── raw/                    ← Your downloaded data
    │   ├── tfl/tfl-journeys-type.csv
    │   ├── tourism/international-visitors-london-raw.csv
    │   ├── weather/london_weather.csv
    │   └── events/uk_holidays.json
    │
    └── cleaned/                ← Output (4 files only)
        ├── tfl_monthly.csv
        ├── tourism_quarterly.csv
        ├── weather_monthly.csv
        └── holidays_monthly.csv
```

---

## 📊 Output Files

### Individual Source Files (4 files):

| Script | Output File | Granularity | Content |
|--------|-------------|-------------|---------|
| `01a_clean_tfl.R` | `tfl_monthly.csv` | Monthly | Transport journeys |
| `01b_clean_tourism.R` | `tourism_quarterly.csv` | Quarterly | Visitor statistics |
| `01c_clean_weather.R` | `weather_daily.csv` | **Daily** | Weather data |
| `01d_clean_holidays.R` | `holidays_daily.csv` | **Daily** | Holiday flags |

### ⭐ Main Output (1 file):

| Script | Output File | Content |
|--------|-------------|---------|
| `02_merge_to_daily.R` | **`foot_traffic_daily.csv`** | **Daily framework with all data merged** |

**This is the file you'll use for modeling!** 

Contains:
- Every day from 2019-2024 (~2,000 rows)
- **TfL daily journeys** (millions) - REAL data
- **Tourism quarterly visits** (thousands) - REAL data
- **Weather** (temperature, precipitation, quality) - REAL data
- **Holidays** (flags and weights) - REAL data
- **Normalized indices** (0-1) for each component (for convenience)

---

## 📈 Data Coverage

| Dataset | Time Period | Records |
|---------|-------------|---------|
| **TfL** | 2010-2024 | ~170 months |
| **Tourism** | 2002-2020 | ~75 quarters |
| **Weather** | 2019-2024 | ~60 months |
| **Holidays** | 2024-2027 | ~48 months |

---

## ⚙️ Requirements

```r
install.packages(c("tidyverse", "data.table", "lubridate", "jsonlite"))
```

---

## ✅ Success Check

After running, verify:
```r
list.files("foot_traffic_data/cleaned/")
# Should show 5 files:
# [1] "foot_traffic_daily.csv"      ← MAIN OUTPUT
# [2] "holidays_daily.csv"
# [3] "tfl_monthly.csv"
# [4] "tourism_quarterly.csv"  
# [5] "weather_daily.csv"

# Check the main output
daily_data <- fread("foot_traffic_data/cleaned/foot_traffic_daily.csv")
nrow(daily_data)  # Should be ~2,000 days
head(daily_data)
```

---

## 🚀 Next Steps

After cleaning:
1. **Inspect the daily data**:
   ```r
   library(data.table)
   ft_daily <- fread("foot_traffic_data/cleaned/foot_traffic_daily.csv")
   summary(ft_daily)
   ```

2. **Visualize component patterns**:
   ```r
   library(ggplot2)
   ggplot(ft_daily, aes(x=as.Date(date), y=tfl_daily_avg_m)) +
     geom_line() + labs(title="TfL Daily Journeys")
   ```

3. **Use components for Airbnb pricing**:
   - Access component predictions via `get_components()` function
   - Create your own pricing adjustment logic based on REAL components
   - No synthetic scores - only observable data

---

## 📝 Notes

### Data Granularity Strategy:
- **Daily data**: Weather, Holidays (exact values each day)
- **Monthly average**: TfL transport (each day gets its month's average)
- **Quarterly average**: Tourism (each day gets its quarter's average)
- **Normalized indices**: Each component has its own 0-1 index (for convenience)

### Why Mixed Granularity?
✅ Captures daily variations (weather, weekends, holidays)  
✅ Preserves monthly/seasonal trends (transport, tourism)  
✅ Perfect for daily Airbnb pricing predictions  
✅ Avoids false precision from interpolation

### Data Coverage:
- **Weather**: 2019-2024 (complete daily coverage) ✅
- **TfL**: 2010-2024 (monthly averages applied to days) ✅
- **Tourism**: 2002-2019 (quarterly averages, ~72 quarters) ⚠️
- **Holidays**: 2012-2027 (complete UK bank holidays) ✅

---

**Last Updated**: November 2024  
**Status**: ✅ Production Ready

