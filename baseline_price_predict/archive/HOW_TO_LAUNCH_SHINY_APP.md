# How to Launch the Shiny App

## Quick Start Guide

### Step 1: Open R or RStudio

### Step 2: Navigate to the project folder

```r
setwd("baseline_price_predict")
```

### Step 3: Launch the app

**Easiest way:**
```r
source("shiny_app/run_app.R")
```

**Alternative ways:**
```r
# Method 2
shiny::runApp("shiny_app")

# Method 3
setwd("shiny_app")
shiny::runApp("app.R")
```

### Step 4: The app will open in your browser automatically!

---

## What You'll See

- **Left Column**: Property information input form
- **Middle Column**: 
  - Baseline Price prediction
  - Occupancy Prediction (placeholder)
  - Annual Revenue (placeholder)
- **Right Column**:
  - Amenity Recommendations
  - Location Map

## First Time Setup

The `run_app.R` script will automatically:
- Install missing R packages
- Check Python/PyTorch (optional)
- Launch the app

You don't need to do anything else!

## Need Help?

Check `shiny_app/README.md` for more details.

