# Project Report: Indian Flight Fare Analysis and Prediction

**Student:** Podisetty Balaji  
**Course:** Artificial Intelligence — Mini Project 2  
**Project artifact:** `notebooks/indian_flight_fare_analysis.ipynb`  
**Executed notebook:** [Open the Google Colab notebook](https://colab.research.google.com/drive/1OxewIGefpY0ChUTlHiep4m0tcgaoK3OT)

## 1. Executive summary

This project explores historical Indian flight listings to describe fare distributions and variation across airlines, routes, booking windows, stops, duration, and travel class. It also compares an Extra Trees regression model with a simple median-fare baseline.

## 2. Problem statement and objectives

The assignment asks for a reproducible fare analysis and prediction exercise. The notebook:

- Acquires the public dataset or assignment mirror and normalizes common column names.
- Standardizes flight durations and creates directional route labels.
- Audits shape, types, missing values, category counts, duplicates, and fare validity.
- Summarizes fare distributions and variation by airline, route, booking window, stops, duration, timing, and class.
- Reports group sizes and uses route/class-aware summaries where available.
- Fits a prediction pipeline and compares held-out metrics with a median baseline.

## 3. Data source and scope

The source is the Kaggle Flight Price Prediction dataset; the assignment-provided Google Drive mirror is a fallback: [Kaggle dataset page](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction). The executed dataset contains 300,153 records and 12 columns. The observations are historical listings, not guaranteed current or future prices.

## 4. Approach and implementation

The notebook attempts the Kaggle source and falls back to the assignment mirror. It standardizes column aliases, converts duration to hours, constructs directional routes, and reports integrity checks. Exploration includes fare distributions, airline and route summaries, booking-window bins, stops, timing, duration, and travel class.

A fixed-seed train/test split feeds a pipeline with categorical encoding and numeric preprocessing for an Extra Trees regressor. Flight codes are excluded as predictive features. A training-set median fare serves as a simple baseline. The held-out set is scored using MAE, RMSE, and R².

## 5. Outcomes

- The data contains **300,153 rows and 12 columns**, with no missing values or exact duplicate rows found by the audit.
- Overall median fare: **Rs 7,425**.
- Vistara had the highest airline median fare (**Rs 15,543; n = 127,859**); AirAsia had the lowest (**Rs 3,276; n = 16,098**).
- Delhi to Mumbai median fare was **Rs 7,262 (n = 15,289)**; Mumbai to Delhi was **Rs 6,300 (n = 14,809)**.
- Median fare was **Rs 16,572** for bookings 0–3 days ahead and **Rs 5,896** for 46–60 days ahead, a 64.4% difference in this sample.
- Pooled median fares were Rs 4,499 for non-stop, Rs 7,959 for one-stop, and Rs 8,307 for two-or-more-stop flights. Overall duration/fare Spearman rho was 0.319.
- Median fares were **Rs 53,164** for Business and **Rs 5,772** for Economy.
- On the held-out split, Extra Trees achieved **MAE Rs 1,494.28, RMSE Rs 3,100.03, R² 0.98**. The median baseline scored **MAE Rs 16,155.22, RMSE Rs 26,376.38, R² -0.35**.

These numbers describe this historical sample and split. They do not guarantee future prices or model performance.

## 6. Tools and reproducibility

Python 3, Google Colab/Jupyter, pandas, NumPy, requests, Matplotlib, Seaborn, SciPy, and scikit-learn. Fixed random seeds are used for repeatable sampling and modeling steps. The notebook ran top to bottom in Colab; the linked version contains saved results. The repository notebook is a clean source copy without embedded outputs and can be rerun with **Runtime → Run all**.

## 7. Limitations and interpretation

- Historical listing prices may not reflect current availability, fees, or future fares.
- Pooled comparisons can be confounded by route, class, and booking window; stratification helps but does not eliminate every confounder.
- A random held-out split from the same dataset measures performance on similar sampled records and may not capture future market shifts.
- The model is an educational baseline, not a live booking recommendation system.

## 8. Conclusion

The sample shows substantial fare differences by airline, route direction, advance booking window, number of stops, and class. The Extra Trees model outperformed a median-fare baseline on this test split. Because the data is historical and observational, these patterns should be used to describe this dataset rather than to promise a future fare.

## 9. References

- [Kaggle Flight Price Prediction dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- Course assignment brief provided through Training Central.
