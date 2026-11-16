"""Exported code for Indian Flight Fare Analysis and Prediction."""

# Recommended execution environment: Google Colab.


# Indian Flight Fare Analysis and Prediction
from pathlib import Path
from zipfile import ZipFile, is_zipfile
import re, warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)

# Download and normalize the data
DATA_DIR = Path("/content/flight_price_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/shubhambathwal/flight-price-prediction"
DRIVE_ID = "1BgLVYBvAQdeG2KMqB7817lXcWGM8t2f0"

def download_data():
    """Download the public archive, with the assignment Drive mirror as fallback."""
    try:
        response = requests.get(KAGGLE_URL, timeout=120)
        if response.ok and response.content[:2] == b"PK":
            path = DATA_DIR / "flight_data.zip"
            path.write_bytes(response.content)
            with ZipFile(path) as archive:
                archive.extractall(DATA_DIR)
            return
    except requests.RequestException as exc:
        print("Kaggle endpoint unavailable:", exc)
    try:
        import gdown
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
    path = DATA_DIR / "flight_data_mirror"
    downloaded = gdown.download(id=DRIVE_ID, output=str(path), quiet=False, fuzzy=True)
    if not downloaded:
        raise RuntimeError("Dataset download failed. Open the Kaggle or Drive source link and retry.")
    if is_zipfile(downloaded):
        with ZipFile(downloaded) as archive:
            archive.extractall(DATA_DIR)
    else:
        Path(downloaded).rename(DATA_DIR / "flight_data_mirror.csv")

if not list(DATA_DIR.rglob("*.csv")):
    download_data()

def clean_header(value):
    """Standardize headers to lowercase snake_case."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

aliases = {
    "airline": ["airline", "carrier"], "flight": ["flight", "flight_code"],
    "source_city": ["source_city", "source", "origin", "from"],
    "departure_time": ["departure_time", "departure", "dep_time"],
    "stops": ["stops", "total_stops", "stop"],
    "arrival_time": ["arrival_time", "arrival", "arr_time"],
    "destination_city": ["destination_city", "destination", "dest_city", "to"],
    "class": ["class", "travel_class", "cabin_class"],
    "duration": ["duration", "flight_duration", "travel_duration"],
    "days_left": ["days_left", "daysleft", "booking_window", "days_before_departure"],
    "price": ["price", "fare", "ticket_price"],
}
alias_lookup = {alias: name for name, names in aliases.items() for alias in names}
paths = list(DATA_DIR.rglob("*.csv"))
candidates = []
for path in paths:
    try:
        sample = pd.read_csv(path, nrows=3)
        cols = [alias_lookup.get(clean_header(c), clean_header(c)) for c in sample.columns]
        if "price" in cols:
            candidates.append((path.stat().st_size, path))
    except Exception:
        pass
if not candidates:
    raise FileNotFoundError(f"No CSV with a fare column found under {DATA_DIR}")
data_path = max(candidates, key=lambda x: x[0])[1]
raw = pd.read_csv(data_path)
raw.columns = [alias_lookup.get(clean_header(c), clean_header(c)) for c in raw.columns]
for col in list(raw.columns):
    if col.startswith("unnamed") or col in {"index", "level_0"}:
        raw = raw.drop(columns=col)
required = ["airline", "source_city", "destination_city", "price"]
missing = [c for c in required if c not in raw.columns]
if missing:
    raise ValueError(f"Missing required columns {missing}; found {list(raw.columns)}")

def duration_hours(value):
    """Convert numeric hours or a string such as 2h 50m to decimal hours."""
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float, np.number)): return float(value)
    text = str(value).strip().lower()
    h, m = re.search(r"(\d+(?:\.\d+)?)\s*h", text), re.search(r"(\d+(?:\.\d+)?)\s*m", text)
    if h or m: return (float(h.group(1)) if h else 0) + (float(m.group(1)) if m else 0)/60
    try: return float(text)
    except ValueError: return np.nan

df = raw.copy()
for col in ["price", "days_left"]:
    if col in df: df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
if "duration" in df:
    df["duration_hours"] = df.pop("duration").map(duration_hours)
for col in ["airline", "flight", "source_city", "departure_time", "stops",
            "arrival_time", "destination_city", "class"]:
    if col in df: df[col] = df[col].astype("string").str.strip()
if "stops" in df:
    df["stops"] = df["stops"].astype("string").str.strip().str.lower().replace({
        "zero": "non-stop", "0": "non-stop", "nonstop": "non-stop",
        "one": "1 stop", "1": "1 stop", "two": "2 stops", "2": "2 stops"})
    df["stops"] = df["stops"].str.replace(" or more", "+", regex=False)
df["route"] = df["source_city"].astype(str) + " → " + df["destination_city"].astype(str)
print("Source file:", data_path, "| Shape:", df.shape, "| Columns:", list(df.columns))
display(df.head(10))
display(df.tail(10))

# A. Data acquisition and integrity — Questions 1–10
print(f"Rows={len(df):,}, columns={df.shape[1]}, memory={df.memory_usage(deep=True).sum()/1e6:.2f} MB")
display(df.dtypes.rename("dtype").to_frame())
display(pd.DataFrame({"unique_values": df.nunique(dropna=False),
                      "missing_count": df.isna().sum(),
                      "missing_pct": 100*df.isna().mean()}).sort_values("unique_values", ascending=False))
duplicates = int(df.duplicated().sum())
print(f"Exact duplicate rows: {duplicates:,} ({100*duplicates/len(df):.2f}%).")
if duplicates: display(df[df.duplicated(keep=False)].head(10))
print("Do not delete duplicates automatically: repeated observed listings may be valid; no booking ID is supplied.")

semantic_types = {
    "airline":"categorical", "flight":"identifier/categorical", "source_city":"categorical",
    "departure_time":"categorical time-of-day", "stops":"ordinal/categorical",
    "arrival_time":"categorical time-of-day", "destination_city":"categorical",
    "class":"categorical", "duration_hours":"numeric continuous",
    "days_left":"numeric time-to-departure offset", "price":"numeric continuous target",
    "route":"engineered categorical pair"
}
display(pd.DataFrame([{"feature":c,"storage_dtype":str(df[c].dtype),
    "analysis_type":semantic_types.get(c,"numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical")}
    for c in df.columns]))

df_optimized = df.copy()
memory_before = df_optimized.memory_usage(deep=True).sum()
for col in df_optimized.select_dtypes(include=["object", "string"]).columns:
    if df_optimized[col].nunique(dropna=False) / max(len(df_optimized), 1) < 0.5:
        df_optimized[col] = df_optimized[col].astype("category")
memory_after = df_optimized.memory_usage(deep=True).sum()
print(f"Memory after low-cardinality categorical encoding: {memory_before/1e6:.2f} MB → "
      f"{memory_after/1e6:.2f} MB ({100*(memory_before-memory_after)/memory_before:.1f}% reduction).")

for col in df.select_dtypes(include=["object", "string", "category"]).columns:
    print(col, "examples:", df[col].dropna().unique()[:15].tolist())

q1, q3 = df["price"].quantile([.25, .75])
iqr = q3-q1
fare_flags = (df["price"] < q1-1.5*iqr) | (df["price"] > q3+1.5*iqr)
print(f"Negative fares={(df.price<0).sum()}, zero fares={(df.price==0).sum()}, "
      f"IQR fare flags={fare_flags.sum():,}; 1st–99th percentile="
      f"₹{df.price.quantile(.01):,.0f}–₹{df.price.quantile(.99):,.0f}.")
meanings = {"airline":"Operating carrier", "flight":"Flight identifier",
 "source_city":"Departure city", "departure_time":"Departure time category",
 "stops":"Stop count/category", "arrival_time":"Arrival time category",
 "destination_city":"Arrival city", "class":"Travel class",
 "duration_hours":"Journey duration in hours", "days_left":"Days until journey",
 "price":"Fare in Indian rupees; prediction target", "route":"Engineered directional source-destination pair"}
display(pd.DataFrame([{"feature":c, "dtype":str(df[c].dtype),
    "meaning":meanings.get(c,"Source field; inspect values"),
    "missing_pct":100*df[c].isna().mean(), "unique_values":df[c].nunique(dropna=True)}
    for c in df.columns]))

# B. Fare distribution — Questions 11–15
price = df["price"].dropna()
summary = price.describe(percentiles=[.05,.25,.50,.75,.90,.95,.99]).to_frame("fare_rupees")
summary.loc["mode"] = price.mode().iloc[0] if not price.mode().empty else np.nan
summary.loc["variance"] = price.var()
summary.loc["skewness"] = price.skew()
summary.loc["kurtosis"] = price.kurt()
display(summary.round(2))
fig, axes = plt.subplots(1,2,figsize=(13,4))
sns.histplot(price,bins=60,kde=True,ax=axes[0]); axes[0].set_title("Fare histogram and KDE"); axes[0].set_xlabel("Price (₹)")
sns.boxplot(x=price,ax=axes[1]); axes[1].set_title("Fare spread and IQR flags")
plt.tight_layout(); plt.show()
price_band = pd.cut(df["price"],[-np.inf,3000,5000,10000,20000,np.inf],
    labels=["Under ₹3,000","₹3,000–5,000","₹5,000–10,000","₹10,000–20,000","Over ₹20,000"],
    right=False)
display(price_band.value_counts().rename("count").to_frame()
    .assign(percent=lambda x:100*x["count"]/len(df)).sort_index())

# C. Network mix and airline analysis — Questions 16–28
def frequency_table(column, n=25):
    """Return count and share for the most frequent labels in a column."""
    counts = df[column].value_counts(dropna=False).head(n)
    return pd.DataFrame({"count":counts,"percent":100*counts/len(df)})
for col in ["airline","source_city","destination_city","stops","departure_time","arrival_time","class"]:
    if col in df: print(col); display(frequency_table(col))
for col in ["duration_hours","days_left"]:
    if col in df: display(df[col].describe(percentiles=[.05,.25,.5,.75,.95]).to_frame(col))

airline = df.groupby("airline").agg(observations=("price","size"),mean_fare=("price","mean"),
    median_fare=("price","median"),min_fare=("price","min"),max_fare=("price","max"),
    fare_std=("price","std")).sort_values("median_fare",ascending=False)
display(airline.round(2))
most_variable_airline = airline.fare_std.idxmax()
print(f"Highest fare standard deviation: {most_variable_airline} "
      f"(₹{airline.loc[most_variable_airline, 'fare_std']:,.0f}; n={airline.loc[most_variable_airline, 'observations']:,}).")
fig, axes = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(data=df,x="airline",y="price",order=airline.index,showfliers=False,ax=axes[0])
axes[0].tick_params(axis="x",rotation=30); axes[0].set_title("Fare by airline")
sns.countplot(data=df,x="airline",order=df.airline.value_counts().index,ax=axes[1])
axes[1].tick_params(axis="x",rotation=30); axes[1].set_title("Listing count by airline")
plt.tight_layout(); plt.show()
rank = airline[["median_fare","mean_fare","observations"]].copy()
rank["median_rank"] = rank.median_fare.rank(ascending=False,method="min")
rank["mean_rank"] = rank.mean_fare.rank(ascending=False,method="min")
display(rank.sort_values("median_rank"))
df["below_overall_median"] = df.price < df.price.median()
below = df.groupby("airline").agg(n=("price","size"),below_median_share=("below_overall_median","mean"),
                                 median_fare=("price","median"))
below["below_median_share"] *= 100
display(below.sort_values("below_median_share",ascending=False).round(2))
display(df.groupby(["airline","class"]).agg(n=("price","size"),mean_fare=("price","mean"),
    median_fare=("price","median")).round(2))
fig, axes = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(data=df,x="airline",y="price",hue="class",showfliers=False,ax=axes[0])
axes[0].tick_params(axis="x",rotation=30); axes[0].set_title("Economy and Business by airline")
stop_airline=df.groupby(["airline","stops"]).price.median().unstack()
sns.heatmap(stop_airline,annot=True,fmt=".0f",cmap="YlGnBu",ax=axes[1])
axes[1].set_title("Median fare by airline and stops")
plt.tight_layout(); plt.show()

# D. Route and city analysis — Questions 29–38
routes=df.groupby("route").agg(observations=("price","size"),mean_fare=("price","mean"),
    median_fare=("price","median"),mean_duration=("duration_hours","mean")).sort_values("observations",ascending=False)
print("Directional routes:",df.route.nunique()); display(routes.head(25).round(2))
for origin,destination in [("Delhi","Mumbai"),("Mumbai","Delhi")]:
    part=df[(df.source_city.str.lower()==origin.lower())&(df.destination_city.str.lower()==destination.lower())]
    print(f"{origin} → {destination}: n={len(part):,}, mean=₹{part.price.mean():,.0f}, median=₹{part.price.median():,.0f}")
pivot=df.pivot_table(index="source_city",columns="destination_city",values="price",aggfunc="median")
display(pivot.round(0))
plt.figure(figsize=(10,7)); sns.heatmap(pivot,annot=True,fmt=".0f",cmap="YlGnBu",linewidths=.4)
plt.title("Median fare by source and destination"); plt.tight_layout(); plt.show()
fig,ax=plt.subplots(figsize=(9,5))
sns.scatterplot(data=routes,x="observations",y="median_fare",size="observations",sizes=(30,350),alpha=.65,ax=ax)
ax.set_xscale("log"); ax.set_title("Route listing volume vs median fare"); plt.tight_layout(); plt.show()
ra=df.groupby(["route","airline"]).price.agg(["size","median"]).reset_index()
ra=ra[ra["size"]>=20]
cheap=ra.loc[ra.groupby("route")["median"].idxmin()].merge(routes[["median_fare"]],left_on="route",right_index=True)
cheap["difference_vs_route_median"]=cheap["median"]-cheap["median_fare"]
print("Least expensive airline per route (minimum 20 observations per airline-route pair):")
display(cheap.sort_values("difference_vs_route_median").head(20))
expensive=ra.loc[ra.groupby("route")["median"].idxmax()].merge(routes[["median_fare"]],left_on="route",right_index=True)
expensive["difference_vs_route_median"]=expensive["median"]-expensive["median_fare"]
print("Most expensive airline per route (same minimum observation threshold):")
display(expensive.sort_values("difference_vs_route_median",ascending=False).head(20))
eligible_routes=routes[routes.observations>=20]
print("Highest median fares among routes with at least 20 listings:")
display(eligible_routes.nlargest(10,"median_fare")[["observations","median_fare","mean_fare","mean_duration"]].round(2))
print("Lowest median fares among routes with at least 20 listings:")
display(eligible_routes.nsmallest(10,"median_fare")[["observations","median_fare","mean_fare","mean_duration"]].round(2))
route_stops=df.groupby(["route","stops"]).agg(n=("price","size"),median_fare=("price","median")).reset_index()
display(route_stops[route_stops.n>=20].pivot(index="route",columns="stops",values="median_fare").head(20).round(0))
print("Route listing counts are not a complete schedule or seat-inventory measure.")

# E. Booking window and days left — Questions 39–46
if "days_left" in df:
    fig,ax=plt.subplots(figsize=(8,4))
    sns.histplot(data=df,x="days_left",bins=30,ax=ax)
    ax.set_title("Distribution of days left at booking")
    ax.set_xlabel("Days before departure")
    plt.tight_layout(); plt.show()
    sample=df.sample(min(10000,len(df)),random_state=42)
    fig,ax=plt.subplots(figsize=(9,5))
    sns.regplot(data=sample,x="days_left",y="price",lowess=True,
        scatter_kws={"alpha":.18,"s":12},line_kws={"color":"crimson"},ax=ax)
    ax.set_title("Fare versus days left with LOWESS trend"); plt.tight_layout(); plt.show()
    day_summary=df.groupby("days_left").price.agg(["size","mean","median"])
    display(day_summary.head(15)); display(day_summary.tail(15))
    bins=[-np.inf,3,7,15,30,45,60,np.inf]
    labels=["0–3 days","4–7 days","8–15 days","16–30 days","31–45 days","46–60 days","More than 60 days"]
    df["booking_window"]=pd.cut(df.days_left,bins=bins,labels=labels,right=True,include_lowest=True)
    window=df.groupby("booking_window",observed=True).price.agg(["size","mean","median"])
    display(window.round(2))
    print("Highest median:",window["median"].idxmax(),"Lowest median:",window["median"].idxmin())
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    sns.lineplot(data=df,x="days_left",y="price",hue="class",estimator="median",errorbar=None,ax=axes[0])
    axes[0].set_title("Booking trend by class")
    sns.lineplot(data=df,x="days_left",y="price",hue="airline",estimator="median",errorbar=None,ax=axes[1])
    axes[1].set_title("Booking trend by airline")
    plt.tight_layout(); plt.show()
    by_stops=df.groupby(["booking_window","stops"],observed=True).price.median().unstack()
    display(by_stops.round(0))
    top_routes=routes.head(5).index
    by_route=df[df.route.isin(top_routes)].groupby(["route","booking_window"],observed=True).price.median().unstack()
    display(by_route.round(0))
    med=window["median"].dropna()
    if len(med)>=2:
        change=100*(med.iloc[-1]-med.iloc[0])/med.iloc[0]
        print(f"Median change from {med.index[0]} to {med.index[-1]}: {change:.1f}%")
else:
    print("No days_left feature: booking-window questions are unavailable from this dataset.")

# F. Duration, stops, time of day, and class — Questions 47–54
if "duration_hours" in df:
    display(pd.DataFrame({"pearson":[df.duration_hours.corr(df.price,method="pearson")],
                          "spearman":[df.duration_hours.corr(df.price,method="spearman")]},
                         index=["duration vs price"]))
    sample=df.sample(min(12000,len(df)),random_state=7)
    sns.scatterplot(data=sample,x="duration_hours",y="price",hue="class",alpha=.25,s=16)
    plt.title("Duration versus fare"); plt.tight_layout(); plt.show()
    display(df.groupby("stops").agg(n=("price","size"),median_duration=("duration_hours","median"),
        mean_duration=("duration_hours","mean"),median_fare=("price","median")).sort_values("median_duration").round(2))
    route_airline_duration=df.groupby(["route","airline"]).agg(n=("price","size"),
        median_duration=("duration_hours","median"),median_fare=("price","median")).reset_index()
    route_airline_duration=route_airline_duration[route_airline_duration.n>=20]
    fastest=route_airline_duration.loc[route_airline_duration.groupby("route").median_duration.idxmin()]
    display(fastest.sort_values("n",ascending=False).head(20).round(2))
    route_class=df.groupby(["route","class"]).agg(n=("price","size"),
        median_duration=("duration_hours","median"),median_fare=("price","median")).reset_index()
    display(route_class[route_class.n>=30].head(20).round(2))
    within_route_class=df.groupby(["route","class"]).apply(
        lambda g: pd.Series({"n":len(g),
            "duration_price_spearman":g["duration_hours"].corr(g["price"],method="spearman"),
            "median_duration":g["duration_hours"].median(),"median_fare":g["price"].median()}),
        include_groups=False).reset_index()
    within_route_class=within_route_class[within_route_class.n>=30].dropna(subset=["duration_price_spearman"])
    print("Duration-versus-fare association within the same route and travel class (at least 30 listings):")
    display(within_route_class.reindex(within_route_class.duration_price_spearman.abs()
        .sort_values(ascending=False).index).head(20).round(3))
    fig,ax=plt.subplots(figsize=(10,5))
    sns.boxplot(data=df,x="stops",y="duration_hours",showfliers=False,ax=ax)
    ax.set_title("Journey duration by number of stops")
    plt.tight_layout(); plt.show()
for col in ["departure_time","arrival_time"]:
    if col in df:
        timing=df.groupby(col).agg(n=("price","size"),mean_fare=("price","mean"),
            median_fare=("price","median"),median_duration=("duration_hours","median")).sort_values("median_fare")
        print(col); display(timing.round(2))
        sns.barplot(data=timing.reset_index(),x=col,y="median_fare",color="steelblue")
        plt.title("Median fare by "+col); plt.xticks(rotation=25); plt.tight_layout(); plt.show()
display(df.groupby("class").agg(n=("price","size"),
    median_duration=("duration_hours","median"),median_fare=("price","median")).round(2))
class_correlations=df.groupby("class").apply(lambda g: pd.Series({
    "n":len(g),"duration_price_pearson":g["duration_hours"].corr(g["price"],method="pearson"),
    "duration_price_spearman":g["duration_hours"].corr(g["price"],method="spearman")
}),include_groups=False)
print("Duration/price correlations calculated separately within each travel class:")
display(class_correlations.round(3))

# Regression baseline — fare prediction
cat_features=[c for c in ["airline","source_city","destination_city","departure_time",
    "arrival_time","stops","class"] if c in df]
num_features=[c for c in ["duration_hours","days_left"] if c in df]
model_data=df[cat_features+num_features+["price"]].dropna()
X=model_data[cat_features+num_features]; y=model_data["price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
if len(X_train)>60000:
    idx=X_train.sample(60000,random_state=42).index
    X_fit,y_fit=X_train.loc[idx],y_train.loc[idx]
else:
    X_fit,y_fit=X_train,y_train
transformers=[]
if cat_features: transformers.append(("categorical",OneHotEncoder(handle_unknown="ignore"),cat_features))
if num_features: transformers.append(("numeric",StandardScaler(),num_features))
prep=ColumnTransformer(transformers=transformers)
model=Pipeline([("preprocess",prep),("regressor",ExtraTreesRegressor(
    n_estimators=100,max_depth=24,min_samples_leaf=2,max_features=.9,n_jobs=-1,random_state=42))])
model.fit(X_fit,y_fit); prediction=model.predict(X_test)
baseline=np.repeat(y_fit.median(),len(y_test))
metrics=pd.DataFrame([
    {"model":"Median baseline","MAE":mean_absolute_error(y_test,baseline),
     "RMSE":mean_squared_error(y_test,baseline)**.5,"R2":r2_score(y_test,baseline)},
    {"model":"Extra Trees","MAE":mean_absolute_error(y_test,prediction),
     "RMSE":mean_squared_error(y_test,prediction)**.5,"R2":r2_score(y_test,prediction)}])
display(metrics.round(2))
idx=np.random.RandomState(42).choice(len(y_test),min(5000,len(y_test)),replace=False)
plt.figure(figsize=(7,6)); plt.scatter(y_test.iloc[idx],prediction[idx],alpha=.25,s=15)
lo=min(y_test.min(),prediction.min()); hi=max(y_test.max(),prediction.max())
plt.plot([lo,hi],[lo,hi],color="crimson",linestyle="--")
plt.xlabel("Observed fare (₹)"); plt.ylabel("Predicted fare (₹)")
plt.title("Observed versus predicted test fares"); plt.tight_layout(); plt.show()
