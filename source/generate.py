"""
Generate the synthetic teaching datasets for the dataset repository site.

Every dataset is produced twice:
  data/<id>.csv        raw version with realistic, documented data-quality problems
  data/<id>_clean.csv  reference version (ground truth before problems were injected)
and described in catalog.json (variables, issues, suggested exercises, stats).

All people, places, products and organizations are fictional.
Run:  python generate.py
"""
import json, os, string
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site", "data")
os.makedirs(OUT, exist_ok=True)

CATALOG = []

# ----------------------------------------------------------------- helpers
def fake_names(rng, n, parts=("Ka", "Lo", "Mi", "Ra", "Te", "Su", "No", "Vi", "Da", "Ze",
                              "Ba", "Fe", "Jo", "Ni", "Pa", "Sa", "To", "Ye", "Ma", "Li"),
               tails=("ra", "lo", "ne", "son", "vin", "ka", "dri", "mo", "ley", "tan")):
    return [rng.choice(parts) + rng.choice(parts).lower() + rng.choice(tails) for _ in range(n)]

def person_names(rng, n):
    first = ["Amara", "Jalen", "Sofia", "Kofi", "Maya", "Diego", "Aisha", "Noah", "Zara", "Malik",
             "Lena", "Omar", "Imani", "Lucas", "Nia", "Ravi", "Chloe", "Tariq", "Elena", "Kai",
             "Ayo", "Mei", "Andre", "Priya", "Jonah", "Fatou", "Leo", "Tess", "Idris", "Rosa"]
    last = ["Okafor", "Bennett", "Ramos", "Mensah", "Chen", "Diallo", "Park", "Silva", "Adeyemi",
            "Novak", "Haddad", "Brooks", "Kamara", "Ortiz", "Nguyen", "Sow", "Patel", "Grant",
            "Ibrahim", "Moreau", "Tanaka", "Owusu", "Reyes", "Hale", "Toure", "Kim", "Duarte"]
    return [f"{rng.choice(first)} {rng.choice(last)}" for _ in range(n)]

def blank(df, col, frac, rng):
    idx = df.sample(frac=frac, random_state=int(rng.integers(1e9))).index
    df.loc[idx, col] = np.nan
    return idx

def dupes(df, n, rng):
    extra = df.sample(n=n, random_state=int(rng.integers(1e9)))
    return pd.concat([df, extra]).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)

def variants(series, mapping, frac, rng):
    """Replace a fraction of values with messy spellings from mapping {clean: [variants]}."""
    s = series.astype(object).copy()
    mask = rng.random(len(s)) < frac
    for i in np.where(mask)[0]:
        v = s.iat[i]
        if v in mapping:
            s.iat[i] = rng.choice(mapping[v])
    return s

def mixed_dates(dt_series, rng, fmts=("%Y-%m-%d %H:%M", "%m/%d/%Y %I:%M %p", "%d %b %Y %H:%M"),
                probs=(0.6, 0.3, 0.1)):
    out = []
    for d in dt_series:
        if pd.isna(d):
            out.append(np.nan); continue
        out.append(d.strftime(fmts[rng.choice(len(fmts), p=probs)]))
    return out

def infer_type(s):
    if pd.api.types.is_bool_dtype(s):
        return "Binary"
    if pd.api.types.is_integer_dtype(s):
        return "Integer"
    if pd.api.types.is_float_dtype(s):
        return "Continuous"
    return "Categorical"

def describe(raw):
    out = {}
    for c in raw.columns:
        s = raw[c]
        d = {"missing": int(s.isna().sum())}
        num = pd.to_numeric(s, errors="coerce")
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            d.update(min=float(num.min()), max=float(num.max()), mean=round(float(num.mean()), 3))
        else:
            vc = s.astype(str)[s.notna()].value_counts()
            d.update(unique=int(s.nunique()), top=[[str(k), int(v)] for k, v in vc.head(4).items()])
        out[c] = d
    return out

def register(id, name, raw, clean, *, summary, description, domain, tasks, level,
             characteristics, variables, issues, exercises, target=None):
    raw.to_csv(os.path.join(OUT, f"{id}.csv"), index=False)
    clean.to_csv(os.path.join(OUT, f"{id}_clean.csv"), index=False)
    stats = describe(raw)
    names = [v[0] for v in variables]
    assert names == list(raw.columns), (id, set(names) ^ set(raw.columns))
    preview = raw.head(8).astype(object).where(raw.head(8).notna(), None).values.tolist()
    preview = [[(round(x, 4) if isinstance(x, float) else x) for x in row] for row in preview]
    CATALOG.append(dict(
        id=id, name=name, summary=summary, description=description, domain=domain,
        tasks=tasks, level=level, characteristics=characteristics, target=target,
        rows=int(len(raw)), cols=int(raw.shape[1]), clean_rows=int(len(clean)),
        missing_pct=round(float(raw.isna().mean().mean() * 100), 1),
        size_kb=round(os.path.getsize(os.path.join(OUT, f"{id}.csv")) / 1024, 1),
        clean_size_kb=round(os.path.getsize(os.path.join(OUT, f"{id}_clean.csv")) / 1024, 1),
        variables=[dict(name=n, role=r, type=t, description=d, units=u, **stats[n])
                   for n, r, t, d, u in variables],
        issues=issues, exercises=exercises,
        columns=list(raw.columns), preview=preview,
    ))
    print(f"{id:28s} raw {raw.shape}  clean {clean.shape}")

# ======================================================================
# 1. Student success
# ======================================================================
def student_success():
    rng = np.random.default_rng(101); n = 1200
    majors = ["Data Science", "Computer Science", "Statistics", "Mathematics", "Biology", "Economics"]
    df = pd.DataFrame({
        "student_id": [f"S{10000+i}" for i in range(n)],
        "major": rng.choice(majors, n, p=[.28, .22, .12, .1, .15, .13]),
        "class_year": rng.choice(["Freshman", "Sophomore", "Junior", "Senior"], n),
        "age": rng.integers(17, 30, n),
        "study_hours_week": rng.gamma(3, 4, n).round(1),
        "sleep_hours": rng.normal(6.8, 1.1, n).clip(3, 10).round(1),
        "attendance_pct": rng.beta(8, 1.6, n).round(3) * 100,
        "part_time_job": rng.choice(["Yes", "No"], n, p=[.4, .6]),
        "commute_minutes": rng.exponential(18, n).round(0),
        "prior_gpa": rng.normal(3.0, .45, n).clip(1.2, 4.0).round(2),
    })
    df["attendance_pct"] = df["attendance_pct"].round(1)
    z = (0.9 * df.study_hours_week + 0.25 * df.attendance_pct + 9 * df.prior_gpa
         + 2 * (df.sleep_hours - 7) - 4 * (df.part_time_job == "Yes") + rng.normal(0, 6, n))
    df["midterm_score"] = (15 + z * 0.85 + rng.normal(0, 5, n)).clip(20, 100).round(1)
    df["final_score"] = (14 + z * 0.9 + rng.normal(0, 6, n)).clip(15, 100).round(1)
    df["passed"] = np.where(0.4 * df.midterm_score + 0.6 * df.final_score >= 60, "Yes", "No")
    clean = df.copy()
    raw = df.copy()
    raw["major"] = variants(raw["major"], {
        "Data Science": ["data science", "Data Sci", "DS", " Data Science"],
        "Computer Science": ["Comp Sci", "computer science", "CS"],
        "Statistics": ["Stats", "statistics ", "STAT"],
        "Mathematics": ["Math", "mathematics"], "Biology": ["Bio", "biology"],
        "Economics": ["Econ", "economics"]}, .15, rng)
    blank(raw, "sleep_hours", .08, rng); blank(raw, "commute_minutes", .05, rng)
    blank(raw, "prior_gpa", .04, rng)
    i = rng.choice(n, 12, replace=False); raw.loc[i, "attendance_pct"] = raw.loc[i, "attendance_pct"] * 10
    i = rng.choice(n, 6, replace=False); raw.loc[i, "age"] = raw.loc[i, "age"] * 10
    i = rng.choice(n, 5, replace=False); raw.loc[i, "study_hours_week"] = -raw.loc[i, "study_hours_week"]
    raw["passed"] = variants(raw["passed"], {"Yes": ["Y", "yes", "1", "TRUE"], "No": ["N", "no", "0", "FALSE"]}, .12, rng)
    raw = dupes(raw, 15, rng)
    register("student-success", "Student Success & Study Habits", raw, clean,
        summary="Habits, attendance and grades of 1,200 university students.",
        description="Records for 1,200 fictional undergraduates: their major, study and sleep habits, "
                    "attendance, outside work, prior GPA and exam scores. The target is whether the student "
                    "passed the course. A good first dataset: small, familiar, and messy in typical ways.",
        domain="Education", tasks=["Cleaning", "Classification", "Visualization"], level="Beginner",
        characteristics=["Tabular", "Multivariate"], target="passed",
        variables=[
            ("student_id", "ID", "Categorical", "Student identifier", ""),
            ("major", "Feature", "Categorical", "Declared major", ""),
            ("class_year", "Feature", "Categorical", "Freshman, Sophomore, Junior or Senior", ""),
            ("age", "Feature", "Integer", "Age of the student", "years"),
            ("study_hours_week", "Feature", "Continuous", "Self-reported study time per week", "hours"),
            ("sleep_hours", "Feature", "Continuous", "Average sleep per night", "hours"),
            ("attendance_pct", "Feature", "Continuous", "Share of classes attended", "%"),
            ("part_time_job", "Feature", "Binary", "Works a part-time job (Yes/No)", ""),
            ("commute_minutes", "Feature", "Continuous", "One-way commute time", "minutes"),
            ("prior_gpa", "Feature", "Continuous", "GPA before this semester (0-4 scale)", "points"),
            ("midterm_score", "Feature", "Continuous", "Midterm exam score", "points (0-100)"),
            ("final_score", "Feature", "Continuous", "Final exam score", "points (0-100)"),
            ("passed", "Target", "Binary", "Passed the course (weighted score of 60 or more)", ""),
        ],
        issues=["Inconsistent spellings of majors (e.g. 'Data Sci', 'DS', ' Data Science')",
                "Missing values in sleep_hours, commute_minutes and prior_gpa",
                "Attendance values above 100% (decimal-point typos)",
                "A few ages entered with an extra zero (e.g. 190)",
                "Negative study hours", "passed coded several ways (Yes, Y, 1, TRUE ...)",
                "15 duplicated rows"],
        exercises=["Standardize the major names and count students per major",
                   "Find and fix impossible values; justify each fix",
                   "Compare final scores across majors with a box plot",
                   "Scale the numeric features and train a pass/fail classifier"])

# ======================================================================
# 2. Riverton 311
# ======================================================================
def riverton_311():
    rng = np.random.default_rng(202); n = 5000
    types = ["Noise", "Pothole", "Streetlight Out", "Illegal Parking", "Graffiti",
             "Missed Trash Pickup", "Water Leak", "Rodents"]
    p = [.22, .14, .1, .16, .07, .13, .08, .1]
    created = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.uniform(0, 365, n), unit="D")
    created = created.floor("min")
    ctype = rng.choice(types, n, p=p)
    base = {"Noise": 1, "Pothole": 9, "Streetlight Out": 6, "Illegal Parking": .5, "Graffiti": 12,
            "Missed Trash Pickup": 2, "Water Leak": 3, "Rodents": 14}
    days = np.array([rng.gamma(1.5, base[t] / 1.5) for t in ctype])
    closed = created + pd.to_timedelta(days, unit="D")
    status = np.where(rng.random(n) < .9, "Closed", rng.choice(["Open", "In Progress"], n))
    closed = pd.Series(closed).where(status == "Closed", pd.NaT).dt.floor("min")
    district = rng.choice(["North", "South", "East", "West", "Downtown"], n, p=[.2, .2, .18, .17, .25])
    centers = {"North": (40.78, -73.95), "South": (40.70, -73.96), "East": (40.74, -73.90),
               "West": (40.74, -74.01), "Downtown": (40.72, -73.99)}
    lat = np.array([centers[d][0] for d in district]) + rng.normal(0, .012, n)
    lon = np.array([centers[d][1] for d in district]) + rng.normal(0, .012, n)
    df = pd.DataFrame({
        "request_id": [f"R25-{100000+i}" for i in range(n)],
        "created_at": created, "closed_at": closed, "complaint_type": ctype,
        "district": district, "status": status,
        "channel": rng.choice(["Phone", "Mobile App", "Web", "Walk-in"], n, p=[.35, .4, .2, .05]),
        "latitude": lat.round(5), "longitude": lon.round(5),
    }).sort_values("created_at").reset_index(drop=True)
    clean = df.copy()
    clean["created_at"] = clean["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    clean["closed_at"] = pd.to_datetime(clean["closed_at"]).dt.strftime("%Y-%m-%d %H:%M")
    raw = df.copy()
    raw["created_at"] = mixed_dates(raw["created_at"], rng)
    raw["closed_at"] = mixed_dates(pd.to_datetime(raw["closed_at"]), rng)
    # closed before created for some
    idx = rng.choice(np.where(df.status == "Closed")[0], 25, replace=False)
    raw.loc[idx, "closed_at"] = (df.loc[idx, "created_at"] - pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d %H:%M")
    raw["status"] = variants(raw["status"], {"Closed": ["closed", "CLOSED", "Closd", "Closed "],
                                             "Open": ["open", "OPEN"], "In Progress": ["in progress", "In-Progress"]}, .1, rng)
    raw["complaint_type"] = variants(raw["complaint_type"], {"Noise": ["noise", "Noise - Loud Music"],
        "Streetlight Out": ["Street Light Out", "streetlight"], "Missed Trash Pickup": ["Trash", "Missed Collection"]}, .06, rng)
    blank(raw, "district", .04, rng)
    i = rng.choice(n, 30, replace=False); raw.loc[i, ["latitude", "longitude"]] = 0.0
    raw = dupes(raw, 60, rng)
    register("riverton-311", "Riverton 311 Service Requests", raw, clean,
        summary="A year of non-emergency service requests in a fictional city.",
        description="5,000 requests made to the 311 line of the fictional city of Riverton during 2025: "
                    "what was reported, where, through which channel, and when the case was closed. "
                    "Dates arrive in three different formats, which makes it ideal for practicing date parsing.",
        domain="Government", tasks=["Cleaning", "Time series", "Visualization"], level="Intermediate",
        characteristics=["Tabular", "Time-series", "Geospatial"],
        variables=[
            ("request_id", "ID", "Categorical", "Request identifier", ""),
            ("created_at", "Feature", "Date", "When the request was made", "date-time"),
            ("closed_at", "Feature", "Date", "When the case was closed (blank if still open)", "date-time"),
            ("complaint_type", "Feature", "Categorical", "Category of the problem", ""),
            ("district", "Feature", "Categorical", "City district", ""),
            ("status", "Feature", "Categorical", "Open, In Progress or Closed", ""),
            ("channel", "Feature", "Categorical", "How the request was submitted", ""),
            ("latitude", "Feature", "Continuous", "Location latitude", "degrees"),
            ("longitude", "Feature", "Continuous", "Location longitude", "degrees"),
        ],
        issues=["Dates in three formats ('2025-03-04 14:22', '03/04/2025 02:22 PM', '04 Mar 2025 14:22')",
                "25 cases closed before they were created",
                "Status and complaint type spelled inconsistently",
                "Missing districts", "30 locations recorded as (0, 0)", "60 duplicate submissions"],
        exercises=["Parse both date columns and compute days_to_close",
                   "Plot weekly request volume by complaint type",
                   "Which district waits longest for a pothole repair?",
                   "Map the requests and spot the (0, 0) errors"])

# ======================================================================
# 3. Harbor City rentals
# ======================================================================
def harbor_rentals():
    rng = np.random.default_rng(303); n = 3000
    hoods = ["Old Port", "Lighthouse Hill", "Marina", "Riverside", "Market District", "Cedar Park", "North Shore"]
    mult = dict(zip(hoods, [1.35, 1.2, 1.5, 1.0, 1.1, 0.85, 0.95]))
    hood = rng.choice(hoods, n)
    rtype = rng.choice(["Entire home", "Private room", "Shared room"], n, p=[.6, .34, .06])
    acc = np.where(rtype == "Entire home", rng.integers(2, 9, n), rng.integers(1, 3, n))
    beds = np.where(rtype == "Entire home", np.maximum(1, acc // 2), 1)
    baths = np.where(rtype == "Entire home", rng.choice([1, 1.5, 2, 2.5, 3], n, p=[.35, .2, .3, .1, .05]), rng.choice([1, 1.5], n))
    base = np.where(rtype == "Entire home", 95, np.where(rtype == "Private room", 55, 30))
    price = (base * np.array([mult[h] for h in hood]) * (1 + .12 * (acc - 2)) * rng.lognormal(0, .25, n)).round(0)
    reviews = rng.negative_binomial(1, .03, n)
    score = np.where(reviews > 0, rng.normal(4.6, .3, n).clip(2.5, 5).round(2), np.nan)
    host_since = pd.Timestamp("2014-01-01") + pd.to_timedelta(rng.uniform(0, 4000, n), unit="D")
    last_review = pd.Series(pd.Timestamp("2025-09-30") - pd.to_timedelta(rng.exponential(60, n), unit="D")).dt.normalize()
    last_review = last_review.where(reviews > 0)
    adj = ["Sunny", "Cozy", "Modern", "Charming", "Spacious", "Quiet", "Bright", "Historic"]
    noun = {"Entire home": ["loft", "apartment", "townhouse", "flat", "cottage"], "Private room": ["room", "bedroom", "suite"],
            "Shared room": ["shared room", "bunk"]}
    title = [f"{rng.choice(adj)} {rng.choice(noun[t])} in {h}" for t, h in zip(rtype, hood)]
    df = pd.DataFrame({
        "listing_id": rng.choice(np.arange(100000, 999999), n, replace=False),
        "title": title, "neighborhood": hood, "room_type": rtype, "accommodates": acc,
        "bedrooms": beds, "bathrooms": baths, "price_per_night": price,
        "cleaning_fee": (price * rng.uniform(.1, .5, n)).round(0),
        "minimum_nights": rng.choice([1, 2, 3, 5, 7, 30], n, p=[.35, .3, .15, .08, .07, .05]),
        "number_of_reviews": reviews, "review_score": score,
        "host_since": host_since.strftime("%Y-%m-%d"),
        "host_is_superhost": rng.choice([True, False], n, p=[.25, .75]),
        "last_review": last_review.dt.strftime("%Y-%m-%d"),
    })
    clean = df.copy()
    raw = df.copy().astype({"price_per_night": object, "cleaning_fee": object, "bathrooms": object,
                            "review_score": object, "host_is_superhost": object})
    raw["price_per_night"] = [f"${p:,.2f}" for p in df.price_per_night]
    i = rng.choice(n, 8, replace=False); raw.loc[i, "price_per_night"] = "$99,999.00"
    raw["cleaning_fee"] = [("" if rng.random() < .07 else f"${c:,.0f}") for c in df.cleaning_fee]
    raw["cleaning_fee"] = raw["cleaning_fee"].replace("", np.nan)
    def bath_text(b, shared):
        if b == 1.5 and rng.random() < .3: return "1 bath + half-bath"
        word = "shared bath" if shared else ("bath" if b == 1 else "baths")
        return f"{b:g} {word}"
    raw["bathrooms"] = [bath_text(b, t == "Shared room") for b, t in zip(df.bathrooms, df.room_type)]
    # review score: some on a 0-100 scale
    sc = df.review_score.copy()
    m = (rng.random(n) < .2) & sc.notna(); sc[m] = (sc[m] * 20).round(0)
    raw["review_score"] = sc
    raw["host_is_superhost"] = np.where(df.host_is_superhost, "t", "f")
    i = rng.choice(n, 10, replace=False); raw.loc[i, "minimum_nights"] = 1125
    raw = dupes(raw, 25, rng)
    register("harbor-rentals", "Harbor City Short-Term Rentals", raw, clean,
        summary="3,000 vacation-rental listings with prices stored as text.",
        description="Listings from a fictional coastal city's short-term rental market: neighborhood, room type, "
                    "capacity, nightly price, fees, reviews and host details. Prices and bathrooms are stored as "
                    "text, and review scores mix two scales.",
        domain="Business", tasks=["Cleaning", "Regression", "Scaling"], level="Intermediate",
        characteristics=["Tabular", "Multivariate", "Text"], target="price_per_night",
        variables=[
            ("listing_id", "ID", "Integer", "Listing identifier", ""),
            ("title", "Feature", "Text", "Listing headline", ""),
            ("neighborhood", "Feature", "Categorical", "Neighborhood", ""),
            ("room_type", "Feature", "Categorical", "Entire home, Private room or Shared room", ""),
            ("accommodates", "Feature", "Integer", "Maximum number of guests", "guests"),
            ("bedrooms", "Feature", "Integer", "Number of bedrooms", ""),
            ("bathrooms", "Feature", "Text", "Bathrooms, written as text (e.g. '1.5 baths')", ""),
            ("price_per_night", "Target", "Text", "Nightly price as text (e.g. '$1,250.00')", "USD"),
            ("cleaning_fee", "Feature", "Text", "One-time cleaning fee as text", "USD"),
            ("minimum_nights", "Feature", "Integer", "Minimum stay", "nights"),
            ("number_of_reviews", "Feature", "Integer", "Total reviews", ""),
            ("review_score", "Feature", "Continuous", "Average rating (mostly 0-5, some 0-100)", "points"),
            ("host_since", "Feature", "Date", "Date the host joined", "date"),
            ("host_is_superhost", "Feature", "Binary", "Superhost status ('t' / 'f')", ""),
            ("last_review", "Feature", "Date", "Date of the most recent review", "date"),
        ],
        issues=["Prices and fees stored as text with '$' and commas",
                "8 placeholder prices of $99,999", "bathrooms written as text ('1 bath + half-bath', 'shared bath')",
                "review_score mixes a 0-5 and a 0-100 scale",
                "Missing review scores and dates for listings with no reviews (legitimately missing)",
                "minimum_nights of 1,125 (a platform default, not a real rule)", "25 duplicated listings"],
        exercises=["Convert prices and bathrooms to numbers",
                   "Put review_score on one scale and explain how you detected the problem",
                   "Compare price distributions by neighborhood and room type",
                   "Scale the features and fit a price model"])

# ======================================================================
# 4. Coffee shop transactions
# ======================================================================
def coffee():
    rng = np.random.default_rng(404); n = 8000
    menu = [("Espresso", "Coffee", 3.0), ("Latte", "Coffee", 4.6), ("Cappuccino", "Coffee", 4.4),
            ("Cold Brew", "Coffee", 4.2), ("Drip Coffee", "Coffee", 2.6), ("Mocha", "Coffee", 4.9),
            ("Chai Latte", "Tea", 4.5), ("Green Tea", "Tea", 3.0), ("Croissant", "Bakery", 3.4),
            ("Blueberry Muffin", "Bakery", 3.1), ("Bagel", "Bakery", 2.8), ("Bottled Water", "Other", 2.0)]
    w = np.array([6, 14, 9, 10, 12, 6, 7, 4, 8, 6, 6, 3], float); w /= w.sum()
    stores = ["Campus", "Downtown", "Airport"]
    day = pd.Timestamp("2025-03-01") + pd.to_timedelta(rng.integers(0, 183, n), unit="D")
    hour = np.clip(rng.normal(11, 3.2, n), 6, 20.9)
    ts = (day + pd.to_timedelta(hour, unit="h")).floor("s")
    pick = rng.choice(len(menu), n, p=w)
    store = rng.choice(stores, n, p=[.4, .38, .22])
    size = [("" if menu[k][1] in ("Bakery", "Other") else rng.choice(["Small", "Medium", "Large"], p=[.3, .45, .25])) for k in pick]
    up = np.array([menu[k][2] for k in pick]) + np.array([{"": 0, "Small": 0, "Medium": .5, "Large": 1.0}[s] for s in size])
    up = up * np.where(store == "Airport", 1.2, 1.0)
    df = pd.DataFrame({
        "transaction_id": [f"T{500000+i}" for i in range(n)], "timestamp": ts, "store": store,
        "product": [menu[k][0] for k in pick], "category": [menu[k][1] for k in pick], "size": size,
        "unit_price": up.round(2), "quantity": rng.choice([1, 1, 1, 2, 2, 3, 4], n),
        "payment_method": rng.choice(["Card", "Mobile", "Cash"], n, p=[.5, .35, .15]),
        "discount_code": rng.choice(["", "STUDENT10", "HAPPYHOUR", "LOYALTY"], n, p=[.8, .1, .05, .05]),
    }).sort_values("timestamp").reset_index(drop=True)
    df["size"] = df["size"].replace("", np.nan); df["discount_code"] = df["discount_code"].replace("", np.nan)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    clean = df.copy()
    raw = df.copy()
    raw["store"] = variants(raw["store"], {"Campus": ["campus", "Campus Store", "CAMPUS"],
        "Downtown": ["Down Town", "downtown", "Downtown "], "Airport": ["airport", "Airport T2"]}, .1, rng)
    i = rng.choice(n, 40, replace=False); raw.loc[i, "unit_price"] = 0.0
    i = rng.choice(n, 70, replace=False); raw.loc[i, "quantity"] = -raw.loc[i, "quantity"]
    i = rng.choice(n, 12, replace=False)
    raw.loc[i, "timestamp"] = [t[:11] + "03:" + t[14:] for t in raw.loc[i, "timestamp"]]
    raw = dupes(raw, 50, rng)
    register("bean-there-coffee", "Bean There Coffee Sales", raw, clean,
        summary="Six months of point-of-sale records from three coffee shops.",
        description="8,000 transactions from a fictional three-store coffee chain (Campus, Downtown, Airport), "
                    "March to August 2025. Each row is one product line on a receipt. Good for revenue "
                    "calculations, time-of-day patterns and store comparisons.",
        domain="Business", tasks=["Cleaning", "Time series", "Visualization"], level="Beginner",
        characteristics=["Tabular", "Time-series"],
        variables=[
            ("transaction_id", "ID", "Categorical", "Receipt line identifier", ""),
            ("timestamp", "Feature", "Date", "Date and time of sale", "date-time"),
            ("store", "Feature", "Categorical", "Store location", ""),
            ("product", "Feature", "Categorical", "Item sold", ""),
            ("category", "Feature", "Categorical", "Coffee, Tea, Bakery or Other", ""),
            ("size", "Feature", "Categorical", "Drink size (blank for food)", ""),
            ("unit_price", "Feature", "Continuous", "Price of one item", "USD"),
            ("quantity", "Feature", "Integer", "Number of items", ""),
            ("payment_method", "Feature", "Categorical", "Card, Mobile or Cash", ""),
            ("discount_code", "Feature", "Categorical", "Promotion used, if any", ""),
        ],
        issues=["Store names spelled several ways", "40 items with a price of 0.00",
                "70 negative quantities (refunds or entry errors: you decide)",
                "12 sales at 3 a.m., when every store is closed",
                "Blank size and discount_code are legitimately empty", "50 duplicated receipt lines"],
        exercises=["Create a revenue column and total it by store and month",
                   "Plot sales by hour of day: when is the morning rush?",
                   "Decide how to treat refunds and justify it",
                   "Which product grew fastest over the six months?"])

# ======================================================================
# 5. Clinic vitals
# ======================================================================
def clinic():
    rng = np.random.default_rng(505); n = 2000
    age = rng.integers(18, 90, n); sex = rng.choice(["F", "M"], n)
    height = np.where(sex == "F", rng.normal(163, 7, n), rng.normal(176, 7.5, n)).round(1)
    bmi = rng.normal(27, 5, n).clip(16, 48)
    weight = (bmi * (height / 100) ** 2).round(1)
    smoker = rng.choice(["Never", "Former", "Current"], n, p=[.55, .27, .18])
    sbp = (100 + .5 * age + .8 * (bmi - 25) + 6 * (smoker == "Current") + rng.normal(0, 12, n)).round(0)
    dbp = (sbp * .62 + rng.normal(0, 6, n)).round(0)
    df = pd.DataFrame({
        "patient_id": [f"P{20000+i}" for i in range(n)],
        "visit_date": (pd.Timestamp("2025-01-02") + pd.to_timedelta(rng.integers(0, 270, n), unit="D")).strftime("%Y-%m-%d"),
        "age": age, "sex": sex, "height": height, "height_unit": "cm", "weight": weight, "weight_unit": "kg",
        "systolic_bp": sbp.astype(int), "diastolic_bp": dbp.astype(int),
        "heart_rate": rng.normal(74, 11, n).round(0).astype(int),
        "temperature": rng.normal(36.8, .35, n).round(1), "smoker": smoker,
        "cholesterol_mg_dl": (170 + .6 * age + rng.normal(0, 30, n)).round(0),
        "glucose_mg_dl": (88 + .25 * age + 1.2 * (bmi - 25) + rng.gamma(2, 6, n)).round(0),
    })
    df["hypertension"] = np.where((df.systolic_bp >= 140) | (df.diastolic_bp >= 90), 1, 0)
    clean = df.copy()
    raw = df.copy().astype({"height": float, "weight": float, "temperature": float})
    m = rng.random(n) < .25
    raw.loc[m, "height"] = (df.loc[m, "height"] / 2.54).round(1); raw.loc[m, "height_unit"] = "in"
    m = rng.random(n) < .3
    raw.loc[m, "weight"] = (df.loc[m, "weight"] * 2.2046).round(1); raw.loc[m, "weight_unit"] = "lb"
    blank(raw, "weight_unit", .05, rng)
    m = rng.random(n) < .2
    raw.loc[m, "temperature"] = (df.loc[m, "temperature"] * 9 / 5 + 32).round(1)
    i = rng.choice(n, 15, replace=False); raw.loc[i, ["systolic_bp", "diastolic_bp"]] = raw.loc[i, ["diastolic_bp", "systolic_bp"]].values
    i = rng.choice(n, 8, replace=False); raw.loc[i, "heart_rate"] = 0
    blank(raw, "cholesterol_mg_dl", .15, rng); blank(raw, "glucose_mg_dl", .06, rng)
    raw["smoker"] = variants(raw["smoker"], {"Never": ["never", "No", "N"], "Current": ["Yes", "current", "Y"],
                                             "Former": ["former", "Ex-smoker"]}, .1, rng)
    register("clinic-vitals", "Community Clinic Vitals", raw, clean,
        summary="Patient vital signs recorded with mixed units of measure.",
        description="Vital signs and lab values for 2,000 fictional adult patients at a community clinic. "
                    "Heights, weights and temperatures were entered in different units depending on the nurse. "
                    "Synthetic data for teaching only: not suitable for any medical conclusion.",
        domain="Health", tasks=["Cleaning", "Scaling", "Classification"], level="Intermediate",
        characteristics=["Tabular", "Multivariate"], target="hypertension",
        variables=[
            ("patient_id", "ID", "Categorical", "Patient identifier", ""),
            ("visit_date", "Feature", "Date", "Date of visit", "date"),
            ("age", "Feature", "Integer", "Age", "years"),
            ("sex", "Feature", "Categorical", "Sex recorded at visit (F/M)", ""),
            ("height", "Feature", "Continuous", "Height, in the unit given in height_unit", "cm or in"),
            ("height_unit", "Feature", "Categorical", "Unit of height", ""),
            ("weight", "Feature", "Continuous", "Weight, in the unit given in weight_unit", "kg or lb"),
            ("weight_unit", "Feature", "Categorical", "Unit of weight (sometimes missing)", ""),
            ("systolic_bp", "Feature", "Integer", "Systolic blood pressure", "mmHg"),
            ("diastolic_bp", "Feature", "Integer", "Diastolic blood pressure", "mmHg"),
            ("heart_rate", "Feature", "Integer", "Resting heart rate", "beats/min"),
            ("temperature", "Feature", "Continuous", "Body temperature (unit not recorded)", "°C or °F"),
            ("smoker", "Feature", "Categorical", "Smoking status", ""),
            ("cholesterol_mg_dl", "Feature", "Continuous", "Total cholesterol", "mg/dL"),
            ("glucose_mg_dl", "Feature", "Continuous", "Fasting glucose", "mg/dL"),
            ("hypertension", "Target", "Binary", "1 if BP ≥ 140/90 at the visit", ""),
        ],
        issues=["Heights in cm and inches, weights in kg and lb", "weight_unit missing for 5% of rows",
                "Temperatures in °C and °F with no unit column", "15 visits with systolic and diastolic swapped",
                "8 heart rates of 0", "Missing cholesterol (15%) and glucose (6%)",
                "Smoking status coded inconsistently ('Yes', 'Ex-smoker', 'N' ...)"],
        exercises=["Convert every measurement to metric units and compute BMI",
                   "Detect Fahrenheit temperatures without a unit column",
                   "Compare standardization and min-max scaling on the vital signs",
                   "Plot blood pressure against age, colored by smoking status"])

# ======================================================================
# 6. Music track features
# ======================================================================
def tracks():
    rng = np.random.default_rng(606); n = 4000
    genres = {"Pop": (118, -6, .7, .7, .15), "Hip-Hop": (95, -6.5, .7, .8, .1), "Rock": (128, -6, .8, .5, .1),
              "Electronic": (126, -5, .85, .72, .05), "Jazz": (110, -11, .4, .55, .7), "Classical": (100, -18, .2, .3, .9),
              "Country": (112, -7, .65, .6, .3), "R&B": (98, -7, .6, .7, .25), "Afrobeats": (106, -6, .75, .82, .15)}
    g = rng.choice(list(genres), n)
    tempo = np.array([genres[x][0] for x in g]) + rng.normal(0, 14, n)
    loud = np.array([genres[x][1] for x in g]) + rng.normal(0, 2.5, n)
    energy = (np.array([genres[x][2] for x in g]) + rng.normal(0, .12, n)).clip(0, 1)
    dance = (np.array([genres[x][3] for x in g]) + rng.normal(0, .12, n)).clip(0, 1)
    acoustic = (np.array([genres[x][4] for x in g]) + rng.normal(0, .15, n)).clip(0, 1)
    words = ["Midnight", "Golden", "River", "Echo", "Neon", "Paper", "Velvet", "Summer", "Electric", "Silent",
             "Fire", "Ocean", "City", "Dream", "Heart", "Signal", "Lagos", "Highway", "Mirror", "Rain"]
    artists = fake_names(rng, 220)
    df = pd.DataFrame({
        "track_id": [f"TRK{rng.integers(16**7):07x}" for _ in range(n)],
        "title": [f"{rng.choice(words)} {rng.choice(words)}" for _ in range(n)],
        "artist": rng.choice(artists, n), "genre": g,
        "release_year": rng.integers(1975, 2026, n),
        "duration_ms": (np.where(g == "Classical", rng.normal(400000, 120000, n), rng.normal(215000, 40000, n))).clip(60000).round(0).astype(int),
        "tempo_bpm": tempo.round(1), "loudness_db": loud.clip(-40, 0).round(2),
        "energy": energy.round(3), "danceability": dance.round(3), "acousticness": acoustic.round(3),
        "valence": rng.beta(2.5, 2.5, n).round(3), "speechiness": np.where(g == "Hip-Hop", rng.beta(3, 7, n), rng.beta(1.2, 18, n)).round(3),
        "key": rng.integers(0, 12, n), "mode": rng.choice(["Major", "Minor"], n, p=[.62, .38]),
    })
    df["popularity"] = (35 + 40 * df.danceability + 10 * df.energy + .35 * (df.release_year - 2000).clip(0)
                        + rng.normal(0, 12, n)).clip(0, 100).round(0).astype(int)
    clean = df.copy()
    raw = df.copy()
    blank(raw, "tempo_bpm", .03, rng)
    raw["genre"] = variants(raw["genre"], {"Hip-Hop": ["hip hop", "HipHop", "Rap/Hip-Hop"], "R&B": ["RnB", "r&b"],
                                           "Electronic": ["EDM", "electronic"]}, .07, rng)
    rep = raw.sample(40, random_state=1).copy(); rep["track_id"] = [f"TRK{rng.integers(16**7):07x}" for _ in range(40)]
    raw = pd.concat([raw, rep]).sample(frac=1, random_state=2).reset_index(drop=True)
    register("track-features", "Music Track Audio Features", raw, clean,
        summary="Audio features of 4,000 songs, on wildly different scales.",
        description="Each row is a fictional song with audio measurements such as tempo, loudness, energy and "
                    "danceability, plus its genre and a popularity score. Features range from 0-1 to over "
                    "400,000 (duration in milliseconds), which makes the dataset ideal for teaching scaling.",
        domain="Entertainment", tasks=["Scaling", "Clustering", "Visualization", "Regression"], level="Beginner",
        characteristics=["Tabular", "Multivariate"], target="popularity",
        variables=[
            ("track_id", "ID", "Categorical", "Track identifier", ""),
            ("title", "Feature", "Text", "Song title", ""),
            ("artist", "Feature", "Categorical", "Performing artist (fictional)", ""),
            ("genre", "Feature", "Categorical", "Main genre", ""),
            ("release_year", "Feature", "Integer", "Year released", "year"),
            ("duration_ms", "Feature", "Integer", "Track length", "milliseconds"),
            ("tempo_bpm", "Feature", "Continuous", "Estimated tempo", "beats/min"),
            ("loudness_db", "Feature", "Continuous", "Average loudness", "dB (negative)"),
            ("energy", "Feature", "Continuous", "Intensity and activity", "0-1"),
            ("danceability", "Feature", "Continuous", "How suitable for dancing", "0-1"),
            ("acousticness", "Feature", "Continuous", "Confidence the track is acoustic", "0-1"),
            ("valence", "Feature", "Continuous", "Musical positiveness", "0-1"),
            ("speechiness", "Feature", "Continuous", "Presence of spoken words", "0-1"),
            ("key", "Feature", "Integer", "Pitch class (0 = C, 1 = C#, ... 11 = B)", ""),
            ("mode", "Feature", "Categorical", "Major or Minor", ""),
            ("popularity", "Target", "Integer", "Popularity score", "0-100"),
        ],
        issues=["Missing tempo for 3% of tracks", "Genre labels spelled several ways ('hip hop', 'RnB', 'EDM')",
                "40 tracks listed twice under different track IDs"],
        exercises=["Compare the ranges of the numeric features before and after StandardScaler and MinMaxScaler",
                   "Cluster tracks with k-means with and without scaling: what changes?",
                   "Plot danceability vs. energy colored by genre",
                   "Find duplicate songs without relying on track_id"])

# ======================================================================
# 7. Craft beer chemistry
# ======================================================================
def beer():
    rng = np.random.default_rng(707); n = 1500
    styles = {"IPA": (6.5, 65, 8, 1.062), "Pale Ale": (5.3, 38, 9, 1.052), "Stout": (6.8, 40, 40, 1.068),
              "Porter": (5.8, 32, 30, 1.058), "Lager": (4.8, 18, 4, 1.046), "Wheat": (5.0, 14, 5, 1.050),
              "Sour": (4.6, 8, 6, 1.045), "Amber": (5.6, 28, 14, 1.055)}
    st = rng.choice(list(styles), n)
    abv = np.array([styles[s][0] for s in st]) + rng.normal(0, .8, n)
    ibu = (np.array([styles[s][1] for s in st]) * rng.lognormal(0, .25, n))
    srm = (np.array([styles[s][2] for s in st]) * rng.lognormal(0, .2, n))
    og = np.array([styles[s][3] for s in st]) + (abv - np.array([styles[s][0] for s in st])) * .0075 + rng.normal(0, .002, n)
    fg = og - abv / 131.25
    ph = np.where(st == "Sour", rng.normal(3.3, .15, n), rng.normal(4.3, .15, n))
    breweries = [f"{w} Brewing" for w in ["Anchor Hill", "Copper Kettle", "Foxglove", "Ironwood", "Lantern", "Mill Creek",
                                         "North Fork", "Quarry", "Red Barn", "Stillwater", "Tidewater", "Wildflower"]]
    df = pd.DataFrame({
        "beer_id": np.arange(1, n + 1), "brewery": rng.choice(breweries, n), "style": st,
        "abv_pct": abv.round(1), "ibu": ibu.round(0), "srm_color": srm.round(1),
        "original_gravity": og.round(3), "final_gravity": fg.round(3), "ph": ph.round(2),
        "price_usd": (8 + abv * 1.1 + rng.normal(0, 2, n)).round(2),
    })
    df["rating"] = (2.2 + .18 * df.abv_pct + .006 * df.ibu - .4 * np.abs(df.ph - 4.2) + rng.normal(0, .35, n)).clip(1, 5).round(2)
    clean = df.copy()
    raw = df.copy()
    m = rng.random(n) < .1; raw.loc[m, "abv_pct"] = (df.loc[m, "abv_pct"] / 100).round(3)
    blank(raw, "ibu", .06, rng); blank(raw, "ph", .04, rng)
    register("craft-brew", "Craft Beer Chemistry", raw, clean,
        summary="Brewing measurements and ratings for 1,500 craft beers.",
        description="Chemistry of fictional craft beers from twelve breweries: alcohol content, bitterness (IBU), "
                    "color (SRM), gravity readings and acidity, with an average customer rating. Features sit on "
                    "very different scales (gravity near 1.05, IBU up to 100+).",
        domain="Food & Drink", tasks=["Scaling", "Regression", "Clustering"], level="Beginner",
        characteristics=["Tabular", "Multivariate"], target="rating",
        variables=[
            ("beer_id", "ID", "Integer", "Beer identifier", ""),
            ("brewery", "Feature", "Categorical", "Brewery (fictional)", ""),
            ("style", "Feature", "Categorical", "Beer style", ""),
            ("abv_pct", "Feature", "Continuous", "Alcohol by volume", "%"),
            ("ibu", "Feature", "Continuous", "International Bitterness Units", "IBU"),
            ("srm_color", "Feature", "Continuous", "Color on the SRM scale (higher = darker)", "SRM"),
            ("original_gravity", "Feature", "Continuous", "Specific gravity before fermentation", "SG"),
            ("final_gravity", "Feature", "Continuous", "Specific gravity after fermentation", "SG"),
            ("ph", "Feature", "Continuous", "Acidity", "pH"),
            ("price_usd", "Feature", "Continuous", "Price of a 4-pack", "USD"),
            ("rating", "Target", "Continuous", "Average customer rating", "1-5"),
        ],
        issues=["About 10% of abv_pct values entered as fractions (0.052 instead of 5.2)",
                "Missing IBU (6%) and pH (4%)"],
        exercises=["Find the ABV values on the wrong scale and fix them",
                   "Scale the chemistry features and fit a rating model",
                   "Do beer styles form clusters? Try k-means on scaled features",
                   "Plot color (SRM) by style"])

# ======================================================================
# 8. Maple County housing
# ======================================================================
def housing():
    rng = np.random.default_rng(808); n = 2500
    hoods = {"Brookside": 1.0, "Elm Grove": 1.15, "Harbor View": 1.45, "Millbrook": .85, "Oakridge": 1.25,
             "Pinecrest": .95, "Stonegate": 1.6, "Willow Bend": 1.05}
    hood = rng.choice(list(hoods), n)
    kind = rng.choice(["Single Family", "Townhouse", "Condo"], n, p=[.62, .2, .18])
    sqft = np.where(kind == "Condo", rng.normal(1050, 250, n), np.where(kind == "Townhouse", rng.normal(1550, 300, n),
                    rng.lognormal(7.6, .3, n))).clip(450).round(0)
    lot = np.where(kind == "Condo", np.nan, (sqft * rng.uniform(2, 6, n)).round(-1))
    beds = np.clip((sqft / 550 + rng.normal(0, .6, n)).round(0), 1, 7).astype(int)
    baths = np.clip(np.round((beds * .6 + rng.normal(.3, .4, n)) * 2) / 2, 1, 5)
    year = rng.integers(1925, 2025, n)
    dist = rng.gamma(3, 2.5, n).round(1)
    cond = rng.choice([1, 2, 3, 4, 5], n, p=[.03, .12, .45, .3, .1])
    school = rng.integers(3, 11, n)
    price = (85 * sqft * np.array([hoods[h] for h in hood]) * (1 + .04 * (cond - 3)) * (1 + .03 * (school - 6))
             * (1 - .012 * dist) * (1 + .0015 * (year - 1975)) * rng.lognormal(0, .12, n)
             + np.nan_to_num(lot) * 4).round(-3)
    df = pd.DataFrame({
        "sale_id": [f"MC-{70000+i}" for i in range(n)],
        "sale_date": (pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")).strftime("%Y-%m-%d"),
        "neighborhood": hood, "property_type": kind, "living_sqft": sqft.astype(int), "lot_sqft": lot,
        "bedrooms": beds, "bathrooms": baths, "year_built": year,
        "garage_spaces": np.where(kind == "Condo", rng.choice([0, 1], n), rng.choice([0, 1, 2, 3], n, p=[.1, .3, .5, .1])),
        "has_pool": rng.choice(["No", "Yes"], n, p=[.85, .15]), "condition": cond, "school_rating": school,
        "miles_to_downtown": dist, "sale_price": price.astype(int),
    })
    clean = df.copy()
    raw = df.copy().astype({"bedrooms": object, "year_built": float})
    m = df.bedrooms >= 5; raw.loc[m & (rng.random(n) < .5), "bedrooms"] = "5+"
    i = rng.choice(n, 10, replace=False); raw.loc[i, "year_built"] = 0
    i = rng.choice(n, 4, replace=False); raw.loc[i, "year_built"] = 2099
    blank(raw, "year_built", .03, rng)
    i = rng.choice(n, 6, replace=False); raw.loc[i, "living_sqft"] = raw.loc[i, "living_sqft"] * 10
    blank(raw, "school_rating", .05, rng)
    raw = dupes(raw, 20, rng)
    register("maple-county-housing", "Maple County Home Sales", raw, clean,
        summary="2,500 residential sales with a right-skewed price.",
        description="Home sales in a fictional county from 2023 to 2025, with size, lot, rooms, age, condition, "
                    "school rating and distance to downtown. Sale prices are strongly right-skewed, so log "
                    "transformation and scaling both matter.",
        domain="Real Estate", tasks=["Regression", "Scaling", "Cleaning", "Visualization"], level="Intermediate",
        characteristics=["Tabular", "Multivariate"], target="sale_price",
        variables=[
            ("sale_id", "ID", "Categorical", "Sale identifier", ""),
            ("sale_date", "Feature", "Date", "Date the sale closed", "date"),
            ("neighborhood", "Feature", "Categorical", "Neighborhood", ""),
            ("property_type", "Feature", "Categorical", "Single Family, Townhouse or Condo", ""),
            ("living_sqft", "Feature", "Integer", "Interior living area", "sq ft"),
            ("lot_sqft", "Feature", "Continuous", "Lot size (blank for condos)", "sq ft"),
            ("bedrooms", "Feature", "Categorical", "Bedrooms (some recorded as '5+')", ""),
            ("bathrooms", "Feature", "Continuous", "Bathrooms (0.5 = half bath)", ""),
            ("year_built", "Feature", "Continuous", "Year of construction", "year"),
            ("garage_spaces", "Feature", "Integer", "Garage capacity", "cars"),
            ("has_pool", "Feature", "Binary", "Has a pool (Yes/No)", ""),
            ("condition", "Feature", "Integer", "Assessor's condition rating", "1-5"),
            ("school_rating", "Feature", "Integer", "Rating of the assigned school", "1-10"),
            ("miles_to_downtown", "Feature", "Continuous", "Distance to downtown", "miles"),
            ("sale_price", "Target", "Integer", "Final sale price", "USD"),
        ],
        issues=["bedrooms stored as text because some are recorded as '5+'",
                "year_built of 0 or 2099, plus missing values", "6 living areas with an extra zero",
                "lot_sqft blank for condos (legitimately missing)", "Missing school ratings", "20 duplicated sales"],
        exercises=["Plot sale_price before and after a log transform",
                   "Decide how to encode '5+' bedrooms",
                   "Compare a regression model with and without feature scaling",
                   "Which neighborhood has the highest price per square foot?"])

# ======================================================================
# 9. Air quality sensors
# ======================================================================
def air():
    rng = np.random.default_rng(909)
    sensors = {"AQ-01": "Riverside Park", "AQ-02": "Central Station", "AQ-03": "Industrial Ave",
               "AQ-04": "University Campus", "AQ-05": "Harbor Road"}
    level = {"AQ-01": .8, "AQ-02": 1.2, "AQ-03": 1.6, "AQ-04": .9, "AQ-05": 1.3}
    hours = pd.date_range("2025-06-01", periods=90 * 24, freq="h")
    rows = []
    for sid, loc in sensors.items():
        t = np.arange(len(hours)); h = hours.hour.values
        rush = 1 + .45 * np.exp(-((h - 8) ** 2) / 4) + .35 * np.exp(-((h - 18) ** 2) / 5)
        weather = np.convolve(rng.normal(0, 1, len(t)), np.ones(48) / 48, mode="same") * 6
        pm25 = (9 * level[sid] * rush + weather + rng.gamma(2, 1.2, len(t))).clip(1)
        temp = 24 + 5 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 1.2, len(t)) + 2 * np.sin(2 * np.pi * t / (90 * 24))
        rows.append(pd.DataFrame({
            "timestamp": hours, "sensor_id": sid, "location": loc, "pm25_ugm3": pm25.round(1),
            "pm10_ugm3": (pm25 * rng.uniform(1.5, 2.1, len(t))).round(1),
            "no2_ppb": (14 * level[sid] * rush + rng.normal(0, 3, len(t))).clip(1).round(1),
            "o3_ppb": (30 + 18 * np.sin(2 * np.pi * (h - 10) / 24) + rng.normal(0, 4, len(t))).clip(2).round(1),
            "temperature_c": temp.round(1), "humidity_pct": (65 - 1.5 * (temp - 24) + rng.normal(0, 5, len(t))).clip(20, 100).round(0),
        }))
    df = pd.concat(rows).sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    clean = df.copy()
    raw = df.copy()
    for c in ["pm25_ugm3", "pm10_ugm3", "no2_ppb"]:
        i = rng.choice(len(raw), 120, replace=False); raw.loc[i, c] = -999.0
    m = (raw.sensor_id == "AQ-03") & (raw.timestamp >= "2025-08-01")
    raw.loc[m, "pm25_ugm3"] = (raw.loc[m, "pm25_ugm3"] + 12).round(1)
    gap = (raw.sensor_id == "AQ-05") & (raw.timestamp >= "2025-07-10") & (raw.timestamp < "2025-07-13")
    raw = raw[~gap]
    raw = raw.drop(raw.sample(frac=.01, random_state=3).index)
    raw = dupes(raw, 80, rng).sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
    register("airwatch-sensors", "AirWatch Sensor Readings", raw, clean,
        summary="Hourly air-quality readings from five city sensors over 90 days.",
        description="Hourly measurements of particulate matter, nitrogen dioxide, ozone, temperature and humidity "
                    "from five fictional sensors, June to August 2025. Contains the classic sensor problems: "
                    "sentinel values for dropouts, gaps, duplicates and a sensor that drifts.",
        domain="Environment", tasks=["Time series", "Cleaning", "Visualization"], level="Advanced",
        characteristics=["Time-series", "Multivariate"],
        variables=[
            ("timestamp", "Feature", "Date", "Start of the hour", "date-time"),
            ("sensor_id", "ID", "Categorical", "Sensor identifier", ""),
            ("location", "Feature", "Categorical", "Sensor location", ""),
            ("pm25_ugm3", "Feature", "Continuous", "Fine particulate matter (PM2.5)", "µg/m³"),
            ("pm10_ugm3", "Feature", "Continuous", "Coarse particulate matter (PM10)", "µg/m³"),
            ("no2_ppb", "Feature", "Continuous", "Nitrogen dioxide", "ppb"),
            ("o3_ppb", "Feature", "Continuous", "Ozone", "ppb"),
            ("temperature_c", "Feature", "Continuous", "Air temperature", "°C"),
            ("humidity_pct", "Feature", "Continuous", "Relative humidity", "%"),
        ],
        issues=["-999 used as a 'no reading' code in pm25, pm10 and no2",
                "Sensor AQ-03 reads about 12 µg/m³ too high for PM2.5 from 1 August (calibration drift)",
                "Sensor AQ-05 is offline 10-12 July (a 3-day gap)", "About 1% of hours missing at random",
                "80 duplicated hourly readings"],
        exercises=["Replace the -999 codes and count the missing hours per sensor",
                   "Resample to daily averages and plot PM2.5 by sensor",
                   "Detect the drifting sensor with a plot, then correct it",
                   "Show the morning and evening rush-hour pattern"])

# ======================================================================
# 10. Telecom churn
# ======================================================================
def churn():
    rng = np.random.default_rng(1010); n = 4000
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2])
    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[.45, .35, .2])
    monthly = (20 + np.where(internet == "Fiber", 55, np.where(internet == "DSL", 30, 0)) + rng.normal(0, 8, n)).clip(18).round(2)
    calls = rng.poisson(1.5, n)
    logit = (-1.2 + 1.4 * (contract == "Month-to-month") - .035 * tenure + .5 * (internet == "Fiber")
             + .3 * calls + .01 * (monthly - 65) + rng.normal(0, .6, n))
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "customer_id": [f"LU-{rng.integers(1e7):07d}" for _ in range(n)],
        "senior_citizen": rng.choice(["No", "Yes"], n, p=[.84, .16]),
        "tenure_months": tenure, "contract": contract, "internet_service": internet,
        "streaming_tv": np.where(internet == "None", "No internet", rng.choice(["Yes", "No"], n)),
        "paperless_billing": rng.choice(["Yes", "No"], n, p=[.6, .4]),
        "payment_method": rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n),
        "monthly_charges": monthly, "total_charges": (monthly * np.maximum(tenure, 0) * rng.uniform(.95, 1.05, n)).round(2),
        "support_calls": calls, "churn": np.where(churned, "Yes", "No"),
    })
    clean = df.copy()
    raw = df.copy().astype({"total_charges": object})
    raw.loc[df.tenure_months == 0, "total_charges"] = " "
    raw["churn"] = variants(raw["churn"], {"Yes": ["yes", "Y", "1"], "No": ["no", "N", "0"]}, .08, rng)
    i = rng.choice(n, 10, replace=False); raw.loc[i, "monthly_charges"] = -raw.loc[i, "monthly_charges"]
    blank(raw, "payment_method", .03, rng)
    register("linkup-churn", "LinkUp Telecom Customer Churn", raw, clean,
        summary="Which telecom customers cancel? An imbalanced classification set.",
        description="Account details for 4,000 customers of a fictional telecom provider and whether they left "
                    "in the last month. About three in ten customers churned, so classes are imbalanced. "
                    "total_charges is stored as text because new customers have a blank space.",
        domain="Business", tasks=["Classification", "Cleaning", "Scaling"], level="Intermediate",
        characteristics=["Tabular", "Multivariate"], target="churn",
        variables=[
            ("customer_id", "ID", "Categorical", "Customer identifier", ""),
            ("senior_citizen", "Feature", "Binary", "Aged 65 or older (Yes/No)", ""),
            ("tenure_months", "Feature", "Integer", "Months as a customer", "months"),
            ("contract", "Feature", "Categorical", "Contract length", ""),
            ("internet_service", "Feature", "Categorical", "Fiber, DSL or None", ""),
            ("streaming_tv", "Feature", "Categorical", "Has streaming TV", ""),
            ("paperless_billing", "Feature", "Binary", "Paperless billing (Yes/No)", ""),
            ("payment_method", "Feature", "Categorical", "How the bill is paid", ""),
            ("monthly_charges", "Feature", "Continuous", "Current monthly bill", "USD"),
            ("total_charges", "Feature", "Text", "Total billed to date (text; blank for new customers)", "USD"),
            ("support_calls", "Feature", "Integer", "Calls to support in the last 6 months", ""),
            ("churn", "Target", "Binary", "Left in the last month", ""),
        ],
        issues=["total_charges stored as text; new customers have ' ' instead of a number",
                "churn coded several ways (Yes, yes, Y, 1 ...)", "10 negative monthly charges",
                "Missing payment method (3%)", "Imbalanced target (about 30% churn)"],
        exercises=["Convert total_charges to a number and decide what blanks should become",
                   "Plot churn rate by contract type and by tenure",
                   "One-hot encode the categorical columns and scale the numeric ones",
                   "Train a classifier and report precision and recall, not just accuracy"])

# ======================================================================
# 11. ShopNest orders
# ======================================================================
def orders():
    rng = np.random.default_rng(1111); n = 6000
    countries = {"United States": ("USD", .45), "United Kingdom": ("GBP", .15), "Germany": ("EUR", .12),
                 "France": ("EUR", .08), "Canada": ("CAD", .1), "Nigeria": ("NGN", .05), "Ghana": ("GHS", .05)}
    rate = {"USD": 1, "GBP": 1.27, "EUR": 1.08, "CAD": .73, "NGN": .00065, "GHS": .065}
    ctry = rng.choice(list(countries), n, p=[v[1] for v in countries.values()])
    cats = {"Electronics": 180, "Books": 22, "Clothing": 48, "Home & Kitchen": 65, "Beauty": 30, "Sports": 70, "Toys": 35}
    cat = rng.choice(list(cats), n)
    items = rng.choice([1, 1, 1, 2, 2, 3, 4, 5], n)
    usd = np.array([cats[c] for c in cat]) * items * rng.lognormal(0, .35, n)
    cur = np.array([countries[c][0] for c in ctry])
    local = usd / np.array([rate[c] for c in cur])
    date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    df = pd.DataFrame({
        "order_id": [f"SN{800000+i}" for i in range(n)], "order_date": date, "customer_id": [f"C{x}" for x in rng.integers(1000, 4000, n)],
        "country": ctry, "currency": cur, "product_category": cat, "items": items,
        "order_total": local.round(2), "order_total_usd": usd.round(2),
        "delivery_days": (rng.gamma(2, 2, n) + np.where(ctry == "United States", 1, 4)).round(0).astype(int),
        "returned": rng.choice(["No", "Yes"], n, p=[.9, .1]),
        "rating": rng.choice([1, 2, 3, 4, 5, np.nan], n, p=[.04, .05, .12, .3, .29, .2]),
    }).sort_values("order_date").reset_index(drop=True)
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    clean = df.copy()
    raw = df.drop(columns="order_total_usd").copy()
    raw["order_date"] = mixed_dates(pd.to_datetime(raw["order_date"]), rng, fmts=("%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y"), probs=(.6, .25, .15))
    raw["country"] = variants(raw["country"], {"United States": ["USA", "US", "U.S.", "united states"],
        "United Kingdom": ["UK", "U.K.", "England"], "Germany": ["DE", "Deutschland"], "Nigeria": ["NG"]}, .15, rng)
    i = rng.choice(n, 30, replace=False); raw.loc[i, "delivery_days"] = -raw.loc[i, "delivery_days"]
    raw = dupes(raw, 45, rng)
    register("shopnest-orders", "ShopNest Online Orders", raw, clean,
        summary="International e-commerce orders priced in six currencies.",
        description="6,000 orders from a fictional online store selling to seven countries in 2025. Totals are in "
                    "each customer's local currency. Convert with these fixed rates to USD: GBP 1.27, EUR 1.08, "
                    "CAD 0.73, NGN 0.00065, GHS 0.065. The reference version includes order_total_usd.",
        domain="Business", tasks=["Cleaning", "Visualization"], level="Intermediate",
        characteristics=["Tabular", "Time-series"],
        variables=[
            ("order_id", "ID", "Categorical", "Order identifier", ""),
            ("order_date", "Feature", "Date", "Order date (three formats)", "date"),
            ("customer_id", "Feature", "Categorical", "Customer identifier (customers can repeat)", ""),
            ("country", "Feature", "Categorical", "Shipping country", ""),
            ("currency", "Feature", "Categorical", "Currency of order_total", ""),
            ("product_category", "Feature", "Categorical", "Main product category", ""),
            ("items", "Feature", "Integer", "Number of items", ""),
            ("order_total", "Feature", "Continuous", "Order value in local currency", "local currency"),
            ("delivery_days", "Feature", "Integer", "Days from order to delivery", "days"),
            ("returned", "Feature", "Binary", "Order returned (Yes/No)", ""),
            ("rating", "Feature", "Continuous", "Customer rating, if given", "1-5"),
        ],
        issues=["Totals in six currencies: must be converted before comparing",
                "Dates in three formats (2025-03-04, 03/04/2025, 04.03.2025)",
                "Country names spelled many ways ('USA', 'U.S.', 'England', 'DE')",
                "30 negative delivery times", "Missing ratings (customers who did not rate)", "45 duplicated orders"],
        exercises=["Convert every order to USD and compare revenue by country",
                   "Parse the dates, keeping in mind that 03/04 and 04.03 mean different things",
                   "Plot monthly revenue by product category",
                   "Do late deliveries get lower ratings?"])

# ======================================================================
# 12. Bike share
# ======================================================================
def bikes():
    rng = np.random.default_rng(1212); n = 7000
    stations = ["Union Square", "City Hall", "Riverfront", "Museum Row", "Campus Gate", "Market St & 5th",
                "Harbor Pier", "Central Library", "Elm & Main", "Stadium", "Botanical Garden", "Tech Park"]
    day = pd.Timestamp("2025-04-01") + pd.to_timedelta(rng.integers(0, 153, n), unit="D")
    member = rng.choice(["Member", "Casual"], n, p=[.68, .32])
    hour = np.where(member == "Member", rng.choice([7.5, 8.5, 12.5, 17.5, 18.5], n) + rng.normal(0, 1, n), rng.normal(14, 3, n)).clip(5, 23.5)
    start = (day + pd.to_timedelta(hour, unit="h")).floor("s")
    bike = rng.choice(["Classic", "Electric"], n, p=[.6, .4])
    dur = np.where(member == "Member", rng.gamma(2.5, 5, n), rng.gamma(2.2, 11, n)) * np.where(bike == "Electric", .8, 1)
    end = start + pd.to_timedelta(dur, unit="m")
    s1 = rng.choice(stations, n); s2 = rng.choice(stations, n)
    temp_by_day = {d: 58 + 25 * np.sin(np.pi * (d - pd.Timestamp('2025-04-01')).days / 200) + rng.normal(0, 5) for d in pd.unique(day)}
    df = pd.DataFrame({
        "trip_id": [f"GL{900000+i}" for i in range(n)], "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end.floor("s").strftime("%Y-%m-%d %H:%M:%S"), "start_station": s1, "end_station": s2,
        "bike_type": bike, "member_type": member, "duration_min": dur.round(1),
        "distance_km": (dur * np.where(bike == "Electric", .3, .22) * rng.uniform(.5, 1.1, n)).round(2),
        "temperature_f": [round(temp_by_day[d], 0) for d in day],
    }).sort_values("start_time").reset_index(drop=True)
    clean = df.copy()
    raw = df.copy()
    i = rng.choice(n, 25, replace=False); raw.loc[i, "duration_min"] = -raw.loc[i, "duration_min"]
    i = rng.choice(n, 15, replace=False); raw.loc[i, "duration_min"] = rng.uniform(1500, 6000, 15).round(1)
    for col in ["start_station", "end_station"]:
        raw[col] = variants(raw[col], {"Market St & 5th": ["Market St and 5th", "Market & 5th St"],
            "Elm & Main": ["Elm and Main", "Elm St & Main St"], "Campus Gate": ["Campus Gate (temp)", "campus gate"]}, .2, rng)
    blank(raw, "end_station", .03, rng); blank(raw, "distance_km", .02, rng)
    register("glide-bikeshare", "Glide Bike-Share Trips", raw, clean,
        summary="Five months of trips on a city bike-share system.",
        description="7,000 trips on a fictional bike-share network, April to August 2025: stations, times, bike "
                    "type, rider type, distance and the day's temperature. Members commute at rush hour; casual "
                    "riders take longer afternoon trips.",
        domain="Transportation", tasks=["Cleaning", "Time series", "Visualization"], level="Beginner",
        characteristics=["Tabular", "Time-series"],
        variables=[
            ("trip_id", "ID", "Categorical", "Trip identifier", ""),
            ("start_time", "Feature", "Date", "Trip start", "date-time"),
            ("end_time", "Feature", "Date", "Trip end", "date-time"),
            ("start_station", "Feature", "Categorical", "Station where the trip began", ""),
            ("end_station", "Feature", "Categorical", "Station where the bike was returned", ""),
            ("bike_type", "Feature", "Categorical", "Classic or Electric", ""),
            ("member_type", "Feature", "Categorical", "Member or Casual rider", ""),
            ("duration_min", "Feature", "Continuous", "Trip length", "minutes"),
            ("distance_km", "Feature", "Continuous", "Estimated distance", "km"),
            ("temperature_f", "Feature", "Continuous", "Average temperature that day", "°F"),
        ],
        issues=["25 negative durations", "15 trips lasting more than a day (bikes not docked)",
                "Station names spelled several ways ('Elm and Main', 'campus gate')",
                "Missing end station (3%) and distance (2%)", "duration_min disagrees with end - start for the bad rows"],
        exercises=["Recompute durations from the timestamps and compare",
                   "Plot trips by hour for members and casual riders",
                   "Find the busiest station pairs",
                   "Does ridership rise with temperature?"])

# ======================================================================
# 13. Fictional nations indicators
# ======================================================================
def nations():
    rng = np.random.default_rng(1313)
    regions = {"Northern Archipelago": 6, "Central Highlands": 6, "Southern Coast": 6, "Eastern Plains": 6, "Western Savanna": 6}
    syll = ["Al", "Bor", "Cal", "Dar", "El", "Fen", "Gal", "Hal", "Is", "Jor", "Kal", "Lum", "Mor", "Nav", "Or", "Pel",
            "Qua", "Ros", "Sel", "Tor", "Ul", "Ver", "Wyn", "Zan"]
    ends = ["ia", "land", "stan", "ora", "ica", "enne", "ovia", "adia"]
    names = set()
    while len(names) < 30:
        names.add(rng.choice(syll) + rng.choice(syll).lower() + rng.choice(ends))
    names = sorted(names); rows = []; k = 0
    for reg, cnt in regions.items():
        rbase = {"Northern Archipelago": 1.4, "Central Highlands": .9, "Southern Coast": 1.1, "Eastern Plains": .7, "Western Savanna": .55}[reg]
        for _ in range(cnt):
            c = names[k]; k += 1
            g0 = rbase * rng.lognormal(8.6, .5); growth = rng.normal(.03, .015)
            pop0 = rng.lognormal(2.5, 1); pg = rng.normal(.015, .008)
            for y in range(2000, 2025):
                t = y - 2000
                gdp = g0 * (1 + growth) ** t * (0.93 if y in (2009, 2020) else 1) * rng.lognormal(0, .02)
                le = 52 + 7 * np.log10(gdp) - 10 + .15 * t + rng.normal(0, .3)
                rows.append(dict(country=c, region=reg, year=y, population_millions=round(pop0 * (1 + pg) ** t, 2),
                                 gdp_per_capita_usd=round(gdp, 0), life_expectancy=round(min(le, 86), 1),
                                 literacy_rate_pct=round(min(99.5, 40 + 12 * np.log10(gdp) - 20 + .6 * t + rng.normal(0, 1)), 1),
                                 co2_tonnes_per_capita=round(max(.1, .0006 * gdp * rng.lognormal(0, .1)), 2),
                                 internet_users_pct=round(100 / (1 + np.exp(-(t - 12 + np.log10(gdp) * 2 - 8) / 3)), 1),
                                 urban_pct=round(min(95, 30 + 5 * np.log10(gdp) + .5 * t + rng.normal(0, 1)), 1)))
    df = pd.DataFrame(rows)
    clean = df.copy()
    raw = df.copy().astype({"gdp_per_capita_usd": object})
    raw["gdp_per_capita_usd"] = [f"{v:,.0f}" for v in df.gdp_per_capita_usd]
    for c in ["literacy_rate_pct", "co2_tonnes_per_capita", "internet_users_pct"]:
        blank(raw, c, .07, rng)
    raw = raw.drop(raw.sample(frac=.03, random_state=5).index)
    raw["region"] = variants(raw["region"], {"Western Savanna": ["W. Savanna", "Western savanna"],
                                             "Northern Archipelago": ["N. Archipelago"]}, .1, rng)
    raw = raw.reset_index(drop=True)
    register("nations-indicators", "Fictional Nations Indicators", raw, clean,
        summary="25 years of development indicators for 30 invented countries.",
        description="A panel of population, income, life expectancy, literacy, emissions, internet use and "
                    "urbanization for 30 invented countries in five regions, 2000-2024. Income dips in 2009 and "
                    "2020. Built for line charts, small multiples and reshaping between long and wide formats.",
        domain="Social Science", tasks=["Visualization", "Time series", "Cleaning"], level="Beginner",
        characteristics=["Time-series", "Panel", "Multivariate"],
        variables=[
            ("country", "ID", "Categorical", "Country (fictional)", ""),
            ("region", "Feature", "Categorical", "World region (fictional)", ""),
            ("year", "Feature", "Integer", "Year", "year"),
            ("population_millions", "Feature", "Continuous", "Population", "millions"),
            ("gdp_per_capita_usd", "Feature", "Text", "GDP per person, as text with commas", "USD"),
            ("life_expectancy", "Feature", "Continuous", "Life expectancy at birth", "years"),
            ("literacy_rate_pct", "Feature", "Continuous", "Adult literacy rate", "%"),
            ("co2_tonnes_per_capita", "Feature", "Continuous", "CO₂ emissions per person", "tonnes"),
            ("internet_users_pct", "Feature", "Continuous", "Share of people online", "%"),
            ("urban_pct", "Feature", "Continuous", "Share living in cities", "%"),
        ],
        issues=["gdp_per_capita_usd stored as text with thousands separators",
                "About 3% of country-years missing entirely", "Missing literacy, CO₂ and internet values (7% each)",
                "Region names abbreviated in some rows"],
        exercises=["Plot life expectancy over time for each region (small multiples)",
                   "Make a scatter of GDP per capita (log scale) vs. life expectancy for 2024",
                   "Pivot the data to wide format with one column per year, then back",
                   "Fill missing years by interpolation within each country"])

# ======================================================================
# 14. Restaurant inspections
# ======================================================================
def inspections():
    rng = np.random.default_rng(1414); n_rest = 600; n = 3500
    cuisines = ["American", "Pizza", "Chinese", "Mexican", "Caribbean", "Ethiopian", "Indian", "Italian", "Café", "Soul Food", "Japanese", "West African"]
    pre = ["Golden", "Mama's", "Blue", "Uncle", "Corner", "Lucky", "Royal", "Little", "Sunset", "Happy"]
    noun = ["Kitchen", "Grill", "Garden", "Diner", "Spot", "House", "Bistro", "Pot", "Table", "Plate"]
    rest = pd.DataFrame({"restaurant_id": [f"RV{3000+i}" for i in range(n_rest)],
                         "restaurant_name": [f"{rng.choice(pre)} {rng.choice(noun)}" for _ in range(n_rest)],
                         "cuisine": rng.choice(cuisines, n_rest),
                         "address": [f"{rng.integers(1, 2400)} {rng.choice(['Main St', 'Oak Ave', 'River Rd', 'Harbor Blvd', 'Elm St', 'King Ave', 'Market St'])}" for _ in range(n_rest)],
                         "quality": rng.normal(0, 1, n_rest)})
    viol = [("04L", "Evidence of mice or live mice in food areas", "Critical"),
            ("02B", "Hot food item not held at or above 140 °F", "Critical"),
            ("06C", "Food not protected from potential contamination", "Critical"),
            ("10F", "Non-food contact surface improperly maintained", "Not Critical"),
            ("08A", "Facility not vermin proof", "Not Critical"),
            ("10B", "Plumbing not properly installed or maintained", "Not Critical"),
            ("02G", "Cold food item held above 41 °F", "Critical"),
            ("09C", "Food contact surface not properly maintained", "Not Critical")]
    pick = rest.sample(n, replace=True, random_state=7).reset_index(drop=True)
    score = (14 - 6 * pick.quality + rng.gamma(2, 3, n)).clip(0, 60).round(0).astype(int)
    v = rng.integers(0, len(viol), n)
    df = pd.DataFrame({
        "inspection_id": [f"INSP-{600000+i}" for i in range(n)], "restaurant_id": pick.restaurant_id,
        "restaurant_name": pick.restaurant_name, "cuisine": pick.cuisine, "address": pick.address,
        "inspection_date": (pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")).strftime("%Y-%m-%d"),
        "score": score, "grade": np.where(score <= 13, "A", np.where(score <= 27, "B", "C")),
        "violation_code": [viol[k][0] for k in v], "violation_description": [viol[k][1] for k in v],
        "critical_flag": [viol[k][2] for k in v],
    }).sort_values("inspection_date").reset_index(drop=True)
    clean = df.copy()
    raw = df.copy()
    def mess(name):
        r = rng.random()
        if r < .08: return name.upper()
        if r < .14: return name.replace("'", "")
        if r < .18: return name + " "
        if r < .21: return name.replace(" ", "  ")
        return name
    raw["restaurant_name"] = [mess(x) for x in raw.restaurant_name]
    raw["cuisine"] = variants(raw["cuisine"], {"Café": ["Cafe", "Coffee/Tea"], "Soul Food": ["Soul food", "Southern"],
                                               "West African": ["African", "Nigerian"], "Pizza": ["Pizza/Italian"]}, .12, rng)
    i = rng.choice(n, 60, replace=False); raw.loc[i, "grade"] = rng.choice(["A", "B", "C"], 60)
    blank(raw, "grade", .06, rng)
    raw["critical_flag"] = variants(raw["critical_flag"], {"Critical": ["Y", "critical"], "Not Critical": ["N", "Not critical"]}, .08, rng)
    register("riverton-inspections", "Riverton Restaurant Inspections", raw, clean,
        summary="Health inspection results for 600 restaurants, with messy names.",
        description="3,500 inspections of 600 fictional restaurants in Riverton, 2023-2025. Lower scores are "
                    "better: 0-13 earns an A, 14-27 a B, 28+ a C. The same restaurant is often spelled several "
                    "ways, so matching names is part of the job.",
        domain="Government", tasks=["Cleaning", "Visualization", "Text"], level="Intermediate",
        characteristics=["Tabular", "Text"],
        variables=[
            ("inspection_id", "ID", "Categorical", "Inspection identifier", ""),
            ("restaurant_id", "ID", "Categorical", "Restaurant identifier", ""),
            ("restaurant_name", "Feature", "Text", "Restaurant name as typed by the inspector", ""),
            ("cuisine", "Feature", "Categorical", "Type of cuisine", ""),
            ("address", "Feature", "Text", "Street address", ""),
            ("inspection_date", "Feature", "Date", "Date of inspection", "date"),
            ("score", "Feature", "Integer", "Violation points (lower is better)", "points"),
            ("grade", "Target", "Categorical", "Letter grade (A, B or C)", ""),
            ("violation_code", "Feature", "Categorical", "Code of the main violation", ""),
            ("violation_description", "Feature", "Text", "Description of the main violation", ""),
            ("critical_flag", "Feature", "Categorical", "Whether the violation is critical", ""),
        ],
        issues=["Restaurant names in different case, with or without apostrophes, with extra spaces",
                "Cuisine labels overlap ('Cafe' vs 'Café', 'African' vs 'West African')",
                "60 grades that do not match the score rules", "Missing grades (6%)",
                "critical_flag coded several ways"],
        exercises=["Standardize restaurant names and check that each restaurant_id has one name",
                   "Recompute grades from scores and count the mismatches",
                   "Which cuisines have the highest average scores?",
                   "Plot the share of A grades by month"])

# ======================================================================
# 15. Employee records
# ======================================================================
def hr():
    rng = np.random.default_rng(1515); n = 1800
    depts = {"Engineering": 105000, "Data Science": 112000, "Sales": 72000, "Marketing": 76000, "Finance": 88000,
             "Human Resources": 68000, "Customer Support": 52000, "Operations": 64000}
    dept = rng.choice(list(depts), n, p=[.22, .1, .17, .1, .09, .06, .14, .12])
    level = rng.choice([1, 2, 3, 4, 5], n, p=[.25, .3, .25, .14, .06])
    age = (22 + level * 4 + rng.normal(0, 5, n)).clip(21, 66).round(0).astype(int)
    hire = pd.Timestamp("2025-06-30") - pd.to_timedelta(rng.gamma(2, 700, n).clip(30, 12000), unit="D")
    salary = (np.array([depts[d] for d in dept]) * (1 + .22 * (level - 2)) * rng.lognormal(0, .08, n)).round(-2)
    perf = rng.choice([1, 2, 3, 4, 5], n, p=[.04, .14, .48, .26, .08])
    sat = (rng.normal(3.4, .9, n) + .2 * (perf - 3)).clip(1, 5).round(1)
    tenure = (pd.Timestamp("2025-06-30") - hire).days / 365
    logit = -1.5 - .9 * (sat - 3.4) - .12 * tenure + .4 * (dept == "Customer Support") - .2 * (level - 2) + rng.normal(0, .5, n)
    df = pd.DataFrame({
        "employee_id": [f"E{4000+i}" for i in range(n)], "department": dept, "job_level": level, "age": age,
        "gender": rng.choice(["Female", "Male", "Non-binary"], n, p=[.47, .5, .03]),
        "education": rng.choice(["High School", "Bachelor's", "Master's", "PhD"], n, p=[.1, .55, .3, .05]),
        "hire_date": hire.strftime("%Y-%m-%d"), "salary_usd": salary.astype(int),
        "bonus_pct": (rng.gamma(2, 2.5, n) * (1 + .1 * perf)).round(1), "performance_rating": perf,
        "remote_status": rng.choice(["On-site", "Hybrid", "Remote"], n, p=[.35, .45, .2]),
        "training_hours": rng.poisson(18, n), "satisfaction": sat,
        "left_company": np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No"),
    })
    clean = df.copy()
    raw = df.copy().astype({"salary_usd": float})
    i = rng.choice(n, 90, replace=False); raw.loc[i, "salary_usd"] = (raw.loc[i, "salary_usd"] / 1000).round(1)
    raw["department"] = variants(raw["department"], {"Human Resources": ["HR", "Human Res."], "Customer Support": ["Support", "Cust. Support"],
                                                     "Data Science": ["Data Sci", "Analytics"], "Engineering": ["Eng", "engineering"]}, .1, rng)
    i = rng.choice(n, 8, replace=False); raw.loc[i, "hire_date"] = "2026-03-15"
    blank(raw, "satisfaction", .07, rng); blank(raw, "education", .03, rng)
    register("summit-hr", "Summit Analytics Employee Records", raw, clean,
        summary="HR records for 1,800 employees, with salaries on two scales.",
        description="Employee records from a fictional analytics company: department, level, pay, performance, "
                    "work arrangement, satisfaction and whether the person left. Some salaries were entered in "
                    "thousands, which is a scaling problem students must spot before any modeling.",
        domain="Business", tasks=["Scaling", "Classification", "Cleaning", "Visualization"], level="Intermediate",
        characteristics=["Tabular", "Multivariate"], target="left_company",
        variables=[
            ("employee_id", "ID", "Categorical", "Employee identifier", ""),
            ("department", "Feature", "Categorical", "Department", ""),
            ("job_level", "Feature", "Integer", "Seniority level (1 = entry, 5 = director)", "1-5"),
            ("age", "Feature", "Integer", "Age", "years"),
            ("gender", "Feature", "Categorical", "Self-reported gender", ""),
            ("education", "Feature", "Categorical", "Highest degree", ""),
            ("hire_date", "Feature", "Date", "Start date", "date"),
            ("salary_usd", "Feature", "Continuous", "Annual base salary (some entered in thousands)", "USD"),
            ("bonus_pct", "Feature", "Continuous", "Last bonus as a share of salary", "%"),
            ("performance_rating", "Feature", "Integer", "Latest review rating", "1-5"),
            ("remote_status", "Feature", "Categorical", "On-site, Hybrid or Remote", ""),
            ("training_hours", "Feature", "Integer", "Training completed this year", "hours"),
            ("satisfaction", "Feature", "Continuous", "Engagement survey score", "1-5"),
            ("left_company", "Target", "Binary", "Left within the following year", ""),
        ],
        issues=["90 salaries entered in thousands (e.g. 85.4 instead of 85,400)",
                "Department names abbreviated ('HR', 'Eng', 'Analytics')", "8 hire dates in the future",
                "Missing satisfaction (7%) and education (3%)"],
        exercises=["Find the salaries on the wrong scale with a histogram, then fix them",
                   "Compute tenure from hire_date",
                   "Compare salary distributions by department and level",
                   "Scale the features and predict who leaves"])

# ======================================================================
# 16. Basketball league
# ======================================================================
def hoops():
    rng = np.random.default_rng(1616)
    teams = ["Riverton Rapids", "Harbor City Gulls", "Maple Valley Owls", "Summit Peaks", "Cedar Park Lynx",
             "Old Port Anchors", "Lighthouse Beacons", "North Shore Waves"]
    players = person_names(rng, 160)
    pos = rng.choice(["Guard", "Forward", "Center"], 160, p=[.42, .38, .2])
    skill = rng.normal(0, 1, 160)
    rows = []
    for season in range(2021, 2026):
        for i, p in enumerate(players):
            if rng.random() < .15: continue
            g = int(rng.integers(20, 41)); mins = float(np.clip(22 + 6 * skill[i] + rng.normal(0, 4), 6, 38))
            ps = pos[i]
            rows.append(dict(player=p, team=teams[i % 8] if rng.random() > .08 else rng.choice(teams), position=ps, season=season,
                             age=int(20 + (i % 15) + season - 2021), height_cm=round({"Guard": 188, "Forward": 201, "Center": 211}[ps] + rng.normal(0, 4), 0),
                             games=g, minutes_per_game=round(mins, 1),
                             points_per_game=round(max(0, mins * (.45 + .08 * skill[i]) + rng.normal(0, 2)), 1),
                             rebounds_per_game=round(max(0, mins * {"Guard": .12, "Forward": .22, "Center": .32}[ps] + rng.normal(0, 1)), 1),
                             assists_per_game=round(max(0, mins * {"Guard": .18, "Forward": .08, "Center": .05}[ps] + rng.normal(0, .8)), 1),
                             fg_pct=round(float(np.clip({"Guard": .44, "Forward": .47, "Center": .54}[ps] + .02 * skill[i] + rng.normal(0, .03), .3, .7)), 3),
                             three_pct=(round(float(np.clip(.35 + .02 * skill[i] + rng.normal(0, .04), .15, .5)), 3) if ps != "Center" or rng.random() < .3 else np.nan),
                             ft_pct=round(float(np.clip(.76 + rng.normal(0, .07), .45, .95)), 3)))
    df = pd.DataFrame(rows)
    clean = df.copy()
    raw = df.copy()
    m = (raw.season == 2023)
    for c in ["fg_pct", "three_pct", "ft_pct"]:
        raw.loc[m, c] = (raw.loc[m, c] * 100).round(1)
    i = rng.choice(len(raw), 12, replace=False); raw.loc[i, "height_cm"] = (raw.loc[i, "height_cm"] / 2.54).round(0)
    raw = dupes(raw, 15, rng)
    register("metro-hoops", "Metro Basketball League Stats", raw, clean,
        summary="Five seasons of per-game averages for a fictional league.",
        description="Season averages for about 150 fictional players on eight teams, 2021-2025: minutes, points, "
                    "rebounds, assists and shooting percentages. Great for comparing positions, ranking players "
                    "and clustering playing styles.",
        domain="Sports", tasks=["Visualization", "Clustering", "Scaling"], level="Beginner",
        characteristics=["Tabular", "Panel"],
        variables=[
            ("player", "ID", "Categorical", "Player name (fictional)", ""),
            ("team", "Feature", "Categorical", "Team that season", ""),
            ("position", "Feature", "Categorical", "Guard, Forward or Center", ""),
            ("season", "Feature", "Integer", "Season year", "year"),
            ("age", "Feature", "Integer", "Age during the season", "years"),
            ("height_cm", "Feature", "Continuous", "Height", "cm"),
            ("games", "Feature", "Integer", "Games played", ""),
            ("minutes_per_game", "Feature", "Continuous", "Average minutes", "minutes"),
            ("points_per_game", "Feature", "Continuous", "Average points", "points"),
            ("rebounds_per_game", "Feature", "Continuous", "Average rebounds", ""),
            ("assists_per_game", "Feature", "Continuous", "Average assists", ""),
            ("fg_pct", "Feature", "Continuous", "Field-goal percentage", "share (0-1)"),
            ("three_pct", "Feature", "Continuous", "Three-point percentage (blank if no attempts)", "share (0-1)"),
            ("ft_pct", "Feature", "Continuous", "Free-throw percentage", "share (0-1)"),
        ],
        issues=["The 2023 season stores percentages as 0-100 instead of 0-1",
                "12 heights entered in inches", "three_pct blank for centers who never shoot threes (legitimately missing)",
                "15 duplicated player-seasons"],
        exercises=["Detect the season with percentages on the wrong scale",
                   "Compare points per game by position with box plots",
                   "Scale the per-game stats and cluster players into styles",
                   "Plot the top scorer's points across the five seasons"])

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "more_datasets.py")).read())

for f in [student_success, riverton_311, harbor_rentals, coffee, clinic, tracks, beer, housing, air,
          churn, orders, bikes, nations, inspections, hr, hoops,
          malaria, tb, dementia, heart, covid, flu, cold, crime, gun_violence, flights, fuel]:
    f()

with open(os.path.join(os.path.dirname(OUT), "catalog.json"), "w") as fh:
    json.dump(CATALOG, fh, indent=1, default=str)
print(len(CATALOG), "datasets;", round(sum(c["size_kb"] + c["clean_size_kb"] for c in CATALOG) / 1024, 2), "MB")
