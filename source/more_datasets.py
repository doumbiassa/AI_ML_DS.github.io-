# Additional datasets (health, public safety, transportation, energy).
# Imported by generate.py; uses its helpers (blank, dupes, variants, mixed_dates, register, person_names).

def _seasonal(t, period, phase=0):
    return np.sin(2 * np.pi * (t - phase) / period)

# ======================================================================
# 17. Malaria surveillance
# ======================================================================
def malaria():
    rng = np.random.default_rng(1717)
    regions = {"Lakeshore": ["Kumbi", "Tanaro", "Selu", "Mbaro"], "Highlands": ["Ondasi", "Kerewa", "Lutoro", "Ivuna"],
               "Savanna": ["Bamako Flats", "Dizani", "Owera", "Kisanto"], "Coast": ["Port Alira", "Masuva", "Tembe", "Nyali"],
               "Delta": ["Rufaro", "Zimba", "Okolo", "Ebute"]}
    burden = {"Lakeshore": 1.5, "Highlands": .35, "Savanna": 1.0, "Coast": 1.2, "Delta": 1.7}
    months = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
    rows = []
    for reg, dists in regions.items():
        for d in dists:
            pop0 = rng.integers(120_000, 900_000); itn0 = rng.uniform(35, 60)
            for k, m in enumerate(months):
                rain = max(0, 110 + 95 * _seasonal(m.month, 12, 1) + rng.normal(0, 25))
                temp = 24 + 3 * _seasonal(m.month, 12, 0) - (6 if reg == "Highlands" else 0) + rng.normal(0, .8)
                itn = min(92, itn0 + k * .45 + rng.normal(0, 2))
                lag_rain = 110 + 95 * _seasonal(m.month - 2, 12, 1)
                inc = burden[reg] * (3 + .12 * lag_rain) * (1 - itn / 140) * rng.lognormal(0, .18)  # per 1,000
                pop = int(pop0 * (1.025 ** (k / 12)))
                susp = int(pop * inc / 1000 * rng.uniform(1.3, 1.7))
                tests = int(susp * rng.uniform(.85, 1.0))
                conf = int(min(tests, pop * inc / 1000))
                rows.append(dict(region=reg, district=d, year=m.year, month=m.month, population=pop,
                                 rainfall_mm=round(rain, 1), avg_temp_c=round(temp, 1), bednet_coverage_pct=round(itn, 1),
                                 suspected_cases=susp, tests_done=tests, confirmed_cases=conf,
                                 deaths=int(rng.poisson(conf * .0012))))
    df = pd.DataFrame(rows)
    clean = df.copy()
    raw = df.copy().astype({"month": object, "population": object})
    mnames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m = rng.random(len(raw)) < .3
    raw.loc[m, "month"] = [mnames[int(x) - 1] for x in df.loc[m, "month"]]
    m = rng.random(len(raw)) < .2
    raw.loc[m, "population"] = [f"{p/1e6:.2f}M" if p >= 1e6 else f"{p/1e3:.0f}K" for p in df.loc[m, "population"]]
    i = rng.choice(len(raw), 30, replace=False); raw.loc[i, "confirmed_cases"] = raw.loc[i, "tests_done"] + rng.integers(50, 500, 30)
    for c in ["confirmed_cases", "tests_done", "rainfall_mm"]:
        blank(raw, c, .04, rng)
    raw["district"] = variants(raw["district"], {"Port Alira": ["Pt. Alira", "PORT ALIRA"], "Bamako Flats": ["Bamako flats", "Bamako-Flats"]}, .25, rng)
    raw = dupes(raw, 20, rng)
    register("kalindra-malaria", "Kalindra Malaria Surveillance", raw, clean,
        summary="Monthly malaria cases, tests and rainfall for 20 districts.",
        description="Six years (2019-2024) of monthly malaria reports from 20 districts of the fictional country of "
                    "Kalindra, with rainfall, temperature and bed-net coverage. Cases peak about two months after the "
                    "rains, and the highlands see far fewer cases. Synthetic data for teaching only.",
        domain="Health", tasks=["Time series", "Cleaning", "Visualization"], level="Intermediate",
        characteristics=["Time-series", "Panel", "Multivariate"],
        variables=[
            ("region", "Feature", "Categorical", "Region", ""), ("district", "ID", "Categorical", "District", ""),
            ("year", "Feature", "Integer", "Year", "year"), ("month", "Feature", "Categorical", "Month (number or short name)", ""),
            ("population", "Feature", "Text", "District population (some written as '1.2M' or '450K')", "people"),
            ("rainfall_mm", "Feature", "Continuous", "Total rainfall that month", "mm"),
            ("avg_temp_c", "Feature", "Continuous", "Average temperature", "°C"),
            ("bednet_coverage_pct", "Feature", "Continuous", "Households with insecticide-treated nets", "%"),
            ("suspected_cases", "Feature", "Integer", "Patients with suspected malaria", "cases"),
            ("tests_done", "Feature", "Continuous", "Diagnostic tests performed", "tests"),
            ("confirmed_cases", "Target", "Continuous", "Laboratory or rapid-test confirmed cases", "cases"),
            ("deaths", "Feature", "Integer", "Malaria deaths", "deaths"),
        ],
        issues=["month written as a number in some rows and a short name ('Jul') in others",
                "population written as text like '1.20M' or '450K' in 20% of rows",
                "30 rows with more confirmed cases than tests (impossible)",
                "Missing confirmed cases, tests and rainfall (4% each)",
                "District names spelled several ways ('Pt. Alira', 'Bamako-Flats')", "20 duplicated reports"],
        exercises=["Build a proper date column from year and month",
                   "Compute incidence per 1,000 people and test positivity",
                   "Plot cases and rainfall together: how long is the lag?",
                   "Compare districts with small multiples or a heatmap of month × district"])

# ======================================================================
# 18. Tuberculosis treatment outcomes
# ======================================================================
def tb():
    rng = np.random.default_rng(1818); n = 3000
    age = rng.gamma(5, 7, n).clip(5, 88).round(0).astype(int)
    sex = rng.choice(["Male", "Female"], n, p=[.62, .38])
    hiv = rng.choice(["Negative", "Positive", "Unknown"], n, p=[.7, .18, .12])
    mdr = rng.random(n) < .07
    adh = (rng.beta(8, 1.5, n) * 100).round(1)
    months = np.where(mdr, rng.choice([9, 18, 20], n), 6)
    w0 = rng.normal(52, 9, n).clip(25, 95).round(1)
    logit = (2.2 + .04 * (adh - 85) - 1.1 * mdr - .9 * (hiv == "Positive") - .015 * (age - 35) + rng.normal(0, .6, n))
    success = rng.random(n) < 1 / (1 + np.exp(-logit))
    outcome = np.where(success, rng.choice(["Cured", "Treatment completed"], n, p=[.6, .4]),
                       rng.choice(["Failed", "Died", "Lost to follow-up"], n, p=[.25, .35, .4]))
    w1 = np.where(success, w0 + rng.normal(4, 2, n), w0 + rng.normal(-1, 3, n)).round(1)
    diag = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 1095, n), unit="D")
    df = pd.DataFrame({
        "patient_id": [f"TB{30000+i}" for i in range(n)], "age": age, "sex": sex,
        "region": rng.choice(["North", "South", "East", "West", "Capital"], n),
        "diagnosis_date": diag.strftime("%Y-%m-%d"), "hiv_status": hiv, "diabetes": rng.choice(["No", "Yes"], n, p=[.88, .12]),
        "smear_result": rng.choice(["Positive", "Negative", "Not done"], n, p=[.6, .3, .1]),
        "drug_resistance": np.where(mdr, "MDR", "Drug-sensitive"), "planned_months": months,
        "weight_start_kg": w0, "weight_end_kg": w1, "adherence_pct": adh, "outcome": outcome,
    })
    df["treatment_success"] = np.where(success, 1, 0)
    clean = df.copy()
    raw = df.copy()
    raw["diagnosis_date"] = mixed_dates(pd.to_datetime(df.diagnosis_date), rng, fmts=("%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"), probs=(.6, .3, .1))
    raw["hiv_status"] = variants(raw["hiv_status"], {"Positive": ["pos", "+", "HIV+"], "Negative": ["neg", "-", "HIV-"], "Unknown": ["?", "Not tested"]}, .15, rng)
    raw["outcome"] = variants(raw["outcome"], {"Lost to follow-up": ["LTFU", "Defaulted"], "Treatment completed": ["Completed"], "Died": ["Deceased"]}, .15, rng)
    i = rng.choice(n, 12, replace=False); raw.loc[i, "weight_start_kg"] = 0
    i = rng.choice(n, 15, replace=False); raw.loc[i, "adherence_pct"] = raw.loc[i, "adherence_pct"] + 100
    blank(raw, "weight_end_kg", .1, rng); blank(raw, "adherence_pct", .05, rng)
    register("tb-outcomes", "Tuberculosis Treatment Outcomes", raw, clean,
        summary="3,000 TB patients, their risk factors and how treatment ended.",
        description="Patient-level records from a fictional national TB program, 2021-2023: age, HIV status, drug "
                    "resistance, adherence, weight change and the treatment outcome. The target treatment_success "
                    "is 1 for Cured or Treatment completed. Synthetic data for teaching only.",
        domain="Health", tasks=["Classification", "Cleaning", "Visualization"], level="Intermediate",
        characteristics=["Tabular", "Multivariate"], target="treatment_success",
        variables=[
            ("patient_id", "ID", "Categorical", "Patient identifier", ""), ("age", "Feature", "Integer", "Age at diagnosis", "years"),
            ("sex", "Feature", "Categorical", "Sex", ""), ("region", "Feature", "Categorical", "Treatment region", ""),
            ("diagnosis_date", "Feature", "Date", "Date of diagnosis (three formats)", "date"),
            ("hiv_status", "Feature", "Categorical", "HIV status", ""), ("diabetes", "Feature", "Binary", "Has diabetes", ""),
            ("smear_result", "Feature", "Categorical", "Sputum smear result", ""),
            ("drug_resistance", "Feature", "Categorical", "Drug-sensitive or multidrug-resistant (MDR)", ""),
            ("planned_months", "Feature", "Integer", "Planned treatment length", "months"),
            ("weight_start_kg", "Feature", "Continuous", "Weight at start of treatment", "kg"),
            ("weight_end_kg", "Feature", "Continuous", "Weight at end of treatment", "kg"),
            ("adherence_pct", "Feature", "Continuous", "Doses taken as planned", "%"),
            ("outcome", "Feature", "Categorical", "Final outcome", ""),
            ("treatment_success", "Target", "Binary", "1 = Cured or Treatment completed", ""),
        ],
        issues=["Diagnosis dates in three formats; note that 03/04/2022 here means 3 April (day first)",
                "HIV status coded many ways ('pos', '+', 'HIV+', '?')",
                "Outcome labels with synonyms ('LTFU', 'Defaulted', 'Deceased')",
                "12 starting weights of 0", "15 adherence values above 100%",
                "Missing end weight (10%) and adherence (5%)"],
        exercises=["Standardize HIV status and outcome labels",
                   "Compare success rates by HIV status and drug resistance",
                   "Plot weight change for successful and unsuccessful treatment",
                   "Train a classifier for treatment_success; which features matter most?"])

# ======================================================================
# 19. Dementia cohort
# ======================================================================
def dementia():
    rng = np.random.default_rng(1919); n = 600
    rows = []
    for p in range(n):
        age0 = rng.integers(60, 86); sex = rng.choice(["Female", "Male"], p=[.56, .44]); edu = int(rng.choice(range(6, 21)))
        apoe = rng.random() < .26; htn = rng.random() < .45
        risk = .02 * (age0 - 70) + .8 * apoe + .3 * htn - .05 * (edu - 12) + rng.normal(0, .7)
        decline = max(0, .35 + .55 * risk + rng.normal(0, .2))
        mmse0 = min(30, 28.5 - .6 * max(risk, 0) + rng.normal(0, 1))
        vol0 = 4200 - 18 * (age0 - 60) - 250 * max(risk, 0) + rng.normal(0, 200)
        visits = rng.integers(2, 6); d0 = pd.Timestamp("2016-01-01") + pd.Timedelta(days=int(rng.integers(0, 700)))
        for v in range(visits):
            yrs = v * rng.uniform(.9, 1.1) * 1.5
            mmse = round(float(np.clip(mmse0 - decline * yrs * 1.6 + rng.normal(0, .8), 0, 30)), 0)
            moca = round(float(np.clip(mmse - 2.5 + rng.normal(0, 1.2), 0, 30)), 0)
            cdr = 0 if mmse >= 27 else (.5 if mmse >= 22 else (1 if mmse >= 16 else 2))
            dx = "Normal" if cdr == 0 else ("MCI" if cdr == .5 else "Dementia")
            rows.append(dict(participant_id=f"DC{5000+p}", visit=v + 1, visit_date=(d0 + pd.Timedelta(days=int(yrs * 365))).strftime("%Y-%m-%d"),
                             age=round(age0 + yrs, 1), sex=sex, education_years=edu, apoe4_carrier="Yes" if apoe else "No",
                             hypertension="Yes" if htn else "No", bmi=round(float(rng.normal(26.5, 4)), 1),
                             mmse=int(mmse), moca=int(moca), cdr=cdr,
                             hippocampal_volume_mm3=round(vol0 - 40 * yrs * (1 + decline) + rng.normal(0, 60), 0), diagnosis=dx))
    df = pd.DataFrame(rows)
    clean = df.copy()
    raw = df.copy().astype({"education_years": object, "hippocampal_volume_mm3": float})
    m = rng.random(len(raw)) < .1; raw.loc[m, "education_years"] = [f"{x} years" for x in df.loc[m, "education_years"]]
    m = rng.random(len(raw)) < .12; raw.loc[m, "hippocampal_volume_mm3"] = (df.loc[m, "hippocampal_volume_mm3"] / 1000).round(2)
    i = rng.choice(len(raw), 8, replace=False); raw.loc[i, "mmse"] = raw.loc[i, "mmse"] + 30
    blank(raw, "moca", .08, rng); blank(raw, "hippocampal_volume_mm3", .1, rng); blank(raw, "bmi", .05, rng)
    raw["diagnosis"] = variants(raw["diagnosis"], {"MCI": ["Mild cognitive impairment", "mci"], "Dementia": ["AD", "dementia"], "Normal": ["CN", "Healthy"]}, .1, rng)
    register("memory-cohort", "Memory & Aging Cohort", raw, clean,
        summary="Repeated memory tests and brain measures for 600 older adults.",
        description="A fictional long-term study following 600 adults aged 60+ for up to six years. At each visit "
                    "participants took two memory tests (MMSE and MoCA, both 0-30) and had a brain scan measuring "
                    "hippocampal volume. Diagnosis moves from Normal to Mild Cognitive Impairment (MCI) to Dementia. "
                    "Synthetic data for teaching only.",
        domain="Health", tasks=["Classification", "Scaling", "Time series", "Visualization"], level="Advanced",
        characteristics=["Longitudinal", "Multivariate"], target="diagnosis",
        variables=[
            ("participant_id", "ID", "Categorical", "Participant identifier (repeats across visits)", ""),
            ("visit", "Feature", "Integer", "Visit number", ""), ("visit_date", "Feature", "Date", "Date of visit", "date"),
            ("age", "Feature", "Continuous", "Age at visit", "years"), ("sex", "Feature", "Categorical", "Sex", ""),
            ("education_years", "Feature", "Text", "Years of schooling (some written as '12 years')", "years"),
            ("apoe4_carrier", "Feature", "Binary", "Carries the APOE-ε4 gene variant", ""),
            ("hypertension", "Feature", "Binary", "Diagnosed high blood pressure", ""),
            ("bmi", "Feature", "Continuous", "Body-mass index", "kg/m²"),
            ("mmse", "Feature", "Integer", "Mini-Mental State Examination score", "0-30"),
            ("moca", "Feature", "Continuous", "Montreal Cognitive Assessment score", "0-30"),
            ("cdr", "Feature", "Continuous", "Clinical Dementia Rating (0, 0.5, 1, 2)", ""),
            ("hippocampal_volume_mm3", "Feature", "Continuous", "Hippocampal volume (some recorded in cm³)", "mm³"),
            ("diagnosis", "Target", "Categorical", "Normal, MCI or Dementia", ""),
        ],
        issues=["education_years stored as text in 10% of rows ('12 years')",
                "12% of hippocampal volumes recorded in cm³ (e.g. 3.85 instead of 3850)",
                "8 MMSE scores above the maximum of 30", "Missing MoCA (8%), volume (10%) and BMI (5%)",
                "Diagnosis labels with abbreviations ('CN', 'mci', 'AD')"],
        exercises=["Plot each participant's MMSE over time (spaghetti plot), colored by final diagnosis",
                   "Find and fix the volumes recorded in the wrong unit",
                   "Scale the features and classify diagnosis at the first visit",
                   "Why must you split train and test by participant, not by row?"])

# ======================================================================
# 20. Heart disease
# ======================================================================
def heart():
    rng = np.random.default_rng(2020); n = 1800
    age = rng.normal(55, 9, n).clip(29, 80).round(0).astype(int)
    sex = rng.choice(["Male", "Female"], n, p=[.66, .34])
    cp = rng.choice(["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], n, p=[.08, .17, .28, .47])
    bp = (118 + .5 * (age - 50) + rng.normal(0, 16, n)).round(0)
    chol = (210 + .8 * (age - 50) + rng.normal(0, 45, n)).clip(110, 520).round(0)
    fbs = rng.random(n) < .15
    maxhr = (205 - .9 * age + rng.normal(0, 18, n)).round(0)
    exang = rng.random(n) < .33
    oldpeak = np.abs(rng.gamma(1.2, .8, n) * (1 + exang)).round(1)
    vessels = rng.choice([0, 1, 2, 3], n, p=[.58, .22, .13, .07])
    logit = (-5 + .05 * age + .9 * (sex == "Male") + 1.3 * (cp == "Asymptomatic") + .012 * (bp - 120) + .004 * (chol - 200)
             - .025 * (maxhr - 150) + 1.0 * exang + .6 * oldpeak + .8 * vessels + rng.normal(0, .5, n))
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1), "age": age, "sex": sex, "chest_pain_type": cp, "resting_bp_mmhg": bp.astype(int),
        "cholesterol_mg_dl": chol.astype(int), "fasting_sugar_over_120": np.where(fbs, "Yes", "No"),
        "resting_ecg": rng.choice(["Normal", "ST-T abnormality", "LV hypertrophy"], n, p=[.5, .35, .15]),
        "max_heart_rate": maxhr.astype(int), "exercise_angina": np.where(exang, "Yes", "No"), "st_depression": oldpeak,
        "st_slope": rng.choice(["Upsloping", "Flat", "Downsloping"], n, p=[.45, .45, .1]), "major_vessels": vessels,
        "thallium_test": rng.choice(["Normal", "Fixed defect", "Reversible defect"], n, p=[.55, .07, .38]),
        "heart_disease": target,
    })
    clean = df.copy()
    raw = df.copy().astype({"chest_pain_type": object, "cholesterol_mg_dl": float, "resting_bp_mmhg": float})
    m = rng.random(n) < .2
    raw.loc[m, "chest_pain_type"] = df.loc[m, "chest_pain_type"].map({"Typical angina": "1", "Atypical angina": "2", "Non-anginal pain": "3", "Asymptomatic": "4"})
    i = rng.choice(n, 70, replace=False); raw.loc[i, "cholesterol_mg_dl"] = 0
    i = rng.choice(n, 5, replace=False); raw.loc[i, "resting_bp_mmhg"] = 0
    blank(raw, "major_vessels", .03, rng); blank(raw, "thallium_test", .02, rng)
    raw = dupes(raw, 18, rng)
    register("heart-disease", "Heart Disease Risk Study", raw, clean,
        summary="Clinical test results for 1,800 patients assessed for heart disease.",
        description="Results of a fictional cardiology clinic's assessments: chest-pain type, blood pressure, "
                    "cholesterol, ECG, exercise test and imaging, plus whether heart disease was diagnosed. Inspired "
                    "by the structure of classic heart-disease studies, but every record is synthetic.",
        domain="Health", tasks=["Classification", "Scaling", "Cleaning"], level="Beginner",
        characteristics=["Tabular", "Multivariate"], target="heart_disease",
        variables=[
            ("patient_id", "ID", "Integer", "Patient identifier", ""), ("age", "Feature", "Integer", "Age", "years"),
            ("sex", "Feature", "Categorical", "Sex", ""),
            ("chest_pain_type", "Feature", "Categorical", "Type of chest pain (text, or codes 1-4 in some rows)", ""),
            ("resting_bp_mmhg", "Feature", "Continuous", "Resting systolic blood pressure", "mmHg"),
            ("cholesterol_mg_dl", "Feature", "Continuous", "Serum cholesterol", "mg/dL"),
            ("fasting_sugar_over_120", "Feature", "Binary", "Fasting blood sugar above 120 mg/dL", ""),
            ("resting_ecg", "Feature", "Categorical", "Resting electrocardiogram result", ""),
            ("max_heart_rate", "Feature", "Integer", "Highest heart rate in the exercise test", "beats/min"),
            ("exercise_angina", "Feature", "Binary", "Chest pain during exercise", ""),
            ("st_depression", "Feature", "Continuous", "ST depression induced by exercise", "mm"),
            ("st_slope", "Feature", "Categorical", "Slope of the peak exercise ST segment", ""),
            ("major_vessels", "Feature", "Continuous", "Major vessels colored by imaging (0-3)", ""),
            ("thallium_test", "Feature", "Categorical", "Thallium stress-test result", ""),
            ("heart_disease", "Target", "Binary", "1 = heart disease diagnosed", ""),
        ],
        issues=["chest_pain_type recorded as codes 1-4 in 20% of rows (1 = Typical, 2 = Atypical, 3 = Non-anginal, 4 = Asymptomatic)",
                "70 cholesterol values of 0 (a missing-value code)", "5 resting blood pressures of 0",
                "Missing major_vessels (3%) and thallium_test (2%)", "18 duplicated patients"],
        exercises=["Replace the 0 codes with missing values and decide how to impute them",
                   "Compare max heart rate for patients with and without heart disease",
                   "One-hot encode the categories, scale the numbers, and fit logistic regression",
                   "Does scaling change the logistic regression coefficients? Why?"])

# ======================================================================
# 21. COVID-19 daily reports
# ======================================================================
def covid():
    rng = np.random.default_rng(2121)
    counties = {"Ashford": 210_000, "Bellmont": 95_000, "Carver": 480_000, "Dunmore": 62_000, "Eastlake": 330_000,
                "Fairview": 150_000, "Glenwood": 41_000, "Hartley": 720_000, "Ironside": 88_000, "Juniper": 260_000,
                "Kingsbury": 125_000, "Linden": 54_000}
    days = pd.date_range("2020-03-01", "2022-12-31", freq="D")
    t = np.arange(len(days))
    waves = [(40, 25, .8), (300, 35, 3.0), (420, 30, 1.2), (540, 30, 2.2), (670, 25, 9.0), (790, 40, 2.5)]
    base = sum(h * np.exp(-((t - c) ** 2) / (2 * w ** 2)) for c, w, h in waves) + .2
    rows = []
    for c, pop in counties.items():
        shift = rng.integers(-10, 10); scale = rng.uniform(.7, 1.3)
        curve = np.roll(base, shift) * scale
        vacc = np.clip(80 / (1 + np.exp(-(t - 330) / 45)) * rng.uniform(.75, 1.05), 0, 92)
        for k, d in enumerate(days):
            weekend = .6 if d.dayofweek >= 5 else 1.0
            cases = rng.poisson(pop / 1e5 * curve[k] * 12 * weekend)
            cfr = .02 if k < 300 else (.012 if k < 600 else .004)
            rows.append(dict(date=d.strftime("%Y-%m-%d"), county=c, population=pop, new_cases=int(cases),
                             new_deaths=int(rng.poisson(cases * cfr)), tests=int(cases * rng.uniform(6, 14) + pop * .0005),
                             hospitalized=int(rng.poisson(cases * (.08 if k < 600 else .03) * 3)),
                             pct_fully_vaccinated=round(float(vacc[k]), 1)))
    df = pd.DataFrame(rows)
    clean = df.copy()
    raw = df.copy()
    # negative corrections and backlog dumps
    i = rng.choice(len(raw), 40, replace=False); raw.loc[i, "new_cases"] = -rng.integers(5, 120, 40)
    i = rng.choice(len(raw), 25, replace=False); raw.loc[i, "new_cases"] = raw.loc[i, "new_cases"] * 8
    raw = raw.drop(raw.sample(frac=.015, random_state=9).index)
    blank(raw, "hospitalized", .05, rng); blank(raw, "tests", .03, rng)
    raw["date"] = mixed_dates(pd.to_datetime(raw["date"]), rng, fmts=("%Y-%m-%d", "%m/%d/%Y"), probs=(.8, .2))
    raw = dupes(raw, 50, rng)
    register("covid19-counties", "COVID-19 County Daily Reports", raw, clean,
        summary="Daily cases, deaths, tests and vaccination for 12 counties, 2020-2022.",
        description="Daily public-health reports for 12 fictional counties from March 2020 through 2022, showing "
                    "several epidemic waves, weekend reporting dips and rising vaccination. The numbers imitate "
                    "real reporting patterns but describe no real place. Synthetic data for teaching only.",
        domain="Health", tasks=["Time series", "Cleaning", "Visualization"], level="Intermediate",
        characteristics=["Time-series", "Panel"],
        variables=[
            ("date", "Feature", "Date", "Report date (two formats)", "date"), ("county", "ID", "Categorical", "County (fictional)", ""),
            ("population", "Feature", "Integer", "County population", "people"),
            ("new_cases", "Feature", "Integer", "New confirmed cases reported that day", "cases"),
            ("new_deaths", "Feature", "Integer", "New deaths reported that day", "deaths"),
            ("tests", "Feature", "Continuous", "Tests reported that day", "tests"),
            ("hospitalized", "Feature", "Continuous", "Patients in hospital with COVID-19", "patients"),
            ("pct_fully_vaccinated", "Feature", "Continuous", "Share of residents fully vaccinated", "%"),
        ],
        issues=["40 negative daily case counts (corrections of earlier over-counts)",
                "25 backlog days with about eight times the normal count",
                "Weekend dips in reporting (real pattern, not an error)",
                "About 1.5% of county-days missing", "Missing tests (3%) and hospitalizations (5%)",
                "Dates in two formats", "50 duplicated reports"],
        exercises=["Compute a 7-day rolling average per 100,000 residents and plot the waves",
                   "Decide how to handle negative counts and backlog spikes",
                   "Compare case fatality across the three years",
                   "Plot vaccination against cases with small multiples by county"])

# ======================================================================
# 22. Influenza surveillance
# ======================================================================
def flu():
    rng = np.random.default_rng(2222)
    regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
    ages = ["0-4", "5-24", "25-64", "65+"]
    rows = []
    for s0 in range(2019, 2024):
        season = f"{s0}-{str(s0+1)[2:]}"
        start = pd.Timestamp(f"{s0}-10-01"); start = start + pd.Timedelta(days=(6 - start.dayofweek) % 7)
        peak = rng.normal(17, 3); height = {2019: 1.0, 2020: .05, 2021: .5, 2022: 1.3, 2023: .9}[s0]
        for wk in range(33):
            ws = start + pd.Timedelta(weeks=wk); mmwr = (40 + wk - 1) % 52 + 1
            for r in regions:
                rshift = regions.index(r) - 2
                for a in ages:
                    amult = {"0-4": 1.8, "5-24": 1.5, "25-64": .8, "65+": .9}[a]
                    visits = int(rng.normal({"0-4": 9000, "5-24": 14000, "25-64": 26000, "65+": 11000}[a], 600))
                    ili_rate = (1.2 + 6 * height * amult * np.exp(-((wk - peak - rshift) ** 2) / 18)) / 100
                    ili = int(rng.binomial(visits, min(ili_rate, .3)))
                    tests = int(ili * rng.uniform(.25, .4))
                    pos = min(.45, .02 + .35 * height * np.exp(-((wk - peak - rshift) ** 2) / 22))
                    fa = int(rng.binomial(tests, pos * .75)); fb = int(rng.binomial(tests, pos * .25))
                    hosp = int(rng.poisson((fa + fb) * {"0-4": .06, "5-24": .02, "25-64": .05, "65+": .2}[a]))
                    rows.append(dict(season=season, mmwr_week=mmwr, week_start=ws.strftime("%Y-%m-%d"), region=r, age_group=a,
                                     total_visits=visits, ili_visits=ili, ili_pct=round(100 * ili / visits, 2),
                                     lab_tests=tests, flu_a_positive=fa, flu_b_positive=fb, hospitalizations=hosp))
    df = pd.DataFrame(rows)
    clean = df.copy()
    raw = df.copy().astype({"ili_pct": object, "lab_tests": object})
    i = rng.choice(len(raw), 40, replace=False); raw.loc[i, "ili_pct"] = (df.loc[i, "ili_pct"] / 100).round(4)
    i = rng.choice(len(raw), 120, replace=False); raw.loc[i, "lab_tests"] = "N/A"
    raw["age_group"] = variants(raw["age_group"], {"65+": ["65 and over", ">=65"], "0-4": ["0 to 4", "00-04"]}, .1, rng)
    raw = raw.drop(raw.sample(frac=.02, random_state=11).index).reset_index(drop=True)
    register("flu-surveillance", "Influenza-Like Illness Surveillance", raw, clean,
        summary="Weekly flu activity by region and age group over five seasons.",
        description="Weekly reports from a fictional network of outpatient clinics, October to May for five flu "
                    "seasons (2019-20 to 2023-24): visits for influenza-like illness (ILI), lab tests, flu A and B "
                    "positives and hospitalizations, by region and age group. The 2020-21 season is almost flat, as "
                    "happened in many countries. Synthetic data for teaching only.",
        domain="Health", tasks=["Time series", "Visualization", "Cleaning"], level="Intermediate",
        characteristics=["Time-series", "Panel"],
        variables=[
            ("season", "Feature", "Categorical", "Flu season (e.g. 2022-23)", ""),
            ("mmwr_week", "Feature", "Integer", "Epidemiological week number (40 = early October)", "week"),
            ("week_start", "Feature", "Date", "Sunday that starts the week", "date"),
            ("region", "Feature", "Categorical", "Region", ""), ("age_group", "Feature", "Categorical", "Patient age group", "years"),
            ("total_visits", "Feature", "Integer", "All outpatient visits", "visits"),
            ("ili_visits", "Feature", "Integer", "Visits for influenza-like illness", "visits"),
            ("ili_pct", "Target", "Continuous", "ILI visits as a share of all visits", "%"),
            ("lab_tests", "Feature", "Text", "Specimens tested for flu ('N/A' when not reported)", "tests"),
            ("flu_a_positive", "Feature", "Integer", "Positive for influenza A", "tests"),
            ("flu_b_positive", "Feature", "Integer", "Positive for influenza B", "tests"),
            ("hospitalizations", "Feature", "Integer", "Flu hospital admissions", "patients"),
        ],
        issues=["40 ili_pct values written as fractions (0.034 instead of 3.4)",
                "120 lab_tests values of 'N/A', which makes the whole column text",
                "Age groups labeled several ways ('65 and over', '>=65', '0 to 4')",
                "About 2% of region-age-weeks missing",
                "Week numbers restart at 1 in January, so sorting by mmwr_week alone is wrong"],
        exercises=["Recompute ili_pct from the counts and find the rows on the wrong scale",
                   "Plot ILI % by week for each season on one chart: which season peaked highest?",
                   "Compare flu A and B by age group with stacked bars",
                   "Build a heatmap of region × week for one season"])

# ======================================================================
# 23. Common cold diary
# ======================================================================
def cold():
    rng = np.random.default_rng(2323); n = 2500
    symptoms_all = ["runny nose", "sore throat", "cough", "sneezing", "congestion", "headache", "fatigue", "fever"]
    probs = [.85, .6, .55, .65, .7, .35, .5, .15]
    season = rng.choice(["Winter", "Spring", "Summer", "Fall"], n, p=[.42, .2, .1, .28])
    age = rng.integers(5, 80, n)
    sleep = rng.normal(7, 1.1, n).clip(3.5, 10).round(1)
    vitc = rng.choice(["Yes", "No"], n, p=[.35, .65]); zinc = rng.choice(["Yes", "No"], n, p=[.2, .8])
    smoker = rng.choice(["No", "Yes"], n, p=[.85, .15])
    sym = [[s for s, p in zip(symptoms_all, probs) if rng.random() < p] or ["runny nose"] for _ in range(n)]
    dur = (7.5 - .35 * (sleep - 7) + 1.2 * (smoker == "Yes") - .5 * (zinc == "Yes") + .4 * np.array([len(s) for s in sym]) - 2 + rng.gamma(2, 1, n)).clip(2).round(0).astype(int)
    sev = np.clip((2 + .7 * np.array([len(s) for s in sym]) + 1.5 * np.array(["fever" in s for s in sym]) + rng.normal(0, 1, n)).round(0), 1, 10).astype(int)
    onset = []
    for s in season:
        mon = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Fall": [9, 10, 11]}[s]
        m = int(rng.choice(mon)); y = 2024 if m >= 9 else 2025
        onset.append(pd.Timestamp(year=y, month=m, day=int(rng.integers(1, 28))).strftime("%Y-%m-%d"))
    df = pd.DataFrame({
        "episode_id": [f"CC{60000+i}" for i in range(n)], "participant_id": [f"P{x}" for x in rng.integers(100, 1600, n)],
        "age": age, "sex": rng.choice(["Female", "Male"], n), "onset_date": onset, "season": season,
        "symptoms": [", ".join(s) for s in sym], "severity_1_10": sev, "duration_days": dur,
        "sleep_hours": sleep, "took_vitamin_c": vitc, "took_zinc": zinc, "smoker": smoker,
        "days_missed": np.minimum(dur, rng.poisson(sev / 3)).astype(int),
    })
    clean = df.copy()
    raw = df.copy()
    def mess(s):
        r = rng.random()
        if r < .15: return s.replace(", ", ";")
        if r < .25: return s.title()
        if r < .32: return s.replace(", ", " / ")
        if r < .36: return s.replace("runny nose", "runny-nose").replace("sore throat", "sore throat ")
        return s
    raw["symptoms"] = [mess(s) for s in df.symptoms]
    m = rng.random(n) < .15; raw.loc[m, "severity_1_10"] = np.ceil(df.loc[m, "severity_1_10"] / 2).astype(int)
    i = rng.choice(n, 12, replace=False); raw.loc[i, "duration_days"] = -raw.loc[i, "duration_days"]
    blank(raw, "sleep_hours", .07, rng); blank(raw, "took_vitamin_c", .04, rng)
    register("common-cold-diary", "Common Cold Symptom Diary", raw, clean,
        summary="2,500 self-reported colds with symptoms listed in one text column.",
        description="Diary entries from a fictional study in which volunteers logged each cold over a year: when "
                    "it started, which symptoms they had, how bad and how long it lasted, sleep, supplements and "
                    "days missed from work or school. Symptoms are all in one text column, so splitting them into "
                    "yes/no columns is the main cleaning task. Synthetic data for teaching only.",
        domain="Health", tasks=["Cleaning", "Text", "Visualization"], level="Beginner",
        characteristics=["Tabular", "Text"], target="duration_days",
        variables=[
            ("episode_id", "ID", "Categorical", "Cold episode identifier", ""),
            ("participant_id", "Feature", "Categorical", "Participant (some had several colds)", ""),
            ("age", "Feature", "Integer", "Age", "years"), ("sex", "Feature", "Categorical", "Sex", ""),
            ("onset_date", "Feature", "Date", "First day of symptoms", "date"), ("season", "Feature", "Categorical", "Season of onset", ""),
            ("symptoms", "Feature", "Text", "All symptoms reported, in one text field", ""),
            ("severity_1_10", "Feature", "Integer", "Self-rated severity (some answered on a 1-5 scale)", "1-10"),
            ("duration_days", "Target", "Integer", "Length of the cold", "days"),
            ("sleep_hours", "Feature", "Continuous", "Average sleep while sick", "hours"),
            ("took_vitamin_c", "Feature", "Binary", "Took vitamin C", ""), ("took_zinc", "Feature", "Binary", "Took zinc", ""),
            ("smoker", "Feature", "Binary", "Smokes", ""), ("days_missed", "Feature", "Integer", "Days of work or school missed", "days"),
        ],
        issues=["symptoms separated by commas, semicolons or slashes, in mixed case, sometimes hyphenated",
                "15% of severity scores given on a 1-5 scale instead of 1-10 (cannot be told apart row by row)",
                "12 negative durations", "Missing sleep (7%) and vitamin C answers (4%)"],
        exercises=["Split symptoms into one yes/no column per symptom",
                   "Plot how often each symptom occurs, overall and by season",
                   "Does sleep relate to how long a cold lasts?",
                   "Explain why the mixed severity scale is hard to fix"])

# ======================================================================
# 24. Crime incidents
# ======================================================================
def crime():
    rng = np.random.default_rng(2424); n = 9000
    off = {"Larceny/Theft": .3, "Burglary": .1, "Assault": .14, "Vandalism": .11, "Motor Vehicle Theft": .08,
           "Robbery": .06, "Fraud": .09, "Drug Offense": .07, "Weapons Violation": .05}
    offense = rng.choice(list(off), n, p=list(off.values()))
    district = rng.choice(["North", "South", "East", "West", "Downtown"], n, p=[.18, .24, .2, .15, .23])
    centers = {"North": (40.78, -73.95), "South": (40.70, -73.96), "East": (40.74, -73.90), "West": (40.74, -74.01), "Downtown": (40.72, -73.99)}
    occ = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.uniform(0, 1035, n), unit="D")
    hour_shift = np.where(np.isin(offense, ["Assault", "Robbery", "Weapons Violation"]), rng.normal(22, 3, n), rng.normal(14, 5, n)) % 24
    occ = (occ.normalize() + pd.to_timedelta(hour_shift, unit="h")).floor("min")
    rep = occ + pd.to_timedelta(np.where(offense == "Fraud", rng.exponential(240, n), rng.exponential(6, n)), unit="h")
    loc_type = rng.choice(["Street", "Residence", "Retail store", "Parking lot", "Park", "Restaurant/Bar", "School", "Transit"], n,
                          p=[.3, .27, .13, .1, .05, .07, .03, .05])
    arrest_p = {"Drug Offense": .8, "Weapons Violation": .7, "Assault": .4, "Robbery": .25, "Larceny/Theft": .12, "Burglary": .1,
                "Vandalism": .08, "Motor Vehicle Theft": .07, "Fraud": .15}
    df = pd.DataFrame({
        "incident_id": [f"RPD-{23000000+i}" for i in range(n)], "occurred_at": occ.strftime("%Y-%m-%d %H:%M"),
        "reported_at": rep.floor("min").strftime("%Y-%m-%d %H:%M"), "offense": offense, "district": district, "location_type": loc_type,
        "latitude": (np.array([centers[d][0] for d in district]) + rng.normal(0, .012, n)).round(5),
        "longitude": (np.array([centers[d][1] for d in district]) + rng.normal(0, .012, n)).round(5),
        "arrest_made": np.where(rng.random(n) < np.array([arrest_p[o] for o in offense]), "Yes", "No"),
        "weapon_involved": np.where(np.isin(offense, ["Robbery", "Weapons Violation"]) | ((offense == "Assault") & (rng.random(n) < .4)), "Yes", "No"),
        "victims": np.where(np.isin(offense, ["Drug Offense", "Weapons Violation"]), 0, rng.choice([1, 1, 1, 1, 2, 3], n)),
    }).sort_values("occurred_at").reset_index(drop=True)
    clean = df.copy()
    raw = df.copy()
    raw["offense"] = variants(raw["offense"], {"Larceny/Theft": ["THEFT", "Larceny", "theft"], "Motor Vehicle Theft": ["Auto Theft", "MVT"],
                                               "Assault": ["ASSAULT", "Simple Assault"], "Drug Offense": ["Narcotics", "Drugs"]}, .12, rng)
    i = rng.choice(n, 40, replace=False)
    raw.loc[i, "reported_at"], raw.loc[i, "occurred_at"] = raw.loc[i, "occurred_at"].values, raw.loc[i, "reported_at"].values
    blank(raw, "latitude", .05, rng); raw.loc[raw.latitude.isna(), "longitude"] = np.nan
    blank(raw, "location_type", .04, rng)
    raw["district"] = variants(raw["district"], {"Downtown": ["Central", "DOWNTOWN"], "North": ["N", "Northside"]}, .06, rng)
    raw = dupes(raw, 70, rng)
    register("riverton-crime", "Riverton Crime Incidents", raw, clean,
        summary="Three years of reported crime incidents with times and locations.",
        description="9,000 incidents reported to the police department of the fictional city of Riverton, January "
                    "2023 to October 2025: offense, when it happened and when it was reported, district, location "
                    "type, coordinates and whether an arrest was made. Synthetic data for teaching only; it does not "
                    "describe any real city.",
        domain="Public Safety", tasks=["Cleaning", "Time series", "Visualization"], level="Intermediate",
        characteristics=["Tabular", "Time-series", "Geospatial"],
        variables=[
            ("incident_id", "ID", "Categorical", "Incident number", ""),
            ("occurred_at", "Feature", "Date", "When the incident happened (best estimate)", "date-time"),
            ("reported_at", "Feature", "Date", "When it was reported", "date-time"),
            ("offense", "Feature", "Categorical", "Offense category", ""), ("district", "Feature", "Categorical", "Police district", ""),
            ("location_type", "Feature", "Categorical", "Type of place", ""),
            ("latitude", "Feature", "Continuous", "Approximate latitude (block level)", "degrees"),
            ("longitude", "Feature", "Continuous", "Approximate longitude (block level)", "degrees"),
            ("arrest_made", "Target", "Binary", "An arrest was made", ""),
            ("weapon_involved", "Feature", "Binary", "A weapon was involved", ""),
            ("victims", "Feature", "Integer", "Number of victims (0 for offenses without a direct victim)", ""),
        ],
        issues=["Offense names spelled several ways ('THEFT', 'Larceny', 'Auto Theft', 'Narcotics')",
                "40 incidents where occurred_at and reported_at are swapped (reported before it happened)",
                "Missing coordinates (5%) and location type (4%)",
                "District labels with synonyms ('Central', 'Northside', 'N')", "70 duplicated incidents"],
        exercises=["Standardize offense names and plot counts by category",
                   "Make an hour-of-day × day-of-week heatmap of incidents",
                   "How long does fraud take to be reported compared with other offenses?",
                   "Which offenses have the highest arrest rates?"])

# ======================================================================
# 25. Gun violence incidents
# ======================================================================
def gun_violence():
    rng = np.random.default_rng(2525); n = 3200
    cities = {"Riverton": ("Ash County", .3), "Harbor City": ("Bay County", .22), "Maple Valley": ("Ash County", .1),
              "Old Port": ("Bay County", .12), "Cedar Park": ("Elm County", .09), "Summit": ("Elm County", .07),
              "North Shore": ("Bay County", .06), "Lakeview": ("Pine County", .04)}
    city = rng.choice(list(cities), n, p=[v[1] for v in cities.values()])
    itype = rng.choice(["Assault", "Domestic dispute", "Robbery", "Accidental discharge", "Self-defense", "Drive-by", "Unknown"], n,
                       p=[.32, .15, .12, .07, .06, .1, .18])
    date = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.uniform(0, 2465, n), unit="D")
    summer = 1 + .5 * np.sin(2 * np.pi * (date.dayofyear.values - 100) / 365)
    keep = rng.random(n) < summer / 1.5
    killed = rng.choice([0, 0, 0, 1, 1, 2], n); injured = rng.choice([0, 1, 1, 1, 2, 3], n)
    injured = np.where((killed + injured) == 0, 1, injured)
    df = pd.DataFrame({
        "incident_id": [f"GV{700000+i}" for i in range(n)], "date": date.strftime("%Y-%m-%d"), "city": city,
        "county": [cities[c][0] for c in city], "incident_type": itype, "killed": killed, "injured": injured,
        "victim_age_group": rng.choice(["Child 0-11", "Teen 12-17", "Adult 18-34", "Adult 35-64", "Senior 65+"], n, p=[.03, .1, .5, .32, .05]),
        "suspect_arrested": rng.choice(["Yes", "No", "Unknown"], n, p=[.35, .45, .2]),
        "school_related": np.where(rng.random(n) < .015, "Yes", "No"),
        "source": rng.choice(["Police report", "News report", "Hospital record"], n, p=[.55, .35, .1]),
    })[keep].reset_index(drop=True)
    clean = df.sort_values("date").reset_index(drop=True)
    raw = clean.copy().astype({"killed": object, "injured": object})
    words = {0: "zero", 1: "one", 2: "two", 3: "three"}
    m = rng.random(len(raw)) < .06; raw.loc[m, "killed"] = [words[int(x)] for x in clean.loc[m, "killed"]]
    m = rng.random(len(raw)) < .06; raw.loc[m, "injured"] = [words[int(x)] for x in clean.loc[m, "injured"]]
    raw["date"] = mixed_dates(pd.to_datetime(raw["date"]), rng, fmts=("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"), probs=(.65, .25, .1))
    blank(raw, "victim_age_group", .12, rng)
    # same incident reported by two sources
    rep = raw.sample(90, random_state=4).copy(); rep["source"] = "News report"
    rep["incident_id"] = [f"GV{900000+i}" for i in range(90)]
    raw = pd.concat([raw, rep]).sample(frac=1, random_state=5).reset_index(drop=True)
    raw["city"] = variants(raw["city"], {"Harbor City": ["Harbor city", "Harbour City"], "Old Port": ["Oldport"]}, .08, rng)
    register("gun-violence-incidents", "Gun Violence Incidents", raw, clean,
        summary="Incidents involving firearms across eight fictional cities, 2019-2025.",
        description="A compiled record of firearm incidents in eight fictional cities, drawn from police, news and "
                    "hospital sources: date, place, type of incident, people killed and injured, and whether a suspect "
                    "was arrested. Incidents peak in summer. Synthetic data for teaching only; it describes no real "
                    "events or people.",
        domain="Public Safety", tasks=["Cleaning", "Visualization", "Time series"], level="Intermediate",
        characteristics=["Tabular", "Time-series"],
        variables=[
            ("incident_id", "ID", "Categorical", "Incident identifier", ""), ("date", "Feature", "Date", "Date of incident (three formats)", "date"),
            ("city", "Feature", "Categorical", "City (fictional)", ""), ("county", "Feature", "Categorical", "County", ""),
            ("incident_type", "Feature", "Categorical", "Circumstance of the incident", ""),
            ("killed", "Feature", "Text", "People killed (sometimes written as a word)", "people"),
            ("injured", "Feature", "Text", "People injured (sometimes written as a word)", "people"),
            ("victim_age_group", "Feature", "Categorical", "Age group of the main victim", ""),
            ("suspect_arrested", "Feature", "Categorical", "Yes, No or Unknown", ""),
            ("school_related", "Feature", "Binary", "Happened at or near a school", ""),
            ("source", "Feature", "Categorical", "Where the record came from", ""),
        ],
        issues=["killed and injured written as words ('two') in about 6% of rows",
                "Dates in three formats ('2023-07-04', '07/04/2023', 'July 04, 2023')",
                "90 incidents recorded twice, once from a news report, with a different incident_id",
                "Missing victim age group (12%)", "City names spelled several ways ('Harbour City', 'Oldport')"],
        exercises=["Convert killed and injured to numbers and compute total casualties",
                   "Find incidents recorded twice without using incident_id",
                   "Plot incidents by month: is there a seasonal pattern?",
                   "Compare cities using rates rather than counts: what extra data would you need?"])

# ======================================================================
# 26. Flights
# ======================================================================
def flights():
    rng = np.random.default_rng(2626); n = 12000
    airports = {"RVT": (0, 0), "HBC": (420, 80), "MPV": (180, 300), "SMT": (900, 650), "CDP": (300, -250),
                "OPT": (650, -120), "NSH": (120, 520), "LKV": (1300, 200)}
    airlines = {"Riverline Air": "RL", "Skyward Express": "SX", "Coastal Jet": "CJ", "Summit Airways": "SA"}
    codes = list(airports)
    o = rng.choice(codes, n); d = rng.choice(codes, n)
    same = o == d; d[same] = np.array([codes[(codes.index(x) + 1) % len(codes)] for x in o[same]])
    dist = np.array([np.hypot(airports[a][0] - airports[b][0], airports[a][1] - airports[b][1]) for a, b in zip(o, d)]).round(0) + 60
    al = rng.choice(list(airlines), n, p=[.35, .3, .2, .15])
    date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 273, n), unit="D")
    sched_dep = (rng.choice(np.arange(6, 22), n) * 60 + rng.choice([0, 10, 15, 25, 30, 40, 45, 55], n))
    air_time = (dist / 7.8 + 18 + rng.normal(0, 6, n)).round(0)
    sched_arr = sched_dep + air_time + 22
    month_factor = np.where(np.isin(date.month, [6, 7]), 1.6, 1.0)
    delay = np.where(rng.random(n) < .35, rng.gamma(1.4, 28, n) * month_factor, rng.normal(-4, 5, n)).round(0)
    cancelled = rng.random(n) < .02 * month_factor
    def hhmm(m):
        m = int(m) % 1440; return (m // 60) * 100 + m % 60
    dep_act = sched_dep + delay; arr_act = sched_arr + delay + rng.normal(0, 6, n).round(0)
    cause = np.where(delay >= 15, rng.choice(["Carrier", "Weather", "Air traffic", "Late aircraft", "Security"], n, p=[.3, .12, .25, .31, .02]), "")
    df = pd.DataFrame({
        "flight_date": date.strftime("%Y-%m-%d"), "airline": al, "flight_number": [f"{airlines[a]}{rng.integers(100, 2999)}" for a in al],
        "origin": o, "destination": d, "scheduled_departure": [hhmm(x) for x in sched_dep],
        "actual_departure": [np.nan if c else hhmm(x) for x, c in zip(dep_act, cancelled)],
        "departure_delay_min": np.where(cancelled, np.nan, delay), "scheduled_arrival": [hhmm(x) for x in sched_arr],
        "actual_arrival": [np.nan if c else hhmm(x) for x, c in zip(arr_act, cancelled)],
        "arrival_delay_min": np.where(cancelled, np.nan, (arr_act - sched_arr).round(0)),
        "distance_miles": dist, "air_time_min": np.where(cancelled, np.nan, air_time),
        "cancelled": np.where(cancelled, 1, 0), "delay_cause": np.where(cancelled, "", cause),
        "aircraft": rng.choice(["Regional jet 76", "Narrowbody 150", "Narrowbody 180", "Turboprop 70"], n, p=[.3, .35, .25, .1]),
    })
    df["delay_cause"] = df["delay_cause"].replace("", np.nan)
    df = df.sort_values(["flight_date", "scheduled_departure"]).reset_index(drop=True)
    clean = df.copy()
    raw = df.copy()
    m = raw.airline == "Coastal Jet"; raw.loc[m, "distance_miles"] = (df.loc[m, "distance_miles"] * 1.609).round(0)
    raw["airline"] = variants(raw["airline"], {"Skyward Express": ["Skyward", "SKYWARD EXPRESS"], "Riverline Air": ["Riverline", "RiverLine Air"]}, .08, rng)
    blank(raw, "air_time_min", .03, rng)
    raw = dupes(raw, 60, rng)
    register("regional-flights", "Regional Flight Delays", raw, clean,
        summary="12,000 flights between eight airports, with delays and causes.",
        description="Nine months of flights (January to September 2025) on four fictional airlines between eight "
                    "fictional airports. Times are stored as HHMM numbers, as in many real flight datasets "
                    "(5 means 00:05, 1430 means 14:30). Delays are worse in June and July. "
                    "Synthetic data for teaching only.",
        domain="Transportation", tasks=["Cleaning", "Visualization", "Regression", "Scaling"], level="Intermediate",
        characteristics=["Tabular", "Time-series"], target="arrival_delay_min",
        variables=[
            ("flight_date", "Feature", "Date", "Date of the flight", "date"), ("airline", "Feature", "Categorical", "Airline (fictional)", ""),
            ("flight_number", "ID", "Categorical", "Flight number", ""),
            ("origin", "Feature", "Categorical", "Origin airport code (fictional)", ""),
            ("destination", "Feature", "Categorical", "Destination airport code (fictional)", ""),
            ("scheduled_departure", "Feature", "Integer", "Scheduled departure time as HHMM", "HHMM"),
            ("actual_departure", "Feature", "Continuous", "Actual departure time as HHMM (blank if cancelled)", "HHMM"),
            ("departure_delay_min", "Feature", "Continuous", "Departure delay (negative = early)", "minutes"),
            ("scheduled_arrival", "Feature", "Integer", "Scheduled arrival time as HHMM", "HHMM"),
            ("actual_arrival", "Feature", "Continuous", "Actual arrival time as HHMM (blank if cancelled)", "HHMM"),
            ("arrival_delay_min", "Target", "Continuous", "Arrival delay (negative = early)", "minutes"),
            ("distance_miles", "Feature", "Continuous", "Route distance (one airline reports km)", "miles"),
            ("air_time_min", "Feature", "Continuous", "Time in the air", "minutes"),
            ("cancelled", "Feature", "Binary", "1 = cancelled", ""),
            ("delay_cause", "Feature", "Categorical", "Main cause when delayed 15+ minutes", ""),
            ("aircraft", "Feature", "Categorical", "Aircraft type and seats", ""),
        ],
        issues=["Times stored as HHMM integers; late flights wrap past midnight (2350 + 30 min = 20)",
                "Coastal Jet reports distance in kilometers, the others in miles",
                "Airline names spelled several ways",
                "Blank times and delays for cancelled flights (legitimately missing)",
                "Missing air time (3%)", "60 duplicated flights"],
        exercises=["Convert HHMM times to real datetimes, handling flights that cross midnight",
                   "Detect the airline with distances in the wrong unit",
                   "Plot average delay by month and by hour of day",
                   "Which airline and route have the worst on-time rate (arrival delay under 15 minutes)?"])

# ======================================================================
# 27. Fuel prices
# ======================================================================
def fuel():
    rng = np.random.default_rng(2727)
    weeks = pd.date_range("2015-01-05", "2025-09-29", freq="W-MON")
    t = np.arange(len(weeks)); yrs = 2015 + t / 52.18
    crude = (55 + 10 * np.sin(t / 40)
             - 30 * np.exp(-((yrs - 2020.3) ** 2) / .02)
             + 45 * np.exp(-((yrs - 2022.45) ** 2) / .08)
             + 12 * (yrs > 2021) - 5 * (yrs > 2023.5)
             + np.cumsum(rng.normal(0, .9, len(t))) * .6).clip(18)
    regions = {"Northeast": (.18, .32), "Southeast": (-.12, .12), "Midwest": (-.05, .18), "Gulf Coast": (-.2, .05), "West Coast": (.75, .5)}
    rows = []
    for r, (gas_add, dsl_add) in regions.items():
        noise = np.cumsum(rng.normal(0, .01, len(t)))
        gas = 1.15 + crude / 42 * .95 + gas_add + .12 * np.sin(2 * np.pi * (weeks.dayofyear / 365 - .25)) + noise
        dsl = 1.25 + crude / 42 * 1.05 + dsl_add + .5 * np.exp(-((yrs - 2022.6) ** 2) / .05) + noise
        ker = dsl + .35 + rng.normal(0, .03, len(t))
        heat = dsl - .1 + .15 * np.cos(2 * np.pi * weeks.dayofyear / 365) + rng.normal(0, .03, len(t))
        for k in range(len(t)):
            rows.append(dict(week_of=weeks[k].strftime("%Y-%m-%d"), region=r, crude_oil_usd_bbl=round(float(crude[k]), 2),
                             gasoline_usd_gal=round(float(gas[k]), 3), diesel_usd_gal=round(float(dsl[k]), 3),
                             kerosene_usd_gal=round(float(ker[k]), 3), heating_oil_usd_gal=round(float(heat[k]), 3)))
    df = pd.DataFrame(rows)
    clean = df.copy()
    raw = df.copy().astype({"gasoline_usd_gal": object, "diesel_usd_gal": float})
    m = rng.random(len(raw)) < .12; raw.loc[m, "gasoline_usd_gal"] = [f"${v:.3f}" for v in df.loc[m, "gasoline_usd_gal"]]
    m = (raw.region == "West Coast") & (raw.week_of >= "2019-01-01") & (raw.week_of < "2020-01-01")
    raw.loc[m, "diesel_usd_gal"] = (df.loc[m, "diesel_usd_gal"] / 3.785).round(3)
    for c in ["kerosene_usd_gal", "heating_oil_usd_gal"]:
        blank(raw, c, .06, rng)
    raw = raw.drop(raw.sample(frac=.02, random_state=13).index).reset_index(drop=True)
    register("fuel-prices", "Fuel & Crude Oil Prices", raw, clean,
        summary="Weekly gasoline, diesel, kerosene and heating-oil prices, 2015-2025.",
        description="Weekly retail fuel prices in five regions of a fictional national market, with the crude oil "
                    "benchmark price, January 2015 to September 2025. The series imitates real market shapes, such as "
                    "a crash in early 2020 and a spike in 2022, but the numbers are synthetic and must not be used as "
                    "real prices.",
        domain="Energy", tasks=["Time series", "Visualization", "Scaling", "Regression"], level="Beginner",
        characteristics=["Time-series", "Panel"],
        variables=[
            ("week_of", "Feature", "Date", "Monday of the reporting week", "date"), ("region", "ID", "Categorical", "Market region", ""),
            ("crude_oil_usd_bbl", "Feature", "Continuous", "Crude oil benchmark price", "USD per barrel"),
            ("gasoline_usd_gal", "Feature", "Text", "Regular gasoline (some values stored as '$3.459')", "USD per gallon"),
            ("diesel_usd_gal", "Feature", "Continuous", "On-highway diesel", "USD per gallon"),
            ("kerosene_usd_gal", "Feature", "Continuous", "Kerosene", "USD per gallon"),
            ("heating_oil_usd_gal", "Feature", "Continuous", "Residential heating oil", "USD per gallon"),
        ],
        issues=["12% of gasoline prices stored as text with a '$' sign",
                "West Coast diesel for 2019 reported per liter instead of per gallon",
                "Missing kerosene and heating-oil prices (6% each)", "About 2% of region-weeks missing",
                "Crude oil is per barrel (42 gallons) while retail prices are per gallon"],
        exercises=["Plot all four fuels over time for one region",
                   "Find the year and region reported in the wrong unit",
                   "How closely does diesel follow crude oil? Try a scatter plot and a correlation",
                   "Index every series to 100 in January 2015 so they can share one axis"])
