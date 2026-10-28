import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime
import io
import re
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.misc import derivative

# ReportLab Imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# -----------------------------------------------------------------------------
# PAGE CONFIG & TITLE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hispanic Heritage Historical Data Explorer",
    page_icon="📈",
    layout="wide"
)

APP_AUTHOR = "Chinomso Ikwecheghe"

st.title("📈 Hispanic Heritage Historical Data Explorer")
st.caption(f"Developed by **{APP_AUTHOR}** | Polynomial Trend Analysis & Demographic Modeling Lab")

st.markdown("""
> **Mexican Demographic Focus Note**: 
> When selecting **Mexico (MX)** in this application, indicators such as **Net Migration** and **Crude Birth Rate** directly reflect the historical demographic dynamics, population pushes, and economic conditions that historically shaped **Mexican-born immigrant communities** in North America.
""")

# -----------------------------------------------------------------------------
# CONSTANTS & METADATA MAPPING
# -----------------------------------------------------------------------------
COUNTRY_MAP = {
    "MX": "Mexico (MX) [Focus: Mexican-Born Immigrants]",
    "BR": "Brazil (BR) [Latin American Comparison]",
    "AR": "Argentina (AR)",
    "CO": "Colombia (CO)",
    "CL": "Chile (CL)",
    "PE": "Peru (PE)",
    "VE": "Venezuela (VE)",
    "EC": "Ecuador (EC)",
    "UY": "Uruguay (UY)"
}

INDICATOR_MAP = {
    "SP.DYN.LE00.IN": {
        "name": "Life Expectancy at Birth",
        "units": "Years",
        "desc": "Total life expectancy at birth indicates the number of years a newborn infant would live if prevailing mortality patterns at the time of birth stay the same.",
        "min": 0, "max": 120
    },
    "SL.UEM.TOTL.NE.ZS": {
        "name": "Unemployment Rate (National Estimate)",
        "units": "% of Total Labor Force",
        "desc": "Share of the labor force without work but available for and seeking employment (national estimate).",
        "min": 0, "max": 100
    },
    "NY.GDP.PCAP.CD": {
        "name": "GDP Per Capita",
        "units": "Current US$",
        "desc": "Gross Domestic Product divided by midyear population. Note: Reflects per-capita economic output, NOT average household income or individual wealth.",
        "min": 0, "max": 1000000
    },
    "SP.DYN.CBRT.IN": {
        "name": "Crude Birth Rate",
        "units": "Births per 1,000 People",
        "desc": "Number of live births occurring during the year per 1,000 midyear population. Note: This is a rate, NOT total birth count.",
        "min": 0, "max": 100
    },
    "SM.POP.NETM": {
        "name": "Net Migration",
        "units": "Total Net Migrants",
        "desc": "Net total of migrants (immigrants minus emigrants) over 5-year periods. Note: This is net flow, NOT emigration count.",
        "min": -50000000, "max": 50000000
    },
    "VC.IHR.PSRC.P5": {
        "name": "Intentional Homicide Rate",
        "units": "Per 100,000 People",
        "desc": "Unlawful deaths purposefully inflicted on a person by another person, per 100,000 population.",
        "min": 0, "max": 1000
    },
    "FP.CPI.TOTL.ZG": {
        "name": "Consumer Price Inflation",
        "units": "Annual %",
        "desc": "Annual percentage change in the cost to the average consumer of acquiring a basket of goods and services.",
        "min": -100, "max": 100000
    },
    "GC.DOD.TOTL.GD.ZS": {
        "name": "Central Government Debt",
        "units": "% of GDP",
        "desc": "Gross debt of central government expressed as a percentage of nominal GDP.",
        "min": 0, "max": 1000
    },
    "FI.RES.TOTL.CD": {
        "name": "Total Reserves (Includes Gold)",
        "units": "Current US$",
        "desc": "Total international reserves comprising holdings of monetary gold, SDRs, and foreign exchange.",
        "min": 0, "max": 1e14
    },
    "NE.RSB.GNFS.ZS": {
        "name": "External Balance on Goods & Services",
        "units": "% of GDP",
        "desc": "Exports of goods and services minus imports of goods and services expressed as a percentage of GDP.",
        "min": -100, "max": 100
    }
}

CURRENT_YEAR = datetime.now().year
START_YEAR = CURRENT_YEAR - 70
END_YEAR = CURRENT_YEAR - 1

# -----------------------------------------------------------------------------
# WORLD BANK API DATA RETRIEVAL (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank_data(country_code, indicator_code):
    """
    Fetches official historical data from World Bank API v2.
    Strictly requests (CURRENT_YEAR - 70) to (CURRENT_YEAR - 1).
    Does NOT substitute fake data on failure.
    """
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {
        "date": f"{START_YEAR}:{END_YEAR}",
        "format": "json",
        "per_page": 1000
    }
    retrieval_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        json_data = resp.json()
        
        if not json_data or len(json_data) < 2 or json_data[1] is None:
            return pd.DataFrame(columns=["Year", "Value"]), retrieval_date, f"http://data.worldbank.org/indicator/{indicator_code}"
        
        records = []
        for entry in json_data[1]:
            yr = int(entry["date"])
            val = entry["value"]
            if val is not None:
                records.append({"Year": int(yr), "Value": float(val)})
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("Year").reset_index(drop=True)
        return df, retrieval_date, f"http://data.worldbank.org/indicator/{indicator_code}"
    except Exception as e:
        return pd.DataFrame(columns=["Year", "Value"]), retrieval_date, f"http://data.worldbank.org/indicator/{indicator_code}"

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "edited_df" not in st.session_state:
    st.session_state.edited_df = None
if "is_modified" not in st.session_state:
    st.session_state.is_modified = False
if "current_selection_key" not in st.session_state:
    st.session_state.current_selection_key = ""
if "calculated_results" not in st.session_state:
    st.session_state.calculated_results = None

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Controls & Parameters")

selected_country = st.sidebar.selectbox(
    "Primary Country",
    options=list(COUNTRY_MAP.keys()),
    format_func=lambda x: COUNTRY_MAP[x],
    index=0
)

selected_indicator = st.sidebar.selectbox(
    "Indicator",
    options=list(INDICATOR_MAP.keys()),
    format_func=lambda x: f"{INDICATOR_MAP[x]['name']} ({x})"
)

poly_degree = st.sidebar.slider("Polynomial Degree (d)", min_value=3, max_value=8, value=3, step=1)
sample_step = st.sidebar.slider("Graph Sample Increment (Years)", min_value=1, max_value=10, value=1, step=1)
forecast_horizon = st.sidebar.slider("Forward Display Horizon (Years)", min_value=0, max_value=50, value=10, step=1)

comparison_country = st.sidebar.selectbox(
    "Comparison Country (Optional)",
    options=["None"] + [c for c in COUNTRY_MAP.keys() if c != selected_country],
    format_func=lambda x: "None" if x == "None" else COUNTRY_MAP[x]
)

printer_mode = st.sidebar.checkbox("🖨️ Printer-Friendly Report View", value=False)

# Track selection key to auto-reset on change
selection_key = f"{selected_country}_{selected_indicator}"
if st.session_state.current_selection_key != selection_key:
    st.session_state.current_selection_key = selection_key
    st.session_state.edited_df = None
    st.session_state.is_modified = False
    st.session_state.calculated_results = None

# Fetch Source Data
raw_df, fetch_time, source_url = fetch_world_bank_data(selected_country, selected_indicator)

if st.session_state.edited_df is None:
    st.session_state.edited_df = raw_df.copy()

# -----------------------------------------------------------------------------
# DATA VALIDATION & RESET
# -----------------------------------------------------------------------------
ind_meta = INDICATOR_MAP[selected_indicator]

st.subheader(f"📊 Historical Data: {COUNTRY_MAP[selected_country]}")

col_meta1, col_meta2 = st.columns([2, 1])
with col_meta1:
    st.markdown(f"**Indicator Unit:** `{ind_meta['units']}`")
    st.markdown(f"**Description:** {ind_meta['desc']}")
    st.markdown(f"**Official Source:** [World Bank API Indicator {selected_indicator}]({source_url})")
    st.caption(f"API Retrieval Timestamp: {fetch_time} | Requested Range: {START_YEAR}–{END_YEAR}")
with col_meta2:
    st.metric("Total API Coverage Count", f"{len(raw_df)} obs", f"Out of {70} max possible years")

st.markdown("---")
st.write("### Data Table & Validation Controls")

if st.session_state.is_modified:
    st.warning("⚠️ **User-Modified Scenario Active**: You are currently working with custom-edited dataset values.")

c_btn1, c_btn2 = st.columns([1, 4])
with c_btn1:
    if st.button("🔄 Reset to Source Data"):
        st.session_state.edited_df = raw_df.copy()
        st.session_state.is_modified = False
        st.session_state.calculated_results = None
        st.rerun()

edited_data_input = st.data_editor(
    st.session_state.edited_df,
    num_rows="dynamic",
    key="editor_table",
    use_container_width=True
)

# Validate Edits
validation_errors = []
working_df = pd.DataFrame()

if edited_data_input is not None and not edited_data_input.empty:
    temp_df = edited_data_input.copy()
    
    # Check nulls
    if temp_df["Year"].isnull().any() or temp_df["Value"].isnull().any():
        validation_errors.append("Table contains null or unassigned Year/Value rows.")
    
    # Numeric conversion
    try:
        temp_df["Year"] = temp_df["Year"].astype(int)
        temp_df["Value"] = temp_df["Value"].astype(float)
    except Exception:
        validation_errors.append("Year must be integer and Value must be numeric.")
    
    # Duplicates
    if temp_df["Year"].duplicated().any():
        dups = temp_df["Year"][temp_df["Year"].duplicated()].unique()
        validation_errors.append(f"Duplicate years detected: {list(dups)}")
    
    # Indicator range constraints
    min_val, max_val = ind_meta["min"], ind_meta["max"]
    invalid_range = temp_df[(temp_df["Value"] < min_val) | (temp_df["Value"] > max_val)]
    if not invalid_range.empty:
        validation_errors.append(f"Some values fall outside domain rules for {ind_meta['name']} ({min_val} to {max_val}).")
    
    if not validation_errors:
        working_df = temp_df.sort_values("Year").reset_index(drop=True)
        if not working_df.equals(raw_df):
            st.session_state.is_modified = True
        st.session_state.edited_df = working_df
    else:
        for err in validation_errors:
            st.error(f"❌ Validation Error: {err}")

# -----------------------------------------------------------------------------
# POLYNOMIAL FITTING ENGINE
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("🧮 Polynomial Model Fitting & In-Sample Metrics")

if len(working_df) < (poly_degree + 1):
    st.error(f"Insufficient data to fit degree {poly_degree} polynomial. Requires at least {poly_degree + 1} distinct observation years, but only {len(working_df)} available.")
    st.stop()

# Centered time transformation: t = year - base_year
base_year = int(working_df["Year"].min())
years = working_df["Year"].values
values = working_df["Value"].values
t_obs = years - base_year

# Polynomial fit on centered t
coeffs = np.polyfit(t_obs, values, poly_degree)
poly_func = np.poly1d(coeffs)

# In-sample predictions
y_pred_in = poly_func(t_obs)
r2_in = r2_score(values, y_pred_in)
mae_in = mean_absolute_error(values, y_pred_in)
residuals = values - y_pred_in

# Formatting Polynomial Equation String
terms = []
for idx, c in enumerate(coeffs):
    pwr = poly_degree - idx
    if pwr == 0:
        terms.append(f"{c:+.4e}")
    elif pwr == 1:
        terms.append(f"{c:+.4e}·t")
    else:
        terms.append(f"{c:+.4e}·t^{pwr}")
eq_str = "y(t) = " + " ".join(terms) + f"  [where t = Year - {base_year}]"

# Holdout validation (Chronological Split: last 20% if >= 10 points)
holdout_text = "N/A (Requires >= 10 points)"
if len(working_df) >= 10:
    split_idx = int(len(working_df) * 0.8)
    train_df = working_df.iloc[:split_idx]
    test_df = working_df.iloc[split_idx:]
    
    t_tr = train_df["Year"].values - base_year
    c_tr = np.polyfit(t_tr, train_df["Value"].values, min(poly_degree, len(train_df)-1))
    p_tr = np.poly1d(c_tr)
    
    t_te = test_df["Year"].values - base_year
    te_pred = p_tr(t_te)
    r2_holdout = r2_score(test_df["Value"].values, te_pred)
    mae_holdout = mean_absolute_error(test_df["Value"].values, te_pred)
    holdout_text = f"Out-of-Sample Holdout R²: {r2_holdout:.4f} | Holdout MAE: {mae_holdout:.4f}"

# Metric Displays
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("In-Sample R² Score", f"{r2_in:.4f}")
col_m2.metric("In-Sample MAE", f"{mae_in:.4f} {ind_meta['units']}")
col_m3.metric("Historical Obs Count", f"{len(working_df)}")
col_m4.metric("Observed Mean", f"{np.mean(values):.2f}")

st.code(f"Fitted Equation ({poly_degree}th Degree Polynomial):\n{eq_str}", language="text")
st.caption(f"**Chronological Holdout Validation**: {holdout_text}")

# -----------------------------------------------------------------------------
# CALCULATE VALUE: SINGLE YEAR & BATCH
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("🔮 Prediction & Calculation Engine")

calc_tab1, calc_tab2 = st.tabs(["Single-Year Calculation", "Batch Calculation"])

with calc_tab1:
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        single_yr_input = st.number_input("Target Calculation Year (1950–2100)", min_value=1950, max_value=2100, value=2025, step=1)
        calc_single_btn = st.button("Calculate Single Value")

with calc_tab2:
    st.write("Calculate trend estimates using a structured range or custom comma-separated years.")
    batch_mode = st.radio("Batch Input Type", ["Year Range", "Custom Comma-Separated List"])
    
    if batch_mode == "Year Range":
        c_r1, c_r2, c_r3 = st.columns(3)
        b_start = c_r1.number_input("Start Year", 1950, 2100, 2020)
        b_end = c_r2.number_input("End Year", 1950, 2100, 2030)
        b_step = c_r3.slider("Range Step (Years)", 1, 10, 1)
        target_years_batch = list(range(int(b_start), int(b_end) + 1, int(b_step)))
    else:
        custom_str = st.text_input("Enter comma-separated years (e.g. 1995, 2005, 2025, 2040)", "1990, 2000, 2010, 2025, 2030")
        target_years_batch = []
        invalid_tokens = []
        for token in custom_str.split(","):
            token_str = token.strip()
            if token_str.isdigit():
                val_y = int(token_str)
                if 1950 <= val_y <= 2100:
                    target_years_batch.append(val_y)
                else:
                    invalid_tokens.append(f"{token_str} (out of 1950-2100 bounds)")
            elif token_str:
                invalid_tokens.append(f"'{token_str}' (non-numeric)")
        if invalid_tokens:
            st.info(f"Skipped invalid inputs: {', '.join(invalid_tokens)}")

    calc_batch_btn = st.button("Calculate Batch Values")

def compute_estimates(year_list):
    res_list = []
    min_obs_yr, max_obs_yr = working_df["Year"].min(), working_df["Year"].max()
    
    for yr in year_list:
        t_val = yr - base_year
        est = float(poly_func(t_val))
        
        # Classification
        if yr in working_df["Year"].values:
            classification = "Observed Historical Year"
            actual_val = working_df[working_df["Year"] == yr]["Value"].values[0]
        elif min_obs_yr <= yr <= max_obs_yr:
            classification = "In-Sample Interpolation"
            actual_val = np.nan
        else:
            classification = "Out-of-Sample Extrapolation"
            actual_val = np.nan
            
        # Implausibility Flag
        implausible = False
        if est < ind_meta["min"] or est > ind_meta["max"]:
            implausible = True
            
        res_list.append({
            "Year": yr,
            "Fitted Value Estimate": est,
            "Units": ind_meta["units"],
            "Domain Classification": classification,
            "Actual Observed Value": actual_val,
            "Extrapolation Flag": "⚠️ Implausible Extrapolation" if implausible else "Normal"
        })
    return pd.DataFrame(res_list)

if calc_single_btn:
    st.session_state.calculated_results = compute_estimates([single_yr_input])

if calc_batch_btn:
    st.session_state.calculated_results = compute_estimates(target_years_batch)

if st.session_state.calculated_results is not None:
    st.subheader("Calculation Engine Output")
    st.dataframe(st.session_state.calculated_results, use_container_width=True)
    
    # Plausibility Warnings
    implausibles = st.session_state.calculated_results[st.session_state.calculated_results["Extrapolation Flag"] != "Normal"]
    if not implausibles.empty:
        st.error("⚠️ **Polynomial Extrapolation Warning**: The mathematical curve produces values outside physical/indicator domain bounds. Polynomial predictions outside the observed period diverge rapidly and should be treated with caution.")

# -----------------------------------------------------------------------------
# MATHEMATICAL & PEDAGOGICAL CURRICULUM EXPLANATIONS
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("📐 Mathematical Foundations & Curriculum Analysis")

curric_tab1, curric_tab2, curric_tab3 = st.tabs(["Algebra 2", "Precalculus", "Calculus"])

with curric_tab1:
    st.subheader("Algebra 2 Core Concepts")
    st.markdown(f"""
    - **Domain**: Historical observed year domain is $[{working_df['Year'].min()}, {working_df['Year'].max()}]$. Display horizon domain spans $[{working_df['Year'].min()}, {working_df['Year'].max() + forecast_horizon}]$.
    - **Range**: Fitted values over historical domain range from `{poly_func(t_obs).min():.2f}` to `{poly_func(t_obs).max():.2f}` {ind_meta['units']}.
    - **End Behavior**: Leading coefficient $a_{{{poly_degree}}} = {coeffs[0]:.4e}$.
        - As $t \\to +\\infty$, $y(t) \\to {'+\\infty' if coeffs[0] > 0 else '-\\infty'}$.
        - As $t \\to -\\infty$, $y(t) \\to {'+\\infty' if (coeffs[0] > 0 and poly_degree % 2 == 0) or (coeffs[0] < 0 and poly_degree % 2 != 0) else '-\\infty'}$.
    - **Average Rate of Change**: Over the entire historical span ($t_1={working_df['Year'].min()}$ to $t_2={working_df['Year'].max()}$):
      $$\\text{{AROC}} = \\frac{{{values[-1]:.2f} - {values[0]:.2f}}}{{{working_df['Year'].max() - working_df['Year'].min()}}} = \\frac{{{values[-1] - values[0]:.2f}}}{{{working_df['Year'].max() - working_df['Year'].min()}}} = {((values[-1] - values[0]) / (working_df['Year'].max() - working_df['Year'].min())):.4f} \\text{{ {ind_meta['units']}/year}}$$
    """)

with curric_tab2:
    st.subheader("Precalculus Structural Analysis")
    # Turning points via derivative roots
    d1 = np.polyder(poly_func)
    r_d1 = np.roots(d1)
    real_turning = [r + base_year for r in r_d1 if np.isreal(r) and (0 <= r <= (working_df['Year'].max() - base_year))]
    
    st.markdown(f"""
    - **Polynomial Degree Effect**: Degree $d = {poly_degree}$ allows up to $d-1 = {poly_degree-1}$ local turning points and at most $d-2 = {poly_degree-2}$ inflection points.
    - **Turning Points in Observed Domain**: Detected at years: `{ [round(x, 1) for x in real_turning] if real_turning else 'None in observed range' }`.
    - **Concavity Dynamics**: Changing average rates of change demonstrate local concavity shifts across historical intervals.
    """)

with curric_tab3:
    st.subheader("Calculus Analysis (Derivatives, Extrema & Inflection)")
    d1_func = np.polyder(poly_func, 1)
    d2_func = np.polyder(poly_func, 2)
    
    crit_t = np.roots(d1_func)
    real_crit = [r for r in crit_t if np.isreal(r) and (0 <= r <= (working_df['Year'].max() - base_year))]
    
    crit_info = []
    for r in real_crit:
        yr_val = r + base_year
        d2_val = d2_func(r)
        c_type = "Local Minimum" if d2_val > 0 else ("Local Maximum" if d2_val < 0 else "Inconclusive")
        crit_info.append(f"Year {yr_val:.1f}: {c_type} (f''(t) = {d2_val:.3e})")
        
    inflect_t = np.roots(d2_func)
    real_inflect = [r + base_year for r in inflect_t if np.isreal(r) and (0 <= r <= (working_df['Year'].max() - base_year))]
    
    st.markdown(f"""
    - **First Derivative $y'(t)$**: Describes instantaneous rate of change of {ind_meta['name']}.
    - **Second Derivative $y''(t)$**: Describes acceleration / concavity.
    - **Classified Critical Points**:
      { chr(10).join(['- ' + ci for ci in crit_info]) if crit_info else '- No interior critical points found in historical window.' }
    - **Verified Inflection Points**:
      { f"- Years: {[round(x, 1) for x in real_inflect]}" if real_inflect else "- No inflection points within observed window." }
    """)

# -----------------------------------------------------------------------------
# VISUALIZATION & COMPARISON
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("📉 Data Visualization & Comparison")

fig, ax = plt.subplots(figsize=(10, 5))

# Graph Sample Increment Filtering
sample_df = working_df.iloc[::sample_step]

# Plot Observed Data
ax.scatter(sample_df["Year"], sample_df["Value"], color="navy", label=f"Observed Sample (Step={sample_step}yr)", zorder=3)
ax.plot(working_df["Year"], working_df["Value"], color="blue", alpha=0.3, linestyle="--", label="Historical Path")

# Continuous Curve Generation
full_years = np.linspace(working_df["Year"].min(), working_df["Year"].max() + forecast_horizon, 300)
full_t = full_years - base_year
full_pred = poly_func(full_t)

# Historical vs Forecast split
hist_mask = full_years <= working_df["Year"].max()
ax.plot(full_years[hist_mask], full_pred[hist_mask], color="darkgreen", linewidth=2, label=f"Fitted Poly (deg={poly_degree})")
if forecast_horizon > 0:
    ax.plot(full_years[~hist_mask], full_pred[~hist_mask], color="red", linestyle=":", linewidth=2, label="Extrapolated Forecast")

# Plot Single/Batch Calculation Markers
if st.session_state.calculated_results is not None:
    calc_df = st.session_state.calculated_results
    ax.scatter(calc_df["Year"], calc_df["Fitted Value Estimate"], color="crimson", marker="X", s=100, label="Calculated Points", zorder=5)

# Comparison Country Overlay
if comparison_country != "None":
    comp_df, _, _ = fetch_world_bank_data(comparison_country, selected_indicator)
    if not comp_df.empty:
        ax.plot(comp_df["Year"], comp_df["Value"], color="orange", linestyle="-.", label=f"Comparison: {comparison_country}")
    else:
        st.warning(f"No comparison data available for {comparison_country}.")

# Formatting Chart
ax.axvline(working_df["Year"].max(), color="gray", linestyle="--", alpha=0.7, label="Historical Boundary")
ax.set_title(f"{ind_meta['name']} Trend - {COUNTRY_MAP[selected_country]}", fontsize=12)
ax.set_xlabel("Year")
ax.set_ylabel(ind_meta["units"])
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="best", fontsize=8)

if printer_mode:
    st.info("🖨️ **Printer-Friendly View Active**: Clean panel rendering for direct document printing.")
    st.pyplot(fig)
else:
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# EXPORT ENGINE (CSV, EXCEL, PDF)
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("📤 Export Options")

col_ex1, col_ex2, col_ex3 = st.columns(3)

# 1. CSV EXPORT
csv_buffer = io.StringIO()
working_df.to_csv(csv_buffer, index=False)
col_ex1.download_button(
    label="📄 Download CSV Data",
    data=csv_buffer.getvalue(),
    file_name=f"{selected_country}_{selected_indicator}_data.csv",
    mime="text/csv"
)

# 2. EXCEL EXPORT (.xlsx via XlsxWriter)
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    # Summary Sheet
    summary_df = pd.DataFrame([
        {"Metric": "Author", "Value": APP_AUTHOR},
        {"Metric": "Country", "Value": COUNTRY_MAP[selected_country]},
        {"Metric": "Indicator", "Value": ind_meta["name"]},
        {"Metric": "Units", "Value": ind_meta["units"]},
        {"Metric": "Polynomial Degree", "Value": poly_degree},
        {"Metric": "Equation", "Value": eq_str},
        {"Metric": "In-Sample R2", "Value": r2_in},
        {"Metric": "In-Sample MAE", "Value": mae_in},
        {"Metric": "User Modified Data Flag", "Value": st.session_state.is_modified}
    ])
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    # Working Data
    working_df.to_excel(writer, sheet_name="Working Data", index=False)
    
    # Calculated Results
    if st.session_state.calculated_results is not None:
        st.session_state.calculated_results.to_excel(writer, sheet_name="Calculated Results", index=False)

col_ex2.download_button(
    label="📊 Download Excel (.xlsx)",
    data=excel_buffer.getvalue(),
    file_name=f"{selected_country}_{selected_indicator}_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# 3. GENUINE PDF EXPORT (ReportLab)
def generate_pdf_report():
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.HexColor('#1E3A8A'))
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=12, leading=16, textColor=colors.HexColor('#1E3A8A'))
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9, leading=12)
    
    # Header
    story.append(Paragraph("Hispanic Heritage Historical Data Analysis Report", title_style))
    story.append(Paragraph(f"Author: {APP_AUTHOR} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 10))
    
    # Metadata Table
    meta_data = [
        ["Country", COUNTRY_MAP[selected_country]],
        ["Indicator", f"{ind_meta['name']} ({selected_indicator})"],
        ["Units", ind_meta["units"]],
        ["Model Poly Degree", str(poly_degree)],
        ["In-Sample R²", f"{r2_in:.4f}"],
        ["In-Sample MAE", f"{mae_in:.4f}"],
        ["Data Status", "User-Modified Scenario" if st.session_state.is_modified else "Official Unedited Source"]
    ]
    t_meta = Table(meta_data, colWidths=[150, 350])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#E5E7EB')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('PADDING', (0,0), (-1,-1), 4)
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))
    
    # Equation
    story.append(Paragraph("Fitted Model Equation", h2_style))
    story.append(Paragraph(f"<font fontName='Courier'>{eq_str}</font>", body_style))
    story.append(Spacer(1, 10))
    
    # Chart Image
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=500, height=250))
    story.append(Spacer(1, 10))
    
    # Methodology & Findings
    story.append(Paragraph("Findings & Modeling Limitations", h2_style))
    limitations_text = (
        "<b>Model Uncertainty & Limitations:</b> High-degree polynomial regression is subject to severe edge-effects "
        "and Runge's phenomenon. Extrapolation beyond observed ranges carries significant mathematical uncertainty. "
        "Calculated trendlines reflect polynomial fitting over time (t = Year - BaseYear) and do not imply causal mechanisms."
    )
    story.append(Paragraph(limitations_text, body_style))
    story.append(Spacer(1, 10))
    
    # Calculated Results Table if available
    if st.session_state.calculated_results is not None:
        story.append(Paragraph("Calculated Output Predictions", h2_style))
        calc_df = st.session_state.calculated_results.head(10) # limit to top 10 for PDF view
        calc_data = [["Year", "Estimate", "Units", "Classification"]]
        for _, row in calc_df.iterrows():
            calc_data.append([str(row["Year"]), f"{row['Fitted Value Estimate']:.2f}", str(row["Units"]), str(row["Domain Classification"])])
        
        t_calc = Table(calc_data, colWidths=[60, 80, 120, 240])
        t_calc.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('PADDING', (0,0), (-1,-1), 3)
        ]))
        story.append(t_calc)

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

col_ex3.download_button(
    label="📕 Download PDF Report",
    data=generate_pdf_report(),
    file_name=f"{selected_country}_{selected_indicator}_report.pdf",
    mime="application/pdf"
)
