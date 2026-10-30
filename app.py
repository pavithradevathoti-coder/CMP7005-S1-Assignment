
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.proportion import proportion_confint

st.set_page_config(page_title="Hypertension population explorer", layout="wide")
folder = Path(__file__).parent
d = pd.read_csv(folder / "population.csv")
info = json.loads((folder / "details.json").read_text())
for column, levels in info["categories"].items():
    d[column] = pd.Categorical(d[column], categories=levels, ordered=True)
BLUE, ORANGE = "#2A78D6", "#EB6834"
label_colours = {"No": BLUE, "Yes": ORANGE}
gender_colours = {"Female": "#008000", "Male": "#6850DF"}
labels = info["labels"]
FACTORS = ["Physical_Activity", "Salt_Intake", "Smoking_Status", "Family_History_HTN", "BMI_Group",
           "Age_Band", "Age_Group", "Gender", "Alcohol_Consumption", "Stress_Level", "Sleep_Group", "Count_Group", "Glucose_Band", "Cholesterol_Band"]
labels.update({"Age_Group": "Age group", "Count_Group": "Selected factor count", "Glucose_Band": "Glucose band", "Cholesterol_Band": "Cholesterol band"})
MEASURES = ["Age", "BMI", "Waist_Circumference_cm", "Sleep_Hours", "Fasting_Glucose_mg_dL",
            "Cholesterol_mg_dL", "Creatinine_mg_dL", "Systolic_BP_mmHg", "Diastolic_BP_mmHg", "Pulse_Pressure", "MAP"]
d["Count_Group"] = d.Risk_Count.map(lambda v: "Unknown" if pd.isna(v) else ("5–6" if v >= 5 else str(int(v))))
d["Count_Group"] = pd.Categorical(d.Count_Group, categories=["0", "1", "2", "3", "4", "5–6", "Unknown"], ordered=True)
d["Glucose_Band"] = pd.cut(d.Fasting_Glucose_mg_dL, [0, 100, 126, np.inf], right=False, labels=["<100", "100–125", "≥126"])
d["Cholesterol_Band"] = pd.cut(d.Cholesterol_mg_dL, [0, 200, 240, np.inf], right=False, labels=["<200", "200–239", "≥240"])
for column in ["Glucose_Band", "Cholesterol_Band"]:
    d[column] = d[column].cat.add_categories(["Unknown"]).fillna("Unknown")

def rates(data, group, outcome):
    table = data.groupby(group, observed=True)[outcome].agg(n="count", events="sum").reset_index()
    table = table[table.n > 0].copy()
    low, high = proportion_confint(table.events, table.n, method="wilson")
    table["percent"], table["low"], table["high"] = 100*table.events/table.n, 100*low, 100*high
    table["plus"] = (table.high-table.percent).clip(lower=0)
    table["minus"] = (table.percent-table.low).clip(lower=0)
    return table

st.title("Hypertension population explorer")
st.caption("Explore population patterns • Educational use • No individual diagnosis or prediction")
with st.sidebar:
    st.header("Population filters")
    ages = st.slider("Age range", 18, 90, (18, 90))
    genders = st.multiselect("Recorded gender", ["Female", "Male"], default=["Female", "Male"])
    bmi_groups = st.multiselect("BMI bands", list(d.BMI_Group.cat.categories), default=list(d.BMI_Group.cat.categories))
    family = st.selectbox("Family history", ["All", "No", "Yes"])
    st.caption("Filters apply to both sections. Empty selections show no records.")
s = d[d.Age.between(*ages) & d.Gender.isin(genders) & d.BMI_Group.isin(bmi_groups)].copy()
if family != "All":
    s = s[s.Family_History_HTN == family]
if s.empty:
    st.info("No records match these filters. Widen the age range or select a gender.")
    st.stop()

overview, explorer = st.tabs(["1. Data overview", "2. Explore patterns"])
with overview:
    cards = st.columns(4)
    cards[0].metric("Selected records", f"{len(s):,}")
    cards[1].metric("Recorded label", f"{100*s.HTN.mean():.1f}%")
    cards[2].metric("140/90 comparator", f"{100*s.BP_140_90.mean():.1f}%")
    cards[3].metric("Median age", f"{s.Age.median():.1f} years")
    st.write("The recorded label matches SBP ≥130 or DBP ≥80. The 140/90 comparator is another single-reading rule; neither establishes a confirmed UK diagnosis.")
    stage = s.groupby("BP_Category", observed=True).size().rename("Records").reset_index()
    fig = px.bar(stage, x="BP_Category", y="Records", text="Records", color_discrete_sequence=[BLUE])
    fig.update_layout(template="plotly_white", height=340, xaxis_title="Recorded BP category", font=dict(size=14, color="#111111"))
    st.plotly_chart(fig, width="stretch", theme=None)
    with st.expander("Data quality and study limitations"):
        st.write(f"Original records: {info['source_rows']:,}. Exact duplicates removed: {info['duplicates']}. Numeric gaps remain missing; categorical gaps are Unknown. No scaling was applied.")
        st.write(f"Unknown alcohol responses in this selection: {s.Alcohol_Consumption.eq('Unknown').sum():,}.")
        st.dataframe(pd.DataFrame(info["missing"].items(), columns=["Original variable", "Missing in full source"]), hide_index=True)
        st.write("Dates, treatment, repeat BP and follow-up outcomes are unavailable. Associations may reflect age, selection and other confounding. Exploratory eGFR uses an unverified Gender-to-sex coefficient mapping.")

with explorer:
    mode = st.selectbox("Choose a chart", ["Group percentages", "Age comparison", "Two factors", "Scatter", "Measurements",
                                          "BP-category profile", "Population sunburst", "Population treemap", "Profile radar"])
    outcome = "HTN"
    if mode in ["Group percentages", "Age comparison", "Two factors"]:
        outcome = st.selectbox("Outcome", ["HTN", "BP_140_90"], format_func=lambda c: "Recorded label" if c == "HTN" else "140/90 comparator")
    age_style = st.radio("Age view", ["Age and gender trend", "Within-age comparison"], horizontal=True) if mode == "Age comparison" else None
    measurement_style = st.radio("Measurement view", ["Laboratory panel", "Single measurement"], horizontal=True) if mode == "Measurements" else None
    caption = "Current population associations only; filtering does not fully adjust for confounding."

    if mode == "Group percentages":
        group = st.selectbox("Compare groups", FACTORS, format_func=lambda c: labels[c])
        table = rates(s, group, outcome)
        table = table[table.n >= 20].copy()
        table["colour"] = [gender_colours.get(str(v), BLUE) if group == "Gender" else ("#777777" if str(v) == "Unknown" else BLUE) for v in table[group]]
        fig = go.Figure(go.Bar(x=table.percent, y=table[group].astype(str), orientation="h",
            marker_color=table.colour, error_x=dict(type="data", array=table.plus, arrayminus=table.minus),
            customdata=table[["n", "events"]], hovertemplate="%{y}: %{x:.1f}%<br>n=%{customdata[0]}<br>Events=%{customdata[1]}<extra></extra>"))
        fig.update_xaxes(range=[0, 100], title="Selected outcome (%)")
        fig.update_yaxes(autorange="reversed")
        table = table.drop(columns=["colour", "plus", "minus"])
        caption += " Whiskers show Wilson 95% confidence intervals. Groups with fewer than 20 records are omitted."

    elif mode == "Age comparison" and age_style == "Age and gender trend":
        table = rates(s, ["Age_Group", "Gender"], outcome)
        table = table[table.n >= 20].copy()
        fig = go.Figure()
        order = info["categories"]["Age_Group"]
        for gender, colour in gender_colours.items():
            part = table[table.Gender == gender].set_index("Age_Group").reindex(order)
            if part.percent.notna().sum() == 0:
                continue
            fill = "rgba(0,128,0,0.14)" if gender == "Female" else "rgba(104,80,223,0.14)"
            fig.add_trace(go.Scatter(x=order, y=part.low, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=order, y=part.high, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=fill, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=order, y=part.percent, mode="lines+markers", name=gender,
                line=dict(color=colour, width=3), marker=dict(size=9, line=dict(color="white", width=2)),
                customdata=part[["n", "low", "high"]], hovertemplate="%{x}: %{y:.1f}%<br>n=%{customdata[0]}<br>95% CI %{customdata[1]:.1f}–%{customdata[2]:.1f}%<extra>%{fullData.name}</extra>"))
        fig.update_yaxes(range=[0, 100], title="Selected outcome (%)")
        fig.update_xaxes(title="Age group (years)", categoryorder="array", categoryarray=order)
        caption += " Shading shows 95% confidence intervals. Groups below n=20 leave gaps. This is not a trajectory of individual patients."

    elif mode in ["Age comparison", "Two factors"]:
        group = st.selectbox("Column groups", FACTORS, format_func=lambda c: labels[c])
        second = "Age_Band" if mode == "Age comparison" else st.selectbox("Row groups", FACTORS, index=5, format_func=lambda c: labels[c])
        if group == second:
            st.info("Choose two different variables."); st.stop()
        table = rates(s, [second, group], outcome)
        table = table[table.n >= 20].copy()
        values = table.pivot(index=second, columns=group, values="percent")
        sizes = table.pivot(index=second, columns=group, values="n")
        fig = go.Figure(go.Heatmap(z=values, x=values.columns.astype(str), y=values.index.astype(str),
            zmin=0, zmax=100, colorscale="Blues", customdata=sizes, texttemplate="%{z:.1f}%",
            hoverongaps=False, hovertemplate="%{y} / %{x}<br>%{z:.1f}%<br>n=%{customdata}<extra></extra>"))
        fig.update_xaxes(title=labels[group]); fig.update_yaxes(title=labels[second], autorange="reversed")
        caption += " Hover for group counts. Cells with fewer than 20 records are omitted."

    elif mode == "Scatter":
        x = st.selectbox("X measurement", MEASURES, index=8, format_func=lambda c: labels[c])
        y = st.selectbox("Y measurement", MEASURES, index=7, format_func=lambda c: labels[c])
        split = st.selectbox("Colour by", ["Hypertension", "Gender", "BP_Category"], format_func=lambda c: labels[c])
        part = s.dropna(subset=[x, y]).copy()
        part = part[part.groupby(split, observed=True).HTN.transform("size") >= 20].copy()
        rng = np.random.default_rng(7)
        bp_view = x == "Diastolic_BP_mmHg" and y == "Systolic_BP_mmHg"
        part["X display"], part["Y display"] = part[x], part[y]
        if bp_view:
            part["X display"] += rng.uniform(-0.45, 0.45, len(part))
            part["Y display"] += rng.uniform(-0.45, 0.45, len(part))
        colours = label_colours if split == "Hypertension" else gender_colours
        fig = px.scatter(part, x="X display", y="Y display", color=split, color_discrete_map=colours,
            category_orders={split: info["categories"][split]}, opacity=0.55, render_mode="svg")
        fig.update_traces(marker_size=4)
        if bp_view:
            for sbp, dbp, dash in [(130, 80, "dash"), (140, 90, "dot")]:
                fig.add_hline(y=sbp, line_dash=dash, line_color="#444444", annotation_text=f"SBP {sbp}")
                fig.add_vline(x=dbp, line_dash=dash, line_color="#444444", annotation_text=f"DBP {dbp}")
        fig.update_xaxes(title=labels[x]); fig.update_yaxes(title=labels[y])
        table = part.groupby(split, observed=True)[list(dict.fromkeys([x, y]))].agg(["count", "median"])
        table.columns = [" ".join(c) for c in table.columns]
        table = table.reset_index()
        caption += " Groups below n=20 are omitted. The BP view uses display-only jitter; BP separation by label partly reflects its definition."
        if x == y:
            caption += " Both axes show the same measurement; the diagonal is an identity, not an association."

    elif mode == "Measurements" and measurement_style == "Single measurement":
        measure = st.selectbox("Measurement", MEASURES, index=4, format_func=lambda c: labels[c])
        split = st.selectbox("Compare by", ["Hypertension", "Gender", "BP_Category"], format_func=lambda c: labels[c])
        part = s.dropna(subset=[measure]).copy()
        part = part[part.groupby(split, observed=True).HTN.transform("size") >= 20]
        fig = px.box(part, x=split, y=measure, color=split, points=False,
                     color_discrete_map=label_colours if split == "Hypertension" else gender_colours,
                     category_orders={split: info["categories"][split]}, labels=labels)
        table = part.groupby(split, observed=True)[measure].agg(n="count", median="median", minimum="min", maximum="max").reset_index()
        caption += " Box plots show the median and interquartile range. Groups below n=20 are omitted. BP comparisons by label are partly circular."

    elif mode in ["Population sunburst", "Population treemap"]:
        group = st.selectbox("Outer groups", FACTORS, format_func=lambda c: labels[c])
        second = st.selectbox("Inner groups", FACTORS, index=3, format_func=lambda c: labels[c])
        if group == second:
            st.info("Choose two different grouping variables."); st.stop()
        table = rates(s, [second, group], "HTN")
        table = table[table.n >= 20].copy()
        if table.empty:
            st.info("No population cells have at least 20 records. Widen the filters."); st.stop()
        for column in [second, group]:
            table[column] = table[column].astype(str)
        draw = px.sunburst if mode == "Population sunburst" else px.treemap
        fig = draw(table, path=[second, group], values="n", color="percent",
                   color_continuous_scale="Blues", range_color=[0, 100], labels=labels)
        caption = "Area shows record counts; colour shows recorded-label percentage. Cells below n=20 are omitted, so displayed totals may be smaller than the selected population."

    elif mode in ["BP-category profile", "Profile radar"]:
        specs = [("Sedentary", "Physical_Activity", ["Sedentary"]), ("High salt", "Salt_Intake", ["High", "Very High"]),
                 ("Family history", "Family_History_HTN", ["Yes"]), ("High stress", "Stress_Level", ["High", "Very High"]),
                 ("Current smoking", "Smoking_Status", ["Current"]), ("BMI ≥30", "BMI_Group", ["BMI ≥30"])]
        rows = []
        for stage in info["categories"]["BP_Category"]:
            for name, column, levels in specs:
                part = s[(s.BP_Category == stage) & s[column].notna() & s[column].ne("Unknown")]
                if len(part) >= 20:
                    rows.append({"Stage": str(stage), "Characteristic": name, "n": len(part),
                                 "Percent": 100*part[column].isin(levels).mean()})
        table = pd.DataFrame(rows, columns=["Stage", "Characteristic", "n", "Percent"])
        if mode == "BP-category profile":
            fig = px.bar(table, x="Characteristic", y="Percent", color="Stage", barmode="group", hover_data=["n"])
            fig.update_yaxes(range=[0, 100])
        else:
            fig = go.Figure()
            complete_stages = []
            for stage, part in table.groupby("Stage", sort=False):
                if len(part) == len(specs):
                    complete_stages.append(stage)
                    fig.add_trace(go.Scatterpolar(r=list(part.Percent)+[part.Percent.iloc[0]],
                        theta=list(part.Characteristic)+[part.Characteristic.iloc[0]], mode="lines+markers", name=stage))
            table = table[table.Stage.isin(complete_stages)]
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
        caption = "All axes show percentages of available responses, not a risk score. Groups below n=20 are omitted; radar requires all six characteristics. BP categories compare different records, not progression over time."

    else:
        labs = ["Fasting_Glucose_mg_dL", "Cholesterol_mg_dL", "Creatinine_mg_dL", "eGFR"]
        names = ["Glucose", "Cholesterol", "Creatinine", "Exploratory eGFR"]
        fig = make_subplots(rows=1, cols=4, subplot_titles=names, horizontal_spacing=0.08)
        rows = []
        for i, column in enumerate(labs, 1):
            for status in ["No", "Yes"]:
                values = s.loc[s.Hypertension == status, column].dropna()
                if len(values) < 20:
                    continue
                rows.append({"Measurement": labels[column], "Label": status, "n": len(values), "Median": values.median()})
                fig.add_trace(go.Violin(x=[status]*len(values), y=values, name=status, legendgroup=status,
                    showlegend=(i==1), line_color=label_colours[status], fillcolor=label_colours[status],
                    opacity=0.8, box_visible=True, points=False, spanmode="hard"), row=1, col=i)
            fig.update_xaxes(categoryorder="array", categoryarray=["No", "Yes"], row=1, col=i)
            fig.update_yaxes(title_text="mg/dL" if i < 4 else "mL/min/1.73m²", row=1, col=i)
        table = pd.DataFrame(rows)
        caption = "Laboratory groups use available measurements and require n≥20. Exploratory eGFR includes age and assumes recorded Gender represents the equation's sex coefficient; it does not diagnose kidney disease."

    st.caption(caption)
    if table.empty:
        st.info("No groups have enough recorded observations. Widen the filters.")
    else:
        fig.update_layout(template="plotly_white", height=460, font=dict(size=14, color="#111111"), margin=dict(l=35, r=25, t=45, b=40))
        st.plotly_chart(fig, width="stretch", theme=None,
            config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "hypertension_chart"}})
        with st.expander("View counts and chart values"):
            st.dataframe(table.round(2), hide_index=True)
        st.download_button("Download summary table", table.to_csv(index=False).encode(), "population_summary.csv", "text/csv")

