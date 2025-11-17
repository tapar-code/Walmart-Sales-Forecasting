# dash_dashboard.py  — portable + production-ready
import os
import pandas as pd
import numpy as np
from datetime import datetime
import dash
from dash import dcc, html, Input, Output
from dash import dash_table
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Paths — ABSOLUTE paths for your Mac
# =========================

# 🚨 EDIT THIS to your actual project root folder
PROJECT_ROOT = "./"

RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results")
DATA_DIR    = os.path.join(PROJECT_ROOT, "Data")
RAW_DIR     = os.path.join(DATA_DIR, "Raw")
PROC_DIR    = os.path.join(DATA_DIR, "Processed")

PATHS = {
    "metrics":   os.path.join(RESULTS_DIR, "all_models_metrics.csv"),
    "forecasts": os.path.join(RESULTS_DIR, "all_models_forecast_comparison.csv"),
    "holiday":   os.path.join(RESULTS_DIR, "holiday_uplift_summary.csv"),
    "markdown":  os.path.join(RESULTS_DIR, "markdown_uplift_summary.csv"),
    "inventory": os.path.join(RESULTS_DIR, "inventory_simulation_results.csv"),
    "future":    os.path.join(RESULTS_DIR, "future_models_forecast_8w.csv"),

    # Raw data
    "raw_train":    os.path.join(RAW_DIR,  "train.csv"),
    "raw_features": os.path.join(RAW_DIR,  "features.csv"),
    "raw_stores":   os.path.join(RAW_DIR,  "stores.csv"),

    # Processed
    "cleaned":      os.path.join(PROC_DIR, "cleaned_walmart_sales.csv"),
}

# =========================
# Safe loaders
# =========================
def read_csv_safe(path, parse_dates=None):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except Exception:
        return None

def none_fig(title="No data found"):
    fig = go.Figure()
    fig.update_layout(
        title=title, height=420,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=title, showarrow=False, font=dict(size=16))]
    )
    return fig

# =========================
# Load data
# =========================
metrics_df = read_csv_safe(PATHS["metrics"])
fc_df      = read_csv_safe(PATHS["forecasts"], parse_dates=["Date"])
holiday_df = read_csv_safe(PATHS["holiday"])
markdown_df= read_csv_safe(PATHS["markdown"])
inv_df     = read_csv_safe(PATHS["inventory"])
future_df  = read_csv_safe(PATHS["future"], parse_dates=["Date"])

# Optional pipeline sources (only if files exist)
raw_train   = read_csv_safe(PATHS["raw_train"], parse_dates=["Date"])
raw_feats   = read_csv_safe(PATHS["raw_features"], parse_dates=["Date"])
raw_stores  = read_csv_safe(PATHS["raw_stores"])
cleaned_df  = read_csv_safe(PATHS["cleaned"], parse_dates=["Date"])

# Standardize columns
if fc_df is not None:
    fc_df = fc_df.rename(columns={"Date": "date"})
if inv_df is not None and "Date" in inv_df.columns:
    inv_df = inv_df.rename(columns={"Date": "date"})

# =========================
# Brand / Style
# =========================
USC = {
    "cardinal": "#990000",
    "gold": "#FFCC00",
    "black": "#000000",
    "white": "#FFFFFF",
    "gray":  "#F7F7F8",
    "blue":  "#0B1F3B",
}
TAB_STYLE          = {"backgroundColor": USC["gold"], "color": USC["black"], "fontWeight": "600"}
TAB_SELECTED_STYLE = {"backgroundColor": USC["cardinal"], "color": USC["white"], "fontWeight": "700"}

CARD_STYLE = {
    "background": USC["gray"], "borderRadius": "16px", "padding": "14px 16px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.06)", "border": "1px solid #eee"
}

# =========================
# Helpers
# =========================
def fmt_money(x):
    try:
        v = float(x)
        if abs(v) >= 1e6:  return f"${v/1e6:,.1f}M"
        if abs(v) >= 1e3:  return f"${v:,.0f}"
        return f"${v:,.2f}"
    except Exception:
        return str(x)

def fmt_pct(x, decimals=2):
    """
    Safe percentage formatter:
    - Returns 'None' for None / NaN / bad values
    - Otherwise '{:.decimalsf}%'
    """
    try:
        if x is None:
            return None
        v = float(x)
        # handle NaN (NaN != NaN)
        if v != v:
            return None
        return f"{v:.{decimals}f}%"
    except Exception:
        return None

def kpi_card(title, value, sub=None):
    return html.Div([
        html.Div(title, style={"fontSize": "12px", "opacity": 0.75, "marginBottom": "6px"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": 700, "color": USC["cardinal"]}),
        html.Div(sub or "", style={"fontSize": "11px", "opacity": 0.7, "marginTop": "4px"})
    ], style=CARD_STYLE)

def table_from_df(df, max_rows=10):
    if df is None or df.empty:
        return html.Div("No data available.", style={"opacity": 0.6})
    preview = df.head(max_rows)
    return dash_table.DataTable(
        data=preview.to_dict("records"),
        columns=[{"name": c, "id": c} for c in preview.columns],
        page_size=max_rows,
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Inter, Arial", "fontSize": 12, "whiteSpace": "nowrap", "textAlign": "left"},
        style_header={"fontWeight": "700", "backgroundColor": USC["gray"]},
    )

def compute_pipeline_stats():
    out = {}
    if cleaned_df is not None and "Date" in cleaned_df.columns:
        out["clean_rows"]  = len(cleaned_df)
        out["clean_cols"]  = len(cleaned_df.columns)
        out["clean_start"] = cleaned_df["Date"].min()
        out["clean_end"]   = cleaned_df["Date"].max()
        key_cols = ["Store","Dept","Date","Weekly_Sales","IsHoliday","CPI","Unemployment","Fuel_Price"]
        have_cols = [c for c in key_cols if c in cleaned_df.columns]
        out["na_summary"] = cleaned_df[have_cols].isna().sum().rename("Nulls").reset_index().rename(columns={"index":"Column"})
        if set(["Store","Dept","Date"]).issubset(cleaned_df.columns):
            out["dupes"] = int(cleaned_df.duplicated(subset=["Store","Dept","Date"]).sum())
    if raw_train is not None:
        out["raw_rows"] = len(raw_train)
        out["raw_cols"] = len(raw_train.columns)
    return out

PIPE = compute_pipeline_stats()

def insights_from_data():
    out = {}
    if metrics_df is not None and {"Model","RMSE_test"}.issubset(metrics_df.columns):
        best_row = metrics_df.loc[metrics_df["RMSE_test"].idxmin()]
        out["best_model"] = str(best_row["Model"])
        out["best_rmse"]  = float(best_row["RMSE_test"])
        out["best_mape"]  = float(best_row.get("MAPE test (%)", np.nan))
    else:
        # ensure keys always exist and are numeric / NaN
        out["best_model"] = None
        out["best_rmse"]  = np.nan
        out["best_mape"]  = np.nan

    if holiday_df is not None and {"Holiday","Avg_Uplift"}.issubset(holiday_df.columns):
        hol_sorted = holiday_df.sort_values("Avg_Uplift", ascending=False)
        out["top_holiday"] = hol_sorted.head(1).to_dict("records")[0] if not hol_sorted.empty else None
        out["low_holiday"] = hol_sorted.tail(1).to_dict("records")[0] if len(hol_sorted) > 0 else None

    if inv_df is not None and {"Service_Level","Avg_Cost","Avg_Stockouts","Model"}.issubset(inv_df.columns):
        # be tolerant of string service levels
        snap = inv_df.copy()
        snap["Service_Level"] = pd.to_numeric(snap["Service_Level"], errors="coerce")
        snap95 = snap[np.isclose(snap["Service_Level"], 1.64)]
        if not snap95.empty:
            out["inv95_best_cost"]     = snap95.loc[snap95["Avg_Cost"].idxmin()].to_dict()
            out["inv95_best_stockout"] = snap95.loc[snap95["Avg_Stockouts"].idxmin()].to_dict()
    return out

INS = insights_from_data()

# =========================
# App
# =========================
app = dash.Dash(__name__, suppress_callback_exceptions=True)   # allow callbacks to targets inside tabs
app.title = "Walmart Sales Forecasting — Executive Analytics"
server = app.server  # for gunicorn / Docker

# =========================
# Layout
# =========================
app.layout = html.Div(style={"backgroundColor": USC["white"], "fontFamily": "Inter, Arial, sans-serif"}, children=[
    # Header
    html.Div([
        html.Div([
            html.Img(
                src="https://identity.usc.edu/wp-content/uploads/2020/01/Primary-Wordmark_CardOnTrans_RGB.png",
                style={"height": "56px", "marginRight": "18px"}
            ),
            html.Div([
                html.H1("Walmart Sales Forecasting — Executive Analytics",
                        style={"margin":"0","color":USC["cardinal"],"fontSize":"26px","letterSpacing":"0.2px"}),
                html.Div("Data pipeline · Forecast accuracy · Promotion impact · Inventory trade-offs",
                         style={"color": USC["blue"], "opacity": 0.85, "marginTop":"4px"})
            ])
        ], style={"display":"flex","alignItems":"center"})
    ], style={"padding":"14px 18px","borderBottom":f"4px solid {USC['gold']}","background":USC["white"]}),

    dcc.Tabs(style={"borderBottom": f"2px solid {USC['gray']}"}, children=[
        # Data Pipeline
        dcc.Tab(label="Data Pipeline", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div([
                html.Div([
                    html.Div("Source → Cleaning → Modeling", style={"fontWeight":"700","color":USC["blue"],"marginBottom":"8px"}),
                    html.Ul([
                        html.Li("Raw inputs: train.csv, features.csv, stores.csv (Kaggle)."),
                        html.Li("Merged on (Store, Date) then (Store) → sorted by (Store, Dept, Date)."),
                        html.Li("IsHoliday unified; MarkDowns filled with 0; CPI/Unemployment ffill/bfill per Store."),
                        html.Li("Output: cleaned_walmart_sales.csv (weekly chain-level ready)."),
                    ], style={"lineHeight":"1.7"})
                ], style={"flex":1, "marginRight":"16px"}),

                html.Div([
                    html.Div([
                        kpi_card("Raw rows", f"{PIPE.get('raw_rows', '—'):,}" if PIPE.get("raw_rows") else "—",
                                 sub=f"{PIPE.get('raw_cols','—')} columns" if PIPE.get("raw_cols") else None),
                        kpi_card("Cleaned rows", f"{PIPE.get('clean_rows','—'):,}" if PIPE.get("clean_rows") else "—",
                                 sub=f"{PIPE.get('clean_cols','—')} columns" if PIPE.get("clean_cols") else None),
                        kpi_card("Date coverage",
                                 f"{PIPE.get('clean_start','—')} → {PIPE.get('clean_end','—')}" if PIPE.get("clean_start") else "—",
                                 sub="Cleaned dataset (weekly)"),
                        kpi_card("Duplicates (Store,Dept,Date)", f"{PIPE.get('dupes', 0):,}" if PIPE.get("dupes") is not None else "—"),
                    ], style={"display":"grid","gridTemplateColumns":"repeat(4, minmax(180px,1fr))","gap":"12px"})
                ], style={"flex":1})
            ], style={"display":"flex","padding":"16px"}),

            html.Hr(),
            html.Div([
                html.Div([
                    html.H4("Raw: train.csv (preview)"),
                    table_from_df(raw_train)
                ], style={"flex":1,"marginRight":"12px"}),
                html.Div([
                    html.H4("Raw: features.csv (preview)"),
                    table_from_df(raw_feats)
                ], style={"flex":1,"marginLeft":"12px"}),
            ], style={"display":"flex","padding":"0 16px 16px"}),

            html.Div([
                html.H4("Cleaned dataset (preview)"),
                table_from_df(cleaned_df)
            ], style={"padding":"0 16px 16px"}),

            html.Div([
                html.H4("Nulls in key columns (cleaned)"),
                table_from_df(PIPE.get("na_summary"))
            ], style={"padding":"0 16px 16px"})
        ]),

        # Data Quality
        dcc.Tab(label="Data Quality", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div([
                html.Div("Quality checks for modeling integrity:", style={"fontWeight":"700","color":USC["blue"]}),
                html.Ul([
                    html.Li("No nulls in critical fields (Store, Dept, Date, CPI, Unemployment, Fuel_Price)."),
                    html.Li("MarkDown* missing → 0 (means no promo that week)."),
                    html.Li("CPI/Unemployment ffill+bfill by Store to keep macro continuity."),
                    html.Li("Duplicates on (Store, Dept, Date) removed/flagged."),
                ], style={"lineHeight":"1.7"})
            ], style={"padding":"16px"}),

            html.Div([
                dcc.Graph(
                    id="dq_sales_by_year",
                    figure=(
                        none_fig("No cleaned file found") if cleaned_df is None else
                        px.bar(
                            cleaned_df.assign(year=cleaned_df["Date"].dt.year)
                                      .groupby("year", as_index=False)["Weekly_Sales"].sum(),
                            x="year", y="Weekly_Sales",
                            title="Coverage Check: Total Weekly Sales by Year",
                            color_discrete_sequence=px.colors.qualitative.Set2
                        ).update_layout(paper_bgcolor="white", plot_bgcolor="white", height=420)
                    )
                )
            ], style={"padding":"0 16px 16px"})
        ]),

        # Forecast Comparison
        dcc.Tab(label="Forecast Comparison", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div([
                html.Div([
                    html.Label("Select forecast models", style={"fontWeight":"600"}),
                    dcc.Dropdown(
                        id="cmp_models",
                        options=[{"label": m, "value": m} for m in ["naive","seasonal_naive","prophet","sarimax","xgb"]],
                        value=["naive","prophet","sarimax"],
                        multi=True, clearable=False
                    )
                ], style={"maxWidth":"420px"}),

                dcc.Graph(id="cmp_graph", style={"marginTop":"8px"}),

                html.Div([
                    html.Div([
                        dcc.Graph(
                            id="rmse_bar",
                            figure=(
                                none_fig("No metrics file found") if metrics_df is None else
                                px.bar(metrics_df, x="Model", y="RMSE_test", color="Model",
                                       title="Model RMSE (Lower is Better)",
                                       color_discrete_sequence=px.colors.qualitative.Set2
                                ).update_layout(height=360, paper_bgcolor="white", plot_bgcolor="white")
                            )
                        )
                    ], style={"flex":1,"paddingRight":"8px"}),

                    html.Div([
                        dcc.Graph(
                            id="mape_bar",
                            figure=(
                                none_fig("No metrics file or MAPE column") if metrics_df is None or "MAPE test (%)" not in metrics_df.columns else
                                px.bar(metrics_df, x="Model", y="MAPE test (%)", color="Model",
                                       title="Model MAPE (Lower is Better)",
                                       color_discrete_sequence=px.colors.qualitative.Set2
                                ).update_layout(height=360, paper_bgcolor="white", plot_bgcolor="white")
                            )
                        )
                    ], style={"flex":1,"paddingLeft":"8px"}),
                ], style={"display":"flex","gap":"12px"})
            ], style={"padding":"16px"})
        ]),

        # Uplift
        dcc.Tab(label="Holiday & Markdown Uplift", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div([
                html.Div([
                    dcc.Graph(
                        id="holiday_uplift",
                        figure=(
                            none_fig("No holiday_uplift_summary.csv") if holiday_df is None else
                            px.bar(holiday_df, x="Holiday", y="Avg_Uplift", color="Holiday",
                                   title="Average Sales Uplift by Holiday",
                                   color_discrete_sequence=px.colors.qualitative.Vivid
                            ).update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
                        )
                    )
                ], style={"flex":1,"paddingRight":"8px"}),

                html.Div([
                    dcc.Graph(
                        id="markdown_uplift",
                        figure=(
                            none_fig("No markdown_uplift_summary.csv") if markdown_df is None else
                            px.bar(markdown_df, x="Markdown", y="Avg_Uplift", color="Markdown",
                                   title="Average Sales Uplift by Markdown",
                                   color_discrete_sequence=px.colors.qualitative.Alphabet
                            ).update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
                        )
                    )
                ], style={"flex":1,"paddingLeft":"8px"}),
            ], style={"display":"flex","gap":"12px","padding":"16px"}),

            html.Div(id="uplift_narrative", style={"padding":"0 16px 16px 16px","color":USC["blue"]})
        ]),

        # Inventory Simulation
        dcc.Tab(label="Inventory Simulation", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div([
                html.Div([
                    dcc.Graph(
                        id="inv_cost",
                        figure=(
                            none_fig("No inventory_simulation_results.csv") if inv_df is None else
                            px.line(inv_df, x="Service_Level", y="Avg_Cost", color="Model",
                                    title="Cost vs Service Level (Safety Factor z)"
                            ).update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white")
                        )
                    )
                ], style={"flex":1,"paddingRight":"8px"}),

                html.Div([
                    dcc.Graph(
                        id="inv_stockouts",
                        figure=(
                            none_fig("No inventory_simulation_results.csv") if inv_df is None else
                            px.line(inv_df, x="Service_Level", y="Avg_Stockouts", color="Model",
                                    title="Stockouts vs Service Level (Safety Factor z)"
                            ).update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white")
                        )
                    )
                ], style={"flex":1,"paddingLeft":"8px"}),
            ], style={"display":"flex","gap":"12px","padding":"16px"}),

            html.Div(id="inv_narrative", style={"padding":"0 16px 16px 16px","color":USC["blue"]})
        ]),

        # Future Forecasts
        dcc.Tab(label="Future Forecasts (8W)", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div([ dcc.Graph(id="future_graph") ], style={"padding":"16px"})
        ]),

        # Executive Report
        dcc.Tab(label="Executive Report", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div([
                html.Div([
                    kpi_card("Best model (RMSE)", INS.get("best_model","—"),
                             sub=f"RMSE {fmt_money(INS.get('best_rmse'))}" if fmt_money(INS.get("best_rmse")) else None),
                    kpi_card(
                        "MAPE (best model)",
                        fmt_pct(INS.get("best_mape")) or "—",
                        sub="Lower is better"
                    ),
                    kpi_card("Top holiday uplift",
                             fmt_money(INS.get("top_holiday",{}).get("Avg_Uplift")) if INS.get("top_holiday") else "—",
                             sub=INS.get("top_holiday",{}).get("Holiday","")),
                    kpi_card("95%: lowest cost model",
                             INS.get("inv95_best_cost",{}).get("Model","—"),
                             sub=(fmt_money(INS.get("inv95_best_cost",{}).get("Avg_Cost")) + "/wk")
                                 if INS.get("inv95_best_cost") else None),
                ], style={"display":"grid","gridTemplateColumns":"repeat(4, minmax(220px,1fr))","gap":"12px","marginBottom":"12px"}),

                html.H2("Executive Summary", style={"color": USC["cardinal"], "marginTop": "4px"}),
                html.Div(id="report_html", style={"lineHeight":"1.6","color":USC["blue"],"maxWidth":"1100px"}),

                html.Hr(),
                html.H3("Key Visuals", style={"color": USC["cardinal"]}),
                html.Div([
                    dcc.Graph(id="report_compact_cmp"),
                    dcc.Graph(id="report_compact_inv"),
                ])
            ], style={"padding":"16px"})
        ]),
    ])
])

# =========================
# Callbacks
# =========================
@app.callback(Output("cmp_graph","figure"), Input("cmp_models","value"))
def update_comparison(models_selected):
    if fc_df is None or "date" not in fc_df.columns or "y_true" not in fc_df.columns:
        return none_fig("No all_models_forecast_comparison.csv")
    models_selected = [m for m in (models_selected or []) if m in fc_df.columns]
    ycols = ["y_true"] + models_selected
    fig = px.line(fc_df, x="date", y=ycols, title="Actual vs Forecasts (Last 8 Weeks)",
                  color_discrete_map={"y_true":"black"})
    for tr in fig.data:
        if tr.name == "y_true":
            tr.name = "Actual"; tr.line.update(width=3, dash="solid", color="black")
        else:
            tr.line.update(dash="dash", width=2)
    fig.update_layout(
        height=520, paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        xaxis_title="Date", yaxis_title="Weekly Sales"
    )
    return fig

@app.callback(Output("uplift_narrative","children"), Input("holiday_uplift","figure"))
def render_uplift_text(_):
    if holiday_df is None or markdown_df is None:
        return html.Div("Upload uplift CSVs to see narrative.", style={"opacity":0.7})
    parts = []
    if INS.get("top_holiday"):
        h = INS["top_holiday"]["Holiday"]; v = INS["top_holiday"]["Avg_Uplift"]
        parts.append(f"• **Top holiday uplift:** {h} with an average lift of **{fmt_money(v)}** per week.")
    if INS.get("low_holiday"):
        h = INS["low_holiday"]["Holiday"]; v = INS["low_holiday"]["Avg_Uplift"]
        parts.append(f"• **Lowest holiday uplift:** {h} averaging **{fmt_money(v)}**.")
    parts.append("• Markdown types show material differences—prioritize those with positive uplift and re-time those with negative or neutral impact.")
    return dcc.Markdown("\n\n".join(parts))

@app.callback(Output("inv_narrative","children"), Input("inv_cost","figure"))
def render_inventory_text(_):
    if inv_df is None:
        return html.Div("Upload inventory_simulation_results.csv to see narrative.", style={"opacity":0.7})
    try:
        inv_df["Service_Level"] = pd.to_numeric(inv_df["Service_Level"], errors="coerce")
    except Exception:
        pass
    bullets = []
    snap95 = inv_df[np.isclose(inv_df["Service_Level"], 1.64)]
    if not snap95.empty:
        best_cost = snap95.loc[snap95["Avg_Cost"].idxmin()]
        best_stock = snap95.loc[snap95["Avg_Stockouts"].idxmin()]
        bullets.append(f"• At **95% service** (z=1.64), **{best_cost['Model']}** minimizes weekly cost at **{fmt_money(best_cost['Avg_Cost'])}**.")
        bullets.append(f"• **Fewest stockouts** at the same level: **{best_stock['Model']}** (~{best_stock['Avg_Stockouts']:,.0f}/week).")
    bullets.append("• Choose z to balance holding cost vs. service. Recalibrate quarterly with realized metrics.")
    return dcc.Markdown("\n\n".join(bullets))

@app.callback(Output("future_graph","figure"), Input("cmp_models","value"))
def update_future(_):
    if future_df is None or not {"Date","Prophet","SARIMA","XGBoost"}.issubset(set(future_df.columns)):
        return none_fig("No future_models_forecast_8w.csv found")
    f = future_df.copy().sort_values("Date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Prophet"], name="Prophet",  mode="lines",         line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["SARIMA"],  name="SARIMAX",  mode="lines",         line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["XGBoost"], name="XGBoost",  mode="lines+markers", line=dict(width=2, dash="dash"), marker=dict(size=6)))
    if fc_df is not None and "date" in fc_df.columns:
        forecast_start = pd.to_datetime(fc_df["date"].max())
    else:
        forecast_start = pd.to_datetime(f["Date"].min()) - pd.Timedelta(days=1)
    forecast_end = pd.to_datetime(f["Date"].max())
    fig.add_vline(x=forecast_start, line_width=2, line_dash="dot", line_color="black")
    fig.add_vrect(x0=forecast_start, x1=forecast_end, fillcolor=USC["gray"], opacity=0.3, layer="below", line_width=0)
    fig.update_layout(
        title="Future Model Forecasts (Next 8 Weeks)",
        height=500, paper_bgcolor="white", plot_bgcolor="white",
        xaxis_title="Date", yaxis_title="Weekly Sales",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig

@app.callback(
    Output("report_html","children"),
    Output("report_compact_cmp","figure"),
    Output("report_compact_inv","figure"),
    Input("cmp_models","value")
)
def render_report(_):
    bullets = []
    if INS.get("best_model") is not None:
        rmse_txt = fmt_money(INS.get("best_rmse"))
        mape_txt = fmt_pct(INS.get("best_mape")) or "N/A"
        bullets.append(f"**Forecast accuracy.** **{INS['best_model']}** achieved lowest test RMSE (**{rmse_txt}**) and MAPE (**{mape_txt}**).")
    else:
        bullets.append("**Forecast accuracy.** Baseline + advanced models compared on last 8 test weeks.")
    if INS.get("top_holiday"):
        t = INS["top_holiday"]
        bullets.append(f"**Promotions/holidays.** Largest uplift: **{t['Holiday']}** (~{fmt_money(t['Avg_Uplift'])}/week).")
    if INS.get("inv95_best_cost") and INS.get("inv95_best_stockout"):
        c = INS["inv95_best_cost"]; s = INS["inv95_best_stockout"]
        bullets.append(f"**Inventory trade-offs (z=1.64).** **{c['Model']}** minimizes cost ("
                       f"{fmt_money(c['Avg_Cost'])}/wk) while **{s['Model']}** has fewest stockouts (~{s['Avg_Stockouts']:,.0f}/wk).")
    bullets.append("**Recommendation.** Use best model in production, add holiday/markdown adjustments, start at ~95% service, monitor & retune quarterly.")
    report_md = dcc.Markdown("\n\n".join([f"- {b}" for b in bullets]))

    if fc_df is None:
        cmp_fig = none_fig("No comparison data")
    else:
        small = ["y_true","seasonal_naive","prophet","sarimax","xgb"]
        cols = [c for c in small if c in fc_df.columns]
        cmp_fig = px.line(fc_df, x="date", y=cols, title="Actual vs Models (Compact View)",
                          color_discrete_map={"y_true":"black"})
        for tr in cmp_fig.data:
            if tr.name == "y_true":
                tr.name = "Actual"; tr.line.update(width=3, color="black")
            else:
                tr.line.update(dash="dash", width=2)
        cmp_fig.update_layout(height=360, paper_bgcolor="white", plot_bgcolor="white",
                              legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))

    if inv_df is None:
        inv_fig = none_fig("No inventory data")
    else:
        inv_fig = px.line(inv_df, x="Service_Level", y="Avg_Cost", color="Model",
                          title="Cost vs Service Level (Compact)").update_layout(
                              height=360, paper_bgcolor="white", plot_bgcolor="white",
                              legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    return report_md, cmp_fig, inv_fig

# =========================
# Run (local dev only)
# =========================
if __name__ == "__main__":
    app.run(debug=True)
