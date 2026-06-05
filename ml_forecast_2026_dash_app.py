from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html


ROOT = Path(r"C:\Users\user\Documents\Codex\2026-06-05\files-mentioned-by-the-user-ml")
OUTPUT_DIR = ROOT / "outputs"
DATA_PATH = OUTPUT_DIR / "ml_forecast_2026_ridge.xlsx"


def money_short(value: float) -> str:
    if value is None or pd.isna(value):
        return "-"
    if abs(value) >= 100_000_000:
        return f"{value / 100_000_000:,.1f}억"
    if abs(value) >= 10_000:
        return f"{value / 10_000:,.1f}만"
    return f"{value:,.0f}"


def pct(value: float) -> str:
    return "-" if value is None or pd.isna(value) else f"{value:.1%}"


def load_data() -> dict[str, pd.DataFrame]:
    sheet_names = [
        "요약",
        "전체예측_2026남은달",
        "모델검증",
        "품목별예측요약",
        "품목별월별예측",
        "병원별예측요약",
        "병원별월별예측",
        "서울대예측요약",
        "거래처예측요약",
        "메토젝트예측요약",
        "팀별목표보조예측",
        "검산",
    ]
    return {name: pd.read_excel(DATA_PATH, sheet_name=name) for name in sheet_names}


def style_table(
    df: pd.DataFrame,
    table_id: str,
    page_size: int = 10,
    selectable: bool = False,
) -> dash_table.DataTable:
    kwargs = {}
    if selectable:
        kwargs["row_selectable"] = "single"
        kwargs["selected_rows"] = []
    return dash_table.DataTable(
        id=table_id,
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "Arial, sans-serif",
            "fontSize": 13,
            "padding": "8px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={"backgroundColor": "#17324d", "color": "white", "fontWeight": "700"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fb"}],
        **kwargs,
    )


def kpi_card(title: str, value: str, note: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value"),
            html.Div(note, className="kpi-note"),
        ],
        className="kpi-card",
    )


def download_button(label: str, button_id: str) -> html.Button:
    return html.Button(label, id=button_id, type="button", className="download-button")


data = load_data()
summary = data["요약"].iloc[0]
overall_forecast = data["전체예측_2026남은달"].copy()
model_check = data["모델검증"].copy()
item_scores = data["품목별예측요약"].copy()
item_monthly = data["품목별월별예측"].copy()
hospital_scores = data["병원별예측요약"].copy()
team_projection = data["팀별목표보조예측"].copy()
validation = data["검산"].copy()

overall_forecast["month"] = pd.to_datetime(overall_forecast["month"])
item_monthly["month"] = pd.to_datetime(item_monthly["month"])

forecast_min_month = overall_forecast["month"].min()
forecast_max_month = overall_forecast["month"].max()
month_options = [
    {"label": m.strftime("%Y-%m"), "value": m.strftime("%Y-%m")}
    for m in pd.date_range(forecast_min_month, forecast_max_month, freq="MS")
]
year_options = [{"label": str(y), "value": str(y)} for y in sorted(overall_forecast["month"].dt.year.unique())]

team_table_base = team_projection[
    [
        "부서",
        "actual_2026_1_4",
        "target_2026_1_4",
        "jan_apr_ar",
        "target_2026_5_12",
        "hybrid_share",
        "ridge_scaled_projection",
    ]
].rename(
    columns={
        "actual_2026_1_4": "2026 1~4월 실제",
        "target_2026_1_4": "2026 1~4월 목표",
        "jan_apr_ar": "1~4월 달성률",
        "target_2026_5_12": "5~12월 목표",
        "hybrid_share": "배분 비중",
        "ridge_scaled_projection": "5~12월 예측",
    }
)

item_table_base = item_scores[
    ["group", "recent_2026_1_4_actual", "forecast_2026_5_12", "model", "mape", "r2"]
].rename(
    columns={
        "group": "품목",
        "recent_2026_1_4_actual": "2026 1~4월 실제",
        "forecast_2026_5_12": "5~12월 예측",
        "model": "모델",
        "mape": "MAPE",
        "r2": "R2",
    }
)

item_options = [{"label": x, "value": x} for x in sorted(item_table_base["품목"].dropna().astype(str).unique())]
team_options = [{"label": x, "value": x} for x in sorted(team_table_base["부서"].dropna().astype(str).unique())]


def month_filter(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    out = df.copy()
    if start_date:
        out = out[out["month"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["month"] <= pd.Timestamp(end_date)]
    return out


def month_year_filter(df: pd.DataFrame, selected_months: list[str] | None, selected_years: list[str] | None) -> pd.DataFrame:
    out = df.copy()
    if selected_years:
        years = {int(y) for y in selected_years}
        out = out[out["month"].dt.year.isin(years)]
    if selected_months:
        months = {pd.Timestamp(m + "-01") for m in selected_months}
        out = out[out["month"].isin(months)]
    return out


def get_filtered_items(selected_months: list[str] | None, selected_years: list[str] | None, selected_items: list[str] | None) -> pd.DataFrame:
    detail = month_year_filter(item_monthly, selected_months, selected_years)
    if selected_items:
        detail = detail[detail["group"].isin(selected_items)]
    period_sum = detail.groupby("group", as_index=False)["forecast_sales"].sum()
    out = item_table_base.copy()
    if selected_items:
        out = out[out["품목"].isin(selected_items)]
    out = out.merge(period_sum, left_on="품목", right_on="group", how="left")
    out["5~12월 예측"] = out["forecast_sales"].fillna(0)
    out = out.drop(columns=[c for c in ["group", "forecast_sales"] if c in out.columns])
    return out.sort_values("5~12월 예측", ascending=False)


def get_filtered_teams(selected_teams: list[str] | None) -> pd.DataFrame:
    out = team_table_base.copy()
    if selected_teams:
        out = out[out["부서"].isin(selected_teams)]
    return out.sort_values("5~12월 예측", ascending=False)


def make_forecast_fig(filtered_forecast: pd.DataFrame):
    fig = px.line(
        filtered_forecast,
        x="month",
        y="forecast_sales",
        markers=True,
        title="선택 기간 전체 Ridge 예측",
    )
    fig.update_traces(line={"color": "#256f8f", "width": 3})
    fig.update_layout(xaxis_title="월", yaxis_title="예측 실적", margin={"l": 48, "r": 24, "t": 54, "b": 40})
    return fig


def make_item_fig(df: pd.DataFrame):
    view = df.sort_values("5~12월 예측", ascending=False).head(15)
    fig = px.bar(
        view.sort_values("5~12월 예측"),
        x="5~12월 예측",
        y="품목",
        orientation="h",
        text=view.sort_values("5~12월 예측")["5~12월 예측"].map(money_short),
        color="MAPE",
        color_continuous_scale="Blues",
        title="선택 품목/기간 예측 상위",
    )
    fig.update_layout(xaxis_title="예측 실적", yaxis_title="품목", coloraxis_colorbar_title="MAPE")
    return fig


def make_team_fig(df: pd.DataFrame):
    view = df.sort_values("5~12월 예측", ascending=False).head(15)
    fig = px.bar(
        view.sort_values("5~12월 예측"),
        x="5~12월 예측",
        y="부서",
        orientation="h",
        text=view.sort_values("5~12월 예측")["5~12월 예측"].map(money_short),
        color="1~4월 달성률",
        color_continuous_scale="RdYlGn",
        title="선택 팀 예측",
    )
    fig.update_layout(xaxis_title="Ridge 총액 배분 예측", yaxis_title="팀", coloraxis_colorbar_title="1~4월 달성률")
    return fig


def make_hospital_fig():
    view = hospital_scores.sort_values("forecast_2026_5_12", ascending=False).head(15)
    fig = px.bar(
        view.sort_values("forecast_2026_5_12"),
        x="forecast_2026_5_12",
        y="group",
        orientation="h",
        text=view.sort_values("forecast_2026_5_12")["forecast_2026_5_12"].map(money_short),
        color="mape",
        color_continuous_scale="Purples",
        title="병원별 2026년 5~12월 예측 상위 15",
    )
    fig.update_layout(xaxis_title="예측 실적", yaxis_title="병원", coloraxis_colorbar_title="MAPE")
    return fig


valid_pass = bool(validation["pass"].fillna(True).all()) if "pass" in validation.columns else False
model_display = pd.DataFrame(
    [
        {
            "모델": summary["best_model"],
            "MAE": f"{summary['mae']:,.0f}",
            "MAPE": pct(summary["mape"]),
            "R2": f"{summary['r2']:.3f}",
            "학습 행수": int(model_check.iloc[0]["train_rows_after_refit"]),
        }
    ]
)
feature_text = str(model_check.iloc[0]["feature_cols"]).replace(", ", " · ")

initial_items = get_filtered_items(None, None, None)
initial_teams = get_filtered_teams(None)

app = Dash(__name__)
app.title = "2026 Ridge 예측 대시보드"
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      body { margin: 0; background: #eef3f6; color: #18212b; }
      .page { max-width: 1360px; margin: 0 auto; padding: 28px; font-family: Arial, sans-serif; }
      .header { margin-bottom: 22px; }
      .eyebrow { color: #496579; font-size: 13px; font-weight: 700; letter-spacing: 0; }
      h1 { margin: 6px 0 8px; font-size: 30px; line-height: 1.2; }
      .sub { color: #526272; font-size: 14px; }
      .kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 18px 0; }
      .kpi-card { background: white; border: 1px solid #d8e0e6; border-radius: 8px; padding: 16px; }
      .kpi-title { color: #607181; font-size: 13px; font-weight: 700; }
      .kpi-value { font-size: 25px; font-weight: 800; margin-top: 8px; color: #17324d; }
      .kpi-note { min-height: 18px; margin-top: 6px; color: #6b7785; font-size: 12px; }
      .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      .section { background: white; border: 1px solid #d8e0e6; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
      .section h2 { margin: 0 0 12px; font-size: 18px; }
      .note { color: #53606d; font-size: 13px; line-height: 1.5; }
      .filters { display: grid; grid-template-columns: 1.1fr 1fr 1fr; gap: 14px; align-items: end; }
      .filters.one-col { grid-template-columns: 1fr; margin-bottom: 10px; }
      .filter-label { color: #607181; font-size: 13px; font-weight: 700; margin-bottom: 6px; }
      .section-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
      .section-head h2 { margin: 0; }
      .download-button { border: 1px solid #1f5d7a; background: #256f8f; color: #fff; border-radius: 6px; padding: 8px 12px; font-size: 13px; font-weight: 700; cursor: pointer; }
      .download-button:hover { background: #1d5d78; }
      .drilldown { border: 1px solid #d8e0e6; background: #f8fbfd; border-radius: 8px; padding: 12px; margin-top: 12px; }
      .mini-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }
      .mini-card { background: white; border: 1px solid #e1e7ec; border-radius: 6px; padding: 10px; }
      .mini-label { color: #647585; font-size: 12px; font-weight: 700; }
      .mini-value { color: #17324d; font-size: 18px; font-weight: 800; margin-top: 5px; }
      @media (max-width: 950px) {
        .kpi-grid, .grid-2, .filters, .mini-grid { grid-template-columns: 1fr; }
        .page { padding: 16px; }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div("2026 Ridge Forecast", className="eyebrow"),
                html.H1("2026년 남은 달 ML 예측 대시보드"),
                html.Div("기간, 품목, 팀을 선택하면 차트와 표가 함께 필터링됩니다.", className="sub"),
            ],
            className="header",
        ),
        html.Div(
            [
                kpi_card("학습 기간", f"{summary['actual_min_month']} ~ {summary['actual_max_month']}", "제공 데이터 기준"),
                kpi_card("예측 기간", f"{summary['forecast_start']} ~ {summary['forecast_end']}", "2026년 남은 달"),
                kpi_card("1~4월 실제", money_short(summary["actual_2026_1_4"]), "2026년 실제 누적"),
                kpi_card("5~12월 예측", money_short(summary["forecast_2026_5_12"]), f"MAPE {pct(summary['mape'])}"),
            ],
            className="kpi-grid",
        ),
        html.Div(
            [
                html.Div([dcc.Graph(id="forecast-graph", figure=make_forecast_fig(overall_forecast))], className="section"),
                html.Div(
                    [
                        html.H2("모델 검증"),
                        html.Div(
                            f"선택 모델은 {summary['best_model']}입니다. MAPE는 {summary['mape']:.2%}, "
                            f"MAE는 {summary['mae']:,.0f}, R2는 {summary['r2']:.3f}입니다. "
                            f"검산 결과는 {'통과' if valid_pass else '확인 필요'}입니다.",
                            className="note",
                        ),
                        style_table(model_display, "model-table", page_size=4),
                        html.Div(f"사용 변수: {feature_text}", className="note", style={"marginTop": "10px"}),
                    ],
                    className="section",
                ),
            ],
            className="grid-2",
        ),
        html.Div(
            [
                html.Div([dcc.Graph(id="team-graph", figure=make_team_fig(initial_teams))], className="section"),
                html.Div([dcc.Graph(id="item-graph", figure=make_item_fig(initial_items))], className="section"),
            ],
            className="grid-2",
        ),
        html.Div([dcc.Graph(figure=make_hospital_fig())], className="section"),
        html.Div(
            [
                html.Div([html.H2("팀별 보조 예측 표"), download_button("CSV 다운로드", "download-team-button")], className="section-head"),
                dcc.Download(id="download-team"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("팀 필터", className="filter-label"),
                                dcc.Dropdown(id="team-filter", options=team_options, multi=True, placeholder="전체 팀"),
                            ]
                        ),
                    ],
                    className="filters one-col",
                ),
                html.Div("팀을 선택하면 해당 팀만 표시됩니다. 행을 클릭하면 실제/목표/예측 배분 구조가 아래에 표시됩니다.", className="note"),
                style_table(initial_teams, "team-table", page_size=10, selectable=True),
                html.Div(id="team-drilldown", className="drilldown"),
            ],
            className="section",
        ),
        html.Div(
            [
                html.Div([html.H2("품목별 예측 요약"), download_button("CSV 다운로드", "download-item-button")], className="section-head"),
                dcc.Download(id="download-item"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("연도 필터", className="filter-label"),
                                dcc.Dropdown(id="year-filter", options=year_options, multi=True, placeholder="전체 연도"),
                            ]
                        ),
                        html.Div(
                            [
                                html.Div("월 필터", className="filter-label"),
                                dcc.Dropdown(id="month-filter", options=month_options, multi=True, placeholder="전체 월"),
                            ]
                        ),
                        html.Div(
                            [
                                html.Div("품목 필터", className="filter-label"),
                                dcc.Dropdown(id="item-filter", options=item_options, multi=True, placeholder="전체 품목"),
                            ]
                        ),
                    ],
                    className="filters",
                ),
                html.Div("연도/월/품목을 선택하면 품목 그래프와 표가 함께 바뀝니다. 행을 클릭하면 선택 기간 내 월별 예측 추이가 아래에 표시됩니다.", className="note"),
                style_table(initial_items, "item-table", page_size=10, selectable=True),
                html.Div(id="item-drilldown", className="drilldown"),
            ],
            className="section",
        ),
        html.Div(
            [
                html.Div([html.H2("검산"), download_button("CSV 다운로드", "download-validation-button")], className="section-head"),
                dcc.Download(id="download-validation"),
                html.Div("검산 표도 표 머리글 아래 필터칸에 검색어를 넣어 필요한 항목만 볼 수 있습니다.", className="note"),
                style_table(validation, "validation-table", page_size=10, selectable=True),
                html.Div(id="validation-drilldown", className="drilldown"),
            ],
            className="section",
        ),
    ],
    className="page",
)


@app.callback(
    Output("forecast-graph", "figure"),
    Output("item-graph", "figure"),
    Output("team-graph", "figure"),
    Output("item-table", "data"),
    Output("team-table", "data"),
    Input("month-filter", "value"),
    Input("year-filter", "value"),
    Input("item-filter", "value"),
    Input("team-filter", "value"),
)
def update_filters(selected_months, selected_years, selected_items, selected_teams):
    filtered_forecast = month_year_filter(overall_forecast, selected_months, selected_years)
    filtered_items = get_filtered_items(selected_months, selected_years, selected_items)
    filtered_teams = get_filtered_teams(selected_teams)
    return (
        make_forecast_fig(filtered_forecast),
        make_item_fig(filtered_items),
        make_team_fig(filtered_teams),
        filtered_items.to_dict("records"),
        filtered_teams.to_dict("records"),
    )


@app.callback(
    Output("download-team", "data"),
    Input("download-team-button", "n_clicks"),
    State("team-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def download_team(_, rows):
    df = pd.DataFrame(rows or team_table_base.to_dict("records"))
    return dcc.send_data_frame(df.to_csv, "team_projection_filtered.csv", index=False, encoding="utf-8-sig")


@app.callback(
    Output("download-item", "data"),
    Input("download-item-button", "n_clicks"),
    State("item-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def download_item(_, rows):
    df = pd.DataFrame(rows or item_table_base.to_dict("records"))
    return dcc.send_data_frame(df.to_csv, "item_forecast_filtered.csv", index=False, encoding="utf-8-sig")


@app.callback(
    Output("download-validation", "data"),
    Input("download-validation-button", "n_clicks"),
    State("validation-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def download_validation(_, rows):
    df = pd.DataFrame(rows or validation.to_dict("records"))
    return dcc.send_data_frame(df.to_csv, "validation_filtered.csv", index=False, encoding="utf-8-sig")


@app.callback(
    Output("team-drilldown", "children"),
    Input("team-table", "selected_rows"),
    State("team-table", "derived_virtual_data"),
)
def update_team_drilldown(selected_rows, rows):
    if not selected_rows or not rows:
        return html.Div("팀 행을 클릭하면 세부 비교가 여기에 표시됩니다.", className="note")
    row = rows[selected_rows[0]]
    chart_df = pd.DataFrame(
        [
            {"구분": "2026 1~4월 실제", "값": row["2026 1~4월 실제"]},
            {"구분": "2026 1~4월 목표", "값": row["2026 1~4월 목표"]},
            {"구분": "2026 5~12월 예측", "값": row["5~12월 예측"]},
        ]
    )
    fig = px.bar(chart_df, x="구분", y="값", text=chart_df["값"].map(money_short), title=f"{row['부서']} 세부 비교")
    fig.update_traces(marker_color=["#256f8f", "#8aa6b8", "#d77a2d"])
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Div("1~4월 달성률", className="mini-label"), html.Div(pct(row["1~4월 달성률"]), className="mini-value")], className="mini-card"),
                    html.Div([html.Div("배분 비중", className="mini-label"), html.Div(pct(row["배분 비중"]), className="mini-value")], className="mini-card"),
                    html.Div([html.Div("5~12월 예측", className="mini-label"), html.Div(money_short(row["5~12월 예측"]), className="mini-value")], className="mini-card"),
                ],
                className="mini-grid",
            ),
            dcc.Graph(figure=fig),
        ]
    )


@app.callback(
    Output("item-drilldown", "children"),
    Input("item-table", "selected_rows"),
    State("item-table", "derived_virtual_data"),
    State("month-filter", "value"),
    State("year-filter", "value"),
)
def update_item_drilldown(selected_rows, rows, selected_months, selected_years):
    if not selected_rows or not rows:
        return html.Div("품목 행을 클릭하면 월별 예측 추이가 여기에 표시됩니다.", className="note")
    row = rows[selected_rows[0]]
    detail = item_monthly[item_monthly["group"].eq(row["품목"])].copy()
    detail = month_year_filter(detail, selected_months, selected_years)
    if detail.empty:
        return html.Div(f"{row['품목']}의 선택 기간 월별 예측 데이터가 없습니다.", className="note")
    fig = px.line(detail, x="month", y="forecast_sales", markers=True, title=f"{row['품목']} 월별 예측")
    fig.update_traces(line={"color": "#d77a2d", "width": 3})
    fig.update_layout(xaxis_title="월", yaxis_title="예측 실적")
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Div("2026 1~4월 실제", className="mini-label"), html.Div(money_short(row["2026 1~4월 실제"]), className="mini-value")], className="mini-card"),
                    html.Div([html.Div("선택 기간 예측", className="mini-label"), html.Div(money_short(row["5~12월 예측"]), className="mini-value")], className="mini-card"),
                    html.Div([html.Div("MAPE", className="mini-label"), html.Div(pct(row["MAPE"]), className="mini-value")], className="mini-card"),
                ],
                className="mini-grid",
            ),
            dcc.Graph(figure=fig),
        ]
    )


@app.callback(
    Output("validation-drilldown", "children"),
    Input("validation-table", "selected_rows"),
    State("validation-table", "derived_virtual_data"),
)
def update_validation_drilldown(selected_rows, rows):
    if not selected_rows or not rows:
        return html.Div("검산 행을 클릭하면 원자료 합계와 변환 합계 차이를 확인할 수 있습니다.", className="note")
    row = rows[selected_rows[0]]
    status = "통과" if bool(row.get("pass")) else "확인 필요"
    return html.Div(
        [
            html.Div(f"검산 항목: {row.get('check')}", className="note"),
            html.Div(
                [
                    html.Div([html.Div("원자료 합계", className="mini-label"), html.Div(money_short(row.get("source_sum")), className="mini-value")], className="mini-card"),
                    html.Div([html.Div("변환 합계/행 수", className="mini-label"), html.Div(f"{row.get('converted_sum'):,.2f}" if pd.notna(row.get("converted_sum")) else "-", className="mini-value")], className="mini-card"),
                    html.Div([html.Div("차이", className="mini-label"), html.Div(f"{row.get('difference'):,.6f}" if pd.notna(row.get("difference")) else "-", className="mini-value")], className="mini-card"),
                ],
                className="mini-grid",
            ),
            html.Div(f"결과: {status}", className="note", style={"marginTop": "10px"}),
        ]
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8051, debug=False)
