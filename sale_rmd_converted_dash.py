from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html


BASE_DIR = Path(r"C:\Users\user\Documents\Codex\2026-06-05\files-mentioned-by-the-user-ml\outputs\performance_aggregate_python")
TABLE_DIR = BASE_DIR / "tables"
MAX_ROWS = 1000


def load_table(name: str) -> pd.DataFrame:
    path = TABLE_DIR / f"{name}.csv.gz"
    return pd.read_csv(path, compression="gzip")


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


def options_from(df: pd.DataFrame, col: str) -> list[dict[str, str]]:
    if col not in df.columns:
        return []
    vals = sorted(df[col].dropna().astype(str).unique())
    return [{"label": v, "value": v} for v in vals]


def filter_in(df: pd.DataFrame, col: str, values: list[str] | None) -> pd.DataFrame:
    if not values or col not in df.columns:
        return df
    return df[df[col].astype(str).isin(values)]


def truncate(df: pd.DataFrame, n: int = MAX_ROWS) -> pd.DataFrame:
    return df.head(n).copy()


def is_percent_col(col: str, series: pd.Series) -> bool:
    name = str(col).lower()
    percent_words = ["ar", "rate", "ratio", "pct", "percent", "mape", "growth", "_gr", "달성", "성장", "증감", "점유", "비율", "률"]
    if name != "gr" and not any(word in name for word in percent_words):
        return False
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return numeric.empty or numeric.abs().max() <= 5


def is_plain_integer_col(col: str) -> bool:
    name = str(col).lower()
    return any(word in name for word in ["year", "month", "rank", "idx", "code", "id", "년", "월", "순위"])


def format_number(value, col: str, series: pd.Series):
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value

    if is_percent_col(col, series):
        return f"{number * 100:,.2f}%"
    if is_plain_integer_col(col):
        return f"{number:.0f}"
    if float(number).is_integer() or abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:,.2f}"


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    view = truncate(df)
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value, c=col, s=view[col]: format_number(value, c, s))
    return view


def table_records(df: pd.DataFrame) -> list[dict]:
    return display_table(df).to_dict("records")


def data_table(df: pd.DataFrame, table_id: str, page_size: int = 12) -> dash_table.DataTable:
    view = display_table(df)
    return dash_table.DataTable(
        id=table_id,
        data=view.to_dict("records"),
        columns=[{"name": str(c), "id": str(c)} for c in view.columns],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        export_format="csv",
        export_headers="display",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "Arial, sans-serif",
            "fontSize": 13,
            "padding": "8px",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "80px",
            "maxWidth": "220px",
        },
        style_header={"backgroundColor": "#005FAF", "color": "white", "fontWeight": "700"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fb"}],
    )


def table_section(
    title: str,
    table_id: str,
    df: pd.DataFrame,
    download_id: str,
    note: str = "",
    filters: list | None = None,
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.H3(title), html.Div(note, className="note") if note else None]),
                    html.Button("CSV 다운로드", id=download_id, type="button", className="download-button"),
                ],
                className="section-head",
            ),
            html.Div(filters, className="inline-filters") if filters else None,
            data_table(df, table_id),
        ],
        className="section",
    )


def filter_block(children: list) -> html.Div:
    return html.Div(children, className="section inline-filters")


def dropdown(label: str, component_id: str, options: list[dict[str, str]], multi: bool = True, wide: bool = False) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="filter-label"),
            dcc.Dropdown(
                id=component_id,
                options=options,
                multi=multi,
                searchable=True,
                clearable=True,
                closeOnSelect=False if multi else True,
                placeholder=f"전체 {label}",
                className="filter-dropdown",
            ),
            html.Div(
                [
                    html.Button("전체 선택", id=f"{component_id}-select-all", type="button", className="filter-action"),
                    html.Button("선택 해제", id=f"{component_id}-clear", type="button", className="filter-action"),
                ],
                className="filter-actions",
            )
            if multi
            else None,
        ],
        className="filter-field filter-field-wide" if wide else "filter-field",
    )


def kpi(title: str, value: str, note: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value"),
            html.Div(note, className="kpi-note"),
        ],
        className="kpi-card",
    )


# Core tables converted from 실적장표_집계_원내외추가.RData
dept_sale = load_table("dept_sale")
uppdept_sale = load_table("uppdept_sale")
sale_aggr = load_table("sale_aggr")
dept_table = load_table("dept_table")
sale_aim_mr = load_table("sale_aim_mr")
hosp_sale_rank = load_table("hosp_sale_rank")
brand_hosp_sale_rank = load_table("brand_hosp_sale_rank")
hosp_aggr = load_table("hosp_aggr")
item_aggr = load_table("item_aggr")
item_hosp_aggr = load_table("item_hosp_aggr")
hosp_item_aggr = load_table("hosp_item_aggr")
mr_hosp_sale = load_table("mr_hosp_sale")

sale_aggr["date"] = pd.to_datetime(sale_aggr["date"], errors="coerce")

overall_sale = float(uppdept_sale["anual_sale"].sum()) if "anual_sale" in uppdept_sale else 0.0
overall_aim = float(uppdept_sale["anual_aim"].sum()) if "anual_aim" in uppdept_sale else 0.0
overall_gap = overall_sale - overall_aim
overall_ar = overall_sale / overall_aim if overall_aim else 0.0

uppdept_options = options_from(dept_table, "upp_dept")
dept_options = options_from(dept_table, "dept")
mr_options = options_from(sale_aim_mr, "MR")
hospital_options = options_from(hosp_sale_rank, "HospitalName")
brand_options = options_from(item_aggr, "Brand")
year_options = options_from(item_aggr, "Year")

FILTER_OPTIONS = {
    "uppdept-filter": uppdept_options,
    "team-uppdept-filter": uppdept_options,
    "team-dept-filter": dept_options,
    "mr-uppdept-filter": uppdept_options,
    "mr-dept-filter": dept_options,
    "mr-filter": mr_options,
    "hospital-filter": hospital_options,
    "hospital-brand-filter": brand_options,
    "hospital-year-filter": year_options,
    "item-brand-filter": brand_options,
    "item-hospital-filter": hospital_options,
    "item-year-filter": year_options,
    "hosp-sale-hospital-filter": hospital_options,
    "hosp-sale-brand-filter": brand_options,
    "hosp-rank-uppdept-filter": uppdept_options,
    "hosp-rank-dept-filter": dept_options,
    "hosp-rank-hospital-filter": hospital_options,
    "brand-rank-brand-filter": brand_options,
    "brand-rank-hospital-filter": hospital_options,
    "target-uppdept-filter": uppdept_options,
    "target-dept-filter": dept_options,
    "target-mr-filter": mr_options,
}


def overall_trend_fig() -> px.line:
    trend = sale_aggr.groupby("date", as_index=False)["sale"].sum().sort_values("date")
    fig = px.line(trend, x="date", y="sale", markers=True, title="전체 실적 추이")
    fig.update_traces(line={"color": "#005FAF", "width": 3})
    fig.update_layout(xaxis_title="월", yaxis_title="실적")
    return fig


def uppdept_bar_fig(df: pd.DataFrame | None = None):
    src = uppdept_sale if df is None else df
    fig = px.bar(
        src.sort_values("anual_sale"),
        x="anual_sale",
        y="upp_dept",
        orientation="h",
        text=src.sort_values("anual_sale")["anual_sale"].map(money_short),
        title="본부별 실적",
    )
    fig.update_layout(xaxis_title="실적", yaxis_title="본부")
    return fig


def dept_bar_fig(df: pd.DataFrame | None = None):
    src = dept_sale if df is None else df
    view = src.sort_values("anual_sale", ascending=False).head(20).sort_values("anual_sale")
    fig = px.bar(
        view,
        x="anual_sale",
        y="dept",
        orientation="h",
        text=view["anual_sale"].map(money_short),
        color="anual_ar",
        color_continuous_scale="RdYlGn",
        title="팀별 실적 상위 20",
    )
    fig.update_layout(xaxis_title="실적", yaxis_title="팀", coloraxis_colorbar_title="달성률")
    return fig


def item_bar_fig(df: pd.DataFrame | None = None):
    src = item_aggr if df is None else df
    if "실적" not in src.columns:
        return px.bar(title="품목별 실적")
    grouped = src.groupby("Brand", as_index=False)["실적"].sum().sort_values("실적", ascending=False).head(20)
    fig = px.bar(
        grouped.sort_values("실적"),
        x="실적",
        y="Brand",
        orientation="h",
        text=grouped.sort_values("실적")["실적"].map(money_short),
        title="품목별 실적 상위 20",
    )
    fig.update_layout(xaxis_title="실적", yaxis_title="품목")
    return fig


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Pharmbio Python Dash"
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
      .page { max-width: 1480px; margin: 0 auto; padding: 22px; font-family: Arial, sans-serif; }
      .header { margin-bottom: 16px; }
      h1 { margin: 0 0 6px; font-size: 30px; color: #17324d; }
      .sub { color: #526272; font-size: 14px; }
      .tabs .tab { padding: 12px 14px !important; font-weight: 700; }
      .tabs .tab--selected { background: #005FAF !important; color: #fff !important; border-top: 3px solid #003f75 !important; }
      .kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 14px 0; }
      .kpi-card, .section { background: white; border: 1px solid #d8e0e6; border-radius: 8px; padding: 16px; margin-bottom: 14px; }
      .kpi-title { color: #607181; font-size: 13px; font-weight: 700; }
      .kpi-value { font-size: 24px; font-weight: 800; margin-top: 8px; color: #17324d; }
      .kpi-note { min-height: 18px; margin-top: 6px; color: #6b7785; font-size: 12px; }
      .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
      .filters, .inline-filters { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 10px 0 14px; }
      .inline-filters { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); padding-top: 2px; }
      .filter-field-wide { grid-column: 1 / -1; }
      .filter-label { color: #607181; font-size: 13px; font-weight: 700; margin-bottom: 6px; }
      .filter-dropdown .Select-control,
      .filter-dropdown .Select--multi .Select-control { min-height: 42px; border: 1px solid #8b9ab5; border-radius: 5px; }
      .filter-dropdown .Select-placeholder,
      .filter-dropdown .Select-value-label { font-size: 15px; line-height: 40px; }
      .filter-dropdown .Select-menu-outer { border: 1px solid #8b7cf6; border-radius: 5px; box-shadow: 0 10px 24px rgba(25, 42, 70, 0.16); max-height: 280px; z-index: 1000; }
      .filter-dropdown .VirtualizedSelectOption { padding: 10px 12px; border-bottom: 1px solid #eef1f4; font-size: 14px; }
      .filter-dropdown .VirtualizedSelectFocusedOption { background: #f2f6ff; }
      .filter-actions { display: flex; gap: 14px; margin-top: 8px; }
      .filter-action { border: 0; background: transparent; color: #46566b; cursor: pointer; font-weight: 800; font-size: 13px; padding: 0; }
      .filter-action:hover { color: #005FAF; text-decoration: underline; }
      .section-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
      .section-head h3 { margin: 0; font-size: 18px; }
      .note { color: #53606d; font-size: 13px; line-height: 1.45; }
      .download-button { border: 1px solid #1f5d7a; background: #256f8f; color: white; border-radius: 6px; padding: 9px 14px; font-size: 13px; font-weight: 800; cursor: pointer; min-width: 140px; }
      .download-button:hover { background: #1d5d78; }
      @media (max-width: 980px) { .kpi-grid, .grid-2, .filters, .inline-filters { grid-template-columns: 1fr; } .page { padding: 14px; } }
    </style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
"""


def layout_summary() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    kpi("전체 실적", money_short(overall_sale), "본부 합계"),
                    kpi("전체 목표", money_short(overall_aim), "본부 목표 합계"),
                    kpi("달성률", pct(overall_ar), "실적 / 목표"),
                    kpi("GAP", money_short(overall_gap), "실적 - 목표"),
                ],
                className="kpi-grid",
            ),
            html.Div([dcc.Graph(figure=overall_trend_fig()), dcc.Graph(figure=uppdept_bar_fig())], className="grid-2"),
            table_section("본부별 실적 테이블", "summary-uppdept-table", uppdept_sale, "download-summary-uppdept"),
            table_section("부서별 실적 테이블", "summary-dept-table", dept_sale, "download-summary-dept"),
            table_section("품목별 실적 순위", "summary-item-table", item_aggr.groupby("Brand", as_index=False)["실적"].sum().sort_values("실적", ascending=False).head(100), "download-summary-item"),
        ]
    )


def layout_uppdept() -> html.Div:
    return html.Div(
        [
            html.Div([dcc.Graph(id="uppdept-graph", figure=uppdept_bar_fig())], className="section"),
            table_section(
                "본부별 실적 현황",
                "uppdept-table",
                uppdept_sale,
                "download-uppdept",
                filters=[dropdown("본부 필터", "uppdept-filter", uppdept_options, wide=True)],
            ),
            table_section("본부별 월별 실적", "uppdept-month-table", sale_aggr, "download-uppdept-month"),
        ]
    )


def layout_team() -> html.Div:
    return html.Div(
        [
            html.Div([dcc.Graph(id="team-graph", figure=dept_bar_fig())], className="section"),
            table_section(
                "팀별 실적 현황",
                "team-table",
                dept_table,
                "download-team",
                filters=[
                    dropdown("본부 필터", "team-uppdept-filter", uppdept_options),
                    dropdown("팀 필터", "team-dept-filter", dept_options),
                ],
            ),
            table_section("팀별 병원/품목 실적 현황", "team-hosp-item-table", mr_hosp_sale, "download-team-hosp-item"),
        ]
    )


def layout_mr() -> html.Div:
    return html.Div(
        [
            table_section(
                "담당자별 실적 현황",
                "mr-table",
                sale_aim_mr,
                "download-mr",
                filters=[
                    dropdown("본부 필터", "mr-uppdept-filter", uppdept_options),
                    dropdown("팀 필터", "mr-dept-filter", dept_options),
                    dropdown("담당자 필터", "mr-filter", mr_options),
                ],
            ),
            table_section("담당자별 병원 실적", "mr-hosp-table", mr_hosp_sale, "download-mr-hosp"),
        ]
    )


def layout_hospital_item() -> html.Div:
    return html.Div(
        [
            table_section(
                "병원별 실적",
                "hospital-table",
                hosp_aggr,
                "download-hospital",
                filters=[
                    dropdown("병원 필터", "hospital-filter", hospital_options),
                    dropdown("품목 필터", "hospital-brand-filter", brand_options),
                    dropdown("연도 필터", "hospital-year-filter", year_options),
                ],
            ),
            table_section("선택 병원 품목별 실적", "hospital-item-table", item_hosp_aggr, "download-hospital-item"),
        ]
    )


def layout_item_hospital() -> html.Div:
    return html.Div(
        [
            html.Div([dcc.Graph(id="item-graph", figure=item_bar_fig())], className="section"),
            table_section(
                "품목별 실적",
                "item-table",
                item_aggr,
                "download-item",
                filters=[
                    dropdown("품목 필터", "item-brand-filter", brand_options),
                    dropdown("병원 필터", "item-hospital-filter", hospital_options),
                    dropdown("연도 필터", "item-year-filter", year_options),
                ],
            ),
            table_section("선택 품목 병원별 실적", "item-hospital-table", item_hosp_aggr, "download-item-hospital"),
        ]
    )


def layout_hosp_sale() -> html.Div:
    return html.Div(
        [
            table_section(
                "거래처별 실적",
                "hosp-sale-table",
                hosp_aggr,
                "download-hosp-sale",
                filters=[
                    dropdown("병원 필터", "hosp-sale-hospital-filter", hospital_options),
                    dropdown("품목 필터", "hosp-sale-brand-filter", brand_options),
                ],
            ),
        ]
    )


def layout_hosp_rank() -> html.Div:
    return html.Div(
        [
            table_section(
                "거래처 순위",
                "hosp-rank-table",
                hosp_sale_rank,
                "download-hosp-rank",
                filters=[
                    dropdown("본부 필터", "hosp-rank-uppdept-filter", uppdept_options),
                    dropdown("팀 필터", "hosp-rank-dept-filter", dept_options),
                    dropdown("병원 필터", "hosp-rank-hospital-filter", hospital_options),
                ],
            ),
        ]
    )


def layout_brand_hosp_rank() -> html.Div:
    return html.Div(
        [
            table_section(
                "브랜드별 거래처 순위",
                "brand-hosp-rank-table",
                brand_hosp_sale_rank,
                "download-brand-hosp-rank",
                filters=[
                    dropdown("품목 필터", "brand-rank-brand-filter", brand_options),
                    dropdown("병원 필터", "brand-rank-hospital-filter", hospital_options),
                ],
            ),
        ]
    )


def layout_target() -> html.Div:
    return html.Div(
        [
            table_section(
                "담당자별 목표 현황",
                "target-mr-table",
                sale_aim_mr,
                "download-target-mr",
                filters=[
                    dropdown("본부 필터", "target-uppdept-filter", uppdept_options),
                    dropdown("팀 필터", "target-dept-filter", dept_options),
                    dropdown("담당자 필터", "target-mr-filter", mr_options),
                ],
            ),
            table_section("팀별 목표 현황", "target-team-table", dept_table, "download-target-team"),
        ]
    )


app.layout = html.Div(
    [
        html.Div([html.H1("Pharmbio 실적 대시보드"), html.Div("R Markdown/flexdashboard 구조를 Python Dash 탭형 레이아웃으로 변환했습니다.", className="sub")], className="header"),
        dcc.Tabs(
            id="main-tabs",
            value="실적 요약",
            className="tabs",
            children=[
                dcc.Tab(label="실적 요약", value="실적 요약"),
                dcc.Tab(label="본부별", value="본부별"),
                dcc.Tab(label="팀별", value="팀별"),
                dcc.Tab(label="담당자별", value="담당자별"),
                dcc.Tab(label="전체 병원/품목별", value="전체 병원/품목별"),
                dcc.Tab(label="전체 품목/병원별", value="전체 품목/병원별"),
                dcc.Tab(label="거래처별 실적", value="거래처별 실적"),
                dcc.Tab(label="거래처 순위", value="거래처 순위"),
                dcc.Tab(label="브랜드별 거래처 순위", value="브랜드별 거래처 순위"),
                dcc.Tab(label="목표", value="목표"),
            ],
        ),
        html.Div(id="tab-content", className="page"),
    ]
)


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab: str):
    if tab == "실적 요약":
        return layout_summary()
    if tab == "본부별":
        return layout_uppdept()
    if tab == "팀별":
        return layout_team()
    if tab == "담당자별":
        return layout_mr()
    if tab == "전체 병원/품목별":
        return layout_hospital_item()
    if tab == "전체 품목/병원별":
        return layout_item_hospital()
    if tab == "거래처별 실적":
        return layout_hosp_sale()
    if tab == "거래처 순위":
        return layout_hosp_rank()
    if tab == "브랜드별 거래처 순위":
        return layout_brand_hosp_rank()
    if tab == "목표":
        return layout_target()
    return layout_summary()


@app.callback(
    Output("uppdept-graph", "figure"),
    Output("uppdept-table", "data"),
    Output("uppdept-month-table", "data"),
    Input("uppdept-filter", "value"),
    prevent_initial_call=True,
)
def update_uppdept(values):
    df = filter_in(uppdept_sale, "upp_dept", values)
    month_df = filter_in(sale_aggr.copy(), "upp_dept", values)
    return uppdept_bar_fig(df), table_records(df), table_records(month_df)


@app.callback(
    Output("team-graph", "figure"),
    Output("team-table", "data"),
    Output("team-hosp-item-table", "data"),
    Input("team-uppdept-filter", "value"),
    Input("team-dept-filter", "value"),
    prevent_initial_call=True,
)
def update_team(uppdepts, depts):
    df = dept_table.copy()
    df = filter_in(df, "upp_dept", uppdepts)
    df = filter_in(df, "dept", depts)
    graph_df = dept_sale.copy()
    if uppdepts and "dept" in df.columns:
        graph_df = filter_in(graph_df, "dept", df["dept"].dropna().astype(str).unique().tolist())
    graph_df = filter_in(graph_df, "dept", depts)
    hosp_item_df = mr_hosp_sale.copy()
    hosp_item_df = filter_in(hosp_item_df, "UppDept", uppdepts)
    hosp_item_df = filter_in(hosp_item_df, "Dept", depts)
    return dept_bar_fig(graph_df), table_records(df), table_records(hosp_item_df)


@app.callback(Output("mr-table", "data"), Output("mr-hosp-table", "data"), Input("mr-uppdept-filter", "value"), Input("mr-dept-filter", "value"), Input("mr-filter", "value"), prevent_initial_call=True)
def update_mr(uppdepts, depts, mrs):
    df = sale_aim_mr.copy()
    hosp_df = mr_hosp_sale.copy()
    for target in [df, hosp_df]:
        pass
    df = filter_in(df, "UppDept", uppdepts)
    df = filter_in(df, "Dept", depts)
    df = filter_in(df, "MR", mrs)
    hosp_df = filter_in(hosp_df, "UppDept", uppdepts)
    hosp_df = filter_in(hosp_df, "Dept", depts)
    hosp_df = filter_in(hosp_df, "MR", mrs)
    return table_records(df), table_records(hosp_df)


@app.callback(Output("hospital-table", "data"), Output("hospital-item-table", "data"), Input("hospital-filter", "value"), Input("hospital-brand-filter", "value"), Input("hospital-year-filter", "value"), prevent_initial_call=True)
def update_hospital_item(hospitals, brands, years):
    hosp_df = filter_in(hosp_aggr.copy(), "HospitalName", hospitals)
    hosp_df = filter_in(hosp_df, "Brand", brands)
    hosp_df = filter_in(hosp_df, "Year", years)
    item_df = filter_in(item_hosp_aggr.copy(), "HospitalName", hospitals)
    item_df = filter_in(item_df, "Brand", brands)
    item_df = filter_in(item_df, "Year", years)
    return table_records(hosp_df), table_records(item_df)


@app.callback(Output("item-graph", "figure"), Output("item-table", "data"), Output("item-hospital-table", "data"), Input("item-brand-filter", "value"), Input("item-hospital-filter", "value"), Input("item-year-filter", "value"), prevent_initial_call=True)
def update_item_hospital(brands, hospitals, years):
    item_df = filter_in(item_aggr.copy(), "Brand", brands)
    item_df = filter_in(item_df, "Year", years)
    item_hosp_df = filter_in(item_hosp_aggr.copy(), "Brand", brands)
    item_hosp_df = filter_in(item_hosp_df, "HospitalName", hospitals)
    item_hosp_df = filter_in(item_hosp_df, "Year", years)
    return item_bar_fig(item_df), table_records(item_df), table_records(item_hosp_df)


@app.callback(Output("hosp-sale-table", "data"), Input("hosp-sale-hospital-filter", "value"), Input("hosp-sale-brand-filter", "value"), prevent_initial_call=True)
def update_hosp_sale(hospitals, brands):
    df = filter_in(hosp_aggr.copy(), "HospitalName", hospitals)
    df = filter_in(df, "Brand", brands)
    return table_records(df)


@app.callback(Output("hosp-rank-table", "data"), Input("hosp-rank-uppdept-filter", "value"), Input("hosp-rank-dept-filter", "value"), Input("hosp-rank-hospital-filter", "value"), prevent_initial_call=True)
def update_hosp_rank(uppdepts, depts, hospitals):
    df = filter_in(hosp_sale_rank.copy(), "UppDept", uppdepts)
    df = filter_in(df, "Dept", depts)
    df = filter_in(df, "HospitalName", hospitals)
    return table_records(df)


@app.callback(Output("brand-hosp-rank-table", "data"), Input("brand-rank-brand-filter", "value"), Input("brand-rank-hospital-filter", "value"), prevent_initial_call=True)
def update_brand_hosp_rank(brands, hospitals):
    df = filter_in(brand_hosp_sale_rank.copy(), "Brand", brands)
    df = filter_in(df, "HospitalName", hospitals)
    return table_records(df)


@app.callback(Output("target-mr-table", "data"), Output("target-team-table", "data"), Input("target-uppdept-filter", "value"), Input("target-dept-filter", "value"), Input("target-mr-filter", "value"), prevent_initial_call=True)
def update_target(uppdepts, depts, mrs):
    mr_df = filter_in(sale_aim_mr.copy(), "UppDept", uppdepts)
    mr_df = filter_in(mr_df, "Dept", depts)
    mr_df = filter_in(mr_df, "MR", mrs)
    team_df = filter_in(dept_table.copy(), "upp_dept", uppdepts)
    team_df = filter_in(team_df, "dept", depts)
    return table_records(mr_df), table_records(team_df)


for filter_id, filter_options in FILTER_OPTIONS.items():

    @app.callback(
        Output(filter_id, "value"),
        Input(f"{filter_id}-select-all", "n_clicks"),
        Input(f"{filter_id}-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def _set_filter_values(_, __, filter_options=filter_options):
        if str(ctx.triggered_id).endswith("-clear"):
            return []
        return [option["value"] for option in filter_options]


def download_from_state(rows, fallback: pd.DataFrame, filename: str):
    df = pd.DataFrame(rows) if rows else display_table(fallback)
    return dcc.send_data_frame(df.to_csv, filename, index=False, encoding="utf-8-sig")


DOWNLOADS = {
    "download-summary-uppdept": ("summary-uppdept-table", uppdept_sale, "본부별실적.csv"),
    "download-summary-dept": ("summary-dept-table", dept_sale, "부서별실적.csv"),
    "download-summary-item": ("summary-item-table", item_aggr, "품목별실적순위.csv"),
    "download-uppdept": ("uppdept-table", uppdept_sale, "본부별실적현황.csv"),
    "download-uppdept-month": ("uppdept-month-table", sale_aggr, "본부별월별실적.csv"),
    "download-team": ("team-table", dept_table, "팀별실적현황.csv"),
    "download-team-hosp-item": ("team-hosp-item-table", mr_hosp_sale, "팀별병원품목실적.csv"),
    "download-mr": ("mr-table", sale_aim_mr, "담당자별실적현황.csv"),
    "download-mr-hosp": ("mr-hosp-table", mr_hosp_sale, "담당자별병원실적.csv"),
    "download-hospital": ("hospital-table", hosp_aggr, "병원별실적.csv"),
    "download-hospital-item": ("hospital-item-table", item_hosp_aggr, "병원품목별실적.csv"),
    "download-item": ("item-table", item_aggr, "품목별실적.csv"),
    "download-item-hospital": ("item-hospital-table", item_hosp_aggr, "품목병원별실적.csv"),
    "download-hosp-sale": ("hosp-sale-table", hosp_aggr, "거래처별실적.csv"),
    "download-hosp-rank": ("hosp-rank-table", hosp_sale_rank, "거래처순위.csv"),
    "download-brand-hosp-rank": ("brand-hosp-rank-table", brand_hosp_sale_rank, "브랜드별거래처순위.csv"),
    "download-target-mr": ("target-mr-table", sale_aim_mr, "담당자별목표현황.csv"),
    "download-target-team": ("target-team-table", dept_table, "팀별목표현황.csv"),
}


for button_id, (table_id, fallback_df, filename) in DOWNLOADS.items():
    app.layout.children.append(dcc.Download(id=f"{button_id}-file"))

    @app.callback(
        Output(f"{button_id}-file", "data"),
        Input(button_id, "n_clicks"),
        State(table_id, "derived_virtual_data"),
        prevent_initial_call=True,
    )
    def _download(_, rows, fallback_df=fallback_df, filename=filename):
        return download_from_state(rows, fallback_df, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8052, debug=False)
