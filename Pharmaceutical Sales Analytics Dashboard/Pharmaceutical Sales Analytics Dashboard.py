from __future__ import annotations

from io import BytesIO
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import threading
import webbrowser

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "sales_star_schema.sqlite"
ETL_PATH = BASE_DIR / "rdata_to_star_schema_sqlite.py"
MAX_ROWS = 1000


def ensure_database() -> None:
    needs_build = not DB_PATH.exists() or DB_PATH.stat().st_size == 0
    if not needs_build:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("SELECT 1 FROM fact_sales_performance LIMIT 1").fetchone()
        except sqlite3.Error:
            needs_build = True
    if needs_build:
        subprocess.run([sys.executable, str(ETL_PATH)], check=True)


ensure_database()


def read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params or {})


def sql_in(column: str, values, params: dict, prefix: str) -> str:
    values = list(values or [])
    if not values:
        return ""
    holders = []
    for i, value in enumerate(values):
        key = f"{prefix}_{i}"
        params[key] = value
        holders.append(f":{key}")
    return f" AND {column} IN ({', '.join(holders)})"


def choices(column: str, table: str, where: str = "") -> list[dict[str, str]]:
    df = read_sql(f"SELECT DISTINCT {column} AS value FROM {table} {where} ORDER BY value")
    return [{"label": str(v), "value": str(v)} for v in df["value"].dropna().tolist() if str(v).strip()]


def report_months() -> list[str]:
    df = read_sql(
        """
        SELECT d.yyyymm
        FROM fact_sales_performance f
        JOIN dim_date d ON f.date_key = d.date_key
        GROUP BY d.yyyymm
        HAVING
            SUM(CASE WHEN f.sale_amount IS NOT NULL AND ABS(f.sale_amount) > 0 THEN 1 ELSE 0 END) > 0
            AND
            SUM(CASE WHEN f.aim_amount IS NOT NULL AND ABS(f.aim_amount) > 0 THEN 1 ELSE 0 END) > 0
        ORDER BY d.yyyymm
        """
    )
    return df["yyyymm"].astype(str).tolist()


MONTHS = report_months()
MONTH_OPTIONS = [{"label": m, "value": m} for m in MONTHS]
UPP_OPTIONS = choices("upp_dept", "dim_org", "WHERE upp_dept IS NOT NULL")
DEPT_OPTIONS = choices("dept", "dim_org", "WHERE dept IS NOT NULL")
EMP_OPTIONS = choices("employee_name", "dim_employee", "WHERE employee_name IS NOT NULL")
HOSP_OPTIONS = choices("hospital_name", "dim_hospital", "WHERE hospital_name IS NOT NULL AND hospital_name <> 'UNKNOWN'")


def report_period_label() -> str:
    if not MONTHS:
        return "분석 가능 월 없음"
    return MONTHS[0] if len(MONTHS) == 1 else f"{MONTHS[0]} ~ {MONTHS[-1]}"


def base_fact_sql(where: str = "") -> str:
    return f"""
        SELECT
            d.yyyymm,
            d.year,
            d.month,
            o.upp_dept,
            o.dept,
            h.hospital_name,
            e.employee_name,
            f.sale_amount,
            f.aim_amount,
            f.achievement_rate,
            f.growth_rate,
            f.source_object
        FROM fact_sales_performance f
        JOIN dim_date d ON f.date_key = d.date_key
        LEFT JOIN dim_org o ON f.org_key = o.org_key
        LEFT JOIN dim_hospital h ON f.hospital_key = h.hospital_key
        LEFT JOIN dim_employee e ON f.employee_key = e.employee_key
        WHERE 1=1
        {where}
    """


def query_fact(months=None, uppdepts=None, depts=None, employees=None, hospitals=None) -> pd.DataFrame:
    params = {}
    where = ""
    where += sql_in("d.yyyymm", list(months or []) or MONTHS, params, "m")
    where += sql_in("o.upp_dept", uppdepts, params, "u")
    where += sql_in("o.dept", depts, params, "d")
    where += sql_in("e.employee_name", employees, params, "e")
    where += sql_in("h.hospital_name", hospitals, params, "h")
    return read_sql(base_fact_sql(where), params)


def money_short(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 100_000:
        return f"{value / 100_000:,.1f}억"
    if abs(value) >= 10_000:
        return f"{value / 10_000:,.1f}만"
    return f"{value:,.0f}"


def pct(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{float(value):.1%}"


def add_rate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["달성률"] = np.where(out["목표"].fillna(0).eq(0), np.nan, out["실적"] / out["목표"])
    return out


def display_fact(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "yyyymm": "월",
            "year": "연도",
            "month": "월번호",
            "upp_dept": "본부",
            "dept": "팀",
            "hospital_name": "거래처",
            "employee_name": "담당자",
            "sale_amount": "실적",
            "aim_amount": "목표",
            "achievement_rate": "달성률",
            "growth_rate": "성장률",
            "source_object": "원천",
        }
    )


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["실적", "목표"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    for col in ["달성률", "성장률"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def top_n(df: pd.DataFrame, dimension: str, value: str = "sale_amount", n: int = 20) -> pd.DataFrame:
    if df.empty or dimension not in df or value not in df:
        return pd.DataFrame(columns=[dimension, value])
    work = df[[dimension, value]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work[dimension] = work[dimension].fillna("").astype(str).str.strip()
    work = work[(work[dimension] != "") & (work[dimension] != "UNKNOWN") & work[value].notna()]
    if work.empty:
        return pd.DataFrame(columns=[dimension, value])
    return work.groupby(dimension, as_index=False)[value].sum().nlargest(n, value).sort_values(value)


def chart_layout(fig):
    fig.update_layout(
        template="plotly_white",
        height=330,
        margin={"l": 26, "r": 18, "t": 38, "b": 32},
        font={"family": "Malgun Gothic, Arial, sans-serif", "size": 13},
        colorway=["#005FAF", "#2BB3A3", "#F59E0B", "#7C3AED", "#EF4444", "#64748B"],
        hovermode="x unified",
        legend_title_text="",
    )
    fig.update_yaxes(automargin=True, gridcolor="#edf2f7", zerolinecolor="#d6dee6")
    fig.update_xaxes(automargin=True, gridcolor="#f3f6f9", zerolinecolor="#d6dee6")
    return fig


def empty_fig(message: str = "표시할 데이터가 없습니다."):
    fig = px.scatter(pd.DataFrame({"x": [], "y": []}), x="x", y="y")
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"size": 15, "color": "#607181"})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return chart_layout(fig)


def xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_sheet = (sheet_name or "Sheet1")[:31]
            out = df.copy()
            out.to_excel(writer, index=False, sheet_name=safe_sheet)
            ws = writer.book[safe_sheet]
            ws.freeze_panes = "A2"
            for cells in ws.columns:
                width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in cells)
                ws.column_dimensions[cells[0].column_letter].width = min(max(width + 2, 10), 36)
    return buffer.getvalue()


def table_component(df: pd.DataFrame, page_size: int = 12):
    view = clean_table(df.head(MAX_ROWS))
    return dash_table.DataTable(
        data=view.to_dict("records"),
        columns=[{"name": str(c), "id": str(c)} for c in view.columns],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        export_format="xlsx",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "Malgun Gothic, Arial, sans-serif",
            "fontSize": 13,
            "padding": "8px",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "80px",
            "maxWidth": "220px",
        },
        style_header={"backgroundColor": "#f1f5f9", "fontWeight": "800", "color": "#25364a"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafcff"}],
    )


def dropdown(label: str, component_id: str, options: list[dict[str, str]], multi: bool = True):
    return html.Div(
        [html.Div(label, className="filter-label"), dcc.Dropdown(id=component_id, options=options, multi=multi, searchable=True, clearable=True, placeholder=f"전체 {label}")],
        className="filter-field",
    )


def filters(prefix: str, include_employee: bool = False, include_hospital: bool = False):
    children = [
        dropdown("월", f"{prefix}-month", MONTH_OPTIONS),
        dropdown("본부", f"{prefix}-upp", UPP_OPTIONS),
        dropdown("팀/부서", f"{prefix}-dept", DEPT_OPTIONS),
    ]
    if include_employee:
        children.append(dropdown("담당자", f"{prefix}-emp", EMP_OPTIONS))
    if include_hospital:
        children.append(dropdown("거래처", f"{prefix}-hosp", HOSP_OPTIONS))
    return html.Div(
        [html.Div([html.Div("조건 선택", className="filter-title"), html.Div("월을 선택하지 않으면 실적과 목표가 모두 있는 전체 분석 기간을 사용합니다.", className="filter-note")], className="filter-head"), html.Div(children, className="filter-grid")],
        className="filters",
    )


def page_intro(title: str, body: str):
    return html.Div([html.Div(title, className="intro-title"), html.Div(body, className="intro-body")], className="page-intro")


def panel(title: str, child, download_id: str | None = None):
    return html.Div(
        [
            html.Div(
                [
                    html.H3(title),
                    html.Div(
                        [
                            html.Button("Excel", id=download_id, className="download-button") if download_id else None,
                            dcc.Download(id=f"{download_id}-file") if download_id else None,
                        ],
                        className="download-wrap",
                    ),
                ],
                className="panel-head",
            ),
            child,
        ],
        className="panel",
    )


def kpis(prefix: str):
    return html.Div(
        [
            html.Div([html.Div("총 실적", className="kpi-title"), html.Div(id=f"{prefix}-sales", className="kpi-value")], className="kpi-card"),
            html.Div([html.Div("총 목표", className="kpi-title"), html.Div(id=f"{prefix}-aim", className="kpi-value")], className="kpi-card"),
            html.Div([html.Div("달성률", className="kpi-title"), html.Div(id=f"{prefix}-rate", className="kpi-value")], className="kpi-card"),
            html.Div([html.Div("조회 행 수", className="kpi-title"), html.Div(id=f"{prefix}-rows", className="kpi-value")], className="kpi-card"),
        ],
        className="kpi-grid",
    )


app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


def nav_menu():
    groups = [
        ("실적", [("실적 요약", "/performance/summary"), ("본부별 실적", "/performance/upp"), ("팀별 실적", "/performance/team")]),
        ("거래처", [("거래처별 실적", "/hospital/summary"), ("거래처 Top 20", "/hospital/top20"), ("거래처 상세", "/hospital/detail")]),
        ("담당자", [("담당자별 실적", "/employee/summary"), ("담당자 Top 20", "/employee/top20"), ("담당자 상세", "/employee/detail")]),
        ("분석", [("Drill-down 분석", "/analysis/drill"), ("목표 대비 분석", "/analysis/target")]),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(title, className="menu-title"),
                    html.Div([dcc.Link(label, href=href, className="submenu-link") for label, href in links], className="submenu"),
                ],
                className="menu-group",
            )
            for title, links in groups
        ],
        className="nav-menu",
    )


def page_shell(title: str, body: str, content):
    return [
        html.Div([html.Div(title, className="page-title"), html.Div(body, className="page-subtitle")], className="page-heading"),
        *content,
    ]


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(
            [
                html.Div([html.H1("Pharmbio 실적 대시보드")]),
                html.Div(
                    [
                        html.Div(f"분석 기간 {report_period_label()}", className="header-badge"),
                        html.Div("실적/목표 동시 존재 월 기준", className="header-badge"),
                        html.Div("Excel 다운로드 지원", className="header-badge"),
                    ],
                    className="header-badges",
                ),
            ],
            className="app-header",
        ),
        nav_menu(),
        html.Div(id="page", className="page"),
        dcc.Download(id="download"),
    ]
)

app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Pharmbio 실적 대시보드</title>
{%favicon%}
{%css%}
<style>
body { margin: 0; background: #f6f8fb; color: #1f2933; font-family: "Malgun Gothic", Arial, sans-serif; }
.app-header { background: #005faf; color: white; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; gap: 18px; }
.app-header h1 { margin: 0; font-size: 25px; font-weight: 850; }
.app-header .sub { margin-top: 4px; color: #d9ecff; font-size: 13px; }
.header-badges { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.header-badge { border: 1px solid rgba(255,255,255,0.32); background: rgba(255,255,255,0.11); color: #fff; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 800; white-space: nowrap; }
.nav-menu { background: #2d86df; box-shadow: 0 2px 8px rgba(0,54,112,.18); padding-left: 14px; display: flex; align-items: stretch; min-height: 48px; position: relative; z-index: 20; }
.menu-group { position: relative; }
.menu-title { color: white; padding: 14px 22px; font-weight: 850; cursor: default; border-right: 1px solid rgba(255,255,255,.10); min-width: 112px; text-align: center; }
.menu-group:hover .menu-title { background: #0e64ad; }
.submenu { display: none; position: absolute; left: 0; top: 48px; min-width: 210px; background: white; border: 1px solid #c9d6e3; border-top: 0; box-shadow: 0 12px 24px rgba(15,23,42,.18); z-index: 100; border-radius: 0 0 8px 8px; overflow: hidden; }
.menu-group:hover .submenu { display: block; }
.submenu-link { display: block; padding: 12px 15px; color: #26394d; text-decoration: none; font-weight: 800; border-bottom: 1px solid #edf2f7; white-space: nowrap; }
.submenu-link:hover { background: #eef6ff; color: #005faf; text-decoration: none; }
.page { max-width: 1680px; margin: 0 auto; padding: 18px 14px 26px; }
.page-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 18px; margin: 0 2px 12px; }
.page-title { color: #172334; font-size: 21px; font-weight: 900; }
.page-subtitle { color: #66788a; font-size: 13px; text-align: right; }
.page-intro { display: none; }
.intro-title { color: #1f2933; font-weight: 850; font-size: 17px; margin-bottom: 2px; }
.intro-body { color: #607181; font-size: 12.5px; line-height: 1.4; }
.filters, .panel, .kpi-card { background: white; border: 1px solid #dde5ed; border-radius: 8px; box-shadow: none; }
.filters { padding: 13px 15px; margin-bottom: 14px; }
.filter-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 8px; border-bottom: 1px solid #f0f3f7; padding-bottom: 8px; }
.filter-title { color: #1f2933; font-size: 14px; font-weight: 850; }
.filter-note { color: #6b7a8a; font-size: 12px; }
.filter-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
.filter-label { color: #435466; font-size: 13px; font-weight: 800; margin-bottom: 5px; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
.kpi-card { padding: 14px 16px; position: relative; overflow: hidden; }
.kpi-card:before { content: ""; position: absolute; left: 0; top: 13px; bottom: 13px; width: 3px; border-radius: 3px; background: #005faf; }
.kpi-title { color: #607181; font-size: 12px; font-weight: 800; margin-bottom: 6px; }
.kpi-value { color: #111827; font-size: 24px; font-weight: 850; line-height: 1.2; }
.grid-2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; }
.grid-1 { display: block; }
.hidden-section { display: none; }
.panel { padding: 14px 15px; margin-bottom: 14px; }
.panel-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px; border-bottom: 1px solid #f0f3f7; padding-bottom: 8px; }
.panel h3 { margin: 0; font-size: 16px; font-weight: 850; color: #1f2933; }
.download-wrap { display: flex; align-items: center; gap: 6px; }
.download-button { border-radius: 5px; font-weight: 800; padding: 5px 10px; font-size: 12px; background: #f8fafc; border: 1px solid #c7d2df; color: #24415f; cursor: pointer; }
.download-button:hover { background: #e3eef9; border-color: #7fa8cf; color: #12385f; }
.drill-path { background: #eef6ff; border: 1px solid #b8d8f6; color: #174a7c; border-radius: 8px; padding: 10px 12px; font-weight: 800; margin-bottom: 12px; }
@media (max-width: 1050px) { .grid-2, .kpi-grid { grid-template-columns: 1fr; } .app-header { align-items: flex-start; flex-direction: column; } .header-badges { justify-content: flex-start; } }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""


def section_layout(
    prefix: str,
    title: str,
    body: str,
    include_employee: bool = False,
    include_hospital: bool = False,
    visible_sections: tuple[str, ...] = ("a", "b", "table"),
):
    return [
        page_intro(title, body),
        filters(prefix, include_employee=include_employee, include_hospital=include_hospital),
        kpis(prefix),
        html.Div(
            [
                html.Div(panel("월별 추세", dcc.Graph(id=f"{prefix}-chart-a"), f"{prefix}-download-a"), className="" if "a" in visible_sections else "hidden-section"),
                html.Div(panel("성과 비교", dcc.Graph(id=f"{prefix}-chart-b"), f"{prefix}-download-b"), className="" if "b" in visible_sections else "hidden-section"),
            ],
            className="grid-2" if "a" in visible_sections and "b" in visible_sections else "grid-1",
        ),
        html.Div(panel("상세 테이블", html.Div(id=f"{prefix}-table"), f"{prefix}-download-table"), className="" if "table" in visible_sections else "hidden-section"),
    ]


@app.callback(Output("page", "children"), Input("url", "pathname"))
def render_page(pathname):
    path = pathname or "/performance/summary"
    if path in ["/", "/performance", "/performance/summary"]:
        return page_shell("실적 요약", "실적과 목표가 모두 있는 월만 기준으로 전체 성과를 확인합니다.", section_layout("summary", "전체 실적 현황", "실적과 목표가 모두 있는 월만 기준으로 전체 성과를 확인합니다."))
    if path == "/performance/upp":
        return page_shell("본부별 실적", "본부별 실적 규모와 달성률, 월별 추세를 비교합니다.", section_layout("upp", "본부 단위 성과", "본부별 실적 규모와 달성률, 월별 추세를 비교합니다."))
    if path == "/performance/team":
        return page_shell("팀별 실적", "팀별 실적 Top 20과 달성률 Top 20을 확인합니다.", section_layout("team", "팀/부서 단위 성과", "팀별 실적 Top 20과 달성률 Top 20을 확인합니다."))
    if path in ["/employee/summary", "/employee/top20", "/employee/detail"]:
        titles = {"/employee/summary": "담당자별 실적", "/employee/top20": "담당자 Top 20", "/employee/detail": "담당자 상세"}
        visible = {
            "/employee/summary": ("b",),
            "/employee/top20": ("a",),
            "/employee/detail": ("table",),
        }[path]
        subtitles = {
            "/employee/summary": "담당자별 월별 실적 흐름을 확인합니다.",
            "/employee/top20": "선택 조건 기준 담당자 실적 Top 20을 확인합니다.",
            "/employee/detail": "담당자별 상세 집계 테이블을 확인합니다.",
        }
        return page_shell(titles[path], subtitles[path], section_layout("emp", "담당자/MR 성과", subtitles[path], include_employee=True, visible_sections=visible))
    if path in ["/hospital/summary", "/hospital/top20", "/hospital/detail"]:
        titles = {"/hospital/summary": "거래처별 실적", "/hospital/top20": "거래처 Top 20", "/hospital/detail": "거래처 상세"}
        visible = {
            "/hospital/summary": ("b",),
            "/hospital/top20": ("a",),
            "/hospital/detail": ("table",),
        }[path]
        subtitles = {
            "/hospital/summary": "거래처별 월별 실적 흐름을 확인합니다.",
            "/hospital/top20": "선택 조건 기준 거래처 실적 Top 20을 확인합니다.",
            "/hospital/detail": "거래처별 상세 집계 테이블을 확인합니다.",
        }
        return page_shell(titles[path], subtitles[path], section_layout("hosp", "거래처 성과", subtitles[path], include_hospital=True, visible_sections=visible))
    if path == "/analysis/target":
        return page_shell("목표 대비 분석", "본부와 팀 기준으로 목표 대비 실적과 달성률을 확인합니다.", section_layout("upp", "목표 대비 분석", "본부와 팀 기준으로 목표 대비 실적과 달성률을 확인합니다."))
    return page_shell("Drill-down 분석", "본부 → 팀 → 담당자 → 거래처 순서로 선택 경로의 실적 추세를 확인합니다.", [
        html.Div(
            [
                html.Div("조건 선택", className="filter-title"),
                html.Div(
                    [
                        dropdown("1단계: 본부", "drill-upp", UPP_OPTIONS, multi=False),
                        dropdown("2단계: 팀/부서", "drill-dept", DEPT_OPTIONS, multi=False),
                        dropdown("3단계: 담당자", "drill-emp", EMP_OPTIONS, multi=False),
                        dropdown("4단계: 거래처", "drill-hosp", HOSP_OPTIONS, multi=False),
                    ],
                    className="filter-grid",
                ),
            ],
            className="filters",
        ),
        html.Div(id="drill-path", className="drill-path"),
        html.Div(
            [
                panel("선택 경로 월별 추세", dcc.Graph(id="drill-chart-a"), "drill-download-a"),
                panel("거래처 Top 20", dcc.Graph(id="drill-chart-b"), "drill-download-b"),
            ],
            className="grid-2",
        ),
        panel("드릴다운 상세", html.Div(id="drill-table"), "drill-download-table"),
    ])


def aggregate_for_tab(tab, months, upp, dept, emp=None, hosp=None):
    df = query_fact(months=months, uppdepts=upp, depts=dept, employees=emp, hospitals=hosp)
    total_sale = df["sale_amount"].sum()
    total_aim = df["aim_amount"].sum()
    kpi = [money_short(total_sale), money_short(total_aim), pct(total_sale / total_aim) if total_aim else "-", f"{len(df):,}"]

    if tab == "summary":
        monthly = df.groupby("yyyymm", as_index=False)[["sale_amount", "aim_amount"]].sum()
        long = monthly.melt("yyyymm", ["sale_amount", "aim_amount"], "구분", "금액")
        long["구분"] = long["구분"].map({"sale_amount": "실적", "aim_amount": "목표"})
        fig_a = chart_layout(px.line(long, x="yyyymm", y="금액", color="구분", markers=True)) if not long.empty else empty_fig()
        table = add_rate(df.groupby(["upp_dept", "dept"], dropna=False, as_index=False).agg(실적=("sale_amount", "sum"), 목표=("aim_amount", "sum"), 행수=("sale_amount", "size")).rename(columns={"upp_dept": "본부", "dept": "팀"})).sort_values("실적", ascending=False)
        fig_b_data = table.groupby("본부", as_index=False)[["실적", "목표"]].sum()
        fig_b_data = add_rate(fig_b_data).sort_values("실적")
        fig_b = chart_layout(px.bar(fig_b_data, x="실적", y="본부", orientation="h", color="달성률", color_continuous_scale="RdYlGn")) if not fig_b_data.empty else empty_fig()
        downloads = {"월별추세": long, "성과비교": fig_b_data, "상세": table}
        return kpi, fig_a, fig_b, table_component(table), downloads

    if tab == "upp":
        table = add_rate(df.groupby("upp_dept", dropna=False, as_index=False)[["sale_amount", "aim_amount"]].sum().rename(columns={"upp_dept": "본부", "sale_amount": "실적", "aim_amount": "목표"})).sort_values("실적", ascending=False)
        monthly = df.groupby(["yyyymm", "upp_dept"], dropna=False, as_index=False)["sale_amount"].sum().rename(columns={"yyyymm": "월", "upp_dept": "본부", "sale_amount": "실적"})
        fig_a = chart_layout(px.line(monthly, x="월", y="실적", color="본부", markers=True)) if not monthly.empty else empty_fig()
        fig_b = chart_layout(px.bar(table.sort_values("달성률"), x="달성률", y="본부", orientation="h", color="달성률", color_continuous_scale="RdYlGn")) if not table.empty else empty_fig()
        return kpi, fig_a, fig_b, table_component(table), {"월별추세": monthly, "성과비교": table, "상세": table}

    if tab == "team":
        table = add_rate(df.groupby(["upp_dept", "dept"], dropna=False, as_index=False)[["sale_amount", "aim_amount"]].sum().rename(columns={"upp_dept": "본부", "dept": "팀", "sale_amount": "실적", "aim_amount": "목표"})).sort_values("실적", ascending=False)
        top_sale = table.nlargest(20, "실적").sort_values("실적")
        top_rate = table.dropna(subset=["달성률"]).nlargest(20, "달성률").sort_values("달성률")
        fig_a = chart_layout(px.bar(top_sale, x="실적", y="팀", orientation="h", color="본부")) if not top_sale.empty else empty_fig()
        fig_b = chart_layout(px.bar(top_rate, x="달성률", y="팀", orientation="h", color="달성률", color_continuous_scale="RdYlGn")) if not top_rate.empty else empty_fig()
        return kpi, fig_a, fig_b, table_component(table), {"실적Top20": top_sale, "달성률Top20": top_rate, "상세": table}

    if tab == "emp":
        top_emp = top_n(df, "employee_name", "sale_amount", 20).rename(columns={"employee_name": "담당자", "sale_amount": "실적"})
        monthly = df.groupby(["yyyymm", "employee_name"], dropna=False, as_index=False)["sale_amount"].sum().rename(columns={"yyyymm": "월", "employee_name": "담당자", "sale_amount": "실적"})
        table = add_rate(df.groupby(["upp_dept", "dept", "employee_name"], dropna=False, as_index=False)[["sale_amount", "aim_amount"]].sum().rename(columns={"upp_dept": "본부", "dept": "팀", "employee_name": "담당자", "sale_amount": "실적", "aim_amount": "목표"})).sort_values("실적", ascending=False)
        fig_a = chart_layout(px.bar(top_emp, x="실적", y="담당자", orientation="h")) if not top_emp.empty else empty_fig("선택 조건에 담당자 실적 데이터가 없습니다.")
        fig_b = chart_layout(px.line(monthly, x="월", y="실적", color="담당자", markers=True)) if not monthly.empty else empty_fig()
        return kpi, fig_a, fig_b, table_component(table), {"담당자Top20": top_emp, "월별추세": monthly, "상세": table}

    table = add_rate(df.groupby(["upp_dept", "dept", "hospital_name"], dropna=False, as_index=False)[["sale_amount", "aim_amount"]].sum().rename(columns={"upp_dept": "본부", "dept": "팀", "hospital_name": "거래처", "sale_amount": "실적", "aim_amount": "목표"})).sort_values("실적", ascending=False)
    top_hosp = top_n(df, "hospital_name", "sale_amount", 20).rename(columns={"hospital_name": "거래처", "sale_amount": "실적"})
    monthly = df.groupby(["yyyymm", "hospital_name"], dropna=False, as_index=False)["sale_amount"].sum().rename(columns={"yyyymm": "월", "hospital_name": "거래처", "sale_amount": "실적"})
    fig_a = chart_layout(px.bar(top_hosp, x="실적", y="거래처", orientation="h")) if not top_hosp.empty else empty_fig("선택 조건에 거래처 실적 데이터가 없습니다.")
    fig_b = chart_layout(px.line(monthly, x="월", y="실적", color="거래처", markers=True)) if not monthly.empty else empty_fig()
    return kpi, fig_a, fig_b, table_component(table), {"거래처Top20": top_hosp, "월별추세": monthly, "상세": table}


@app.callback(
    Output("summary-sales", "children"),
    Output("summary-aim", "children"),
    Output("summary-rate", "children"),
    Output("summary-rows", "children"),
    Output("summary-chart-a", "figure"),
    Output("summary-chart-b", "figure"),
    Output("summary-table", "children"),
    Input("summary-month", "value"),
    Input("summary-upp", "value"),
    Input("summary-dept", "value"),
)
def update_summary(months, upp, dept):
    kpi, fig_a, fig_b, table, _ = aggregate_for_tab("summary", months, upp, dept)
    return *kpi, fig_a, fig_b, table


@app.callback(
    Output("upp-sales", "children"), Output("upp-aim", "children"), Output("upp-rate", "children"), Output("upp-rows", "children"),
    Output("upp-chart-a", "figure"), Output("upp-chart-b", "figure"), Output("upp-table", "children"),
    Input("upp-month", "value"), Input("upp-upp", "value"), Input("upp-dept", "value"),
)
def update_upp(months, upp, dept):
    kpi, fig_a, fig_b, table, _ = aggregate_for_tab("upp", months, upp, dept)
    return *kpi, fig_a, fig_b, table


@app.callback(
    Output("team-sales", "children"), Output("team-aim", "children"), Output("team-rate", "children"), Output("team-rows", "children"),
    Output("team-chart-a", "figure"), Output("team-chart-b", "figure"), Output("team-table", "children"),
    Input("team-month", "value"), Input("team-upp", "value"), Input("team-dept", "value"),
)
def update_team(months, upp, dept):
    kpi, fig_a, fig_b, table, _ = aggregate_for_tab("team", months, upp, dept)
    return *kpi, fig_a, fig_b, table


@app.callback(
    Output("emp-sales", "children"), Output("emp-aim", "children"), Output("emp-rate", "children"), Output("emp-rows", "children"),
    Output("emp-chart-a", "figure"), Output("emp-chart-b", "figure"), Output("emp-table", "children"),
    Input("emp-month", "value"), Input("emp-upp", "value"), Input("emp-dept", "value"), Input("emp-emp", "value"),
)
def update_emp(months, upp, dept, emp):
    kpi, fig_a, fig_b, table, _ = aggregate_for_tab("emp", months, upp, dept, emp=emp)
    return *kpi, fig_a, fig_b, table


@app.callback(
    Output("hosp-sales", "children"), Output("hosp-aim", "children"), Output("hosp-rate", "children"), Output("hosp-rows", "children"),
    Output("hosp-chart-a", "figure"), Output("hosp-chart-b", "figure"), Output("hosp-table", "children"),
    Input("hosp-month", "value"), Input("hosp-upp", "value"), Input("hosp-dept", "value"), Input("hosp-hosp", "value"),
)
def update_hosp(months, upp, dept, hosp):
    kpi, fig_a, fig_b, table, _ = aggregate_for_tab("hosp", months, upp, dept, hosp=hosp)
    return *kpi, fig_a, fig_b, table


@app.callback(
    Output("drill-path", "children"),
    Output("drill-chart-a", "figure"),
    Output("drill-chart-b", "figure"),
    Output("drill-table", "children"),
    Input("drill-upp", "value"),
    Input("drill-dept", "value"),
    Input("drill-emp", "value"),
    Input("drill-hosp", "value"),
)
def update_drill(upp, dept, emp, hosp):
    df = query_fact(uppdepts=[upp] if upp else None, depts=[dept] if dept else None, employees=[emp] if emp else None, hospitals=[hosp] if hosp else None)
    monthly = df.groupby("yyyymm", as_index=False)["sale_amount"].sum().rename(columns={"yyyymm": "월", "sale_amount": "실적"})
    top_hosp = top_n(df, "hospital_name", "sale_amount", 20).rename(columns={"hospital_name": "거래처", "sale_amount": "실적"})
    detail = clean_table(display_fact(df).sort_values(["월", "실적"], ascending=[False, False]).head(MAX_ROWS))
    path = " → ".join([upp or "전체 본부", dept or "전체 팀", emp or "전체 담당자", hosp or "전체 거래처"])
    fig_a = chart_layout(px.area(monthly, x="월", y="실적", markers=True)) if not monthly.empty else empty_fig()
    fig_b = chart_layout(px.bar(top_hosp, x="실적", y="거래처", orientation="h")) if not top_hosp.empty else empty_fig("선택 조건에 거래처 실적 데이터가 없습니다.")
    return path, fig_a, fig_b, table_component(detail)


def download_for(tab, which, months, upp, dept, emp=None, hosp=None):
    _, _, _, _, downloads = aggregate_for_tab(tab, months, upp, dept, emp=emp, hosp=hosp)
    key = {"a": list(downloads.keys())[0], "b": list(downloads.keys())[1], "table": "상세"}[which]
    filename = f"{tab}_{key}.xlsx"
    return dcc.send_bytes(xlsx_bytes({key: clean_table(downloads[key])}), filename)


for tab in ["summary", "upp", "team"]:
    for which in ["a", "b", "table"]:
        app.callback(
            Output(f"{tab}-download-{which}-file", "data"),
            Input(f"{tab}-download-{which}", "n_clicks"),
            State(f"{tab}-month", "value"),
            State(f"{tab}-upp", "value"),
            State(f"{tab}-dept", "value"),
            prevent_initial_call=True,
        )(lambda n, months, upp, dept, tab=tab, which=which: download_for(tab, which, months, upp, dept))

for tab in ["emp"]:
    for which in ["a", "b", "table"]:
        app.callback(
            Output(f"{tab}-download-{which}-file", "data"),
            Input(f"{tab}-download-{which}", "n_clicks"),
            State(f"{tab}-month", "value"),
            State(f"{tab}-upp", "value"),
            State(f"{tab}-dept", "value"),
            State(f"{tab}-emp", "value"),
            prevent_initial_call=True,
        )(lambda n, months, upp, dept, emp, tab=tab, which=which: download_for(tab, which, months, upp, dept, emp=emp))

for tab in ["hosp"]:
    for which in ["a", "b", "table"]:
        app.callback(
            Output(f"{tab}-download-{which}-file", "data"),
            Input(f"{tab}-download-{which}", "n_clicks"),
            State(f"{tab}-month", "value"),
            State(f"{tab}-upp", "value"),
            State(f"{tab}-dept", "value"),
            State(f"{tab}-hosp", "value"),
            prevent_initial_call=True,
        )(lambda n, months, upp, dept, hosp, tab=tab, which=which: download_for(tab, which, months, upp, dept, hosp=hosp))


@app.callback(
    Output("drill-download-a-file", "data"),
    Input("drill-download-a", "n_clicks"),
    State("drill-upp", "value"), State("drill-dept", "value"), State("drill-emp", "value"), State("drill-hosp", "value"),
    prevent_initial_call=True,
)
def download_drill_monthly(n, upp, dept, emp, hosp):
    df = query_fact(uppdepts=[upp] if upp else None, depts=[dept] if dept else None, employees=[emp] if emp else None, hospitals=[hosp] if hosp else None)
    monthly = df.groupby("yyyymm", as_index=False)["sale_amount"].sum().rename(columns={"yyyymm": "월", "sale_amount": "실적"})
    return dcc.send_bytes(xlsx_bytes({"월별추세": monthly}), "drill_monthly.xlsx")


@app.callback(
    Output("drill-download-b-file", "data"),
    Input("drill-download-b", "n_clicks"),
    State("drill-upp", "value"), State("drill-dept", "value"), State("drill-emp", "value"), State("drill-hosp", "value"),
    prevent_initial_call=True,
)
def download_drill_hosp(n, upp, dept, emp, hosp):
    df = query_fact(uppdepts=[upp] if upp else None, depts=[dept] if dept else None, employees=[emp] if emp else None, hospitals=[hosp] if hosp else None)
    top_hosp = top_n(df, "hospital_name", "sale_amount", 20).rename(columns={"hospital_name": "거래처", "sale_amount": "실적"})
    return dcc.send_bytes(xlsx_bytes({"거래처Top20": top_hosp}), "drill_hospital_top20.xlsx")


@app.callback(
    Output("drill-download-table-file", "data"),
    Input("drill-download-table", "n_clicks"),
    State("drill-upp", "value"), State("drill-dept", "value"), State("drill-emp", "value"), State("drill-hosp", "value"),
    prevent_initial_call=True,
)
def download_drill_table(n, upp, dept, emp, hosp):
    df = query_fact(uppdepts=[upp] if upp else None, depts=[dept] if dept else None, employees=[emp] if emp else None, hospitals=[hosp] if hosp else None)
    return dcc.send_bytes(xlsx_bytes({"상세": clean_table(display_fact(df).head(MAX_ROWS))}), "drill_detail.xlsx")


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def available_port(host: str, start: int = 8050) -> int:
    for port in range(start, start + 50):
        if not port_open(host, port):
            return port
    raise RuntimeError("No local port available.")


if __name__ == "__main__":
    host = "127.0.0.1"
    port = available_port(host, 8050)
    threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    app.run(debug=False, host=host, port=port)
