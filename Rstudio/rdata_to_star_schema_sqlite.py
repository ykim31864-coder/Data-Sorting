from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyreadr


DEFAULT_RDATA = Path(r"C:\Users\user\AppData\Local\Temp\실적장표_집계_원내외추가.RData")
DEFAULT_DB = Path(r"C:\Users\user\Documents\Codex\2026-06-09\files-mentioned-by-the-user-r-2\outputs\sales_star_schema.sqlite")


def clean_text(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype) == "category":
            out[col] = out[col].astype(str).replace({"nan": None})
        elif out[col].dtype == object:
            out[col] = out[col].map(clean_text)
    return out.replace({np.nan: None})


def r_date_to_timestamp(values: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    return pd.to_datetime(numeric_values, unit="D", origin="1970-01-01", errors="coerce")


def date_key(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%Y%m%d").astype("Int64")


def month_date_key(year: int | str, month: int) -> int:
    return int(f"{int(year):04d}{month:02d}01")


def make_keyed_dimension(df: pd.DataFrame, key_name: str, sort_cols: list[str]) -> pd.DataFrame:
    dim = df.drop_duplicates().sort_values(sort_cols, na_position="last").reset_index(drop=True)
    dim.insert(0, key_name, range(1, len(dim) + 1))
    return dim


def merge_key(df: pd.DataFrame, dim: pd.DataFrame, key_name: str, on: list[str]) -> pd.DataFrame:
    return df.merge(dim[[key_name] + on], on=on, how="left")


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS fact_sales_performance;
        DROP TABLE IF EXISTS bridge_hospital_employee_year;
        DROP TABLE IF EXISTS dim_date;
        DROP TABLE IF EXISTS dim_org;
        DROP TABLE IF EXISTS dim_region;
        DROP TABLE IF EXISTS dim_hospital;
        DROP TABLE IF EXISTS dim_product;
        DROP TABLE IF EXISTS dim_employee;

        CREATE TABLE dim_date (
            date_key INTEGER PRIMARY KEY,
            date_value TEXT,
            year INTEGER NOT NULL,
            quarter INTEGER NOT NULL,
            month INTEGER NOT NULL,
            yyyymm TEXT NOT NULL
        );

        CREATE TABLE dim_org (
            org_key INTEGER PRIMARY KEY,
            upp_dept TEXT,
            dept TEXT,
            charge_team TEXT
        );

        CREATE TABLE dim_region (
            region_key INTEGER PRIMARY KEY,
            region_class TEXT,
            hospital_cat TEXT
        );

        CREATE TABLE dim_hospital (
            hospital_key INTEGER PRIMARY KEY,
            hospital_name TEXT NOT NULL,
            hospital_cat TEXT,
            bed_qty REAL,
            dr REAL,
            ld REAL,
            charge_team TEXT
        );

        CREATE TABLE dim_product (
            product_key INTEGER PRIMARY KEY,
            std_criteria TEXT,
            brand TEXT,
            dc_brand TEXT,
            detail_brand TEXT
        );

        CREATE TABLE dim_employee (
            employee_key INTEGER PRIMARY KEY,
            employee_name TEXT,
            mr TEXT,
            valid_year INTEGER
        );

        CREATE TABLE bridge_hospital_employee_year (
            hospital_key INTEGER NOT NULL,
            employee_key INTEGER NOT NULL,
            year INTEGER NOT NULL,
            PRIMARY KEY (hospital_key, employee_key, year)
        );

        CREATE TABLE fact_sales_performance (
            sales_fact_key INTEGER PRIMARY KEY AUTOINCREMENT,
            date_key INTEGER NOT NULL,
            org_key INTEGER,
            region_key INTEGER,
            hospital_key INTEGER,
            product_key INTEGER,
            employee_key INTEGER,
            sale_amount REAL,
            aim_amount REAL,
            qty REAL,
            achievement_rate REAL,
            growth_rate REAL,
            source_object TEXT,
            load_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX idx_fact_date ON fact_sales_performance(date_key);
        CREATE INDEX idx_fact_org ON fact_sales_performance(org_key);
        CREATE INDEX idx_fact_hospital ON fact_sales_performance(hospital_key);
        CREATE INDEX idx_fact_product ON fact_sales_performance(product_key);
        CREATE INDEX idx_fact_employee ON fact_sales_performance(employee_key);
        """
    )


def collect_month_keys_from_columns(columns: Iterable[str]) -> list[int]:
    keys = []
    for col in columns:
        if len(col) >= 7 and col[:4].isdigit() and col[4] == "-" and col[5:7].isdigit():
            keys.append(int(col[:4] + col[5:7] + "01"))
    return keys


def build_database(rdata_path: Path, db_path: Path, include_item_hospital: bool) -> None:
    result = pyreadr.read_r(str(rdata_path))
    data = {name: normalize_frame(df) for name, df in result.items() if isinstance(df, pd.DataFrame)}

    sale_aim = data["sale_aim"]
    sale_aim_dates = r_date_to_timestamp(pd.Series(sale_aim["date"]))
    date_keys = set(date_key(sale_aim_dates).dropna().astype(int).tolist())
    date_keys.update(collect_month_keys_from_columns(data["mr_hosp_sale"].columns))

    if include_item_hospital:
        for year in data["group_hosp_item_aggr"]["Year"].dropna().unique():
            if str(year).isdigit():
                date_keys.update(month_date_key(year, month) for month in range(1, 13))

    dim_date = pd.DataFrame({"date_key": sorted(date_keys)})
    dim_date["date_value"] = pd.to_datetime(dim_date["date_key"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    dim_date["year"] = dim_date["date_key"].astype(str).str[:4].astype(int)
    dim_date["month"] = dim_date["date_key"].astype(str).str[4:6].astype(int)
    dim_date["quarter"] = ((dim_date["month"] - 1) // 3 + 1).astype(int)
    dim_date["yyyymm"] = dim_date["date_key"].astype(str).str[:4] + "-" + dim_date["date_key"].astype(str).str[4:6]

    org_parts = [
        sale_aim.rename(columns={"upp_dept": "upp_dept"})[["upp_dept", "dept"]].assign(charge_team=None),
        data["mr_hosp_sale"][["UppDept", "Dept"]].rename(columns={"UppDept": "upp_dept", "Dept": "dept"}).assign(charge_team=None),
    ]
    if "team_hospital_input" in data:
        org_parts.append(
            data["team_hospital_input"][["UppDeptName", "Charge_team"]]
            .rename(columns={"UppDeptName": "upp_dept", "Charge_team": "charge_team"})
            .assign(dept=None)[["upp_dept", "dept", "charge_team"]]
        )
    dim_org = make_keyed_dimension(pd.concat(org_parts, ignore_index=True), "org_key", ["upp_dept", "dept", "charge_team"])

    region_parts = [
        sale_aim[["region_class"]].assign(hospital_cat=None),
    ]
    if "hosp_aggr" in data:
        region_parts.append(data["hosp_aggr"][["HospitalCat"]].rename(columns={"HospitalCat": "hospital_cat"}).assign(region_class=None))
    dim_region = make_keyed_dimension(pd.concat(region_parts, ignore_index=True), "region_key", ["region_class", "hospital_cat"])

    hospital_parts = []
    if "hosp_info" in data:
        hospital_parts.append(
            data["hosp_info"].rename(
                columns={
                    "HospitalName": "hospital_name",
                    "BedQty": "bed_qty",
                    "LD": "ld",
                    "Charge_team": "charge_team",
                }
            )[["hospital_name", "bed_qty", "dr", "ld", "charge_team"]].assign(hospital_cat=None)
        )
    hospital_parts.append(
        data["mr_hosp_sale"][["거래처"]].rename(columns={"거래처": "hospital_name"}).assign(
            hospital_cat=None, bed_qty=None, dr=None, ld=None, charge_team=None
        )
    )
    if include_item_hospital:
        hospital_parts.append(
            data["group_hosp_item_aggr"][["HospitalName"]].rename(columns={"HospitalName": "hospital_name"}).assign(
                hospital_cat=None, bed_qty=None, dr=None, ld=None, charge_team=None
            )
        )
    dim_hospital_raw = pd.concat(hospital_parts, ignore_index=True)
    dim_hospital_raw["hospital_name"] = dim_hospital_raw["hospital_name"].fillna("UNKNOWN")
    dim_hospital_raw = dim_hospital_raw.sort_values(["hospital_name", "bed_qty"], na_position="last")
    dim_hospital_raw = dim_hospital_raw.drop_duplicates(subset=["hospital_name"], keep="first")
    dim_hospital = make_keyed_dimension(dim_hospital_raw, "hospital_key", ["hospital_name"])

    product_parts = [
        pd.DataFrame([{"std_criteria": None, "brand": None, "dc_brand": None, "detail_brand": None}])
    ]
    if include_item_hospital:
        product_parts.append(
            data["group_hosp_item_aggr"][["StdCriteria", "Brand", "DCBrand"]].rename(
                columns={"StdCriteria": "std_criteria", "Brand": "brand", "DCBrand": "dc_brand"}
            ).assign(detail_brand=None)
        )
    if "detail_brand" in data:
        product_parts.append(data["detail_brand"].rename(columns={"detail_brand": "detail_brand"}).assign(std_criteria=None, brand=None, dc_brand=None))
    dim_product = make_keyed_dimension(pd.concat(product_parts, ignore_index=True), "product_key", ["brand", "dc_brand", "std_criteria", "detail_brand"])

    employee_parts = [
        data["mr_hosp_sale"][["MR"]].rename(columns={"MR": "employee_name"}).assign(mr=lambda x: x["employee_name"], valid_year=2026)
    ]
    if "Charge_emp" in data:
        employee_parts.append(data["Charge_emp"][["담당자"]].rename(columns={"담당자": "employee_name"}).assign(mr=None, valid_year=None))
    if "hosp_emp" in data:
        for year in ["2023", "2024", "2025", "2026"]:
            employee_parts.append(data["hosp_emp"][[year]].rename(columns={year: "employee_name"}).assign(mr=None, valid_year=int(year)))
    dim_employee_raw = pd.concat(employee_parts, ignore_index=True)
    dim_employee_raw = dim_employee_raw.dropna(subset=["employee_name"]).drop_duplicates(subset=["employee_name", "valid_year"], keep="first")
    dim_employee = make_keyed_dimension(dim_employee_raw, "employee_key", ["employee_name", "valid_year"])

    fact_parts = []

    sale_fact = sale_aim.copy()
    sale_fact["date_value"] = r_date_to_timestamp(pd.Series(sale_fact["date"]))
    sale_fact["date_key"] = date_key(sale_fact["date_value"]).astype("Int64")
    sale_fact = sale_fact.rename(columns={"upp_dept": "upp_dept", "sale": "sale_amount", "aim": "aim_amount"})
    sale_fact = merge_key(sale_fact, dim_org, "org_key", ["upp_dept", "dept", "charge_team"]) if "charge_team" in sale_fact.columns else sale_fact.assign(charge_team=None).merge(dim_org[["org_key", "upp_dept", "dept", "charge_team"]], on=["upp_dept", "dept", "charge_team"], how="left")
    sale_fact = sale_fact.merge(dim_region[["region_key", "region_class", "hospital_cat"]], on=["region_class", "hospital_cat"], how="left") if "hospital_cat" in sale_fact.columns else sale_fact.assign(hospital_cat=None).merge(dim_region[["region_key", "region_class", "hospital_cat"]], on=["region_class", "hospital_cat"], how="left")
    sale_fact["achievement_rate"] = sale_fact["sale_amount"] / sale_fact["aim_amount"]
    sale_fact["growth_rate"] = None
    sale_fact["qty"] = None
    sale_fact["hospital_key"] = None
    sale_fact["product_key"] = None
    sale_fact["employee_key"] = None
    sale_fact["source_object"] = "sale_aim"
    fact_parts.append(sale_fact[["date_key", "org_key", "region_key", "hospital_key", "product_key", "employee_key", "sale_amount", "aim_amount", "qty", "achievement_rate", "growth_rate", "source_object"]])

    mr = data["mr_hosp_sale"]
    monthly_rows = []
    for _, row in mr.iterrows():
        for month in range(1, 13):
            ym = f"2026-{month:02d}"
            monthly_rows.append({
                "date_key": int(f"2026{month:02d}01"),
                "upp_dept": row["UppDept"],
                "dept": row["Dept"],
                "employee_name": row["MR"],
                "hospital_name": row["거래처"],
                "aim_amount": row.get(f"{ym} 목표"),
                "sale_amount": row.get(f"{ym} 실적"),
                "achievement_rate": row.get(f"{ym} 달성률"),
                "growth_rate": row.get(f"{ym} 성장률"),
            })
    mr_fact = pd.DataFrame(monthly_rows).assign(charge_team=None, region_key=None, product_key=None, qty=None, source_object="mr_hosp_sale")
    mr_fact = mr_fact.merge(dim_org[["org_key", "upp_dept", "dept", "charge_team"]], on=["upp_dept", "dept", "charge_team"], how="left")
    mr_fact = mr_fact.merge(dim_hospital[["hospital_key", "hospital_name"]], on="hospital_name", how="left")
    mr_fact = mr_fact.merge(dim_employee[["employee_key", "employee_name", "valid_year"]], left_on=["employee_name"], right_on=["employee_name"], how="left")
    mr_fact = mr_fact[(mr_fact["valid_year"].isna()) | (mr_fact["valid_year"].eq(2026))]
    fact_parts.append(mr_fact[["date_key", "org_key", "region_key", "hospital_key", "product_key", "employee_key", "sale_amount", "aim_amount", "qty", "achievement_rate", "growth_rate", "source_object"]])

    if include_item_hospital:
        item = data["group_hosp_item_aggr"]
        item_rows = []
        for _, row in item.iterrows():
            year = row["Year"]
            if not str(year).isdigit():
                continue
            for month in range(1, 13):
                item_rows.append({
                    "date_key": month_date_key(year, month),
                    "upp_dept": row["UppDept"],
                    "dept": row["Dept"],
                    "hospital_name": row["HospitalName"],
                    "std_criteria": row["StdCriteria"],
                    "brand": row["Brand"],
                    "dc_brand": row["DCBrand"],
                    "qty": row.get(f"{month}월 수량"),
                    "sale_amount": row.get(f"{month}월 실적"),
                    "growth_rate": row.get(f"{month}월 성장률"),
                })
        item_fact = pd.DataFrame(item_rows).assign(charge_team=None, region_key=None, employee_key=None, aim_amount=None, achievement_rate=None, detail_brand=None, source_object="group_hosp_item_aggr")
        item_fact = item_fact.merge(dim_org[["org_key", "upp_dept", "dept", "charge_team"]], on=["upp_dept", "dept", "charge_team"], how="left")
        item_fact = item_fact.merge(dim_hospital[["hospital_key", "hospital_name"]], on="hospital_name", how="left")
        item_fact = item_fact.merge(dim_product[["product_key", "std_criteria", "brand", "dc_brand", "detail_brand"]], on=["std_criteria", "brand", "dc_brand", "detail_brand"], how="left")
        fact_parts.append(item_fact[["date_key", "org_key", "region_key", "hospital_key", "product_key", "employee_key", "sale_amount", "aim_amount", "qty", "achievement_rate", "growth_rate", "source_object"]])

    fact = pd.concat(fact_parts, ignore_index=True).replace({np.nan: None})

    bridge = pd.DataFrame(columns=["hospital_key", "employee_key", "year"])
    if "hosp_emp" in data:
        bridge_rows = []
        hosp_emp = data["hosp_emp"].rename(columns={"거래처": "hospital_name"})
        for _, row in hosp_emp.iterrows():
            for year in ["2023", "2024", "2025", "2026"]:
                if row.get(year):
                    bridge_rows.append({"hospital_name": row["hospital_name"], "employee_name": row[year], "year": int(year)})
        bridge = pd.DataFrame(bridge_rows)
        if not bridge.empty:
            bridge = bridge.merge(dim_hospital[["hospital_key", "hospital_name"]], on="hospital_name", how="left")
            bridge = bridge.merge(dim_employee[["employee_key", "employee_name", "valid_year"]], left_on=["employee_name", "year"], right_on=["employee_name", "valid_year"], how="left")
            bridge = bridge.dropna(subset=["hospital_key", "employee_key"])[["hospital_key", "employee_key", "year"]].drop_duplicates()

    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        create_tables(conn)
        dim_date.to_sql("dim_date", conn, if_exists="append", index=False)
        dim_org.to_sql("dim_org", conn, if_exists="append", index=False)
        dim_region.to_sql("dim_region", conn, if_exists="append", index=False)
        dim_hospital.to_sql("dim_hospital", conn, if_exists="append", index=False)
        dim_product.to_sql("dim_product", conn, if_exists="append", index=False)
        dim_employee.to_sql("dim_employee", conn, if_exists="append", index=False)
        bridge.to_sql("bridge_hospital_employee_year", conn, if_exists="append", index=False)
        fact.to_sql("fact_sales_performance", conn, if_exists="append", index=False, chunksize=100_000)
        conn.executescript(
            """
            DROP VIEW IF EXISTS vw_uppdept_month;
            CREATE VIEW vw_uppdept_month AS
            SELECT
                d.yyyymm,
                o.upp_dept,
                SUM(f.sale_amount) AS sale_amount,
                SUM(f.aim_amount) AS aim_amount,
                CASE WHEN SUM(f.aim_amount) = 0 THEN NULL ELSE SUM(f.sale_amount) / SUM(f.aim_amount) END AS achievement_rate
            FROM fact_sales_performance f
            JOIN dim_date d ON f.date_key = d.date_key
            LEFT JOIN dim_org o ON f.org_key = o.org_key
            GROUP BY d.yyyymm, o.upp_dept;
            """
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RData sales tables to a SQLite star schema.")
    parser.add_argument("--rdata", type=Path, default=DEFAULT_RDATA)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--include-item-hospital", action="store_true", help="Also load group_hosp_item_aggr monthly item/hospital fact rows. This can create millions of rows.")
    args = parser.parse_args()
    build_database(args.rdata, args.db, args.include_item_hospital)
    print(args.db)


if __name__ == "__main__":
    main()
