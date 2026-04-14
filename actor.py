import math
from typing import Dict

import gspread
import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="배우 정량분석 결과 테이블", layout="wide")

RAW_REQUIRED_COLUMNS = [
    "인물명",
    "프로그램명",
    "드라마화제성",
    "배우화제성",
    "랭크인주차",
    "랭크인배우수",
    "작품내랭킹",
]

DEFAULT_WEIGHTS = {
    "production": 0.4,
    "stability": 0.3,
    "contribution": 0.3,
}

DEFAULT_AXIS_THRESHOLDS = {
    "S2+": 0.99,
    "S+": 0.96,
    "S": 0.90,
    "A": 0.70,
    "B": 0.50,
    "C": 0.20,
}


def get_secret_section(name: str) -> Dict:
    try:
        return dict(st.secrets[name])
    except Exception:
        return {}


def get_gspread_client():
    sa = get_secret_section("gcp_service_account")
    if not sa:
        st.error("Secrets에 [gcp_service_account] 설정이 없습니다.")
        st.stop()

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    return gspread.authorize(creds)


@st.cache_data(ttl=600)
def load_raw_from_gsheet() -> pd.DataFrame:
    data_cfg = get_secret_section("data")
    spreadsheet_id = data_cfg.get("spreadsheet_id", "").strip()
    raw_sheet = data_cfg.get("raw_sheet", "RAW").strip() or "RAW"

    if not spreadsheet_id:
        st.error("Secrets의 [data].spreadsheet_id 값이 없습니다.")
        st.stop()

    gc = get_gspread_client()
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(raw_sheet)

    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame(columns=RAW_REQUIRED_COLUMNS)

    header = values[0]
    rows = values[1:]

    fixed_header = []
    for idx, col in enumerate(header, start=1):
        name = (col or "").strip()
        fixed_header.append(name if name else f"unnamed_{idx}")

    max_len = len(fixed_header)
    normalized_rows = []
    for row in rows:
        row = list(row)
        if len(row) < max_len:
            row += [""] * (max_len - len(row))
        normalized_rows.append(row[:max_len])

    df = pd.DataFrame(normalized_rows, columns=fixed_header)

    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"RAW 시트에 필요한 컬럼이 없습니다: {missing}")
        st.stop()

    use_cols = [
        c
        for c in [
            "인물명",
            "프로그램명",
            "드라마화제성",
            "배우화제성",
            "랭크인주차",
            "랭크인배우수",
            "작품내랭킹",
            "점유율",
        ]
        if c in df.columns
    ]
    df = df[use_cols].copy()

    df["인물명"] = df["인물명"].astype(str).str.strip()
    df["프로그램명"] = df["프로그램명"].astype(str).str.strip()

    for col in ["드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹", "점유율"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"": np.nan, "None": np.nan, "nan": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["인물명"].notna() & (df["인물명"] != "")].copy()
    df = df[df["프로그램명"].notna() & (df["프로그램명"] != "")].copy()

    return df


# Google Sheets PERCENTRANK.INC 결과에 가장 가깝게 맞추기 위해 tie 처리 min 사용
def percentrank_inc_min(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if len(s) <= 1:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s.rank(method="min") - 1) / (len(s) - 1)


def axis_grade(p: float, thresholds: Dict[str, float]) -> str:
    if pd.isna(p):
        return ""
    for grade in ["S2+", "S+", "S", "A", "B", "C"]:
        if p >= thresholds[grade]:
            return grade
    return "D"


@st.cache_data(ttl=600)
def build_result_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    weights = DEFAULT_WEIGHTS.copy()
    weights.update(get_secret_section("weights"))

    thresholds = DEFAULT_AXIS_THRESHOLDS.copy()
    thresholds.update(get_secret_section("axis_grading"))

    df = raw_df.copy()
    df["r1"] = (df["작품내랭킹"] == 1).astype(float)
    df["r2"] = (df["작품내랭킹"] == 2).astype(float) * 0.5
    df["r3"] = (df["작품내랭킹"] == 3).astype(float) * 0.3

    grouped = df.groupby("인물명", sort=False)

    result = pd.DataFrame(index=grouped.size().index)
    result["배우"] = result.index
    result["드라마 화제성"] = grouped["드라마화제성"].sum()
    result["배우 화제성"] = grouped["배우화제성"].sum()
    result["출연작품"] = grouped.size().astype(float)
    result["랭크주차"] = grouped["랭크인주차"].sum()
    result["대표작 성과"] = grouped["배우화제성"].max()
    result["출연작"] = grouped["프로그램명"].apply(lambda s: ", ".join(pd.unique(s.dropna().astype(str))))

    result["작품평균"] = result["배우 화제성"] / result["출연작품"]
    global_avg = result["작품평균"].mean()
    result["보정 작품평균"] = (result["배우 화제성"] + 3 * global_avg) / (result["출연작품"] + 3)

    result["히트 분산지수"] = 1 - grouped["배우화제성"].max() / result["배우 화제성"]
    r_min = result["히트 분산지수"].min()
    r_max = result["히트 분산지수"].max()
    if pd.isna(r_min) or pd.isna(r_max) or math.isclose(r_min, r_max):
        result["히트 분산정규화"] = 0.0
    else:
        result["히트 분산정규화"] = (result["히트 분산지수"] - r_min) / (r_max - r_min)

    result["화제성 기여도"] = result["배우 화제성"] / result["드라마 화제성"]
    result["1위배율"] = grouped["r1"].sum() / result["출연작품"]
    result["2위배율"] = grouped["r2"].sum() / result["출연작품"]
    result["3위배율"] = grouped["r3"].sum() / result["출연작품"]

    result["대표작 성과백분위"] = percentrank_inc_min(result["대표작 성과"])

    p_min = result["보정 작품평균"].min()
    p_max = result["보정 작품평균"].max()
    if pd.isna(p_min) or pd.isna(p_max) or math.isclose(p_min, p_max):
        p_norm = pd.Series(0.0, index=result.index)
    else:
        p_norm = (result["보정 작품평균"] - p_min) / (p_max - p_min)

    result["꾸준함지수"] = np.minimum(
        1.0,
        0.25 * p_norm
        + 0.55 * (0.7 * result["히트 분산정규화"] + 0.3 * (result["출연작품"] / (result["출연작품"] + 2)))
        + 0.2 * (result["대표작 성과백분위"] ** 3),
    )

    result["보정기여도"] = 0.5 * p_norm + 0.5 * (
        result["화제성 기여도"] * (result["1위배율"] + result["2위배율"] + result["3위배율"])
    )

    # 축별 백분율
    result["생산력(백분율)"] = percentrank_inc_min(result["배우 화제성"])
    result["안정성(백분율)"] = percentrank_inc_min(result["꾸준함지수"])
    result["기여도(백분율)"] = percentrank_inc_min(result["보정기여도"])

    # 축별 등급
    result["생산력"] = result["생산력(백분율)"].apply(lambda x: axis_grade(x, thresholds))
    result["안정성"] = result["안정성(백분율)"].apply(lambda x: axis_grade(x, thresholds))
    result["기여도"] = result["기여도(백분율)"].apply(lambda x: axis_grade(x, thresholds))

    # 합산은 점수만 표시
    result["합산점수"] = 100 * (
        result["생산력(백분율)"] * float(weights.get("production", 0.4))
        + result["안정성(백분율)"] * float(weights.get("stability", 0.3))
        + result["기여도(백분율)"] * float(weights.get("contribution", 0.3))
    )

    result = result.reset_index(drop=True)
    result.insert(0, "#", np.arange(1, len(result) + 1))

    column_order = [
        "#",
        "배우",
        "생산력",
        "안정성",
        "기여도",
        "합산점수",
        "생산력(백분율)",
        "안정성(백분율)",
        "기여도(백분율)",
        "드라마 화제성",
        "배우 화제성",
        "출연작품",
        "꾸준함지수",
        "보정 작품평균",
        "작품평균",
        "히트 분산지수",
        "히트 분산정규화",
        "보정기여도",
        "화제성 기여도",
        "1위배율",
        "2위배율",
        "3위배율",
        "랭크주차",
        "대표작 성과",
        "대표작 성과백분위",
        "출연작",
    ]
    return result[column_order]


def format_display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    percent_cols = [
        "생산력(백분율)",
        "안정성(백분율)",
        "기여도(백분율)",
        "화제성 기여도",
        "1위배율",
        "2위배율",
        "3위배율",
        "대표작 성과백분위",
    ]

    numeric_cols = [
        "합산점수",
        "드라마 화제성",
        "배우 화제성",
        "출연작품",
        "꾸준함지수",
        "보정 작품평균",
        "작품평균",
        "히트 분산지수",
        "히트 분산정규화",
        "보정기여도",
        "랭크주차",
        "대표작 성과",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:,.2f}"
            )

    for col in percent_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x * 100:,.2f}%"
            )

    if "#" in out.columns:
        out["#"] = out["#"].astype(int).astype(str)

    return out


def main():
    app_cfg = get_secret_section("app")
    st.title(app_cfg.get("title", "배우 정량분석 결과 테이블"))
    st.caption(app_cfg.get("subtitle", "RAW 기반 계산 결과를 먼저 검증하는 화면"))

    raw_df = load_raw_from_gsheet()
    result_df = build_result_table(raw_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("RAW 행 수", f"{len(raw_df):,}")
    c2.metric("배우 수", f"{result_df['배우'].nunique():,}")
    c3.metric("프로그램 수", f"{raw_df['프로그램명'].nunique():,}")

    with st.sidebar:
        st.subheader("필터")
        keyword = st.text_input("배우 검색", "")
        selected_prod = st.multiselect("생산력 등급", options=sorted(result_df["생산력"].dropna().unique().tolist(), reverse=True))
        selected_stab = st.multiselect("안정성 등급", options=sorted(result_df["안정성"].dropna().unique().tolist(), reverse=True))
        selected_cont = st.multiselect("기여도 등급", options=sorted(result_df["기여도"].dropna().unique().tolist(), reverse=True))

    filtered = result_df.copy()
    if keyword.strip():
        filtered = filtered[filtered["배우"].str.contains(keyword.strip(), case=False, na=False)]
    if selected_prod:
        filtered = filtered[filtered["생산력"].isin(selected_prod)]
    if selected_stab:
        filtered = filtered[filtered["안정성"].isin(selected_stab)]
    if selected_cont:
        filtered = filtered[filtered["기여도"].isin(selected_cont)]

    filtered = filtered.sort_values(["합산점수", "배우 화제성"], ascending=[False, False]).reset_index(drop=True)
    filtered["#"] = np.arange(1, len(filtered) + 1)

    st.subheader("결과 테이블")
    st.caption("합산등급은 제외하고, 축별 등급과 합산점수만 표시합니다.")
    st.dataframe(format_display_table(filtered), use_container_width=True, height=760, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "결과 테이블 CSV 다운로드",
        data=csv,
        file_name="actor_quant_result_table.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
