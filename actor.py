from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None


AXIS_GRADE_ORDER = ["S2+", "S+", "S", "A", "B", "C", "D"]


# -----------------------------
# Config
# -----------------------------

def get_secret(path, default=None):
    cur = st.secrets
    for key in path:
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def get_config() -> Dict:
    return {
        "title": get_secret(["app", "title"], "배우 정량분석 결과 테이블"),
        "subtitle": get_secret(["app", "subtitle"], "RAW 시트를 기반으로 생산력·안정성·기여도를 계산한 결과 테이블"),
        "spreadsheet_id": get_secret(["data", "spreadsheet_id"], ""),
        "raw_sheet": get_secret(["data", "raw_sheet"], "RAW"),
        "weights": {
            "production": float(get_secret(["weights", "production"], 0.4)),
            "stability": float(get_secret(["weights", "stability"], 0.3)),
            "contribution": float(get_secret(["weights", "contribution"], 0.3)),
        },
        "axis_thresholds": {
            "S2+": float(get_secret(["axis_grading", "S2P"], 0.99)),
            "S+": float(get_secret(["axis_grading", "SP"], 0.96)),
            "S": float(get_secret(["axis_grading", "S"], 0.90)),
            "A": float(get_secret(["axis_grading", "A"], 0.70)),
            "B": float(get_secret(["axis_grading", "B"], 0.50)),
            "C": float(get_secret(["axis_grading", "C"], 0.20)),
        },
    }


# -----------------------------
# Google Sheets
# -----------------------------

def _get_gspread_client():
    if gspread is None or Credentials is None:
        raise ImportError("gspread 또는 google-auth 패키지가 설치되어 있지 않습니다.")

    svc = dict(st.secrets["gcp_service_account"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(svc, scopes=scopes)
    return gspread.authorize(creds)


@st.cache_data(show_spinner=False)
def load_raw_sheet(spreadsheet_id: str, raw_sheet: str) -> pd.DataFrame:
    client = _get_gspread_client()
    ws = client.open_by_key(spreadsheet_id).worksheet(raw_sheet)
    values = ws.get_all_values()
    if not values:
        raise ValueError("RAW 시트가 비어 있습니다.")

    header = [str(x).strip() for x in values[0]]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)

    # 빈 컬럼 / Unnamed 컬럼 제거
    keep_cols = []
    for c in df.columns:
        cs = str(c).strip()
        if not cs:
            continue
        if cs.lower().startswith("unnamed"):
            continue
        keep_cols.append(c)
    df = df[keep_cols].copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = ["인물명", "프로그램명", "드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"RAW 시트에 필요한 컬럼이 없습니다: {', '.join(missing)}")

    for col in ["드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹", "점유율"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    df["인물명"] = df["인물명"].astype(str).str.strip()
    df["프로그램명"] = df["프로그램명"].astype(str).str.strip()
    df = df[(df["인물명"] != "") & (df["프로그램명"] != "")].copy()

    if "점유율" not in df.columns:
        df["점유율"] = np.nan

    needs_share = df["점유율"].isna() | (df["점유율"] == 0)
    denom = df["드라마화제성"].replace(0, np.nan)
    df.loc[needs_share, "점유율"] = df.loc[needs_share, "배우화제성"] / denom.loc[needs_share]

    return df.reset_index(drop=True)


# -----------------------------
# Calculation
# -----------------------------

def percentrank_inc(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(valid) <= 1:
        out.loc[valid.index] = 1.0
        return out
    ranks = valid.rank(method="max")
    out.loc[valid.index] = (ranks - 1) / (len(valid) - 1)
    return out.clip(0, 1)


def normalize_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def axis_grade_from_pct(value: float, thresholds: Dict[str, float]) -> str:
    if pd.isna(value):
        return "D"
    if value >= thresholds["S2+"]:
        return "S2+"
    if value >= thresholds["S+"]:
        return "S+"
    if value >= thresholds["S"]:
        return "S"
    if value >= thresholds["A"]:
        return "A"
    if value >= thresholds["B"]:
        return "B"
    if value >= thresholds["C"]:
        return "C"
    return "D"


@st.cache_data(show_spinner=False)
def build_result_table(raw_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = raw_df.copy()
    df["점유율_calc"] = df["배우화제성"] / df["드라마화제성"].replace(0, np.nan)

    grouped = df.groupby("인물명", dropna=False)

    result = pd.DataFrame({"배우": grouped.size().index})
    result["드라마 화제성"] = grouped["드라마화제성"].sum().values
    result["배우 화제성"] = grouped["배우화제성"].sum().values
    result["출연작품"] = grouped["프로그램명"].nunique().values
    result["랭크주차"] = grouped["랭크인주차"].sum().values
    result["대표작 성과"] = grouped["배우화제성"].max().values

    work_map = grouped["프로그램명"].agg(lambda x: ", ".join(pd.Series(x).dropna().astype(str).drop_duplicates().tolist()))
    result["출연작"] = result["배우"].map(work_map)

    result["작품평균"] = result["배우 화제성"] / result["출연작품"].replace(0, np.nan)
    overall_work_avg = result["작품평균"].mean(skipna=True)
    prior_strength = 3.0
    result["보정 작품평균"] = (
        result["출연작품"] * result["작품평균"] + prior_strength * overall_work_avg
    ) / (result["출연작품"] + prior_strength)

    max_actor_buzz = grouped["배우화제성"].max()
    total_actor_buzz = grouped["배우화제성"].sum().replace(0, np.nan)
    hit_dispersion = 1 - (max_actor_buzz / total_actor_buzz)
    result["히트 분산지수"] = result["배우"].map(hit_dispersion).fillna(0)
    result["히트 분산정규화"] = normalize_minmax(result["히트 분산지수"])

    result["화제성 기여도"] = result["배우 화제성"] / result["드라마 화제성"].replace(0, np.nan)

    rank1 = grouped["작품내랭킹"].apply(lambda x: (pd.to_numeric(x, errors="coerce") == 1).sum())
    rank2 = grouped["작품내랭킹"].apply(lambda x: (pd.to_numeric(x, errors="coerce") == 2).sum())
    rank3 = grouped["작품내랭킹"].apply(lambda x: (pd.to_numeric(x, errors="coerce") == 3).sum())

    result["1위배율"] = result["배우"].map(rank1) / result["출연작품"].replace(0, np.nan)
    result["2위배율"] = result["배우"].map(rank2) / result["출연작품"].replace(0, np.nan)
    result["3위배율"] = result["배우"].map(rank3) / result["출연작품"].replace(0, np.nan)
    result[["1위배율", "2위배율", "3위배율"]] = result[["1위배율", "2위배율", "3위배율"]].fillna(0)

    # 축 계산
    result["생산력(백분율)"] = percentrank_inc(result["배우 화제성"])

    work_count_factor = result["출연작품"] / (result["출연작품"] + 2)
    weeks_per_work = result["랭크주차"] / result["출연작품"].replace(0, np.nan)
    weeks_factor = normalize_minmax(weeks_per_work.fillna(0))
    quality_factor = percentrank_inc(result["보정 작품평균"])
    dispersion_factor = 1 - result["히트 분산정규화"]
    result["꾸준함지수"] = (
        0.40 * quality_factor
        + 0.25 * weeks_factor
        + 0.20 * work_count_factor
        + 0.15 * dispersion_factor
    )
    result["안정성(백분율)"] = percentrank_inc(result["꾸준함지수"])

    weighted_rank_ratio = (result["1위배율"] * 1.0) + (result["2위배율"] * 0.5) + (result["3위배율"] * 0.3)
    contribution_base = (
        0.55 * percentrank_inc(result["화제성 기여도"])
        + 0.30 * weighted_rank_ratio
        + 0.15 * percentrank_inc(result["대표작 성과"])
    )
    result["보정기여도"] = contribution_base
    result["기여도(백분율)"] = percentrank_inc(result["보정기여도"])

    result["생산력"] = result["생산력(백분율)"].apply(lambda x: axis_grade_from_pct(x, cfg["axis_thresholds"]))
    result["안정성"] = result["안정성(백분율)"].apply(lambda x: axis_grade_from_pct(x, cfg["axis_thresholds"]))
    result["기여도"] = result["기여도(백분율)"].apply(lambda x: axis_grade_from_pct(x, cfg["axis_thresholds"]))

    result["합산점수"] = (
        result["생산력(백분율)"] * cfg["weights"]["production"]
        + result["안정성(백분율)"] * cfg["weights"]["stability"]
        + result["기여도(백분율)"] * cfg["weights"]["contribution"]
    ) * 100

    result["생산력(백분율)"] = result["생산력(백분율)"] * 100
    result["안정성(백분율)"] = result["안정성(백분율)"] * 100
    result["기여도(백분율)"] = result["기여도(백분율)"] * 100
    result["화제성 기여도"] = result["화제성 기여도"] * 100

    result = result.sort_values(["합산점수", "배우 화제성"], ascending=[False, False]).reset_index(drop=True)
    result.insert(0, "#", np.arange(1, len(result) + 1))

    ordered_cols = [
        "#", "배우", "생산력", "안정성", "기여도", "합산점수",
        "생산력(백분율)", "안정성(백분율)", "기여도(백분율)",
        "드라마 화제성", "배우 화제성", "출연작품", "랭크주차",
        "작품평균", "보정 작품평균", "화제성 기여도", "대표작 성과", "출연작"
    ]
    return result[ordered_cols].copy()


# -----------------------------
# UI
# -----------------------------

def main():
    st.set_page_config(page_title="배우 정량분석 결과 테이블", layout="wide")
    cfg = get_config()

    st.title(cfg["title"])
    st.caption(cfg["subtitle"])

    if not cfg["spreadsheet_id"]:
        st.error("secrets.toml의 [data].spreadsheet_id 값이 비어 있습니다.")
        st.stop()

    try:
        raw_df = load_raw_sheet(cfg["spreadsheet_id"], cfg["raw_sheet"])
        result_df = build_result_table(raw_df, cfg)
    except Exception as e:
        st.error(f"데이터 로드 또는 계산 중 오류가 발생했습니다: {e}")
        st.stop()

    with st.sidebar:
        st.subheader("필터")
        keyword = st.text_input("배우 검색", "").strip()
        selected_prod = st.multiselect("생산력 등급", AXIS_GRADE_ORDER)
        selected_stab = st.multiselect("안정성 등급", AXIS_GRADE_ORDER)
        selected_cont = st.multiselect("기여도 등급", AXIS_GRADE_ORDER)
        min_score, max_score = st.slider(
            "합산점수 범위",
            min_value=0.0,
            max_value=100.0,
            value=(0.0, 100.0),
            step=0.01,
        )

    view_df = result_df.copy()
    if keyword:
        view_df = view_df[view_df["배우"].str.contains(keyword, case=False, na=False)]
    if selected_prod:
        view_df = view_df[view_df["생산력"].isin(selected_prod)]
    if selected_stab:
        view_df = view_df[view_df["안정성"].isin(selected_stab)]
    if selected_cont:
        view_df = view_df[view_df["기여도"].isin(selected_cont)]
    view_df = view_df[(view_df["합산점수"] >= min_score) & (view_df["합산점수"] <= max_score)]

    c1, c2, c3 = st.columns(3)
    c1.metric("배우 수", f"{len(view_df):,}")
    c2.metric("RAW 행 수", f"{len(raw_df):,}")
    c3.metric("전체 배우 수", f"{len(result_df):,}")

    st.dataframe(
        view_df,
        use_container_width=True,
        hide_index=True,
        height=820,
        column_config={
            "#": st.column_config.NumberColumn("#", format="%d", width="small"),
            "배우": st.column_config.TextColumn("배우", width="medium"),
            "생산력": st.column_config.TextColumn("생산력", width="small"),
            "안정성": st.column_config.TextColumn("안정성", width="small"),
            "기여도": st.column_config.TextColumn("기여도", width="small"),
            "합산점수": st.column_config.NumberColumn("합산점수", format="%,.2f"),
            "생산력(백분율)": st.column_config.NumberColumn("생산력(백분율)", format="%.2f%%", help="0~100%"),
            "안정성(백분율)": st.column_config.NumberColumn("안정성(백분율)", format="%.2f%%", help="0~100%"),
            "기여도(백분율)": st.column_config.NumberColumn("기여도(백분율)", format="%.2f%%", help="0~100%"),
            "드라마 화제성": st.column_config.NumberColumn("드라마 화제성", format="%,.2f"),
            "배우 화제성": st.column_config.NumberColumn("배우 화제성", format="%,.2f"),
            "출연작품": st.column_config.NumberColumn("출연작품", format="%d"),
            "랭크주차": st.column_config.NumberColumn("랭크주차", format="%,.2f"),
            "작품평균": st.column_config.NumberColumn("작품평균", format="%,.2f"),
            "보정 작품평균": st.column_config.NumberColumn("보정 작품평균", format="%,.2f"),
            "화제성 기여도": st.column_config.NumberColumn("화제성 기여도", format="%.2f%%", help="0~100%"),
            "대표작 성과": st.column_config.NumberColumn("대표작 성과", format="%,.2f"),
            "출연작": st.column_config.TextColumn("출연작", width="large"),
        },
    )

    csv = view_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "결과 테이블 CSV 다운로드",
        data=csv,
        file_name="actor_quant_result_table.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
