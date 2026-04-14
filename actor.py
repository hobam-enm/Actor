from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None


GRADE_ORDER = ["TS", "T1", "T2", "T3", "M1", "M2", "M3", "B1", "B2", "B3"]
TIER_ORDER = ["Top", "Mid", "Base"]
AXIS_GRADE_ORDER = ["S2+", "S+", "S", "A", "B", "C", "D"]
AXIS_LABELS = {
    "S2+": "최상위 1%",
    "S+": "상위 4%",
    "S": "상위 10%",
    "A": "상위 30%",
    "B": "상위 50%",
    "C": "상위 80%",
    "D": "하위 20%",
}


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
        "title": get_secret(["app", "title"], "배우 정량분석 대시보드"),
        "subtitle": get_secret(["app", "subtitle"], "RAW 시트를 기반으로 배우 정량분석을 실시간 계산하는 대시보드"),
        "spreadsheet_id": get_secret(["data", "spreadsheet_id"], ""),
        "raw_sheet": get_secret(["data", "raw_sheet"], "RAW"),
        "weights": {
            "production": float(get_secret(["weights", "production"], 0.4)),
            "stability": float(get_secret(["weights", "stability"], 0.3)),
            "contribution": float(get_secret(["weights", "contribution"], 0.3)),
        },
        "thresholds": {
            "TS": float(get_secret(["grading", "TS"], 99)),
            "T1": float(get_secret(["grading", "T1"], 97)),
            "T2": float(get_secret(["grading", "T2"], 93)),
            "T3": float(get_secret(["grading", "T3"], 85)),
            "M1": float(get_secret(["grading", "M1"], 70)),
            "M2": float(get_secret(["grading", "M2"], 50)),
            "M3": float(get_secret(["grading", "M3"], 30)),
            "B1": float(get_secret(["grading", "B1"], 15)),
            "B2": float(get_secret(["grading", "B2"], 5)),
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
        raise ImportError("gspread/google-auth 패키지가 설치되어 있지 않습니다.")
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

    df = pd.DataFrame(values[1:], columns=values[0])
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]

    required = ["인물명", "프로그램명", "드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"RAW 시트에 필요한 컬럼이 없습니다: {', '.join(missing)}")

    for col in ["드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹", "점유율"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["인물명"] = df["인물명"].astype(str).str.strip()
    df["프로그램명"] = df["프로그램명"].astype(str).str.strip()
    df = df[df["인물명"].ne("")].reset_index(drop=True)

    if "점유율" not in df.columns:
        df["점유율"] = np.nan

    zero_or_blank = df["점유율"].isna() | (df["점유율"] == 0)
    denom = df["드라마화제성"].replace(0, np.nan)
    df.loc[zero_or_blank, "점유율"] = df.loc[zero_or_blank, "배우화제성"] / denom.loc[zero_or_blank]

    return df


# -----------------------------
# Calculation helpers
# -----------------------------
def normalize_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mn) / (mx - mn)


def percentrank_inc(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) <= 1:
        out = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
        out[s.isna()] = np.nan
        return out
    ranks = valid.rank(method="max")
    pct = (ranks - 1) / (len(valid) - 1)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[valid.index] = pct
    return out.clip(0, 1)


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


def final_grade_from_pct(value_pct100: float, thresholds: Dict[str, float]) -> str:
    if pd.isna(value_pct100):
        return "B3"
    if value_pct100 >= thresholds["TS"]:
        return "TS"
    if value_pct100 >= thresholds["T1"]:
        return "T1"
    if value_pct100 >= thresholds["T2"]:
        return "T2"
    if value_pct100 >= thresholds["T3"]:
        return "T3"
    if value_pct100 >= thresholds["M1"]:
        return "M1"
    if value_pct100 >= thresholds["M2"]:
        return "M2"
    if value_pct100 >= thresholds["M3"]:
        return "M3"
    if value_pct100 >= thresholds["B1"]:
        return "B1"
    if value_pct100 >= thresholds["B2"]:
        return "B2"
    return "B3"


def tier_group(grade: str) -> str:
    if grade in ["TS", "T1", "T2", "T3"]:
        return "Top"
    if grade in ["M1", "M2", "M3"]:
        return "Mid"
    return "Base"


@st.cache_data(show_spinner=False)
def build_quant_from_raw(raw_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = raw_df.copy()

    grouped = df.groupby("인물명", dropna=False)
    result = pd.DataFrame({"배우": grouped.size().index})
    result["드라마 화제성"] = grouped["드라마화제성"].sum().values
    result["배우 화제성"] = grouped["배우화제성"].sum().values
    result["출연작품"] = grouped.size().values
    result["랭크주차"] = grouped["랭크인주차"].sum().values
    result["대표작 성과"] = grouped["배우화제성"].max().values

    # Unique work list
    work_map = grouped["프로그램명"].agg(lambda x: ", ".join(pd.Series(x).dropna().astype(str).drop_duplicates().tolist()))
    result["출연작"] = result["배우"].map(work_map)

    # Core metrics from the original quant sheet logic
    result["작품평균"] = result["배우 화제성"] / result["출연작품"].replace(0, np.nan)
    overall_work_avg = result["작품평균"].mean(skipna=True)
    result["보정 작품평균"] = (result["배우 화제성"] + 3 * overall_work_avg) / (result["출연작품"] + 3)

    max_actor_buzz = grouped["배우화제성"].max()
    total_actor_buzz = grouped["배우화제성"].sum().replace(0, np.nan)
    hit_dispersion = 1 - (max_actor_buzz / total_actor_buzz)
    result["히트 분산지수"] = result["배우"].map(hit_dispersion)
    result["히트 분산정규화"] = normalize_minmax(result["히트 분산지수"])

    result["화제성 기여도"] = result["배우 화제성"] / result["드라마 화제성"].replace(0, np.nan)

    rank1 = grouped["작품내랭킹"].apply(lambda x: (pd.to_numeric(x, errors="coerce") == 1).sum())
    rank2 = grouped["작품내랭킹"].apply(lambda x: (pd.to_numeric(x, errors="coerce") == 2).sum())
    rank3 = grouped["작품내랭킹"].apply(lambda x: (pd.to_numeric(x, errors="coerce") == 3).sum())
    result["1위배율"] = result["배우"].map(rank1) / result["출연작품"].replace(0, np.nan)
    result["2위배율"] = result["배우"].map(rank2) / result["출연작품"].replace(0, np.nan) * 0.5
    result["3위배율"] = result["배우"].map(rank3) / result["출연작품"].replace(0, np.nan) * 0.3
    result[["1위배율", "2위배율", "3위배율"]] = result[["1위배율", "2위배율", "3위배율"]].fillna(0)
    result["대표작 성과백분위"] = percentrank_inc(result["대표작 성과"])

    # 꾸준함지수 / 보정기여도
    p_norm = normalize_minmax(result["보정 작품평균"])
    result["꾸준함지수"] = (
        0.25 * p_norm
        + 0.55 * (0.7 * result["히트 분산정규화"] + 0.3 * (result["출연작품"] / (result["출연작품"] + 2)))
        + 0.2 * (result["대표작 성과백분위"] ** 3)
    ).clip(upper=1)

    result["보정기여도"] = (
        0.5 * p_norm
        + 0.5 * (result["화제성 기여도"] * (result["1위배율"] + result["2위배율"] + result["3위배율"]))
    )

    # Percentiles
    result["생산력(백분율)"] = percentrank_inc(result["배우 화제성"])
    result["지속력(백분율)"] = percentrank_inc(result["꾸준함지수"])
    result["기여도(백분율)"] = percentrank_inc(result["보정기여도"])
    result["안정성(백분율)"] = result["지속력(백분율)"]

    # Axis grades
    axis_thr = cfg["axis_thresholds"]
    result["생산력"] = result["생산력(백분율)"].apply(lambda x: axis_grade_from_pct(x, axis_thr))
    result["안정성"] = result["안정성(백분율)"].apply(lambda x: axis_grade_from_pct(x, axis_thr))
    result["기여도"] = result["기여도(백분율)"].apply(lambda x: axis_grade_from_pct(x, axis_thr))

    # Final score and grade
    w = cfg["weights"]
    result["합산점수"] = 100 * (
        result["생산력(백분율)"] * w["production"]
        + result["안정성(백분율)"] * w["stability"]
        + result["기여도(백분율)"] * w["contribution"]
    )
    result["종합백분위"] = percentrank_inc(result["합산점수"]) * 100
    result["최종등급"] = result["종합백분위"].apply(lambda x: final_grade_from_pct(x, cfg["thresholds"]))
    result["Tier"] = result["최종등급"].apply(tier_group)
    result["#"] = np.arange(1, len(result) + 1)

    result = result.sort_values(["합산점수", "배우 화제성"], ascending=[False, False]).reset_index(drop=True)
    result["#"] = np.arange(1, len(result) + 1)

    # display-friendly order
    ordered_cols = [
        "#", "배우", "합산점수", "최종등급", "Tier", "생산력", "안정성", "기여도", "출연작",
        "생산력(백분율)", "안정성(백분율)", "기여도(백분율)", "종합백분위",
        "드라마 화제성", "배우 화제성", "출연작품", "꾸준함지수", "보정 작품평균", "작품평균",
        "히트 분산지수", "히트 분산정규화", "보정기여도", "화제성 기여도",
        "1위배율", "2위배율", "3위배율", "랭크주차", "대표작 성과", "대표작 성과백분위",
    ]
    result = result[[c for c in ordered_cols if c in result.columns]]
    return result


# -----------------------------
# Display helpers
# -----------------------------
def format_pct01(v):
    if pd.isna(v):
        return "-"
    return f"{v*100:.1f}%"


def format_pct100(v):
    if pd.isna(v):
        return "-"
    return f"{v:.1f}%"


def build_actor_comment(row: pd.Series) -> str:
    prod = row.get("생산력(백분율)", np.nan)
    stab = row.get("안정성(백분율)", np.nan)
    contrib = row.get("기여도(백분율)", np.nan)

    if prod >= 0.93 and stab >= 0.85 and contrib >= 0.7:
        return "세 축이 함께 높은 상위권 배우입니다. 총량과 안정성, 작품 내 존재감이 동시에 받쳐주는 타입입니다."
    if prod >= 0.93 and contrib >= 0.8:
        return "생산력과 기여도가 강한 타입입니다. 강한 화제성을 만들고 작품 내 존재감도 큰 편입니다."
    if stab >= 0.85 and prod < 0.85:
        return "폭발력보다 안정성이 돋보이는 타입입니다. 성과 편차가 상대적으로 크지 않고 누적 관리력이 좋습니다."
    if contrib >= 0.85 and prod < 0.8:
        return "총량보다도 작품 내 점유와 존재감이 강한 타입입니다."
    return "세 축의 균형을 함께 보는 것이 적합한 배우입니다."


def render_sidebar(df: pd.DataFrame):
    st.sidebar.header("필터")
    actor_query = st.sidebar.text_input("배우명 검색", "")
    tier_sel = st.sidebar.multiselect("Tier", TIER_ORDER, default=TIER_ORDER)
    grade_sel = st.sidebar.multiselect("최종등급", GRADE_ORDER, default=GRADE_ORDER)

    works_min = int(df["출연작품"].fillna(0).min())
    works_max = int(df["출연작품"].fillna(0).max())
    works_range = st.sidebar.slider("출연작품 수", works_min, works_max, (works_min, works_max))

    score_min = float(np.floor(df["합산점수"].min()))
    score_max = float(np.ceil(df["합산점수"].max()))
    score_range = st.sidebar.slider("합산점수", score_min, score_max, (score_min, score_max))

    return {
        "actor_query": actor_query.strip(),
        "tier_sel": tier_sel,
        "grade_sel": grade_sel,
        "works_range": works_range,
        "score_range": score_range,
    }


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    out = df.copy()
    if filters["actor_query"]:
        out = out[out["배우"].astype(str).str.contains(filters["actor_query"], case=False, na=False)]
    out = out[out["Tier"].isin(filters["tier_sel"])]
    out = out[out["최종등급"].isin(filters["grade_sel"])]
    out = out[out["출연작품"].fillna(0).between(*filters["works_range"])]
    out = out[out["합산점수"].fillna(0).between(*filters["score_range"])]
    return out


def render_overview(df: pd.DataFrame):
    st.subheader("전체 개요")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 배우 수", f"{len(df):,}명")
    c2.metric("Top 비중", f"{df['Tier'].eq('Top').mean()*100:.1f}%")
    c3.metric("Mid 비중", f"{df['Tier'].eq('Mid').mean()*100:.1f}%")
    c4.metric("Base 비중", f"{df['Tier'].eq('Base').mean()*100:.1f}%")

    left, right = st.columns([1.1, 1])
    with left:
        counts = df["최종등급"].value_counts().reindex(GRADE_ORDER).fillna(0).reset_index()
        counts.columns = ["등급", "배우수"]
        fig = px.bar(counts, x="등급", y="배우수", text="배우수", category_orders={"등급": GRADE_ORDER})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        tier_counts = df["Tier"].value_counts().reindex(TIER_ORDER).fillna(0).reset_index()
        tier_counts.columns = ["Tier", "배우수"]
        fig2 = px.pie(tier_counts, names="Tier", values="배우수", hole=0.5)
        fig2.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("생산력 × 안정성 포지셔닝")
    plot_df = df.copy()
    plot_df["버블크기"] = plot_df["기여도(백분율)"].fillna(0).clip(lower=0.01) * 60
    fig3 = px.scatter(
        plot_df,
        x="안정성(백분율)",
        y="생산력(백분율)",
        size="버블크기",
        color="최종등급",
        hover_name="배우",
        hover_data={
            "기여도(백분율)": ':.1%',
            "합산점수": ':.2f',
            "출연작품": True,
            "버블크기": False,
        },
        category_orders={"최종등급": GRADE_ORDER},
    )
    fig3.update_xaxes(title="안정성 백분위", tickformat=".0%")
    fig3.update_yaxes(title="생산력 백분위", tickformat=".0%")
    fig3.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)


def render_actor_detail(filtered: pd.DataFrame, raw_df: pd.DataFrame):
    st.subheader("배우 상세")
    if filtered.empty:
        st.info("현재 필터 조건에 해당하는 배우가 없습니다.")
        return

    actor_names = filtered["배우"].tolist()
    actor = st.selectbox("배우 선택", actor_names, index=0)
    row = filtered.loc[filtered["배우"] == actor].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("최종등급", row["최종등급"])
    c2.metric("합산점수", f"{row['합산점수']:.2f}")
    c3.metric("종합백분위", format_pct100(row["종합백분위"]))
    c4.metric("출연작품", f"{int(row['출연작품'])}")
    c5.metric("대표작 성과", f"{row['대표작 성과']:.2f}")

    left, right = st.columns([1.1, 1])
    with left:
        bars = pd.DataFrame(
            {
                "축": ["생산력", "안정성", "기여도"],
                "백분위": [row["생산력(백분율)"], row["안정성(백분율)"], row["기여도(백분율)"]],
                "등급": [row["생산력"], row["안정성"], row["기여도"]],
            }
        )
        fig = px.bar(bars, x="백분위", y="축", orientation="h", text="등급")
        fig.update_traces(textposition="outside")
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**해석 요약**")
        st.write(build_actor_comment(row))
        st.markdown("**출연작**")
        st.write(row.get("출연작", "-"))

    st.markdown("**등급 판정 근거**")
    explain = pd.DataFrame(
        {
            "항목": ["종합백분위", "생산력", "안정성", "기여도", "Tier"],
            "값": [
                format_pct100(row["종합백분위"]),
                f"{row['생산력']} ({format_pct01(row['생산력(백분율)'])})",
                f"{row['안정성']} ({format_pct01(row['안정성(백분율)'])})",
                f"{row['기여도']} ({format_pct01(row['기여도(백분율)'])})",
                row["Tier"],
            ],
        }
    )
    st.dataframe(explain, use_container_width=True, hide_index=True)

    actor_raw = raw_df[raw_df["인물명"] == actor].copy()
    st.markdown("**작품별 RAW 데이터**")
    if actor_raw.empty:
        st.info("해당 배우의 RAW 데이터가 없습니다.")
    else:
        actor_raw["점유율"] = actor_raw["점유율"].apply(format_pct01)
        cols = [c for c in ["프로그램명", "드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹", "점유율"] if c in actor_raw.columns]
        st.dataframe(actor_raw[cols].sort_values(["배우화제성", "드라마화제성"], ascending=False), use_container_width=True, hide_index=True)


def render_table(df: pd.DataFrame):
    st.subheader("전체 테이블")
    table = df[[
        "#", "최종등급", "배우", "합산점수", "종합백분위", "Tier",
        "생산력", "안정성", "기여도",
        "생산력(백분율)", "안정성(백분율)", "기여도(백분율)",
        "대표작 성과", "출연작품", "배우 화제성", "드라마 화제성"
    ]].copy()

    table["종합백분위"] = table["종합백분위"].map(format_pct100)
    for col in ["생산력(백분율)", "안정성(백분율)", "기여도(백분율)"]:
        table[col] = table[col].map(format_pct01)
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_formula_summary(cfg: Dict):
    with st.expander("계산 로직 보기", expanded=False):
        st.markdown(
            f"""
**최종 합산점수**  
= 생산력 백분위 × {cfg['weights']['production']} + 안정성 백분위 × {cfg['weights']['stability']} + 기여도 백분위 × {cfg['weights']['contribution']}  
그리고 최종적으로 ×100 처리

**생산력** = 배우 화제성 총합의 백분위  
**안정성** = 꾸준함지수의 백분위  
**기여도** = 보정기여도의 백분위

**꾸준함지수**  
= 0.25 × 보정 작품평균 정규화  
+ 0.55 × (0.7 × 히트 분산정규화 + 0.3 × 출연작품/(출연작품+2))  
+ 0.2 × 대표작 성과백분위³

**보정기여도**  
= 0.5 × 보정 작품평균 정규화  
+ 0.5 × (화제성 기여도 × (1위배율 + 2위배율 + 3위배율))
            """
        )


def main():
    st.set_page_config(page_title="배우 정량분석", page_icon="🎭", layout="wide")
    cfg = get_config()

    st.title(cfg["title"])
    st.caption(cfg["subtitle"])
    render_formula_summary(cfg)

    with st.expander("최종 등급 기준", expanded=False):
        thresholds_df = pd.DataFrame(
            {
                "등급": GRADE_ORDER,
                "종합백분위 기준": [
                    f">= {cfg['thresholds']['TS']}%",
                    f">= {cfg['thresholds']['T1']}%",
                    f">= {cfg['thresholds']['T2']}%",
                    f">= {cfg['thresholds']['T3']}%",
                    f">= {cfg['thresholds']['M1']}%",
                    f">= {cfg['thresholds']['M2']}%",
                    f">= {cfg['thresholds']['M3']}%",
                    f">= {cfg['thresholds']['B1']}%",
                    f">= {cfg['thresholds']['B2']}%",
                    f"< {cfg['thresholds']['B2']}%",
                ],
            }
        )
        st.dataframe(thresholds_df, use_container_width=True, hide_index=True)

    try:
        raw_df = load_raw_sheet(cfg["spreadsheet_id"], cfg["raw_sheet"])
        quant_df = build_quant_from_raw(raw_df, cfg)
    except Exception as e:
        st.error("데이터를 불러오거나 계산하지 못했습니다. secrets 설정과 RAW 시트 컬럼명을 확인해 주세요.")
        st.exception(e)
        st.stop()

    filters = render_sidebar(quant_df)
    filtered = apply_filters(quant_df, filters)

    tab1, tab2, tab3 = st.tabs(["개요", "배우 상세", "전체 테이블"])
    with tab1:
        render_overview(filtered)
    with tab2:
        render_actor_detail(filtered, raw_df)
    with tab3:
        render_table(filtered)


if __name__ == "__main__":
    main()
