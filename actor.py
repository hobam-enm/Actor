import math
from typing import Dict, List

import gspread
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="배우 정량분석 대시보드", layout="wide")

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
    "Top-S": 0.99,
    "Top-A": 0.97,
    "Top-B": 0.93,
    "Top-C": 0.85,
    "Middle-A": 0.70,
    "Middle-B": 0.50,
    "Middle-C": 0.30,
    "Base-A": 0.15,
    "Base-B": 0.05,
}

GRADE_ORDER = [
    "Top-S",
    "Top-A",
    "Top-B",
    "Top-C",
    "Middle-A",
    "Middle-B",
    "Middle-C",
    "Base-A",
    "Base-B",
    "Base-C",
]

VISIBLE_COLUMNS = [
    "#",
    "배우",
    "생산력등급",
    "안정성등급",
    "기여도등급",
    "합산점수",
    "생산백분율",
    "안정백분율",
    "기여백분율",
    "배우화제성",
    "출연작품수",
]

COLOR_MAP = {
    "Top-S": "#0B3B91",
    "Top-A": "#1757B0",
    "Top-B": "#2D72D2",
    "Top-C": "#4C8DE8",
    "Middle-A": "#5C7CFA",
    "Middle-B": "#748FFC",
    "Middle-C": "#91A7FF",
    "Base-A": "#F08C00",
    "Base-B": "#FA5252",
    "Base-C": "#E03131",
}


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        .app-subtitle {color:#6b7280; margin-top:-0.2rem; margin-bottom:1rem;}
        .soft-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            height: 100%;
        }
        .metric-label {font-size: 0.92rem; color:#6b7280; font-weight:700; margin-bottom:10px;}
        .metric-value {font-size: 2rem; font-weight:800; color:#111827; line-height:1.1;}
        .metric-sub {font-size: 0.87rem; color:#6b7280; margin-top:8px;}
        .section-title {font-size: 1.25rem; font-weight: 800; color:#111827; margin: 0 0 0.8rem 0.1rem;}
        .pill-wrap {display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;}
        .pill {
            display:inline-block; padding:6px 10px; border-radius:999px;
            background:#eff6ff; color:#1d4ed8; font-weight:700; font-size:0.82rem;
            border:1px solid #dbeafe;
        }
        .mini-card {
            background:#fff; border:1px solid #e5e7eb; border-radius:18px; padding:16px 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05); height:100%;
        }
        .grade-chip {display:inline-block; padding:4px 10px; border-radius:999px; color:#fff; font-size:0.8rem; font-weight:800;}
        .actor-name {font-size:1.15rem; font-weight:800; color:#111827; margin:8px 0 6px 0;}
        .actor-sub {font-size:0.88rem; color:#6b7280;}
        .kv-grid {display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px;}
        .work-card {background:#fff; border:1px solid #e5e7eb; border-radius:18px; padding:16px; box-shadow: 0 8px 24px rgba(15,23,42,0.05);}
        .work-title {font-size:1rem; font-weight:800; color:#111827; margin-bottom:10px;}
        .work-k {font-size:0.82rem; color:#6b7280; margin-bottom:4px;}
        .work-v {font-size:1.1rem; font-weight:800; color:#111827; margin-bottom:10px;}
        .stDataFrame {border-radius:18px; overflow:hidden;}
        .stTextInput > div > div, .stMultiSelect > div > div, .stSelectbox > div > div {
            border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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

    header = [(c or "").strip() or f"unnamed_{i+1}" for i, c in enumerate(values[0])]
    rows = []
    max_len = len(header)
    for row in values[1:]:
        row = list(row)
        if len(row) < max_len:
            row += [""] * (max_len - len(row))
        rows.append(row[:max_len])

    df = pd.DataFrame(rows, columns=header)
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"RAW 시트에 필요한 컬럼이 없습니다: {missing}")
        st.stop()

    keep_cols = [
        c for c in [
            "인물명", "프로그램명", "드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹", "점유율"
        ] if c in df.columns
    ]
    df = df[keep_cols].copy()

    for col in ["인물명", "프로그램명"]:
        df[col] = df[col].astype(str).str.strip()

    for col in ["드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹", "점유율"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(",", "", regex=False).str.strip().replace({"": np.nan, "None": np.nan, "nan": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["인물명"].notna() & (df["인물명"] != "")].copy()
    df = df[df["프로그램명"].notna() & (df["프로그램명"] != "")].copy()
    return df


def percentrank_inc_min(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if len(s) <= 1:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s.rank(method="min") - 1) / (len(s) - 1)


def axis_grade(p: float, thresholds: Dict[str, float]) -> str:
    if pd.isna(p):
        return ""
    for grade in GRADE_ORDER[:-1]:
        if p >= thresholds[grade]:
            return grade
    return "Base-C"


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
    result["드라마화제성"] = grouped["드라마화제성"].sum()
    result["배우화제성"] = grouped["배우화제성"].sum()
    result["출연작품수"] = grouped.size().astype(float)
    result["랭크주차"] = grouped["랭크인주차"].sum()
    result["대표작 성과"] = grouped["배우화제성"].max()
    result["출연작"] = grouped["프로그램명"].apply(lambda s: ", ".join(pd.unique(s.dropna().astype(str))))

    result["작품평균"] = result["배우화제성"] / result["출연작품수"]
    global_avg = result["작품평균"].mean()
    result["보정 작품평균"] = (result["배우화제성"] + 3 * global_avg) / (result["출연작품수"] + 3)

    result["히트 분산지수"] = 1 - grouped["배우화제성"].max() / result["배우화제성"]
    r_min, r_max = result["히트 분산지수"].min(), result["히트 분산지수"].max()
    if pd.isna(r_min) or pd.isna(r_max) or math.isclose(r_min, r_max):
        result["히트 분산정규화"] = 0.0
    else:
        result["히트 분산정규화"] = (result["히트 분산지수"] - r_min) / (r_max - r_min)

    result["화제성 기여도"] = result["배우화제성"] / result["드라마화제성"]
    result["1위배율"] = grouped["r1"].sum() / result["출연작품수"]
    result["2위배율"] = grouped["r2"].sum() / result["출연작품수"]
    result["3위배율"] = grouped["r3"].sum() / result["출연작품수"]
    result["대표작 성과백분위"] = percentrank_inc_min(result["대표작 성과"])

    p_min, p_max = result["보정 작품평균"].min(), result["보정 작품평균"].max()
    if pd.isna(p_min) or pd.isna(p_max) or math.isclose(p_min, p_max):
        p_norm = pd.Series(0.0, index=result.index)
    else:
        p_norm = (result["보정 작품평균"] - p_min) / (p_max - p_min)

    result["꾸준함지수"] = np.minimum(
        1.0,
        0.25 * p_norm
        + 0.55 * (0.7 * result["히트 분산정규화"] + 0.3 * (result["출연작품수"] / (result["출연작품수"] + 2)))
        + 0.2 * (result["대표작 성과백분위"] ** 3),
    )

    base_contribution = 0.5 * p_norm + 0.5 * (
        result["화제성 기여도"] * (result["1위배율"] + result["2위배율"] + result["3위배율"])
    )
    result["작품체급백분위"] = percentrank_inc_min(result["드라마화제성"])
    result["작품체급보정"] = 0.45 + 0.55 * result["작품체급백분위"]
    result["보정기여도"] = base_contribution * result["작품체급보정"]

    result["생산백분율"] = percentrank_inc_min(result["배우화제성"])
    result["안정백분율"] = percentrank_inc_min(result["꾸준함지수"])
    result["기여백분율"] = percentrank_inc_min(result["보정기여도"])

    result["생산력등급"] = result["생산백분율"].apply(lambda x: axis_grade(x, thresholds))
    result["안정성등급"] = result["안정백분율"].apply(lambda x: axis_grade(x, thresholds))
    result["기여도등급"] = result["기여백분율"].apply(lambda x: axis_grade(x, thresholds))

    result["합산점수"] = 100 * (
        result["생산백분율"] * float(weights.get("production", 0.4))
        + result["안정백분율"] * float(weights.get("stability", 0.3))
        + result["기여백분율"] * float(weights.get("contribution", 0.3))
    )

    result = result.sort_values(["합산점수", "배우화제성"], ascending=[False, False]).reset_index(drop=True)
    result.insert(0, "#", np.arange(1, len(result) + 1))
    return result


def format_percent_0(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.0f}%"


def format_int(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:,.0f}"


def format_score(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:,.2f}"


def format_visible_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[VISIBLE_COLUMNS].copy()
    out["합산점수"] = pd.to_numeric(out["합산점수"], errors="coerce").map(format_score)
    for col in ["생산백분율", "안정백분율", "기여백분율"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").map(format_percent_0)
    for col in ["배우화제성", "출연작품수"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").map(format_int)
    out["#"] = out["#"].astype(int).astype(str)
    return out


def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class='soft-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-sub'>{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def actor_card(name: str, grade: str, score: float, sub_text: str):
    color = COLOR_MAP.get(grade, "#334155")
    st.markdown(
        f"""
        <div class='mini-card'>
            <span class='grade-chip' style='background:{color};'>{grade}</span>
            <div class='actor-name'>{name}</div>
            <div class='actor-sub'>합산점수 {format_score(score)}</div>
            <div class='actor-sub' style='margin-top:6px;'>{sub_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def work_card(program: str, drama_score: float, actor_score: float):
    st.markdown(
        f"""
        <div class='work-card'>
            <div class='work-title'>{program}</div>
            <div class='work-k'>드라마화제성</div>
            <div class='work-v'>{format_int(drama_score)}</div>
            <div class='work-k'>배우화제성</div>
            <div class='work-v'>{format_int(actor_score)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_grade_representatives(result_df: pd.DataFrame):
    st.markdown("<div class='section-title'>등급별 대표배우</div>", unsafe_allow_html=True)
    reps = []
    for grade in GRADE_ORDER:
        sub = result_df[result_df["생산력등급"] == grade].sort_values(["합산점수", "배우화제성"], ascending=[False, False]).head(1)
        if len(sub):
            reps.append(sub.iloc[0])

    cols = st.columns(5)
    for idx, row in enumerate(reps[:10]):
        with cols[idx % 5]:
            actor_card(
                row["배우"],
                row["생산력등급"],
                row["합산점수"],
                f"배우화제성 {format_int(row['배우화제성'])} · 출연작 {format_int(row['출연작품수'])}개",
            )


def build_actor_program_summary(raw_df: pd.DataFrame, actor_name: str) -> pd.DataFrame:
    actor_raw = raw_df[raw_df["인물명"] == actor_name].copy()
    if actor_raw.empty:
        return pd.DataFrame(columns=["프로그램명", "드라마화제성", "배우화제성"])
    agg = (
        actor_raw.groupby("프로그램명", as_index=False)[["드라마화제성", "배우화제성"]]
        .sum()
        .sort_values(["배우화제성", "드라마화제성"], ascending=[False, False])
    )
    return agg


def filter_result_df(
    result_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    actor_keyword: str,
    program_keyword: str,
    prod_grades: List[str],
    stab_grades: List[str],
    cont_grades: List[str],
) -> pd.DataFrame:
    filtered = result_df.copy()
    if actor_keyword.strip():
        filtered = filtered[filtered["배우"].str.contains(actor_keyword.strip(), case=False, na=False)]
    if program_keyword.strip():
        actor_pool = raw_df.loc[
            raw_df["프로그램명"].str.contains(program_keyword.strip(), case=False, na=False),
            "인물명",
        ].dropna().unique().tolist()
        filtered = filtered[filtered["배우"].isin(actor_pool)]
    if prod_grades:
        filtered = filtered[filtered["생산력등급"].isin(prod_grades)]
    if stab_grades:
        filtered = filtered[filtered["안정성등급"].isin(stab_grades)]
    if cont_grades:
        filtered = filtered[filtered["기여도등급"].isin(cont_grades)]
    filtered = filtered.sort_values(["합산점수", "배우화제성"], ascending=[False, False]).reset_index(drop=True)
    filtered["#"] = np.arange(1, len(filtered) + 1)
    return filtered


def render_overview(raw_df: pd.DataFrame, result_df: pd.DataFrame):
    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    total_actors = result_df["배우"].nunique()
    total_programs = raw_df["프로그램명"].nunique()
    top_ratio = result_df["생산력등급"].isin(GRADE_ORDER[:4]).mean() * 100
    avg_score = result_df["합산점수"].mean()
    best_actor = result_df.iloc[0]["배우"] if len(result_df) else "-"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("전체 배우", format_int(total_actors), "RAW 기반 계산 결과")
    with c2:
        metric_card("전체 작품", format_int(total_programs), "프로그램 기준")
    with c3:
        metric_card("Top 구간 비중", f"{top_ratio:.1f}%", "Top-S ~ Top-C")
    with c4:
        metric_card("평균 합산점수", format_score(avg_score), "전체 배우 평균")
    with c5:
        metric_card("현재 1위 배우", best_actor, f"합산점수 {format_score(result_df.iloc[0]['합산점수'])}" if len(result_df) else "")

    render_grade_representatives(result_df)

    st.markdown("<div class='section-title'>결과 테이블</div>", unsafe_allow_html=True)
    st.dataframe(format_visible_table(result_df), use_container_width=True, hide_index=True, height=760)


def render_detail(raw_df: pd.DataFrame, result_df: pd.DataFrame):
    st.markdown("<div class='section-title'>상세보기</div>", unsafe_allow_html=True)
    f1, f2, f3, f4, f5 = st.columns([1.2, 1.2, 1, 1, 1])
    with f1:
        actor_keyword = st.text_input("배우 검색", placeholder="배우명 입력")
    with f2:
        program_keyword = st.text_input("작품 검색", placeholder="프로그램명 입력")
    with f3:
        prod_grades = st.multiselect("생산력 등급", GRADE_ORDER)
    with f4:
        stab_grades = st.multiselect("안정성 등급", GRADE_ORDER)
    with f5:
        cont_grades = st.multiselect("기여도 등급", GRADE_ORDER)

    filtered = filter_result_df(result_df, raw_df, actor_keyword, program_keyword, prod_grades, stab_grades, cont_grades)

    s1, s2, s3 = st.columns(3)
    with s1:
        metric_card("조회 배우 수", format_int(len(filtered)), "현재 필터 기준")
    with s2:
        metric_card("평균 합산점수", format_score(filtered["합산점수"].mean()) if len(filtered) else "0.00", "현재 필터 기준")
    with s3:
        metric_card(
            "관련 작품 수",
            format_int(raw_df[raw_df["인물명"].isin(filtered["배우"])]["프로그램명"].nunique()) if len(filtered) else "0",
            "현재 필터 기준",
        )

    if filtered.empty:
        st.info("조건에 맞는 배우가 없습니다.")
        return

    actor_options = filtered["배우"].tolist()
    selected_actor = st.selectbox("배우 선택", actor_options, index=0)
    actor_row = filtered[filtered["배우"] == selected_actor].iloc[0]

    a1, a2, a3, a4 = st.columns([1.35, 1, 1, 1])
    with a1:
        actor_card(
            actor_row["배우"],
            actor_row["생산력등급"],
            actor_row["합산점수"],
            f"생산 {format_percent_0(actor_row['생산백분율'])} · 안정 {format_percent_0(actor_row['안정백분율'])} · 기여 {format_percent_0(actor_row['기여백분율'])}",
        )
    with a2:
        metric_card("생산력 등급", actor_row["생산력등급"], f"항목별 점수 {format_percent_0(actor_row['생산백분율'])}")
    with a3:
        metric_card("안정성 등급", actor_row["안정성등급"], f"항목별 점수 {format_percent_0(actor_row['안정백분율'])}")
    with a4:
        metric_card("기여도 등급", actor_row["기여도등급"], f"항목별 점수 {format_percent_0(actor_row['기여백분율'])}")

    left, right = st.columns([1.05, 1.2])
    with left:
        radar = go.Figure()
        radar.add_trace(
            go.Scatterpolar(
                r=[actor_row["생산백분율"] * 100, actor_row["안정백분율"] * 100, actor_row["기여백분율"] * 100],
                theta=["생산력", "안정성", "기여도"],
                fill="toself",
                name="항목별 점수",
                line=dict(color="#2D72D2", width=3),
                fillcolor="rgba(45,114,210,0.25)",
            )
        )
        radar.update_layout(
            title="항목별 점수",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="")),
            height=420,
            margin=dict(l=10, r=10, t=60, b=20),
            showlegend=False,
        )
        st.plotly_chart(radar, use_container_width=True)
    with right:
        st.markdown("<div class='section-title'>배우 요약</div>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            metric_card("합산점수", format_score(actor_row["합산점수"]), "가중합 기준")
        with g2:
            metric_card("배우화제성", format_int(actor_row["배우화제성"]), "누적 기준")
        g3, g4 = st.columns(2)
        with g3:
            metric_card("출연작품수", format_int(actor_row["출연작품수"]), "누적 작품 수")
        with g4:
            metric_card("대표 생산등급", actor_row["생산력등급"], "축별 등급 기준")
        st.markdown(
            f"""
            <div class='soft-card' style='margin-top:12px;'>
                <div class='metric-label'>출연작 목록</div>
                <div class='pill-wrap'>
                    {''.join([f"<span class='pill'>{p.strip()}</span>" for p in str(actor_row['출연작']).split(',') if p.strip()][:18])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>대표출연작</div>", unsafe_allow_html=True)
    actor_programs = build_actor_program_summary(raw_df, selected_actor).head(6)
    cols = st.columns(3)
    for idx, row in actor_programs.iterrows():
        with cols[idx % 3]:
            work_card(row["프로그램명"], row["드라마화제성"], row["배우화제성"])


def main():
    inject_css()
    app_cfg = get_secret_section("app")
    st.title(app_cfg.get("title", "배우 정량분석 대시보드"))
    st.markdown(
        f"<div class='app-subtitle'>{app_cfg.get('subtitle', 'RAW 시트를 기반으로 배우 정량분석 결과를 계산하고 시각화합니다.')}</div>",
        unsafe_allow_html=True,
    )

    raw_df = load_raw_from_gsheet()
    result_df = build_result_table(raw_df)

    page = st.sidebar.radio("페이지", ["overview", "상세보기"], index=0)
    st.sidebar.caption("RAW 기반 계산 결과")

    if page == "overview":
        render_overview(raw_df, result_df)
    else:
        render_detail(raw_df, result_df)


if __name__ == "__main__":
    main()
