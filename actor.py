import math
import re
from typing import Dict, List, Tuple

import gspread
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="배우 다차원 화제성 지표 - 드라마", layout="wide")

RAW_REQUIRED_COLUMNS = [
    "인물명",
    "프로그램명",
    "드라마화제성",
    "배우화제성",
    "랭크인주차",
    "랭크인배우수",
    "작품내랭킹",
]

DEFAULT_WEIGHTS = {"production": 0.4, "stability": 0.3, "contribution": 0.3}

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
    "Top-S", "Top-A", "Top-B", "Top-C",
    "Middle-A", "Middle-B", "Middle-C",
    "Base-A", "Base-B", "Base-C",
]

TIER_BASE_ORDER = ["Top", "Middle", "Base"]

TIER_COLORS = {
    "Top": "#2456ff",
    "Middle": "#0d9a85",
    "Base": "#ef6a5b",
}
GRADE_BG = {
    "Top-S": "#7a3cff", "Top-A": "#1747d1", "Top-B": "#3f74ff", "Top-C": "#9ab7ff",
    "Middle-A": "#0b8f7c", "Middle-B": "#24baa1", "Middle-C": "#b7efe3",
    "Base-A": "#ef7d1a", "Base-B": "#f45b49", "Base-C": "#ffd0ca",
}

VISIBLE_COLUMNS = [
    "#", "배우", "합산티어", "생산력등급", "안정성등급", "기여도등급",
    "합산점수", "생산백분율", "안정백분율", "기여백분율", "배우화제성", "출연작품수"
]


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {background: #f5f7fb;}
        
[data-testid="stHeader"] {display:none;}
        [data-testid="stToolbar"] {display:none;}
        [data-testid="stDecoration"] {display:none;}
        [data-testid="stSidebarNav"] {display:none;}
        .block-container {padding-top: 0.55rem; padding-bottom: 2rem; max-width: 1500px;}
        h1, h2, h3 {letter-spacing: -0.02em;}
        .page-title {font-size: 2.05rem; font-weight: 900; color:#1f2937; margin-bottom: 1.2rem;}
        .section-title {font-size: 1.35rem; font-weight: 900; color:#232b3a; margin: 0.2rem 0 0.95rem 0.15rem;}
        .spacer-lg {height: 24px;}
        .spacer-md {height: 14px;}
        .card {
            background: linear-gradient(180deg, #ffffff 0%, #fafcff 100%);
            border: 1px solid #e7ebf3;
            border-radius: 22px;
            padding: 20px 22px;
            box-shadow: 0 10px 28px rgba(31,41,55,0.05);
            height: 100%;
        }
        .metric-label {font-size: 0.94rem; color:#6b7280; font-weight:800; margin-bottom:10px;}
        .metric-value {font-size: 2rem; font-weight:900; color:#111827; line-height:1.1;}
        .metric-sub {font-size: 0.82rem; color:#7b8495; margin-top: 8px;}
        .mini-card {
            background:#fff; border:1px solid #e7ebf3; border-radius:18px; padding:14px 16px;
            box-shadow: 0 8px 22px rgba(31,41,55,0.04); min-height: 116px;
        }
        .tiny-card {
            background:#fff; border:1px solid #e7ebf3; border-radius:16px; padding:12px 14px;
            box-shadow: 0 6px 18px rgba(31,41,55,0.04); min-height: 88px;
        }
        .chip {display:inline-block; padding:5px 10px; border-radius:999px; font-size:0.78rem; font-weight:800;}
        .actor-name {font-size:1.05rem; font-weight:900; color:#111827; margin:8px 0 6px 0;}
        .actor-sub {font-size:0.84rem; color:#6b7280; line-height:1.55;}
        .rep-card {
            background:#fff; border:1px solid #e7ebf3; border-radius:20px; padding:16px 18px;
            box-shadow: 0 8px 22px rgba(31,41,55,0.04); min-height: 190px;
        }
        .rep-title {font-size:1rem; font-weight:900; margin-bottom:12px; color:#222b3d;}
        .rep-line {font-size:0.92rem; color:#374151; line-height:1.95;}
        .work-card {
            background: linear-gradient(180deg, #ffffff 0%, #fafcff 100%);
            border:1px solid #e7ebf3; border-radius:20px; padding:18px 18px;
            box-shadow: 0 8px 22px rgba(31,41,55,0.04); min-height: 160px;
        }
        .work-title {font-size:1rem; font-weight:900; color:#111827; margin-bottom:12px;}
        .work-k {font-size:0.8rem; color:#778196; margin-bottom:4px;}
        .work-v {font-size:1.15rem; font-weight:900; color:#111827; margin-bottom:10px;}
        .subtle {color:#6b7280; font-size:0.9rem;}
        .select-hint {font-size:0.88rem; color:#6b7280; margin-top:-0.2rem; margin-bottom:0.9rem;}
        .summary-card {
            background: linear-gradient(180deg, #ffffff 0%, #fafcff 100%);
            border: 1px solid #e7ebf3;
            border-radius: 22px;
            padding: 18px 18px;
            box-shadow: 0 8px 22px rgba(31,41,55,0.04);
            min-height: 126px;
            height: 100%;
        }
        .summary-card.accent {
            border: 2px solid #2158d9;
            box-shadow: 0 10px 28px rgba(33,88,217,0.08);
        }
        .summary-title {
            font-size: 0.96rem;
            font-weight: 800;
            color: #667085;
            margin-bottom: 10px;
        }
        .summary-big {
            font-size: 2.1rem;
            line-height: 1.05;
            font-weight: 900;
            color: #10213a;
            letter-spacing: -0.04em;
        }
        .summary-mid {
            font-size: 0.98rem;
            font-weight: 800;
            color: #10213a;
            margin-top: 4px;
        }
        .summary-sub {
            font-size: 0.88rem;
            color: #7b8495;
            line-height: 1.65;
            margin-top: 10px;
        }

        .stTextInput > div > div, .stSelectbox > div > div, .stRadio > div, .stMultiSelect > div > div {
            border-radius: 14px !important;
        }
        .stDataFrame {border-radius: 18px; overflow: hidden;}
        
        section[data-testid="stSidebar"] {
            background:#f7f8fb;
            border-right:1px solid #e5e7eb;
            min-width: 280px !important;
            max-width: 280px !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top:1.2rem;
            padding-left:0.5rem;
            padding-right:0.5rem;
        }
        div[data-testid="stSidebarUserContent"] .stRadio,
        div[data-testid="stSidebarUserContent"] .stRadio > div {
            width: 100%;
            gap: 0 !important;
        }
        div[data-testid="stSidebarUserContent"] .stRadio label[data-baseweb="radio"] {
            display:flex !important;
            width:100% !important;
            margin:0 !important;
            padding:0 !important;
            border-top:1px solid #d9dee8;
            cursor:pointer !important;
            min-height:auto !important;
        }
        div[data-testid="stSidebarUserContent"] .stRadio label[data-baseweb="radio"]:last-of-type {
            border-bottom:1px solid #d9dee8;
        }
        div[data-testid="stSidebarUserContent"] .stRadio label[data-baseweb="radio"] > div:first-child {
            display:none !important;
        }
        div[data-testid="stSidebarUserContent"] .stRadio label[data-baseweb="radio"] p {
            display:block !important;
            width:100% !important;
            margin:0 !important;
            padding:1rem 0.75rem !important;
            text-align:center !important;
            font-size:1.05rem !important;
            font-weight:800 !important;
            line-height:1.25 !important;
            color:#334155 !important;
            white-space:nowrap !important;
            word-break:keep-all !important;
            letter-spacing:-0.02em !important;
            box-sizing:border-box !important;
        }
        div[data-testid="stSidebarUserContent"] .stRadio label[data-baseweb="radio"]:has(input:checked) p {
            background:#1f64f0 !important;
            color:#ffffff !important;
        }
        .sidebar-footnote {color:#8b919c; font-size:0.9rem; margin-top:2rem;}
        
        .summary-card .actor-sub { line-height: 1.8; }

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


    # ===== 기간 데이터만 'source' 시트에서 별도로 가져오기 =====
    try:
        ws_source = sh.worksheet("source")
        source_values = ws_source.get_all_values()
        # 공백이 아닌 기간 데이터만 중복 없이 리스트로 추출
        period_values = list(set([str(row[0]).strip() for row in source_values[1:] if len(row) > 0 and str(row[0]).strip() != ""]))
    except Exception:
        period_values = []

    keep_cols = [c for c in [
        "인물명", "프로그램명", "드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹", "점유율"
    ] if c in df.columns]
    df = df[keep_cols].copy()
    df["__period_raw"] = "||".join(period_values)

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


def major_tier(grade: str) -> str:
    if str(grade).startswith("Top"):
        return "Top"
    if str(grade).startswith("Middle"):
        return "Middle"
    return "Base"


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
    result["합산백분율"] = percentrank_inc_min(result["합산점수"])
    result["합산티어"] = result["합산백분율"].apply(lambda x: axis_grade(x, thresholds))
    result["대분류티어"] = result["합산티어"].apply(major_tier)

    for col, pct_col in [("생산", "생산백분율"), ("안정", "안정백분율"), ("기여", "기여백분율")]:
        result[f"{col}_대분류내점수"] = (
            result.groupby("대분류티어")[pct_col]
            .transform(lambda s: percentrank_inc_min(s) * 100)
        )
        result[f"{col}_전체점수"] = result[pct_col] * 100

    result = result.sort_values(["합산점수", "배우화제성"], ascending=[False, False]).reset_index(drop=True)
    result.insert(0, "#", np.arange(1, len(result) + 1))
    return result


def format_percent_0(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.0f}%"


def format_int(x: float) -> str:
    return "" if pd.isna(x) else f"{x:,.0f}"


def format_score(x: float) -> str:
    return "" if pd.isna(x) else f"{x:,.2f}"


def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class='card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-sub'>{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top10_card(rank: int, name: str, tier: str, score: float):
    bg = GRADE_BG.get(tier, "#64748b")
    st.markdown(
        f"""
        <div class='tiny-card'>
            {chip_html(f"{rank}위 · {tier}", tier)}
            <div class='actor-name' style='font-size:0.98rem; margin-top:10px;'>{name}</div>
            <div class='actor-sub'>합산점수 {format_score(score)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def representative_card(grade: str, rows: pd.DataFrame):
    bg = GRADE_BG.get(grade, "#64748b")
    lines = "".join([
        f"<div class='rep-line'><b>{r['배우']}</b> ({format_score(r['합산점수'])})</div>" for _, r in rows.iterrows()
    ])
    st.markdown(
        f"""
        <div class='rep-card'>
            <div class='rep-title'>{chip_html(grade, grade)}</div>
            {lines if lines else "<div class='actor-sub'>해당 배우 없음</div>"}
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



def summary_grade_card(title: str, grade: str, accent: bool = False) -> str:
    bg = GRADE_BG.get(str(grade), "#64748b")
    fg = grade_text_color(str(grade))
    card_class = "summary-card accent" if accent else "summary-card"
    return f"""
    <div class='{card_class}'>
        <div class='summary-title'>{title}</div>
        <div class='summary-big' style='color:{bg};'>{grade}</div>
    </div>
    """


def actor_summary_card(row: pd.Series):
    c1, c2, c3, c4, c5 = st.columns([1.45, 1.1, 0.95, 0.95, 0.95])

    with c1:
        st.markdown(
            f"""
            <div class='summary-card'>
                <div class='summary-title'>배우 개요</div>
                <div class='summary-big' style='font-size:1.9rem;'>{row['배우']}</div>
                <div class='summary-sub'>
                    합산점수 <b>{format_score(row['합산점수'])}</b><br>
                    배우화제성 <b>{format_int(row['배우화제성'])}</b><br>
                    출연작품수 <b>{format_int(row['출연작품수'])}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(summary_grade_card("종합등급", row["합산티어"], accent=True), unsafe_allow_html=True)

    with c3:
        st.markdown(summary_grade_card("생산력 등급", row["생산력등급"]), unsafe_allow_html=True)

    with c4:
        st.markdown(summary_grade_card("안정성 등급", row["안정성등급"]), unsafe_allow_html=True)

    with c5:
        st.markdown(summary_grade_card("기여도 등급", row["기여도등급"]), unsafe_allow_html=True)


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


def make_triangle_chart(values: List[float], title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=["생산력", "안정성", "기여도", "생산력"],
            fill="toself",
            line=dict(color="#356AE6", width=3),
            fillcolor="rgba(53,106,230,0.22)",
            marker=dict(size=8, color="#356AE6"),
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        height=360,
        margin=dict(l=20, r=20, t=55, b=20),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10), gridcolor="#dbe4f3"),
            angularaxis=dict(tickfont=dict(size=13, color="#374151")),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig




def grade_text_color(grade: str) -> str:
    grade = str(grade)
    return "#374151" if grade.endswith("C") else "#ffffff"


def chip_html(label: str, grade: str) -> str:
    bg = GRADE_BG.get(str(grade), "#64748b")
    fg = grade_text_color(str(grade))
    return f"<span class='chip' style='background:{bg}; color:{fg};'>{label}</span>"



def parse_week_label(label: str):
    s = str(label).strip()
    if not s:
        return None

    # 숫자만 뽑아 연/월/주를 찾고, 표시는 원문 그대로 사용
    nums = re.findall(r"\d+", s)
    if len(nums) >= 3:
        try:
            yy = int(nums[0])
            mm = int(nums[1])
            ww = int(nums[2])
            
            # 연도가 2자리(예: 26)로 들어오면 4자리(2026)로 변환, 4자리면 그대로 사용
            if yy < 100:
                yy += 2000
                
            # 정상적인 연/월/주 범위인지 체크 (2000~2100년 지원)
            if 2000 <= yy <= 2100 and 1 <= mm <= 12 and 1 <= ww <= 6:
                return (yy, mm, ww)
        except Exception:
            return None
    return None


def get_data_period_caption(raw_df: pd.DataFrame) -> str:
    if "__period_raw" not in raw_df.columns:
        return ""

    vals = raw_df["__period_raw"].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return ""

    # "||"로 묶어둔 전체 기간 데이터를 분리
    raw_str = vals.iloc[0]
    if "||" in raw_str:
        unique_vals = raw_str.split("||")
    else:
        unique_vals = pd.unique(vals).tolist()

    parsed = []
    for v in unique_vals:
        key = parse_week_label(v)
        if key is not None:
            parsed.append((key, v))

    if parsed:
        # 튜플 (연, 월, 주) 순으로 정렬하여 가장 빠른/늦은 날짜 추출
        parsed.sort(key=lambda x: x[0])
        return f"데이터 기준 기간 · {parsed[0][1]} ~ {parsed[-1][1]}"

    return f"데이터 기준 기간 · {unique_vals[0]} ~ {unique_vals[-1]}"


def render_reference():
    st.markdown("<div class='section-title'>참고사항</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='card'>
            <div class='rep-title'>1. 지표 구성 개요</div>
            <div class='actor-sub'>
                다차원 화제성 지표는 <b>FUNDEX 인물 화제성점수</b>를 기반으로 배우별 <b>생산력</b>, <b>안정성</b>, <b>기여도</b>를 계산하고,
                합산점수는 <b>생산력 40% · 안정성 30% · 기여도 30%</b> 가중으로 산출합니다.
                <br><br>
                해당 지표는 배우 화제성을 단순 총량 순위로 보지 않고,
                <br>· 얼마나 크게 성과를 내는지 <b>(생산력)</b>
                <br>· 그 성과가 얼마나 꾸준한지 <b>(안정성)</b>
                <br>· 작품 안에서 얼마나 중심적인 존재감을 보이는지 <b>(기여도)</b>
                <br>세 축으로 나누어 평가합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='spacer-lg'></div>", unsafe_allow_html=True)
    st.markdown("<div class='rep-title'>2. 상세 지표 설명</div>", unsafe_allow_html=True)
    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='summary-card' style='min-height:auto;'>
            <div class='summary-title'>생산력</div>
            <div class='actor-sub'>
                <b>정의</b><br>
                배우가 만들어낸 화제성의 절대 규모
                <br><br>
                <b>계산</b><br>
                배우 화제성 총합을 기준으로 전체 배우 내 상대적 위치 계산
                <br><br>
                <b>산식</b><br>
                생산력 = 배우화제성의 전체 백분위
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='summary-card' style='min-height:auto;'>
            <div class='summary-title'>안정성</div>
            <div class='actor-sub'>
                <b>정의</b><br>
                여러 작품에서 얼마나 꾸준히 성과를 냈는지
                <br><br>
                <b>계산</b><br>
                보정 작품평균, 히트 분산 보정, 작품수 보정, 대표작 성과를 함께 반영
                <br><br>
                <b>산식</b><br>
                안정성 = 꾸준함지수의 전체 백분위
                <br><br>
                꾸준함지수 = MIN(1,<br>
                &nbsp;&nbsp;0.25 × 보정작품평균정규화<br>
                &nbsp;&nbsp;+ 0.55 × (0.7 × 히트분산정규화 + 0.3 × 작품수보정)<br>
                &nbsp;&nbsp;+ 0.20 × (대표작성과백분위³)<br>
                )
                <br><br>
                <b>세부 항목</b><br>
                · 보정작품평균정규화 = (보정작품평균 - 전체 최소 보정작품평균) / (전체 최대 보정작품평균 - 전체 최소 보정작품평균)
                <br>
                · 보정작품평균 = (배우화제성 + 3 × 전체 작품평균 평균) / (출연작품수 + 3)
                <br>
                · 히트분산정규화 = (히트분산지수 - 전체 최소 히트분산지수) / (전체 최대 히트분산지수 - 전체 최소 히트분산지수)
                <br>
                · 히트분산지수 = 1 - (대표작성과 / 배우화제성)
                <br>
                · 작품수보정 = 출연작품수 / (출연작품수 + 2)
                <br>
                · 대표작성과백분위 = 대표작성과의 전체 백분위
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='summary-card' style='min-height:auto;'>
            <div class='summary-title'>기여도</div>
            <div class='actor-sub'>
                <b>정의</b><br>
                배우가 작품 전체 성과 안에서 얼마나 중심적인 존재감을 보였는지
                <br><br>
                <b>계산</b><br>
                작품 내 기여도와 상위 랭킹 비중을 반영하되, 작은 작품 과대평가를 막기 위해 작품 체급 보정을 추가 적용
                <br><br>
                <b>산식</b><br>
                기여도 = 최종기여도의 전체 백분위
                <br><br>
                최종기여도 = 보정기여도 × 작품체급보정
                <br><br>
                보정기여도 =<br>
                &nbsp;&nbsp;0.5 × 보정작품평균정규화<br>
                &nbsp;&nbsp;+ 0.5 × (화제성기여도 × (1위배율 + 2위배율 + 3위배율))
                <br><br>
                <b>세부 항목</b><br>
                · 화제성기여도 = 배우화제성 / 드라마화제성
                <br>
                · 1위배율 = 1위횟수 / 출연작품수
                <br>
                · 2위배율 = 0.5 × (2위횟수 / 출연작품수)
                <br>
                · 3위배율 = 0.3 × (3위횟수 / 출연작품수)
                <br>
                · 작품체급보정 = 0.45 + 0.55 × 작품체급백분위
                <br>
                · 작품체급백분위 = 드라마화제성의 전체 백분위
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='spacer-lg'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='card'>
            <div class='rep-title'>3. 최종 합산점수</div>
            <div class='actor-sub'>
                <b>산식</b><br>
                합산점수 = 100 × (0.4 × 생산력 + 0.3 × 안정성 + 0.3 × 기여도)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='spacer-lg'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='card'>
            <div class='rep-title'>4. 등급 컷</div>
            <div class='actor-sub'>
                Top-S : 상위 99% 이상<br>
                Top-A : 상위 97% 이상<br>
                Top-B : 상위 93% 이상<br>
                Top-C : 상위 85% 이상<br>
                Middle-A : 상위 70% 이상<br>
                Middle-B : 상위 50% 이상<br>
                Middle-C : 상위 30% 이상<br>
                Base-A : 상위 15% 이상<br>
                Base-B : 상위 5% 이상<br>
                Base-C : 그 미만
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def table_styler(df: pd.DataFrame):
    show = df[VISIBLE_COLUMNS].copy()
    show["합산점수"] = pd.to_numeric(show["합산점수"], errors="coerce").map(format_score)
    for c in ["생산백분율", "안정백분율", "기여백분율"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").map(format_percent_0)
    for c in ["배우화제성", "출연작품수"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").map(format_int)
    show["#"] = show["#"].astype(int)

    def bg_color(val):
        grade = str(val)
        color = GRADE_BG.get(grade, "#ffffff")
        text_color = grade_text_color(grade)
        return f"background-color: {color}; color: {text_color}; font-weight: 800; border-radius: 8px;"

    styler = (
        show.style
        .map(bg_color, subset=["합산티어", "생산력등급", "안정성등급", "기여도등급"])
        .set_properties(subset=["합산티어"], **{"font-weight": "900"})
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#f1f5fb"), ("color", "#374151"), ("font-weight", "800")]},
            {"selector": "td", "props": [("padding", "8px 10px")]},
        ])
    )
    return styler


def render_overview(raw_df: pd.DataFrame, result_df: pd.DataFrame):
    st.markdown("<div class='section-title'>OVERVIEW</div>", unsafe_allow_html=True)

    # 정의된 함수를 호출하여 데이터 기준 기간 텍스트를 가져오고 화면에 캡션으로 출력합니다.
    period_caption = get_data_period_caption(raw_df)
    if period_caption:
        st.caption(period_caption)

    total_actors = result_df["배우"].nunique()
    total_programs = raw_df["프로그램명"].nunique()
    top_ratio = (result_df["대분류티어"] == "Top").mean() * 100
    top1 = result_df.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("전체 배우", format_int(total_actors), "분석 대상 배우 수")
    with c2:
        metric_card("전체 작품", format_int(total_programs), "RAW 기준 프로그램 수")
    with c3:
        metric_card("Top 비중", f"{top_ratio:.0f}%", "합산티어 기준")
    with c4:
        metric_card("현재 1위 배우", top1["배우"], f"합산점수 {format_score(top1['합산점수'])}")

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Top 10</div>", unsafe_allow_html=True)
    top10 = result_df.head(10)
    cols = st.columns(5)
    for i, (_, row) in enumerate(top10.iterrows()):
        with cols[i % 5]:
            top10_card(int(row["#"]), row["배우"], row["합산티어"], row["합산점수"])

    st.markdown("<div class='spacer-lg'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>등급별 대표배우</div>", unsafe_allow_html=True)
    rep_cols = st.columns(5)
    for i, grade in enumerate(GRADE_ORDER):
        sub = result_df[result_df["합산티어"] == grade].head(5)
        with rep_cols[i % 5]:
            representative_card(grade, sub)

    st.markdown("<div class='spacer-lg'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>전체 배우 리스트</div>", unsafe_allow_html=True)
    st.dataframe(table_styler(result_df), use_container_width=True, hide_index=True, height=760)


def similar_tier_actors(result_df: pd.DataFrame, row: pd.Series, top_n: int = 4) -> pd.DataFrame:
    pool = result_df[
        (result_df["생산력등급"] == row["생산력등급"]) |
        (result_df["안정성등급"] == row["안정성등급"]) |
        (result_df["기여도등급"] == row["기여도등급"])
    ].copy()
    pool = pool[pool["배우"] != row["배우"]].copy()
    if pool.empty:
        return pool
    match_count = (
        (pool["생산력등급"] == row["생산력등급"]).astype(int)
        + (pool["안정성등급"] == row["안정성등급"]).astype(int)
        + (pool["기여도등급"] == row["기여도등급"]).astype(int)
    )
    pool["match_count"] = match_count
    pool["distance"] = np.sqrt(
        (pool["생산백분율"] - row["생산백분율"]) ** 2
        + (pool["안정백분율"] - row["안정백분율"]) ** 2
        + (pool["기여백분율"] - row["기여백분율"]) ** 2
    )
    return pool.sort_values(["match_count", "distance", "합산점수"], ascending=[False, True, False]).head(top_n)


def render_detail(raw_df: pd.DataFrame, result_df: pd.DataFrame):
    st.markdown("<div class='section-title'>배우 상세 보기</div>", unsafe_allow_html=True)
    st.markdown("<div class='select-hint'>배우명을 검색해서 원하는 배우를 선택해 주세요.</div>", unsafe_allow_html=True)
    names = result_df["배우"].tolist()
    selected_actor = st.selectbox("배우 선택", names, index=0, placeholder="배우명을 검색해 선택")
    row = result_df[result_df["배우"] == selected_actor].iloc[0]

    actor_summary_card(row)
    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_all = make_triangle_chart(
            [row["생산_전체점수"], row["안정_전체점수"], row["기여_전체점수"]],
            "항목별 점수 · 전체 기준",
        )
        st.plotly_chart(fig_all, use_container_width=True)
    with c2:
        fig_tier = make_triangle_chart(
            [row["생산_대분류내점수"], row["안정_대분류내점수"], row["기여_대분류내점수"]],
            f"항목별 점수 · {row['대분류티어']} 등급 기준",
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>대표출연작</div>", unsafe_allow_html=True)
    actor_programs = build_actor_program_summary(raw_df, selected_actor).head(6)
    cols = st.columns(3)
    for idx, r in actor_programs.iterrows():
        with cols[idx % 3]:
            work_card(r["프로그램명"], r["드라마화제성"], r["배우화제성"])

    st.markdown("<div class='section-title'>유사티어 배우</div>", unsafe_allow_html=True)
    sim = similar_tier_actors(result_df, row, 4)
    cols = st.columns(4)
    for i, (_, r) in enumerate(sim.iterrows()):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class='mini-card'>
                    <div class='actor-name'>{r['배우']}</div>
                    <div class='actor-sub'>생산 {r['생산력등급']} · 안정 {r['안정성등급']} · 기여 {r['기여도등급']}</div>
                    <div class='actor-sub' style='margin-top:8px;'>합산티어 <b>{r['합산티어']}</b> · 점수 {format_score(r['합산점수'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def compare_table_rows(result_df: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    cols = ["배우", "합산티어", "생산력등급", "안정성등급", "기여도등급", "합산점수", "배우화제성", "출연작품수"]
    return result_df[result_df["배우"].isin(names)][cols].sort_values("합산점수", ascending=False)


def render_compare(raw_df: pd.DataFrame, result_df: pd.DataFrame):
    st.markdown("<div class='section-title'>배우 모아보기</div>", unsafe_allow_html=True)
    mode = st.radio(
        "보기 옵션",
        ["작품 검색해서 모아보기", "등급 안에서 비교하기", "배우 직접 선택 1대1 비교"],
        horizontal=True,
    )

    selected_names: List[str] = []
    if mode == "작품 검색해서 모아보기":
        program_list = sorted(raw_df["프로그램명"].dropna().astype(str).unique().tolist())
        selected_program = st.selectbox("작품 선택", program_list, index=0, placeholder="작품명을 검색해 선택")
        matched = raw_df[raw_df["프로그램명"] == selected_program]
        selected_names = matched["인물명"].dropna().astype(str).unique().tolist()
        st.caption(f"선택 작품: {selected_program} · 배우 {len(selected_names)}명")
    elif mode == "등급 안에서 비교하기":
        c1, c2 = st.columns(2)
        tier_type_options = {
            "합산티어": "합산티어",
            "생산력등급": "생산력등급",
            "안정성등급": "안정성등급",
            "기여도등급": "기여도등급",
        }
        with c1:
            tier_field_label = st.selectbox("비교 기준", list(tier_type_options.keys()), index=0)
        with c2:
            tier_field = tier_type_options[tier_field_label]
            available = [g for g in GRADE_ORDER if g in result_df[tier_field].dropna().unique().tolist()]
            selected_grade = st.selectbox("등급 선택", available, index=0)
        filtered = result_df[result_df[tier_field] == selected_grade].copy().sort_values(["합산점수", "배우화제성"], ascending=[False, False])
        selected_names = filtered["배우"].tolist()[:20]
        st.caption(f"{tier_field_label}: {selected_grade} · {len(filtered):,}명 중 상위 20명 표시")
    else:
        names = result_df["배우"].tolist()
        left, right = st.columns(2)
        with left:
            actor1 = st.selectbox("배우 1", names, index=0, placeholder="배우명 검색")
        with right:
            default_idx = 1 if len(names) > 1 else 0
            actor2 = st.selectbox("배우 2", names, index=default_idx, placeholder="배우명 검색")
        selected_names = [a for a in [actor1, actor2] if a]

    if not selected_names:
        st.info("조건을 선택하면 비교 대상이 표시됩니다.")
        return

    comp_df = compare_table_rows(result_df, selected_names)
    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    st.dataframe(comp_df.style.format({"합산점수": "{:.2f}", "배우화제성": "{:,.0f}", "출연작품수": "{:,.0f}"}), use_container_width=True, hide_index=True)

    chart_names = comp_df["배우"].tolist()[:8]
    fig = go.Figure()
    all_values = []
    for name in chart_names:
        r = result_df[result_df["배우"] == name].iloc[0]
        vals = [r["생산_전체점수"], r["안정_전체점수"], r["기여_전체점수"], r["생산_전체점수"]]
        all_values.extend(vals[:-1])
        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=["생산력", "안정성", "기여도", "생산력"],
                mode="lines+markers",
                line=dict(width=3),
                marker=dict(size=8),
                name=name,
            )
        )
    if mode == "배우 직접 선택 1대1 비교" and len(chart_names) == 2 and all_values:
        rmin, rmax = min(all_values), max(all_values)
        pad = max(6, (rmax - rmin) * 0.45)
        low = max(0, math.floor((rmin - pad) / 5) * 5)
        high = min(100, math.ceil((rmax + pad) / 5) * 5)
    else:
        low, high = 0, 100
    fig.update_layout(
        title="선택 배우 비교 · 항목별 점수",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        polar=dict(radialaxis=dict(visible=True, range=[low, high])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    inject_css()
    st.markdown("<div class='page-title'>배우 다차원 화제성 지표 - 드라마</div>", unsafe_allow_html=True)

    raw_df = load_raw_from_gsheet()
    result_df = build_result_table(raw_df)

    with st.sidebar:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        page = st.radio("", ["OVERVIEW", "배우 상세보기", "배우 모아보기", "참고사항"], index=0, label_visibility="collapsed")
        st.markdown("<div class='sidebar-footnote'>문의 : 미디어마케팅팀 데이터인사이트파트</div>", unsafe_allow_html=True)

    if page == "OVERVIEW":
        render_overview(raw_df, result_df)
    elif page == "배우 상세보기":
        render_detail(raw_df, result_df)
    elif page == "배우 모아보기":
        render_compare(raw_df, result_df)
    else:
        render_reference()


if __name__ == "__main__":
    main()
