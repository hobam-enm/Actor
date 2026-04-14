import math
from typing import Dict, List

import gspread
import numpy as np
import pandas as pd
import plotly.express as px
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

# 사용자 지정 기준
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
GRADE_ORDER = ["Top-S", "Top-A", "Top-B", "Top-C", "Middle-A", "Middle-B", "Middle-C", "Base-A", "Base-B", "Base-C"]
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


# Google Sheets PERCENTRANK.INC에 가깝게 맞추기 위한 처리

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
    r_min = result["히트 분산지수"].min()
    r_max = result["히트 분산지수"].max()
    if pd.isna(r_min) or pd.isna(r_max) or math.isclose(r_min, r_max):
        result["히트 분산정규화"] = 0.0
    else:
        result["히트 분산정규화"] = (result["히트 분산지수"] - r_min) / (r_max - r_min)

    result["화제성 기여도"] = result["배우화제성"] / result["드라마화제성"]
    result["1위배율"] = grouped["r1"].sum() / result["출연작품수"]
    result["2위배율"] = grouped["r2"].sum() / result["출연작품수"]
    result["3위배율"] = grouped["r3"].sum() / result["출연작품수"]
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
        + 0.55 * (0.7 * result["히트 분산정규화"] + 0.3 * (result["출연작품수"] / (result["출연작품수"] + 2)))
        + 0.2 * (result["대표작 성과백분위"] ** 3),
    )

    result["보정기여도"] = 0.5 * p_norm + 0.5 * (
        result["화제성 기여도"] * (result["1위배율"] + result["2위배율"] + result["3위배율"])
    )

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


def make_grade_count_chart(result_df: pd.DataFrame, grade_col: str, title: str):
    counts = (
        result_df[grade_col]
        .value_counts()
        .reindex(GRADE_ORDER, fill_value=0)
        .reset_index()
    )
    counts.columns = ["등급", "인원수"]
    fig = px.bar(counts, x="등급", y="인원수", text="인원수", title=title)
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), xaxis_title=None, yaxis_title=None)
    fig.update_traces(textposition="outside")
    return fig


def make_axis_mix_chart(result_df: pd.DataFrame):
    summary = pd.DataFrame(
        {
            "축": ["생산력", "안정성", "기여도"],
            "평균 백분율": [
                result_df["생산백분율"].mean() * 100,
                result_df["안정백분율"].mean() * 100,
                result_df["기여백분율"].mean() * 100,
            ],
        }
    )
    fig = px.bar(summary, x="축", y="평균 백분율", text="평균 백분율", title="축별 평균 백분율")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20), xaxis_title=None, yaxis_title=None)
    return fig


def make_scatter(result_df: pd.DataFrame):
    fig = px.scatter(
        result_df,
        x="안정백분율",
        y="기여백분율",
        size="배우화제성",
        color="생산력등급",
        hover_name="배우",
        hover_data={
            "합산점수": ":.2f",
            "생산백분율": ":.2%",
            "안정백분율": ":.2%",
            "기여백분율": ":.2%",
            "배우화제성": ":,.0f",
            "출연작품수": ":,.0f",
        },
        title="안정성 × 기여도 포지셔닝",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20), xaxis_title="안정 백분율", yaxis_title="기여 백분율")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


def make_top10_table(result_df: pd.DataFrame) -> pd.DataFrame:
    top10 = result_df.sort_values(["합산점수", "배우화제성"], ascending=[False, False]).head(10).copy()
    return format_visible_table(top10)


def filter_result_df(result_df: pd.DataFrame, raw_df: pd.DataFrame, actor_keyword: str, program_keyword: str, prod_grades: List[str], stab_grades: List[str], cont_grades: List[str]) -> pd.DataFrame:
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


def render_summary_page(raw_df: pd.DataFrame, result_df: pd.DataFrame):
    st.subheader("요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 배우 수", f"{result_df['배우'].nunique():,}")
    c2.metric("전체 작품 수", f"{raw_df['프로그램명'].nunique():,}")
    c3.metric("Top 구간 비중", f"{((result_df['생산력등급'].isin(GRADE_ORDER[:4])).mean() * 100):.1f}%")
    c4.metric("평균 합산점수", f"{result_df['합산점수'].mean():,.2f}")

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(make_scatter(result_df), use_container_width=True)
    with right:
        st.plotly_chart(make_axis_mix_chart(result_df), use_container_width=True)
        st.markdown("### 합산점수 상위 10")
        st.dataframe(make_top10_table(result_df), use_container_width=True, hide_index=True, height=360)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(make_grade_count_chart(result_df, "생산력등급", "생산력 등급 분포"), use_container_width=True)
    with g2:
        st.plotly_chart(make_grade_count_chart(result_df, "안정성등급", "안정성 등급 분포"), use_container_width=True)
    with g3:
        st.plotly_chart(make_grade_count_chart(result_df, "기여도등급", "기여도 등급 분포"), use_container_width=True)


def render_detail_page(raw_df: pd.DataFrame, result_df: pd.DataFrame):
    st.subheader("상세 탐색")

    with st.sidebar:
        st.markdown("### 상세 필터")
        actor_keyword = st.text_input("배우 검색", "")
        program_keyword = st.text_input("작품 검색", "")
        prod_grades = st.multiselect("생산력 등급", GRADE_ORDER)
        stab_grades = st.multiselect("안정성 등급", GRADE_ORDER)
        cont_grades = st.multiselect("기여도 등급", GRADE_ORDER)

    filtered = filter_result_df(result_df, raw_df, actor_keyword, program_keyword, prod_grades, stab_grades, cont_grades)

    c1, c2, c3 = st.columns(3)
    c1.metric("조회 배우 수", f"{len(filtered):,}")
    c2.metric("평균 합산점수", f"{filtered['합산점수'].mean():,.2f}" if len(filtered) else "0.00")
    c3.metric("조회 작품 수", f"{raw_df[raw_df['인물명'].isin(filtered['배우'])]['프로그램명'].nunique():,}" if len(filtered) else "0")

    st.markdown("### 결과 테이블")
    st.dataframe(format_visible_table(filtered), use_container_width=True, hide_index=True, height=680)

    csv = filtered[VISIBLE_COLUMNS].to_csv(index=False).encode("utf-8-sig")
    st.download_button("결과 테이블 CSV 다운로드", data=csv, file_name="actor_quant_result_table.csv", mime="text/csv")

    if len(filtered) == 0:
        return

    st.markdown("### 배우 상세 보기")
    selected_actor = st.selectbox("배우 선택", filtered["배우"].tolist())
    actor_row = filtered[filtered["배우"] == selected_actor].iloc[0]
    actor_raw = raw_df[raw_df["인물명"] == selected_actor].copy()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("생산력 등급", actor_row["생산력등급"])
    a2.metric("안정성 등급", actor_row["안정성등급"])
    a3.metric("기여도 등급", actor_row["기여도등급"])
    a4.metric("합산점수", f"{actor_row['합산점수']:,.2f}")

    bar_df = pd.DataFrame(
        {
            "축": ["생산", "안정", "기여"],
            "백분율": [actor_row["생산백분율"] * 100, actor_row["안정백분율"] * 100, actor_row["기여백분율"] * 100],
        }
    )
    fig = px.bar(bar_df, x="축", y="백분율", text="백분율", title=f"{selected_actor} 축별 백분율")
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    info1, info2 = st.columns(2)
    with info1:
        st.markdown("### 배우 요약")
        summary_df = pd.DataFrame(
            {
                "항목": ["배우", "배우화제성", "출연작품수", "생산백분율", "안정백분율", "기여백분율", "출연작"],
                "값": [
                    actor_row["배우"],
                    format_int(actor_row["배우화제성"]),
                    format_int(actor_row["출연작품수"]),
                    format_percent_0(actor_row["생산백분율"]),
                    format_percent_0(actor_row["안정백분율"]),
                    format_percent_0(actor_row["기여백분율"]),
                    actor_row["출연작"],
                ],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True, height=290)
    with info2:
        st.markdown("### 출연 작품별 RAW")
        actor_raw_display = actor_raw[[c for c in ["프로그램명", "드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹"] if c in actor_raw.columns]].copy()
        for col in ["드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹"]:
            if col in actor_raw_display.columns:
                actor_raw_display[col] = pd.to_numeric(actor_raw_display[col], errors="coerce").map(format_int)
        st.dataframe(actor_raw_display, use_container_width=True, hide_index=True, height=290)


def main():
    app_cfg = get_secret_section("app")
    st.title(app_cfg.get("title", "배우 정량분석 대시보드"))
    st.caption(app_cfg.get("subtitle", "RAW 시트를 기반으로 배우 정량분석 결과를 계산하고 시각화합니다."))

    raw_df = load_raw_from_gsheet()
    result_df = build_result_table(raw_df)

    summary_tab, detail_tab = st.tabs(["메인", "상세"])
    with summary_tab:
        render_summary_page(raw_df, result_df)
    with detail_tab:
        render_detail_page(raw_df, result_df)


if __name__ == "__main__":
    main()
