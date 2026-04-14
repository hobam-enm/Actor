import json
from io import StringIO
from pathlib import Path

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
GRADE_LABELS = {
    "TS": "최상위 1%",
    "T1": "상위 3% 이내",
    "T2": "상위 7% 이내",
    "T3": "상위 15% 이내",
    "M1": "상위 30% 이내",
    "M2": "상위 50% 이내",
    "M3": "상위 70% 이내",
    "B1": "상위 85% 이내",
    "B2": "상위 95% 이내",
    "B3": "기타 구간",
}


# -----------------------------
# Config helpers
# -----------------------------
def get_secret(path, default=None):
    cur = st.secrets
    for key in path:
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def get_config():
    thresholds = {
        "TS": float(get_secret(["grading", "TS"], 99)),
        "T1": float(get_secret(["grading", "T1"], 97)),
        "T2": float(get_secret(["grading", "T2"], 93)),
        "T3": float(get_secret(["grading", "T3"], 85)),
        "M1": float(get_secret(["grading", "M1"], 70)),
        "M2": float(get_secret(["grading", "M2"], 50)),
        "M3": float(get_secret(["grading", "M3"], 30)),
        "B1": float(get_secret(["grading", "B1"], 15)),
        "B2": float(get_secret(["grading", "B2"], 5)),
    }
    cfg = {
        "title": get_secret(["app", "title"], "배우 정량분석 대시보드"),
        "subtitle": get_secret(["app", "subtitle"], "RAW 및 정량분석 결과를 시각적으로 탐색하는 대시보드"),
        "mode": get_secret(["data", "mode"], "gsheet_private"),
        "excel_file": get_secret(["data", "excel_file"], "data/actors_quant.xlsx"),
        "raw_sheet": get_secret(["data", "raw_sheet"], "RAW"),
        "quant_sheet": get_secret(["data", "quant_sheet"], "정량분석"),
        "raw_csv_url": get_secret(["data", "raw_csv_url"], ""),
        "quant_csv_url": get_secret(["data", "quant_csv_url"], ""),
        "spreadsheet_id": get_secret(["data", "spreadsheet_id"], ""),
        "thresholds": thresholds,
    }
    return cfg


# -----------------------------
# Grade helpers
# -----------------------------
def tier_group(grade: str) -> str:
    if grade in ["TS", "T1", "T2", "T3"]:
        return "Top"
    if grade in ["M1", "M2", "M3"]:
        return "Mid"
    return "Base"


def grade_from_percentile(p: float, thresholds: dict) -> str:
    if pd.isna(p):
        return "B3"
    if p >= thresholds["TS"]:
        return "TS"
    if p >= thresholds["T1"]:
        return "T1"
    if p >= thresholds["T2"]:
        return "T2"
    if p >= thresholds["T3"]:
        return "T3"
    if p >= thresholds["M1"]:
        return "M1"
    if p >= thresholds["M2"]:
        return "M2"
    if p >= thresholds["M3"]:
        return "M3"
    if p >= thresholds["B1"]:
        return "B1"
    if p >= thresholds["B2"]:
        return "B2"
    return "B3"


# -----------------------------
# Data loading
# -----------------------------
def _read_excel_fullsheet(excel_path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(excel_path, sheet_name=sheet_name, header=None)


def _read_public_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url, header=None)


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


def _read_gsheet_private(spreadsheet_id: str, worksheet_name: str) -> pd.DataFrame:
    client = _get_gspread_client()
    ws = client.open_by_key(spreadsheet_id).worksheet(worksheet_name)
    values = ws.get_all_values()
    return pd.DataFrame(values)


@st.cache_data(show_spinner=False)
def load_raw_sheet(cfg: dict) -> pd.DataFrame:
    mode = cfg["mode"]
    if mode == "excel":
        df = pd.read_excel(cfg["excel_file"], sheet_name=cfg["raw_sheet"])
    elif mode == "public_csv":
        df = pd.read_csv(cfg["raw_csv_url"])
    elif mode == "gsheet_private":
        df = _read_gsheet_private(cfg["spreadsheet_id"], cfg["raw_sheet"])
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        raise ValueError(f"지원하지 않는 data.mode: {mode}")

    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    for col in ["드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹", "점유율"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "점유율" not in df.columns and {"배우화제성", "드라마화제성"}.issubset(df.columns):
        denom = df["드라마화제성"].replace(0, np.nan)
        df["점유율"] = df["배우화제성"] / denom
    return df


@st.cache_data(show_spinner=False)
def load_quant_sheet(cfg: dict) -> pd.DataFrame:
    mode = cfg["mode"]
    if mode == "excel":
        raw = _read_excel_fullsheet(cfg["excel_file"], cfg["quant_sheet"])
    elif mode == "public_csv":
        raw = _read_public_csv(cfg["quant_csv_url"])
    elif mode == "gsheet_private":
        raw = _read_gsheet_private(cfg["spreadsheet_id"], cfg["quant_sheet"])
    else:
        raise ValueError(f"지원하지 않는 data.mode: {mode}")

    # Expecting the real header row at Excel row 5 / index 4
    header_row_idx = 4
    header = raw.iloc[header_row_idx].tolist()
    df = raw.iloc[header_row_idx + 1 :].copy()
    df.columns = header
    df = df.dropna(subset=["배우"]).reset_index(drop=True)
    df = df.loc[:, [c for c in df.columns if pd.notna(c)]]

    # Normalize numeric fields
    numeric_cols = [
        "#", "합산점수", "생산력(백분율)", "지속력(백분율)", "기여도(백분율)",
        "드라마 화제성", "배우 화제성", "출연작품", "꾸준함지수", "보정 작품평균", "작품평균",
        "히트 분산지수", "히트 분산정규화", "보정기여도", "화제성 기여도",
        "1위배율", "2위배율", "3위배율", "랭크주차", "대표작 성과", "대표작 성과백분위",
        "단일 대형 히트 보정치",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["생산력", "안정성", "기여도", "배우", "출연작"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Percentiles and grade
    df["종합백분위"] = df["합산점수"].rank(pct=True, method="max") * 100
    df["최종등급"] = df["종합백분위"].apply(lambda x: grade_from_percentile(x, cfg["thresholds"]))
    df["Tier"] = df["최종등급"].apply(tier_group)
    df["등급설명"] = df["최종등급"].map(GRADE_LABELS)

    # User-friendly aliases
    rename_map = {
        "지속력(백분율)": "안정성(백분율)",
        "출연작품": "작품수",
    }
    df = df.rename(columns=rename_map)
    return df


# -----------------------------
# Presentation helpers
# -----------------------------
def format_pct01(v):
    if pd.isna(v):
        return "-"
    return f"{v*100:.1f}%"


def format_pct100(v):
    if pd.isna(v):
        return "-"
    return f"{v:.1f}%"


def build_actor_summary_row(row: pd.Series) -> str:
    lines = []
    prod = row.get("생산력(백분율)", np.nan)
    stab = row.get("안정성(백분율)", np.nan)
    contrib = row.get("기여도(백분율)", np.nan)

    def level(v):
        if pd.isna(v):
            return None
        if v >= 0.9:
            return "최상위권"
        if v >= 0.7:
            return "상위권"
        if v >= 0.5:
            return "중상위권"
        if v >= 0.3:
            return "중위권"
        return "하위권"

    lines.append(f"생산력은 {level(prod)}({format_pct01(prod)})")
    lines.append(f"안정성은 {level(stab)}({format_pct01(stab)})")
    lines.append(f"기여도는 {level(contrib)}({format_pct01(contrib)})")

    if prod >= 0.85 and stab >= 0.7:
        comment = "누적 화제성과 지속성이 모두 강한 편입니다."
    elif prod >= 0.85 and contrib >= 0.7:
        comment = "작품 내 존재감이 강하게 반영되는 상위권 타입입니다."
    elif stab >= 0.75 and prod < 0.7:
        comment = "폭발력보다는 안정적인 성과 유지가 강점인 타입입니다."
    elif contrib >= 0.8 and prod < 0.7:
        comment = "작품 규모 대비 배우 존재감이 돋보이는 타입입니다."
    else:
        comment = "세 축의 균형을 함께 보며 해석하는 것이 적합합니다."

    return " · ".join(lines) + "\n\n" + comment


# -----------------------------
# UI
# -----------------------------
def render_sidebar(quant_df: pd.DataFrame):
    st.sidebar.header("필터")
    actor_query = st.sidebar.text_input("배우명 검색", "")
    tier_sel = st.sidebar.multiselect("Tier", TIER_ORDER, default=TIER_ORDER)
    grade_sel = st.sidebar.multiselect("최종등급", GRADE_ORDER, default=GRADE_ORDER)

    work_min, work_max = int(quant_df["작품수"].fillna(0).min()), int(quant_df["작품수"].fillna(0).max())
    work_range = st.sidebar.slider("작품수", min_value=work_min, max_value=work_max, value=(work_min, work_max))

    score_min, score_max = float(quant_df["합산점수"].min()), float(quant_df["합산점수"].max())
    score_range = st.sidebar.slider("합산점수", min_value=float(np.floor(score_min)), max_value=float(np.ceil(score_max)), value=(float(np.floor(score_min)), float(np.ceil(score_max))))

    return {
        "actor_query": actor_query.strip(),
        "tier_sel": tier_sel,
        "grade_sel": grade_sel,
        "work_range": work_range,
        "score_range": score_range,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    if filters["actor_query"]:
        out = out[out["배우"].astype(str).str.contains(filters["actor_query"], case=False, na=False)]
    out = out[out["Tier"].isin(filters["tier_sel"])]
    out = out[out["최종등급"].isin(filters["grade_sel"])]
    out = out[out["작품수"].fillna(0).between(filters["work_range"][0], filters["work_range"][1])]
    out = out[out["합산점수"].fillna(0).between(filters["score_range"][0], filters["score_range"][1])]
    return out


def render_overview(filtered: pd.DataFrame):
    st.subheader("전체 개요")
    total = len(filtered)
    top_share = (filtered["Tier"].eq("Top").mean() * 100) if total else 0
    mid_share = (filtered["Tier"].eq("Mid").mean() * 100) if total else 0
    base_share = (filtered["Tier"].eq("Base").mean() * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 배우 수", f"{total:,}명")
    c2.metric("Top 비중", f"{top_share:.1f}%")
    c3.metric("Mid 비중", f"{mid_share:.1f}%")
    c4.metric("Base 비중", f"{base_share:.1f}%")

    left, right = st.columns([1.1, 1])
    with left:
        grade_counts = (
            filtered["최종등급"].value_counts()
            .reindex(GRADE_ORDER)
            .fillna(0)
            .rename_axis("등급")
            .reset_index(name="배우수")
        )
        fig = px.bar(grade_counts, x="등급", y="배우수", text="배우수", category_orders={"등급": GRADE_ORDER})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        tier_counts = (
            filtered["Tier"].value_counts()
            .reindex(TIER_ORDER)
            .fillna(0)
            .rename_axis("Tier")
            .reset_index(name="배우수")
        )
        fig2 = px.pie(tier_counts, names="Tier", values="배우수", hole=0.5)
        fig2.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("생산력 × 안정성 포지셔닝")
    pos_df = filtered.copy()
    pos_df["bubble_size"] = pos_df["기여도(백분율)"].fillna(0).clip(lower=0.01) * 60
    fig3 = px.scatter(
        pos_df,
        x="안정성(백분율)",
        y="생산력(백분율)",
        size="bubble_size",
        color="최종등급",
        hover_name="배우",
        hover_data={
            "기여도(백분율)": ':.1%',
            "안정성(백분율)": ':.1%',
            "생산력(백분율)": ':.1%',
            "bubble_size": False,
        },
        category_orders={"최종등급": GRADE_ORDER},
    )
    fig3.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20))
    fig3.update_xaxes(tickformat=".0%", title="안정성 백분위")
    fig3.update_yaxes(tickformat=".0%", title="생산력 백분위")
    st.plotly_chart(fig3, use_container_width=True)


def render_actor_detail(filtered: pd.DataFrame, raw_df: pd.DataFrame):
    st.subheader("배우 상세")
    actor_list = filtered.sort_values(["최종등급", "합산점수"], ascending=[True, False])["배우"].tolist()
    if not actor_list:
        st.info("현재 필터 조건에 해당하는 배우가 없습니다.")
        return

    default_actor = actor_list[0]
    actor = st.selectbox("배우 선택", actor_list, index=0)
    row = filtered.loc[filtered["배우"] == actor].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("최종등급", row["최종등급"])
    c2.metric("합산점수", f"{row['합산점수']:.2f}")
    c3.metric("종합백분위", format_pct100(row["종합백분위"]))
    c4.metric("작품수", f"{int(row['작품수']) if pd.notna(row['작품수']) else '-'}")
    c5.metric("대표작 성과", f"{row['대표작 성과']:.2f}" if pd.notna(row.get("대표작 성과")) else "-")

    left, right = st.columns([1.1, 1])
    with left:
        bars = pd.DataFrame(
            {
                "축": ["생산력", "안정성", "기여도"],
                "백분위": [
                    row.get("생산력(백분율)", np.nan),
                    row.get("안정성(백분율)", np.nan),
                    row.get("기여도(백분율)", np.nan),
                ],
            }
        )
        fig = px.bar(bars, x="백분위", y="축", orientation="h", text="백분위")
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**해석 요약**")
        st.write(build_actor_summary_row(row))
        works = row.get("출연작", "")
        if pd.notna(works) and str(works).strip():
            st.markdown("**출연작**")
            st.write(str(works))

    st.markdown("**등급 판정 근거**")
    explain = pd.DataFrame(
        {
            "항목": ["종합백분위", "생산력", "안정성", "기여도", "Tier", "등급 설명"],
            "값": [
                format_pct100(row["종합백분위"]),
                format_pct01(row.get("생산력(백분율)", np.nan)),
                format_pct01(row.get("안정성(백분율)", np.nan)),
                format_pct01(row.get("기여도(백분율)", np.nan)),
                row.get("Tier", "-"),
                row.get("등급설명", "-"),
            ],
        }
    )
    st.dataframe(explain, use_container_width=True, hide_index=True)

    st.markdown("**작품별 RAW 데이터**")
    actor_raw = raw_df.loc[raw_df["인물명"] == actor].copy()
    if actor_raw.empty:
        st.info("RAW 시트에서 해당 배우의 작품 데이터가 없습니다.")
    else:
        if "점유율" in actor_raw.columns:
            actor_raw["점유율"] = actor_raw["점유율"].apply(format_pct01)
        show_cols = [c for c in ["프로그램명", "드라마화제성", "배우화제성", "랭크인주차", "랭크인배우수", "작품내랭킹", "점유율"] if c in actor_raw.columns]
        st.dataframe(actor_raw[show_cols].sort_values(["배우화제성", "드라마화제성"], ascending=False), use_container_width=True, hide_index=True)


def render_tables(filtered: pd.DataFrame):
    st.subheader("배우 리스트")
    table_cols = [
        "최종등급", "배우", "합산점수", "종합백분위", "Tier",
        "생산력(백분율)", "안정성(백분율)", "기여도(백분율)",
        "대표작 성과", "작품수",
    ]
    table = filtered[table_cols].copy().sort_values(["최종등급", "합산점수"], ascending=[True, False])
    for col in ["종합백분위"]:
        table[col] = table[col].map(format_pct100)
    for col in ["생산력(백분율)", "안정성(백분율)", "기여도(백분율)"]:
        table[col] = table[col].map(format_pct01)
    st.dataframe(table, use_container_width=True, hide_index=True)


# -----------------------------
# Main
# -----------------------------
def main():
    st.set_page_config(page_title="배우 정량분석", page_icon="🎭", layout="wide")
    cfg = get_config()

    st.title(cfg["title"])
    st.caption(cfg["subtitle"])

    with st.expander("현재 등급 기준", expanded=False):
        threshold_df = pd.DataFrame(
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
        st.dataframe(threshold_df, use_container_width=True, hide_index=True)

    try:
        raw_df = load_raw_sheet(cfg)
        quant_df = load_quant_sheet(cfg)
    except Exception as e:
        st.error("데이터를 불러오지 못했습니다. secrets 설정 또는 파일 경로를 확인해 주세요.")
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
        render_tables(filtered)


if __name__ == "__main__":
    main()
