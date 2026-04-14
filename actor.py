from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None


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


def get_config() -> dict:
    return {
        "title": get_secret(["app", "title"], "배우 RAW 확인 대시보드"),
        "subtitle": get_secret(["app", "subtitle"], "비공개 구글시트의 RAW 시트를 그대로 확인하는 화면"),
        "spreadsheet_id": get_secret(["data", "spreadsheet_id"], ""),
        "raw_sheet": get_secret(["data", "raw_sheet"], "RAW"),
    }


# -----------------------------
# Google Sheets
# -----------------------------
def _get_gspread_client():
    if gspread is None or Credentials is None:
        raise ImportError("gspread/google-auth 패키지가 설치되어 있지 않습니다.")

    if "gcp_service_account" not in st.secrets:
        raise KeyError("st.secrets에 [gcp_service_account]가 없습니다.")

    svc = dict(st.secrets["gcp_service_account"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(svc, scopes=scopes)
    return gspread.authorize(creds)


@st.cache_data(show_spinner=False, ttl=300)
def load_raw_sheet(spreadsheet_id: str, raw_sheet: str) -> pd.DataFrame:
    client = _get_gspread_client()
    ws = client.open_by_key(spreadsheet_id).worksheet(raw_sheet)
    values = ws.get_all_values()

    if not values:
        raise ValueError("RAW 시트가 비어 있습니다.")

    # 첫 행을 헤더로 사용
    header = [str(c).strip() for c in values[0]]
    data = values[1:]
    df = pd.DataFrame(data, columns=header)

    # 완전 빈 컬럼 제거
    keep_cols = [c for c in df.columns if str(c).strip() != ""]
    df = df[keep_cols].copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 완전 빈 행 제거
    non_empty_mask = df.apply(lambda row: any(str(v).strip() != "" for v in row), axis=1)
    df = df.loc[non_empty_mask].reset_index(drop=True)

    return df


# -----------------------------
# Helpers
# -----------------------------
def find_actor_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "인물명",
        "배우",
        "배우명",
        "출연자",
        "출연자명",
        "이름",
        "name",
        "actor",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


# -----------------------------
# UI
# -----------------------------
def main():
    st.set_page_config(page_title="배우 RAW 확인", page_icon="📋", layout="wide")
    cfg = get_config()

    st.title(cfg["title"])
    st.caption(cfg["subtitle"])

    with st.expander("현재 설정", expanded=False):
        st.write({
            "spreadsheet_id": cfg["spreadsheet_id"],
            "raw_sheet": cfg["raw_sheet"],
        })

    try:
        raw_df = load_raw_sheet(cfg["spreadsheet_id"], cfg["raw_sheet"])
    except Exception as e:
        st.error("RAW 시트를 불러오지 못했습니다. 시트명, spreadsheet_id, 서비스계정 공유 여부를 확인해 주세요.")
        st.exception(e)
        st.stop()

    actor_col = find_actor_column(raw_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 행 수", f"{len(raw_df):,}")
    c2.metric("전체 컬럼 수", f"{len(raw_df.columns):,}")
    if actor_col:
        actor_series = raw_df[actor_col].astype(str).str.strip()
        actor_series = actor_series[actor_series.ne("")]
        c3.metric(f"고유 배우 수 ({actor_col})", f"{actor_series.nunique():,}")
    else:
        c3.metric("고유 배우 수", "배우 컬럼 미탐지")

    st.subheader("컬럼 목록")
    st.dataframe(pd.DataFrame({"column": raw_df.columns}), use_container_width=True, hide_index=True)

    st.subheader("RAW 테이블")

    left, right = st.columns([2, 1])
    with left:
        query = st.text_input("행 검색", "")
    with right:
        row_limit = st.selectbox("표시 행 수", [50, 100, 300, 1000, "전체"], index=1)

    view_df = raw_df.copy()
    if query:
        mask = view_df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False))
        view_df = view_df[mask.any(axis=1)].copy()

    if row_limit != "전체":
        view_df = view_df.head(int(row_limit))

    st.dataframe(view_df, use_container_width=True, hide_index=True)

    with st.expander("CSV 다운로드", expanded=False):
        csv = raw_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="RAW CSV 다운로드",
            data=csv,
            file_name="raw_export.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
