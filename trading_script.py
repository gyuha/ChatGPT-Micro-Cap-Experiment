"""ChatGPT 마이크로캡 포트폴리오 관리 유틸리티

이 모듈은 원본 스크립트를 다음과 같이 개선:
- Yahoo Finance와 Stooq 간의 견고한 폴백 시스템으로 시장 데이터 중앙 집중화
- 모든 가격 요청이 동일한 접근자를 통하도록 보장 (일관성 확보)
- Yahoo의 빈 데이터프레임 처리 (예외 없이) 하여 실제 폴백이 작동하도록 함
- Stooq 출력을 Yahoo와 유사한 컬럼으로 정규화
- 주말 처리를 일관성 있고 테스트 가능하게 만듦
- 이전 실행과의 동작 및 CSV 형식 호환성 유지

주의사항:
- 일부 티커/인덱스는 Stooq에서 사용할 수 없음 (예: ^RUT). 이들은 Yahoo를 유지.
- Stooq 종료 날짜는 배타적임; 범위에 대해 +1일 추가함.
- "Adj Close"는 다운스트림 기대치와 일치하도록 Stooq에서 "Close"와 동일하게 설정.

[분석] 이 설계는 데이터 소스의 안정성을 크게 향상시킴. 단일 데이터 소스 의존으로 인한
시스템 중단을 방지하고, 여러 소스 간의 데이터 형식 차이를 정규화하여 일관된 분석을 가능하게 함.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
import yfinance as yf

# Optional pandas-datareader import for Stooq access
try:
    _HAS_PDR = True
except Exception:
    _HAS_PDR = False

# -------- 날짜 오버라이드 시스템 --------
# 백테스트 및 테스트 목적으로 특정 날짜를 '오늘'로 설정하는 기능
# [분석] 실제 운용에서는 None, 백테스트 시에는 특정 날짜로 설정하여
# 과거 데이터로 시스템을 테스트할 수 있게 함
ASOF_DATE: pd.Timestamp | None = None


def set_asof(date: str | datetime | pd.Timestamp | None) -> None:
    """전역 'as of' 날짜를 설정하여 스크립트가 해당 날짜를 '오늘'로 처리하도록 함.
    'YYYY-MM-DD' 형식 사용.

    [원인] 백테스트나 과거 시점 분석을 위해 시간을 조작할 필요가 있음
    [분석] 이 기능으로 인해 동일한 코드로 실시간 운용과 백테스트 모두 가능
    """
    global ASOF_DATE
    if date is None:
        print("이전 날짜가 전달되지 않았습니다. 오늘 날짜를 사용합니다...")
        ASOF_DATE = None
        return
    ASOF_DATE = pd.Timestamp(date).normalize()  # 시간 정보 제거하고 날짜만 사용
    pure_date = ASOF_DATE.date()

    print(f"날짜를 {pure_date}로 설정합니다.")


# 환경 변수를 통한 오버라이드 허용:  ASOF_DATE=YYYY-MM-DD python trading_script.py
# [분석] 명령행 실행 시 환경 변수로도 날짜 설정이 가능하여 자동화에 유리함
_env_asof = os.environ.get("ASOF_DATE")
if _env_asof:
    set_asof(_env_asof)


def _effective_now() -> datetime:
    """현재 유효한 시간을 반환 (ASOF_DATE가 설정되어 있으면 그것을, 아니면 실제 현재 시간)

    [분석] 시스템 전체에서 시간 참조를 통일하여 백테스트와 실제 운용 간 일관성 보장
    """
    return ASOF_DATE.to_pydatetime() if ASOF_DATE is not None else datetime.now()


# ------------------------------
# 전역 변수 / 파일 위치 설정
# ------------------------------
# [분석] 스크립트와 동일한 디렉토리에 데이터 파일들을 저장하여 이식성 확보
SCRIPT_DIR = Path(__file__).resolve().parent  # 스크립트가 위치한 절대 경로
DATA_DIR = SCRIPT_DIR  # 기본적으로 스크립트와 같은 위치에 파일 저장
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"  # 포트폴리오 상태 CSV
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"  # 거래 로그 CSV
DEFAULT_BENCHMARKS = ["IWO", "XBI", "SPY", "IWM"]  # 기본 벤치마크 티커들

# ------------------------------
# 구성 도우미 함수들 — 벤치마크 티커 설정 (tickers.json)
# ------------------------------

logger = logging.getLogger(__name__)


def _read_json_file(path: Path) -> Optional[Dict]:
    """JSON 파일을 읽고 파싱함. 성공 시 dict 반환, 찾을 수 없거나 유효하지 않으면 None 반환.

    [원인] 설정 파일 읽기에서 발생할 수 있는 다양한 오류 상황을 모두 처리해야 함
    [분석] 파일 없음, JSON 파싱 오류, 기타 IO 오류를 구분하여 처리하여 시스템 안정성 확보

    - FileNotFoundError -> None 반환
    - JSON 디코드 오류 -> 경고 로그 후 None 반환
    - 기타 IO 오류 -> 경고 로그 후 None 반환
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None  # 파일이 없으면 조용히 None 반환
    except json.JSONDecodeError as exc:
        logger.warning(
            "tickers.json이 존재하지만 형식이 잘못됨: %s -> %s. 기본값으로 되돌아감.",
            path,
            exc,
        )
        return None
    except Exception as exc:
        logger.warning(
            "tickers.json을 읽을 수 없음 (%s): %s. 기본값으로 되돌아감.", path, exc
        )
        return None


def load_benchmarks(script_dir: Path | None = None) -> List[str]:
    """벤치마크 티커 목록을 반환.

    [원인] 사용자가 벤치마크를 커스터마이징할 수 있어야 하지만, 설정 파일이 없어도 동작해야 함
    [분석] 다음 위치에서 `tickers.json` 파일을 찾음:
      - script_dir (제공된 경우) 또는 모듈 SCRIPT_DIR, 그리고
      - script_dir.parent (프로젝트 루트 후보).

    예상되는 스키마:
      {"benchmarks": ["IWO", "XBI", "SPY", "IWM"]}

    동작:
    - 파일이 없거나 형식이 잘못된 경우 -> DEFAULT_BENCHMARKS 복사본 반환.
    - 'benchmarks' 키가 없거나 리스트가 아닌 경우 -> 경고 로그 후 기본값 반환.
    - 티커들을 정규화(strip, upper)하고 순서를 유지하면서 중복 제거.
    """
    base = Path(script_dir) if script_dir else SCRIPT_DIR
    candidates = [base, base.parent]  # 스크립트 위치와 상위 디렉토리에서 검색

    cfg = None
    cfg_path = None
    for c in candidates:
        p = (c / "tickers.json").resolve()
        data = _read_json_file(p)
        if data is not None:
            cfg = data
            cfg_path = p
            break

    if not cfg:
        return DEFAULT_BENCHMARKS.copy()

    benchmarks = cfg.get("benchmarks")
    if not isinstance(benchmarks, list):
        logger.warning(
            "tickers.json at %s에서 'benchmarks' 배열이 없음. 기본값으로 되돌아감.",
            cfg_path,
        )
        return DEFAULT_BENCHMARKS.copy()

    # 티커 정규화 및 중복 제거
    seen = set()
    result: list[str] = []
    for t in benchmarks:
        if not isinstance(t, str):
            continue
        up = t.strip().upper()  # 공백 제거하고 대문자로 변환
        if not up:
            continue
        if up not in seen:
            seen.add(up)
            result.append(up)

    return result if result else DEFAULT_BENCHMARKS.copy()


# ------------------------------
# 날짜 도우미 함수들
# ------------------------------


def last_trading_date(today: datetime | None = None) -> pd.Timestamp:
    """마지막 거래일 반환 (월-금), 토/일을 금요일로 매핑.

    [원인] 주식 시장은 주말에 거래하지 않으므로 가격 데이터도 주말에는 없음
    [분석] 토요일이나 일요일에 스크립트를 실행해도 금요일 데이터를 기준으로 작동하도록 함
    이를 통해 주말에도 포트폴리오 분석이 가능함
    """
    dt = pd.Timestamp(today or _effective_now())
    if dt.weekday() == 5:  # 토 -> 금
        return (dt - pd.Timedelta(days=1)).normalize()
    if dt.weekday() == 6:  # 일 -> 금
        return (dt - pd.Timedelta(days=2)).normalize()
    return dt.normalize()  # 시간 정보 제거하고 날짜만 반환


def check_weekend() -> str:
    """마지막 거래일에 대한 ISO 날짜 문자열을 반환하는 하위 호환성 래퍼.

    [분석] 기존 코드와의 호환성을 위해 문자열 형태로 반환
    """
    return last_trading_date().date().isoformat()


def trading_day_window(
    target: datetime | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """마지막 거래일에 대한 [시작, 종료) 윈도우 (주말에는 금요일).

    [원인] 일간 데이터를 조회할 때 정확한 날짜 범위가 필요함
    [분석] 반배타적 구간 [start, end)를 사용하여 데이터 조회 API와 일치시킴
    """
    d = last_trading_date(target)
    return d, (d + pd.Timedelta(days=1))  # [금요일, 토요일) 형태


# ------------------------------
# 데이터 접근 계층 - 핵심 시스템
# ------------------------------

# 알려진 Stooq 심볼 리매핑 (일반적인 인덱스용)
# [분석] 각 데이터 소스마다 동일한 지수를 다른 심볼로 표기하므로 매핑 테이블 필요
STOOQ_MAP = {
    "^GSPC": "^SPX",  # S&P 500 - Yahoo와 Stooq 간 심볼 차이
    "^DJI": "^DJI",  # 다우존스 - 동일 심볼
    "^IXIC": "^IXIC",  # 나스닥 종합 - 동일 심볼
    # "^RUT": Stooq에 없음; Yahoo 유지
}

# Stooq에서 시도하면 안 되는 심볼들
# [원인] 일부 인덱스는 Stooq에서 제공하지 않아 불필요한 API 호출을 피해야 함
STOOQ_BLOCKLIST = {"^RUT"}


# ------------------------------
# 데이터 접근 계층 (업데이트됨) - 다중 소스 폴백 시스템
# ------------------------------


@dataclass
class FetchResult:
    """데이터 가져오기 결과를 담는 클래스

    [분석] 데이터와 함께 데이터 소스 정보를 저장하여 디버깅과 모니터링에 활용
    """

    df: pd.DataFrame  # 실제 가격 데이터
    source: str  # "yahoo" | "stooq-pdr" | "stooq-csv" | "yahoo:<proxy>-proxy" | "empty"


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 인덱스를 DatetimeIndex로 변환

    [원인] 다양한 데이터 소스에서 날짜 형식이 다를 수 있음
    [분석] 통일된 날짜 인덱스 형식으로 변환하여 후속 처리의 일관성 보장
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass  # 변환 실패 시 원본 유지
    return df


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 데이터를 표준 형식으로 정규화

    [원인] 다른 데이터 소스들이 서로 다른 컬럼명이나 구조를 가질 수 있음
    [분석] 모든 소스의 데이터를 동일한 컬럼 구조로 통일하여 하위 코드의 호환성 보장
    """
    # 예상되는 모든 컬럼이 존재하는지 확인
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan  # 없는 컬럼은 NaN으로 채움
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]  # 수정 종가가 없으면 종가와 동일하게 설정
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return df[cols]


def _yahoo_download(ticker: str, **kwargs: Any) -> pd.DataFrame:
    """실제 User-Agent를 사용하여 yfinance.download 호출하고 모든 출력을 숨김

    [원인] Yahoo Finance는 봇 트래픽을 차단할 수 있고, yfinance는 많은 로그를 출력함
    [분석] 실제 브라우저처럼 보이는 User-Agent 설정과 모든 출력 숨김으로 안정성 향상
    """
    import io
    import logging
    from contextlib import redirect_stderr, redirect_stdout

    import requests

    # 실제 브라우저처럼 보이는 세션 설정
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    kwargs.setdefault("progress", False)  # 진행 표시줄 비활성화
    kwargs.setdefault("threads", False)  # 멀티스레드 비활성화
    kwargs.setdefault("session", sess)

    # yfinance 로깅 완전 차단
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 경고도 모두 숨김
        try:
            with (
                redirect_stdout(buf),
                redirect_stderr(buf),
            ):  # 모든 출력을 버퍼로 리다이렉트
                df = cast(pd.DataFrame, yf.download(ticker, **kwargs))
        except Exception:
            return pd.DataFrame()  # 오류 시 빈 DataFrame 반환
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _stooq_csv_download(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Stooq CSV 엔드포인트에서 OHLCV 가져오기 (일간 데이터). 미국 티커 및 많은 ETF에 적합.

    [원인] pandas-datareader가 없거나 실패할 때를 대비한 직접 CSV 다운로드 방식
    [분석] HTTP 요청으로 직접 CSV를 받아와서 파싱하므로 외부 라이브러리 의존성 최소화
    """
    import io

    import requests

    if ticker in STOOQ_BLOCKLIST:
        return pd.DataFrame()
    t = STOOQ_MAP.get(ticker, ticker)

    # Stooq 일간 CSV: 소문자; 주식/ETF는 .us 사용, 인덱스는 ^ 접두사 유지
    if not t.startswith("^"):
        sym = t.lower()
        if not sym.endswith(".us"):
            sym = f"{sym}.us"  # 미국 주식은 .us 접미사 추가
    else:
        sym = t.lower()  # 인덱스는 소문자로만 변환

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"  # 일간 데이터 CSV 엔드포인트
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()

        # 날짜 컬럼을 datetime으로 변환하고 인덱스로 설정
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # [start, end) 범위로 필터링 (Stooq 종료일은 배타적)
        df = df.loc[(df.index >= start.normalize()) & (df.index < end.normalize())]

        # Yahoo와 유사한 스키마로 정규화
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    except Exception:
        return pd.DataFrame()  # 모든 오류에서 빈 DataFrame 반환


def _stooq_download(
    ticker: str,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
) -> pd.DataFrame:
    """pandas-datareader를 통해 Stooq에서 OHLCV 가져오기; 실패 시 빈 DF 반환.

    [원인] pandas-datareader는 더 안정적인 Stooq 접근 방식이지만 외부 의존성
    [분석] 우선적으로 시도하고 실패하면 직접 CSV 다운로드 방식으로 폴백
    """
    if not _HAS_PDR or ticker in STOOQ_BLOCKLIST:
        return pd.DataFrame()

    t = STOOQ_MAP.get(ticker, ticker)
    if not t.startswith("^"):
        t = t.lower()  # 주식 심볼은 소문자로 변환

    try:
        # pandas-datareader가 전역적으로 사용 불가능한 경우 로컬 import
        if not _HAS_PDR:
            return pd.DataFrame()
        import pandas_datareader.data as pdr_local

        df = cast(pd.DataFrame, pdr_local.DataReader(t, "stooq", start=start, end=end))
        df.sort_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()  # 모든 예외에서 빈 DataFrame 반환


def _weekend_safe_range(
    period: str | None, start: Any, end: Any
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """구체적인 [시작, 종료) 윈도우를 계산.

    [원인] 주말이나 명시적 날짜 범위에 대해 일관된 처리가 필요함
    [분석] 다양한 입력 방식을 통일된 날짜 범위로 변환하여 데이터 조회의 일관성 보장

    - 명시적 시작/종료 제공 시: 그것들을 사용 (종료일에 +1일 추가하여 배타적으로 만듦).
    - period가 '1d'인 경우: 주말에는 마지막 거래일의 [금, 토) 윈도우 사용.
    - period가 '2d'/'5d' 같은 경우: 마지막 거래일에서 끝나는 윈도우 구축.
    """
    if start or end:
        end_ts = (
            pd.Timestamp(end) if end else last_trading_date() + pd.Timedelta(days=1)
        )
        start_ts = pd.Timestamp(start) if start else (end_ts - pd.Timedelta(days=5))
        return start_ts.normalize(), pd.Timestamp(end_ts).normalize()

    # 명시적 날짜 없음; period에서 유도
    if isinstance(period, str) and period.endswith("d"):
        days = int(period[:-1])  # '5d'에서 5 추출
    else:
        days = 1  # 기본값

    # 마지막 거래일에 고정 (일/토에는 금요일)
    end_trading = last_trading_date()
    start_ts = (end_trading - pd.Timedelta(days=days)).normalize()
    end_ts = (end_trading + pd.Timedelta(days=1)).normalize()
    return start_ts, end_ts


def download_price_data(ticker: str, **kwargs: Any) -> FetchResult:
    """다단계 폴백을 사용한 견고한 OHLCV 가져오기:

    [핵심 설계] 데이터 소스 장애에 대비한 4단계 폴백 시스템
    [분석] 단일 소스 의존의 위험을 제거하고 데이터 가용성을 극대화

    순서:
        1) Yahoo Finance (yfinance 사용)
        2) Stooq (pandas-datareader 사용)
        3) Stooq (직접 CSV 다운로드)
        4) 인덱스 프록시 (예: ^GSPC->SPY, ^RUT->IWM) Yahoo 사용

    [Open, High, Low, Close, Adj Close, Volume] 컬럼을 가진 DataFrame 반환.
    """
    # 범위 인수 추출, 주말 안전 윈도우 계산
    period = kwargs.pop("period", None)
    start = kwargs.pop("start", None)
    end = kwargs.pop("end", None)
    kwargs.setdefault("progress", False)  # 진행 표시줄 비활성화
    kwargs.setdefault("threads", False)  # 멀티스레드 비활성화

    s, e = _weekend_safe_range(period, start, end)

    # ---------- 1) Yahoo (날짜 경계 설정) ----------
    df_y = _yahoo_download(ticker, start=s, end=e, **kwargs)
    if isinstance(df_y, pd.DataFrame) and not df_y.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_y)), "yahoo")

    # ---------- 2) pandas-datareader를 통한 Stooq ----------
    df_s = _stooq_download(ticker, start=s, end=e)
    if isinstance(df_s, pd.DataFrame) and not df_s.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_s)), "stooq-pdr")

    # ---------- 3) 직접 Stooq CSV ----------
    df_csv = _stooq_csv_download(ticker, s, e)
    if isinstance(df_csv, pd.DataFrame) and not df_csv.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_csv)), "stooq-csv")

    # ---------- 4) 해당하는 경우 프록시 인덱스 ----------
    proxy_map = {
        "^GSPC": "SPY",
        "^RUT": "IWM",
    }  # S&P500 -> SPY ETF, 러셀2000 -> IWM ETF
    proxy = proxy_map.get(ticker)
    if proxy:
        df_proxy = _yahoo_download(proxy, start=s, end=e, **kwargs)
        if isinstance(df_proxy, pd.DataFrame) and not df_proxy.empty:
            return FetchResult(
                _normalize_ohlcv(_to_datetime_index(df_proxy)), f"yahoo:{proxy}-proxy"
            )

    # ---------- 아무것도 작동하지 않음 ----------
    empty = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    )
    return FetchResult(empty, "empty")


# ------------------------------
# 파일 경로 구성
# ------------------------------


def set_data_dir(data_dir: Path) -> None:
    """데이터 디렉토리와 관련 CSV 파일 경로를 설정

    [원인] 사용자가 데이터 저장 위치를 커스터마이징할 수 있어야 함
    [분석] 전역 변수들을 업데이트하여 모든 파일 작업이 지정된 디렉토리에서 수행되도록 함
    """
    global DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV
    DATA_DIR = Path(data_dir)
    os.makedirs(DATA_DIR, exist_ok=True)  # 디렉토리가 없으면 생성
    PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
    TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"


# ------------------------------
# 포트폴리오 운영 함수들
# ------------------------------


def _ensure_df(
    portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]],
) -> pd.DataFrame:
    """다양한 입력 형식을 DataFrame으로 변환

    [원인] 포트폴리오 데이터가 dict, list, DataFrame 등 다양한 형태로 올 수 있음
    [분석] 입력 형식에 관계없이 일관된 DataFrame으로 변환하여 후속 처리의 일관성 보장
    """
    if isinstance(portfolio, pd.DataFrame):
        return portfolio.copy()
    if isinstance(portfolio, (dict, list)):
        return pd.DataFrame(portfolio)
    raise TypeError("포트폴리오는 DataFrame, dict, 또는 list[dict] 형태여야 합니다")


def process_portfolio(
    portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]],
    cash: float,
    interactive: bool = True,
) -> tuple[pd.DataFrame, float]:
    """포트폴리오 처리의 핵심 함수 - 수동 거래 입력, 가격 업데이트, 손절매 실행 포함

    [핵심 설계] 사용자 상호작용과 자동화된 포트폴리오 관리를 결합
    [분석] 3단계 처리 과정:
    1. 대화형 수동 거래 입력 (매수/매도 주문)
    2. 현재 가격으로 포트폴리오 평가
    3. 손절매 조건 확인 및 실행
    """
    today_iso = last_trading_date().date().isoformat()  # YYYY-MM-DD 형식
    portfolio_df = _ensure_df(portfolio)

    results: list[dict[str, object]] = []  # 결과 저장용 리스트
    total_value = 0.0  # 총 포트폴리오 가치
    total_pnl = 0.0  # 총 손익

    # ------- 대화형 거래 입력 (MOO 지원) -------
    if interactive:
        while True:
            print(portfolio_df)
            action = (
                input(
                    f""" 현금 {cash}원이 있습니다.
수동 거래를 기록하시겠습니까? 매수는 'b', 매도는 's', 계속하려면 Enter: """
                )
                .strip()
                .lower()
            )

            if action == "b":  # 매수 주문
                ticker = input("티커 심볼 입력: ").strip().upper()
                order_type = (
                    input("주문 유형? 'm' = 시장가 개장, 'l' = 지정가: ")
                    .strip()
                    .lower()
                )

                try:
                    shares = float(input("주식 수량 입력: "))
                    if shares <= 0:
                        raise ValueError
                except ValueError:
                    print("유효하지 않은 주식 수량입니다. 매수가 취소되었습니다.")
                    continue

                if order_type == "m":  # 시장가 개장 주문 (Market On Open)
                    try:
                        stop_loss = float(input("손절가 입력 (0은 건너뛰기): "))
                        if stop_loss < 0:
                            raise ValueError
                    except ValueError:
                        print("유효하지 않은 손절가입니다. 매수가 취소되었습니다.")
                        continue

                    # 개장가로 실행하기 위해 당일 데이터 조회
                    s, e = trading_day_window()
                    fetch = download_price_data(
                        ticker, start=s, end=e, auto_adjust=False, progress=False
                    )
                    data = fetch.df
                    if data.empty:
                        print(
                            f"{ticker}의 MOO 매수 실패: 시장 데이터 없음 (소스={fetch.source})."
                        )
                        continue

                    # 개장가 사용, 없으면 종가로 대체
                    o = (
                        float(data["Open"].iloc[-1])
                        if "Open" in data
                        else float(data["Close"].iloc[-1])
                    )
                    exec_price = round(o, 2)
                    notional = exec_price * shares  # 총 매수 금액
                    if notional > cash:
                        print(
                            f"{ticker}의 MOO 매수 실패: 비용 {notional:.2f}이 현금 {cash:.2f}을 초과합니다."
                        )
                        continue

                    # 거래 로그 기록
                    log = {
                        "Date": today_iso,
                        "Ticker": ticker,
                        "Shares Bought": shares,
                        "Buy Price": exec_price,
                        "Cost Basis": notional,
                        "PnL": 0.0,
                        "Reason": "수동 매수 MOO - 체결됨",
                    }
                    # --- 수동 MOO 매수 로깅 ---
                    if os.path.exists(TRADE_LOG_CSV):
                        df_log = pd.read_csv(TRADE_LOG_CSV)
                        if df_log.empty:
                            df_log = pd.DataFrame([log])
                        else:
                            df_log = pd.concat(
                                [df_log, pd.DataFrame([log])], ignore_index=True
                            )
                    else:
                        df_log = pd.DataFrame([log])
                    df_log.to_csv(TRADE_LOG_CSV, index=False)

                    # 포트폴리오 업데이트 (기존 보유 시 평균단가 계산)
                    rows = portfolio_df.loc[
                        portfolio_df["ticker"].astype(str).str.upper() == ticker.upper()
                    ]
                    if rows.empty:  # 신규 매수
                        new_trade = {
                            "ticker": ticker,
                            "shares": float(shares),
                            "stop_loss": float(stop_loss),
                            "buy_price": float(exec_price),
                            "cost_basis": float(notional),
                        }
                        if portfolio_df.empty:
                            portfolio_df = pd.DataFrame([new_trade])
                        else:
                            portfolio_df = pd.concat(
                                [portfolio_df, pd.DataFrame([new_trade])],
                                ignore_index=True,
                            )
                    else:  # 추가 매수 - 평균단가 계산
                        idx = rows.index[0]
                        cur_shares = float(portfolio_df.at[idx, "shares"])
                        cur_cost = float(portfolio_df.at[idx, "cost_basis"])
                        new_shares = cur_shares + float(shares)
                        new_cost = cur_cost + float(notional)
                        avg_price = new_cost / new_shares if new_shares else 0.0
                        portfolio_df.at[idx, "shares"] = new_shares
                        portfolio_df.at[idx, "cost_basis"] = new_cost
                        portfolio_df.at[idx, "buy_price"] = avg_price
                        portfolio_df.at[idx, "stop_loss"] = float(stop_loss)

                    cash -= notional  # 현금 차감
                    print(
                        f"{ticker}의 수동 MOO 매수가 ${exec_price:.2f}에 체결되었습니다 ({fetch.source})."
                    )
                    continue

                elif order_type == "l":  # 지정가 주문
                    try:
                        buy_price = float(input("매수 지정가 입력: "))
                        stop_loss = float(input("손절가 입력 (0은 건너뛰기): "))
                        if buy_price <= 0 or stop_loss < 0:
                            raise ValueError
                    except ValueError:
                        print("유효하지 않은 입력입니다. 지정가 매수가 취소되었습니다.")
                        continue

                    cash, portfolio_df = log_manual_buy(
                        buy_price, shares, ticker, stop_loss, cash, portfolio_df
                    )
                    continue
                else:
                    print("알 수 없는 주문 유형입니다. 'm' 또는 'l'을 사용하세요.")
                    continue

            if action == "s":  # 매도 주문
                try:
                    ticker = input("티커 심볼 입력: ").strip().upper()
                    shares = float(input("매도할 주식 수량 입력 (지정가): "))
                    sell_price = float(input("매도 지정가 입력: "))
                    if shares <= 0 or sell_price <= 0:
                        raise ValueError
                except ValueError:
                    print("유효하지 않은 입력입니다. 수동 매도가 취소되었습니다.")
                    continue

                cash, portfolio_df = log_manual_sell(
                    sell_price, shares, ticker, cash, portfolio_df
                )
                continue

            break  # 거래 입력 완료, 가격 책정으로 진행

    # ------- 일일 가격 책정 + 손절매 실행 -------
    s, e = trading_day_window()  # 오늘의 거래 시간 윈도우
    for _, stock in portfolio_df.iterrows():
        ticker = str(stock["ticker"]).upper()
        shares = int(stock["shares"]) if not pd.isna(stock["shares"]) else 0
        cost = float(stock["buy_price"]) if not pd.isna(stock["buy_price"]) else 0.0
        cost_basis = (
            float(stock["cost_basis"])
            if not pd.isna(stock["cost_basis"])
            else cost * shares
        )
        stop = float(stock["stop_loss"]) if not pd.isna(stock["stop_loss"]) else 0.0

        # 현재 가격 데이터 조회
        fetch = download_price_data(
            ticker, start=s, end=e, auto_adjust=False, progress=False
        )
        data = fetch.df

        if data.empty:
            print(f"{ticker} 데이터가 없습니다 (소스={fetch.source}).")
            row = {
                "Date": today_iso,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": cost,
                "Cost Basis": cost_basis,
                "Stop Loss": stop,
                "Current Price": "",
                "Total Value": "",
                "PnL": "",
                "Action": "데이터 없음",
                "Cash Balance": "",
                "Total Equity": "",
            }
            results.append(row)
            continue

        # OHLC 가격 추출
        o = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
        high = float(data["High"].iloc[-1])
        low = float(data["Low"].iloc[-1])
        c = float(data["Close"].iloc[-1])
        if np.isnan(o):
            o = c  # 개장가가 없으면 종가로 대체

        # 손절매 로직 - 장중 최저가가 손절가 이하로 떨어졌는지 확인
        if stop and low <= stop:
            # 손절매 실행 - 개장가가 손절가 이하면 개장가로, 아니면 손절가로 실행
            exec_price = round(o if o <= stop else stop, 2)
            value = round(exec_price * shares, 2)
            pnl = round((exec_price - cost) * shares, 2)
            action = "매도 - 손절매 실행됨"
            cash += value  # 현금에 매도 대금 추가
            portfolio_df = log_sell(ticker, shares, exec_price, cost, pnl, portfolio_df)
            row = {
                "Date": today_iso,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": cost,
                "Cost Basis": cost_basis,
                "Stop Loss": stop,
                "Current Price": exec_price,
                "Total Value": value,
                "PnL": pnl,
                "Action": action,
                "Cash Balance": "",
                "Total Equity": "",
            }
        else:
            # 보유 지속 - 현재가로 평가
            price = round(c, 2)
            value = round(price * shares, 2)
            pnl = round((price - cost) * shares, 2)
            action = "보유"
            total_value += value  # 총 포트폴리오 가치에 추가
            total_pnl += pnl  # 총 손익에 추가
            row = {
                "Date": today_iso,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": cost,
                "Cost Basis": cost_basis,
                "Stop Loss": stop,
                "Current Price": price,
                "Total Value": value,
                "PnL": pnl,
                "Action": action,
                "Cash Balance": "",
                "Total Equity": "",
            }

        results.append(row)

    # 총계 행 추가
    total_row = {
        "Date": today_iso,
        "Ticker": "TOTAL",
        "Shares": "",
        "Buy Price": "",
        "Cost Basis": "",
        "Stop Loss": "",
        "Current Price": "",
        "Total Value": round(total_value, 2),
        "PnL": round(total_pnl, 2),
        "Action": "",
        "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2),
    }
    results.append(total_row)

    # CSV 파일에 결과 저장 (기존 같은 날짜 데이터는 제거 후 추가)
    df_out = pd.DataFrame(results)
    if PORTFOLIO_CSV.exists():
        existing = pd.read_csv(PORTFOLIO_CSV)
        existing = existing[existing["Date"] != str(today_iso)]  # 오늘 데이터 제거
        print("결과를 CSV에 저장 중...")
        df_out = pd.concat([existing, df_out], ignore_index=True)
    df_out.to_csv(PORTFOLIO_CSV, index=False)

    return portfolio_df, cash


# ------------------------------
# Trade logging
# ------------------------------


def log_sell(
    ticker: str,
    shares: float,
    price: float,
    cost: float,
    pnl: float,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    today = check_weekend()
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Sold": shares,
        "Sell Price": price,
        "Cost Basis": cost,
        "PnL": pnl,
        "Reason": "AUTOMATED SELL - STOPLOSS TRIGGERED",
    }
    print(f"{ticker} stop loss was met. Selling all shares.")
    portfolio = portfolio[portfolio["ticker"] != ticker]

    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)
    return portfolio


def log_manual_buy(
    buy_price: float,
    shares: float,
    ticker: str,
    stoploss: float,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    today = check_weekend()

    if interactive:
        check = input(
            f"You are placing a BUY LIMIT for {shares} {ticker} at ${buy_price:.2f}.\n"
            f"If this is a mistake, type '1': "
        )
        if check == "1":
            print("Returning...")
            return cash, chatgpt_portfolio

    if not isinstance(chatgpt_portfolio, pd.DataFrame) or chatgpt_portfolio.empty:
        chatgpt_portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

    s, e = trading_day_window()
    fetch = download_price_data(
        ticker, start=s, end=e, auto_adjust=False, progress=False
    )
    data = fetch.df
    if data.empty:
        print(
            f"Manual buy for {ticker} failed: no market data available (source={fetch.source})."
        )
        return cash, chatgpt_portfolio

    o = float(data.get("Open", [np.nan])[-1])
    h = float(data["High"].iloc[-1])
    l = float(data["Low"].iloc[-1])
    if np.isnan(o):
        o = float(data["Close"].iloc[-1])

    if o <= buy_price:
        exec_price = o
    elif l <= buy_price:
        exec_price = buy_price
    else:
        print(
            f"Buy limit ${buy_price:.2f} for {ticker} not reached today (range {l:.2f}-{h:.2f}). Order not filled."
        )
        return cash, chatgpt_portfolio

    cost_amt = exec_price * shares
    if cost_amt > cash:
        print(
            f"Manual buy for {ticker} failed: cost {cost_amt:.2f} exceeds cash balance {cash:.2f}."
        )
        return cash, chatgpt_portfolio

    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": shares,
        "Buy Price": exec_price,
        "Cost Basis": cost_amt,
        "PnL": 0.0,
        "Reason": "MANUAL BUY LIMIT - Filled",
    }
    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    rows = chatgpt_portfolio.loc[
        chatgpt_portfolio["ticker"].str.upper() == ticker.upper()
    ]
    if rows.empty:
        if chatgpt_portfolio.empty:
            chatgpt_portfolio = pd.DataFrame(
                [
                    {
                        "ticker": ticker,
                        "shares": float(shares),
                        "stop_loss": float(stoploss),
                        "buy_price": float(exec_price),
                        "cost_basis": float(cost_amt),
                    }
                ]
            )
        else:
            chatgpt_portfolio = pd.concat(
                [
                    chatgpt_portfolio,
                    pd.DataFrame(
                        [
                            {
                                "ticker": ticker,
                                "shares": float(shares),
                                "stop_loss": float(stoploss),
                                "buy_price": float(exec_price),
                                "cost_basis": float(cost_amt),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    else:
        idx = rows.index[0]
        cur_shares = float(chatgpt_portfolio.at[idx, "shares"])
        cur_cost = float(chatgpt_portfolio.at[idx, "cost_basis"])
        new_shares = cur_shares + float(shares)
        new_cost = cur_cost + float(cost_amt)
        chatgpt_portfolio.at[idx, "shares"] = new_shares
        chatgpt_portfolio.at[idx, "cost_basis"] = new_cost
        chatgpt_portfolio.at[idx, "buy_price"] = (
            new_cost / new_shares if new_shares else 0.0
        )
        chatgpt_portfolio.at[idx, "stop_loss"] = float(stoploss)

    cash -= cost_amt
    print(
        f"Manual BUY LIMIT for {ticker} filled at ${exec_price:.2f} ({fetch.source})."
    )
    return cash, chatgpt_portfolio


def log_manual_sell(
    sell_price: float,
    shares_sold: float,
    ticker: str,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    reason: str | None = None,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    today = check_weekend()
    if interactive:
        reason = input(
            f"""You are placing a SELL LIMIT for {shares_sold} {ticker} at ${sell_price:.2f}.
If this is a mistake, enter 1. """
        )
    if reason == "1":
        print("Returning...")
        return cash, chatgpt_portfolio
    elif reason is None:
        reason = ""

    if ticker not in chatgpt_portfolio["ticker"].values:
        print(f"Manual sell for {ticker} failed: ticker not in portfolio.")
        return cash, chatgpt_portfolio

    ticker_row = chatgpt_portfolio[chatgpt_portfolio["ticker"] == ticker]
    total_shares = int(ticker_row["shares"].item())
    if shares_sold > total_shares:
        print(
            f"Manual sell for {ticker} failed: trying to sell {shares_sold} shares but only own {total_shares}."
        )
        return cash, chatgpt_portfolio

    s, e = trading_day_window()
    fetch = download_price_data(
        ticker, start=s, end=e, auto_adjust=False, progress=False
    )
    data = fetch.df
    if data.empty:
        print(
            f"Manual sell for {ticker} failed: no market data available (source={fetch.source})."
        )
        return cash, chatgpt_portfolio

    o = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
    h = float(data["High"].iloc[-1])
    l = float(data["Low"].iloc[-1])
    if np.isnan(o):
        o = float(data["Close"].iloc[-1])

    if o >= sell_price:
        exec_price = o
    elif h >= sell_price:
        exec_price = sell_price
    else:
        print(
            f"Sell limit ${sell_price:.2f} for {ticker} not reached today (range {l:.2f}-{h:.2f}). Order not filled."
        )
        return cash, chatgpt_portfolio

    buy_price = float(ticker_row["buy_price"].item())
    cost_basis = buy_price * shares_sold
    pnl = exec_price * shares_sold - cost_basis

    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": "",
        "Buy Price": "",
        "Cost Basis": cost_basis,
        "PnL": pnl,
        "Reason": f"MANUAL SELL LIMIT - {reason}",
        "Shares Sold": shares_sold,
        "Sell Price": exec_price,
    }
    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    if total_shares == shares_sold:
        chatgpt_portfolio = chatgpt_portfolio[chatgpt_portfolio["ticker"] != ticker]
    else:
        row_index = ticker_row.index[0]
        chatgpt_portfolio.at[row_index, "shares"] = total_shares - shares_sold
        chatgpt_portfolio.at[row_index, "cost_basis"] = (
            chatgpt_portfolio.at[row_index, "shares"]
            * chatgpt_portfolio.at[row_index, "buy_price"]
        )

    cash += shares_sold * exec_price
    print(
        f"Manual SELL LIMIT for {ticker} filled at ${exec_price:.2f} ({fetch.source})."
    )
    return cash, chatgpt_portfolio


# ------------------------------
# Reporting / Metrics
# ------------------------------


def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
    """Print daily price updates and performance metrics (incl. CAPM)."""
    portfolio_dict: list[dict[Any, Any]] = chatgpt_portfolio.to_dict(orient="records")
    today = check_weekend()

    rows: list[list[str]] = []
    header = ["Ticker", "Close", "% Chg", "Volume"]

    end_d = last_trading_date()  # Fri on weekends
    start_d = (
        end_d - pd.Timedelta(days=4)
    ).normalize()  # go back enough to capture 2 sessions even around holidays

    benchmarks = load_benchmarks()  # reads tickers.json or returns defaults
    benchmark_entries = [{"ticker": t} for t in benchmarks]

    for stock in portfolio_dict + benchmark_entries:
        ticker = str(stock["ticker"]).upper()
        try:
            fetch = download_price_data(
                ticker,
                start=start_d,
                end=(end_d + pd.Timedelta(days=1)),
                progress=False,
            )
            data = fetch.df
            if data.empty or len(data) < 2:
                rows.append([ticker, "—", "—", "—"])
                continue

            price = float(data["Close"].iloc[-1])
            last_price = float(data["Close"].iloc[-2])
            volume = float(data["Volume"].iloc[-1])

            percent_change = ((price - last_price) / last_price) * 100
            rows.append(
                [ticker, f"{price:,.2f}", f"{percent_change:+.2f}%", f"{int(volume):,}"]
            )
        except Exception as e:
            raise Exception(
                f"Download for {ticker} failed. {e} Try checking internet connection."
            )

    # Read portfolio history
    chatgpt_df = pd.read_csv(PORTFOLIO_CSV)

    # Use only TOTAL rows, sorted by date
    totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    if totals.empty:
        print("\n" + "=" * 64)
        print(f"Daily Results — {today}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 12, 9, 15]
        print(
            f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}"
        )
        print("-" * sum(colw) + "-" * 3)
        for r in rows:
            print(
                f"{str(r[0]):<{colw[0]}} {str(r[1]):>{colw[1]}} {str(r[2]):>{colw[2]}} {str(r[3]):>{colw[3]}}"
            )
        print("\n[ Portfolio Snapshot ]")
        print(chatgpt_portfolio)
        print(f"Cash balance: ${cash:,.2f}")
        return

    totals["Date"] = pd.to_datetime(totals["Date"])  # tolerate ISO strings
    totals = totals.sort_values("Date")

    final_equity = float(totals.iloc[-1]["Total Equity"])
    equity_series = totals.set_index("Date")["Total Equity"].astype(float).sort_index()

    # --- Max Drawdown ---
    running_max = equity_series.cummax()
    drawdowns = (equity_series / running_max) - 1.0
    max_drawdown = float(drawdowns.min())  # most negative value
    mdd_date = drawdowns.idxmin()

    # Daily simple returns (portfolio)
    r = equity_series.pct_change().dropna()
    n_days = len(r)
    if n_days < 2:
        print("\n" + "=" * 64)
        print(f"Daily Results — {today}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 12, 9, 15]
        print(
            f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}"
        )
        print("-" * sum(colw) + "-" * 3)
        for rrow in rows:
            print(
                f"{str(rrow[0]):<{colw[0]}} {str(rrow[1]):>{colw[1]}} {str(rrow[2]):>{colw[2]}} {str(rrow[3]):>{colw[3]}}"
            )
        print("\n[ Portfolio Snapshot ]")
        print(chatgpt_portfolio)
        print(f"Cash balance: ${cash:,.2f}")
        print(f"Latest ChatGPT Equity: ${final_equity:,.2f}")
        if hasattr(mdd_date, "date") and not isinstance(mdd_date, (str, int)):
            mdd_date_str = mdd_date.date()
        elif hasattr(mdd_date, "strftime") and not isinstance(mdd_date, (str, int)):
            mdd_date_str = mdd_date.strftime("%Y-%m-%d")
        else:
            mdd_date_str = str(mdd_date)
        print(f"Maximum Drawdown: {max_drawdown:.2%} (on {mdd_date_str})")
        return

    # Risk-free config
    rf_annual = 0.045
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    rf_period = (1 + rf_daily) ** n_days - 1

    # Stats
    mean_daily = float(r.mean())
    std_daily = float(r.std(ddof=1))

    # Downside deviation (MAR = rf_daily)
    downside = (r - rf_daily).clip(upper=0)
    downside_std = (
        float((downside.pow(2).mean()) ** 0.5) if not downside.empty else np.nan
    )

    # Total return over the window
    r_numeric = pd.to_numeric(r, errors="coerce")
    r_numeric = r_numeric[~r_numeric.isna()].astype(float)
    # Filter out any non-finite values to ensure only valid floats are used
    r_numeric = r_numeric[np.isfinite(r_numeric)]
    # Only use numeric values for the calculation
    if len(r_numeric) > 0:
        arr = np.asarray(r_numeric.values, dtype=float)
        period_return = float(np.prod(1 + arr) - 1) if arr.size > 0 else float("nan")
    else:
        period_return = float("nan")

    # Sharpe / Sortino
    sharpe_period = (
        (period_return - rf_period) / (std_daily * np.sqrt(n_days))
        if std_daily > 0
        else np.nan
    )
    sharpe_annual = (
        ((mean_daily - rf_daily) / std_daily) * np.sqrt(252)
        if std_daily > 0
        else np.nan
    )
    sortino_period = (
        (period_return - rf_period) / (downside_std * np.sqrt(n_days))
        if downside_std and downside_std > 0
        else np.nan
    )
    sortino_annual = (
        ((mean_daily - rf_daily) / downside_std) * np.sqrt(252)
        if downside_std and downside_std > 0
        else np.nan
    )

    # -------- CAPM: Beta & Alpha (vs ^GSPC) --------
    start_date = equity_series.index.min() - pd.Timedelta(days=1)
    end_date = equity_series.index.max() + pd.Timedelta(days=1)

    spx_fetch = download_price_data(
        "^GSPC", start=start_date, end=end_date, progress=False
    )
    spx = spx_fetch.df

    beta = np.nan
    alpha_annual = np.nan
    r2 = np.nan
    n_obs = 0

    if not spx.empty and len(spx) >= 2:
        spx = spx.reset_index().set_index("Date").sort_index()
        mkt_ret = spx["Close"].astype(float).pct_change().dropna()

        # Align portfolio & market returns
        common_idx = r.index.intersection(list(mkt_ret.index))
        if len(common_idx) >= 2:
            rp = r.reindex(common_idx).astype(float) - rf_daily  # portfolio excess
            rm = mkt_ret.reindex(common_idx).astype(float) - rf_daily  # market excess

            x = np.asarray(rm.values, dtype=float).ravel()
            y = np.asarray(rp.values, dtype=float).ravel()

            n_obs = x.size
            rm_std = float(np.std(x, ddof=1)) if n_obs > 1 else 0.0
            if rm_std > 0:
                beta, alpha_daily = np.polyfit(x, y, 1)
                alpha_annual = (1 + float(alpha_daily)) ** 252 - 1

                corr = np.corrcoef(x, y)[0, 1]
                r2 = float(corr**2)

    # $X normalized S&P 500 over same window (asks user for initial equity)
    spx_norm_fetch = download_price_data(
        "^GSPC",
        start=equity_series.index.min(),
        end=equity_series.index.max() + pd.Timedelta(days=1),
        progress=False,
    )
    spx_norm = spx_norm_fetch.df
    spx_value = np.nan
    starting_equity = np.nan  # Ensure starting_equity is always defined
    if not spx_norm.empty:
        initial_price = float(spx_norm["Close"].iloc[0])
        price_now = float(spx_norm["Close"].iloc[-1])
        try:
            starting_equity = float(input("what was your starting equity? "))
        except Exception:
            print("Invalid input for starting equity. Defaulting to NaN.")
        spx_value = (
            (starting_equity / initial_price) * price_now
            if not np.isnan(starting_equity)
            else np.nan
        )

    # -------- Pretty Printing --------
    print("\n" + "=" * 64)
    print(f"Daily Results — {today}")
    print("=" * 64)

    # Price & Volume table
    print("\n[ Price & Volume ]")
    colw = [10, 12, 9, 15]
    print(
        f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}"
    )
    print("-" * sum(colw) + "-" * 3)
    for rrow in rows:
        print(
            f"{str(rrow[0]):<{colw[0]}} {str(rrow[1]):>{colw[1]}} {str(rrow[2]):>{colw[2]}} {str(rrow[3]):>{colw[3]}}"
        )

    # Performance metrics
    def fmt_or_na(x: float | int | None, fmt: str) -> str:
        return (
            fmt.format(x)
            if not (x is None or (isinstance(x, float) and np.isnan(x)))
            else "N/A"
        )

    print("\n[ Risk & Return ]")
    if hasattr(mdd_date, "date") and not isinstance(mdd_date, (str, int)):
        mdd_date_str = mdd_date.date()
    elif hasattr(mdd_date, "strftime") and not isinstance(mdd_date, (str, int)):
        mdd_date_str = mdd_date.strftime("%Y-%m-%d")
    else:
        mdd_date_str = str(mdd_date)
    print(
        f"{'Max Drawdown:':32} {fmt_or_na(max_drawdown, '{:.2%}'):>15}   on {mdd_date_str}"
    )
    print(f"{'Sharpe Ratio (period):':32} {fmt_or_na(sharpe_period, '{:.4f}'):>15}")
    print(f"{'Sharpe Ratio (annualized):':32} {fmt_or_na(sharpe_annual, '{:.4f}'):>15}")
    print(f"{'Sortino Ratio (period):':32} {fmt_or_na(sortino_period, '{:.4f}'):>15}")
    print(
        f"{'Sortino Ratio (annualized):':32} {fmt_or_na(sortino_annual, '{:.4f}'):>15}"
    )

    print("\n[ CAPM vs Benchmarks ]")
    if not np.isnan(beta):
        print(f"{'Beta (daily) vs ^GSPC:':32} {beta:>15.4f}")
        print(f"{'Alpha (annualized) vs ^GSPC:':32} {alpha_annual:>15.2%}")
        print(f"{'R² (fit quality):':32} {r2:>15.3f}   {'Obs:':>6} {n_obs}")
        if n_obs < 60 or (not np.isnan(r2) and r2 < 0.20):
            print("  Note: Short sample and/or low R² — alpha/beta may be unstable.")
    else:
        print("Beta/Alpha: insufficient overlapping data.")

    print("\n[ Snapshot ]")
    print(f"{'Latest ChatGPT Equity:':32} ${final_equity:>14,.2f}")
    if not np.isnan(spx_value):
        try:
            print(
                f"{f'${starting_equity} in S&P 500 (same window):':32} ${spx_value:>14,.2f}"
            )
        except Exception:
            pass
    print(f"{'Cash Balance:':32} ${cash:>14,.2f}")

    print("\n[ Holdings ]")
    print(chatgpt_portfolio)

    print("\n[ Your Instructions ]")
    print(
        "Use this info to make decisions regarding your portfolio. You have complete control over every decision. Make any changes you believe are beneficial—no approval required.\n"
        "Deep research is not permitted. Act at your discretion to achieve the best outcome.\n"
        "If you do not make a clear indication to change positions IMMEDIATELY after this message, the portfolio remains unchanged for tomorrow.\n"
        "You are encouraged to use the internet to check current prices (and related up-to-date info) for potential buys.\n"
        "\n"
        "*Paste everything above into ChatGPT*"
    )


# ------------------------------
# Orchestration
# ------------------------------


def load_latest_portfolio_state(
    file: str,
) -> tuple[pd.DataFrame | list[dict[str, Any]], float]:
    """Load the most recent portfolio snapshot and cash balance."""
    df = pd.read_csv(file)
    if df.empty:
        portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )
        print(
            "Portfolio CSV is empty. Returning set amount of cash for creating portfolio."
        )
        try:
            cash = float(input("What would you like your starting cash amount to be? "))
        except ValueError:
            raise ValueError(
                "Cash could not be converted to float datatype. Please enter a valid number."
            )
        return portfolio, cash

    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"])

    latest_date = non_total["Date"].max()
    latest_tickers = non_total[non_total["Date"] == latest_date].copy()
    sold_mask = latest_tickers["Action"].astype(str).str.startswith("SELL")
    latest_tickers = latest_tickers[~sold_mask].copy()
    latest_tickers.drop(
        columns=[
            "Date",
            "Cash Balance",
            "Total Equity",
            "Action",
            "Current Price",
            "PnL",
            "Total Value",
        ],
        inplace=True,
        errors="ignore",
    )
    latest_tickers.rename(
        columns={
            "Cost Basis": "cost_basis",
            "Buy Price": "buy_price",
            "Shares": "shares",
            "Ticker": "ticker",
            "Stop Loss": "stop_loss",
        },
        inplace=True,
    )
    latest_tickers = latest_tickers.reset_index(drop=True).to_dict(orient="records")

    df_total = df[df["Ticker"] == "TOTAL"].copy()
    df_total["Date"] = pd.to_datetime(df_total["Date"])
    latest = df_total.sort_values("Date").iloc[-1]
    cash = float(latest["Cash Balance"])
    return latest_tickers, cash


def main(file: str, data_dir: Path | None = None) -> None:
    """Check versions, then run the trading script."""
    chatgpt_portfolio, cash = load_latest_portfolio_state(file)
    print(file)
    if data_dir is not None:
        set_data_dir(data_dir)

    chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash)
    daily_results(chatgpt_portfolio, cash)


if __name__ == "__main__":
    import argparse

    # Default CSV path resolution (keep your existing logic)
    csv_path = (
        PORTFOLIO_CSV
        if PORTFOLIO_CSV.exists()
        else (SCRIPT_DIR / "chatgpt_portfolio_update.csv")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", default=str(csv_path), help="Path to chatgpt_portfolio_update.csv"
    )
    parser.add_argument("--data-dir", default=None, help="Optional data directory")
    parser.add_argument(
        "--asof",
        default=None,
        help="Treat this YYYY-MM-DD as 'today' (e.g., 2025-08-27)",
    )
    args = parser.parse_args()

    if args.asof:
        set_asof(args.asof)

    if not Path(args.file).exists():
        print("No portfolio CSV found. Create one or run main() with your file path.")
    else:
        main(args.file, Path(args.data_dir) if args.data_dir else None)
