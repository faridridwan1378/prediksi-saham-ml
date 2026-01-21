import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

warnings.filterwarnings("ignore", category=FutureWarning)

# Coba import matplotlib untuk plot; jika tidak ada, grafik dilewati
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Peringatan: matplotlib belum terinstal. Grafik tidak akan ditampilkan.")


# ===== INDICATOR FUNCTIONS =====
def hitung_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def tambah_indikator(data):
    close = data['Adj Close']

    data['Return'] = close.pct_change()
    data['RSI'] = hitung_rsi(close)

    data['SMA_20'] = close.rolling(20).mean()
    data['SMA_50'] = close.rolling(50).mean()
    data['SMA_200'] = close.rolling(200).mean()
    data['Ratio_SMA50'] = close / data['SMA_50']
    data['Ratio_SMA200'] = close / data['SMA_200']

    data['Volatility_14'] = data['Return'].rolling(14).std()

    std20 = close.rolling(20).std()
    data['BB_Mid'] = data['SMA_20']
    data['BB_Upper'] = data['BB_Mid'] + 2 * std20
    data['BB_Lower'] = data['BB_Mid'] - 2 * std20
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Mid']
    data['BB_Position'] = (close - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    data['Lag1'] = data['Return'].shift(1)
    data['Lag2'] = data['Return'].shift(2)


# ===== BACKTEST DETAIL =====
def backtest_detailed(rets, probs, prob_threshold=0.60):
    """
    rets: return harian (Series)
    probs: probabilitas naik (Series, index sama)
    """
    rets = rets.copy()
    probs = probs.reindex(rets.index)

    signals = (probs > prob_threshold).astype(int)
    strat_rets = rets * signals

    equity_strat = (1 + strat_rets).cumprod()
    equity_bh = (1 + rets).cumprod()

    total_ret_strat = float(equity_strat.iloc[-1] - 1)
    total_ret_bh = float(equity_bh.iloc[-1] - 1)

    trades = int((signals == 1).sum())
    win_trades = int(((strat_rets > 0) & (signals == 1)).sum())
    win_rate = win_trades / trades if trades > 0 else 0.0

    # Max drawdown
    roll_max = equity_strat.cummax()
    drawdown = equity_strat / roll_max - 1.0
    max_drawdown = float(drawdown.min())

    # CAGR (approx, asumsi 252 hari trading per tahun)
    n_days = len(rets)
    if n_days > 0:
        cagr_strat = float(equity_strat.iloc[-1] ** (252 / n_days) - 1)
        cagr_bh = float(equity_bh.iloc[-1] ** (252 / n_days) - 1)
    else:
        cagr_strat = 0.0
        cagr_bh = 0.0

    # Sharpe ratio (simple, risk-free ~ 0)
    if strat_rets.std() > 0:
        sharpe = float((strat_rets.mean() * 252) / (strat_rets.std() * np.sqrt(252)))
    else:
        sharpe = 0.0

    return {
        "signals": signals,
        "equity_strat": equity_strat,
        "equity_bh": equity_bh,
        "total_ret_strat": total_ret_strat,
        "total_ret_bh": total_ret_bh,
        "trades": trades,
        "win_rate": float(win_rate),
        "max_drawdown": max_drawdown,
        "cagr_strat": cagr_strat,
        "cagr_bh": cagr_bh,
        "sharpe": sharpe,
    }


# ===== MAIN ANALYSIS =====
def analisa_saham_detail(ticker, start_date='2015-01-01', prob_threshold=0.60):
    print(f"\n{'='*70}")
    print(f"ANALISIS DETAIL SAHAM: {ticker}")
    print(f"{'='*70}")

    df = yf.download(ticker, start=start_date, progress=False)
    if df is None or df.empty:
        print("Data kosong atau gagal download.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        print("Kolom harga tidak ditemukan.")
        return

    data = df[[price_col]].copy()
    data.rename(columns={price_col: 'Adj Close'}, inplace=True)

    tambah_indikator(data)
    data['Target'] = np.where(data['Return'] > 0, 1, 0)

    data.dropna(inplace=True)
    if len(data) < 300:
        print(f"Data terlalu sedikit: {len(data)} baris.")
        return

    fitur = [
        'Lag1', 'Lag2',
        'RSI',
        'Ratio_SMA50', 'Ratio_SMA200',
        'Volatility_14',
        'BB_Width', 'BB_Position',
        'MACD', 'MACD_Hist',
    ]

    X = data[fitur]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds_test = model.predict(X_test)
    proba_test = model.predict_proba(X_test)[:, 1]

    akurasi = accuracy_score(y_test, preds_test)
    try:
        presisi = precision_score(y_test, preds_test)
    except ZeroDivisionError:
        presisi = 0.0

    print(f"[PERFORMA MODEL - DATA TEST]")
    print(f"Akurasi : {akurasi*100:.2f}%")
    print(f"Presisi Prediksi Naik : {presisi*100:.2f}%")

    # Prediksi hari berikutnya
    input_next = data.iloc[[-1]][fitur]
    proba_next = model.predict_proba(input_next)[0]
    prob_naik_next = proba_next[1] * 100
    prob_turun_next = proba_next[0] * 100

    last_date = data.index[-1].strftime('%Y-%m-%d')
    last_price = data['Adj Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]

    print(f"\n[PREVIEW HARI BERIKUTNYA]")
    print(f"Tanggal Terakhir : {last_date}")
    print(f"Harga Terakhir   : {last_price:.2f}")
    print(f"RSI Terakhir     : {last_rsi:.2f}")
    print(f"Peluang Naik     : {prob_naik_next:.2f}%")
    print(f"Peluang Turun    : {prob_turun_next:.2f}%")

    # Backtest detail di periode test
    rets_test = data.loc[X_test.index, 'Return']
    probs_test = pd.Series(proba_test, index=X_test.index)

    bt = backtest_detailed(
        rets=rets_test,
        probs=probs_test,
        prob_threshold=prob_threshold
    )

    print(f"\n[BACKTEST STRATEGI (PERIODE TEST - {len(X_test)} HARI)]")
    print(f"Threshold ProbNaik BUY : {prob_threshold*100:.0f}%")
    print(f"Total Return Strategi  : {bt['total_ret_strat']*100:.2f}%")
    print(f"Total Return Buy&Hold  : {bt['total_ret_bh']*100:.2f}%")
    print(f"CAGR Strategi          : {bt['cagr_strat']*100:.2f}% per tahun (approx)")
    print(f"CAGR Buy&Hold          : {bt['cagr_bh']*100:.2f}% per tahun (approx)")
    print(f"Max Drawdown Strategi  : {bt['max_drawdown']*100:.2f}%")
    print(f"Sharpe Ratio (sederhana): {bt['sharpe']:.2f}")
    print(f"Jumlah Hari BUY        : {bt['trades']}")
    print(f"Win Rate BUY           : {bt['win_rate']*100:.2f}%")

    # Simpan ringkasan ke CSV
    summary = {
        "Ticker": ticker,
        "TanggalTerakhir": last_date,
        "HargaTerakhir": last_price,
        "RSI_Terakhir": last_rsi,
        "Akurasi": akurasi,
        "Presisi": presisi,
        "ProbNaik_Next_%": prob_naik_next,
        "ProbTurun_Next_%": prob_turun_next,
        "BT_TotalRet_Strategy_%": bt["total_ret_strat"] * 100,
        "BT_TotalRet_BuyHold_%": bt["total_ret_bh"] * 100,
        "BT_CAGR_Strategy_%": bt["cagr_strat"] * 100,
        "BT_CAGR_BuyHold_%": bt["cagr_bh"] * 100,
        "BT_MaxDrawdown_%": bt["max_drawdown"] * 100,
        "BT_Sharpe": bt["sharpe"],
        "BT_Trades": bt["trades"],
        "BT_WinRate_%": bt["win_rate"] * 100,
    }
    df_summary = pd.DataFrame([summary])
    csv_path = "analisa_saham_detail.csv"
    if os.path.exists(csv_path):
        df_summary.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df_summary.to_csv(csv_path, index=False)
    print(f"\nRingkasan disimpan ke: {csv_path}")

    # Grafik optional
    if HAS_MPL:
        plt.figure(figsize=(12, 5))
        plt.plot(bt['equity_bh'].index, bt['equity_bh'], label="Buy&Hold", color="grey")
        plt.plot(bt['equity_strat'].index, bt['equity_strat'], label="Strategi ProbNaik", color="green")
        plt.title(f"Equity Curve (Periode Test) - {ticker}")
        plt.xlabel("Tanggal")
        plt.ylabel("Equity (relatif)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("\n[INFO] matplotlib tidak tersedia, grafik equity curve dilewati.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        kode = sys.argv[1].strip().upper()
    else:
        kode = "BBRI.JK"  # Default ticker if none provided
    analisa_saham_detail(kode, start_date="2015-01-01", prob_threshold=0.60)
