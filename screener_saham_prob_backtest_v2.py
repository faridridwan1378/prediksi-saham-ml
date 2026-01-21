import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------ FUNGSI BANTUAN ------------------ #
def hitung_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def tambah_indikator(data):
    """Tambahkan indikator teknikal lanjutan ke DataFrame 'data' (kolom 'Adj Close')."""
    close = data['Adj Close']

    # Return harian
    data['Return'] = close.pct_change()

    # RSI
    data['RSI'] = hitung_rsi(close)

    # SMA (trend)
    data['SMA_20'] = close.rolling(20).mean()
    data['SMA_50'] = close.rolling(50).mean()
    data['SMA_200'] = close.rolling(200).mean()
    data['Ratio_SMA50'] = close / data['SMA_50']
    data['Ratio_SMA200'] = close / data['SMA_200']

    # Volatilitas 14 hari
    data['Volatility_14'] = data['Return'].rolling(14).std()

    # Bollinger Bands 20
    std20 = close.rolling(20).std()
    data['BB_Mid'] = data['SMA_20']
    data['BB_Upper'] = data['BB_Mid'] + 2 * std20
    data['BB_Lower'] = data['BB_Mid'] - 2 * std20
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Mid']
    data['BB_Position'] = (close - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # Lag return
    data['Lag1'] = data['Return'].shift(1)
    data['Lag2'] = data['Return'].shift(2)


def backtest_strategy(data, test_index, prob_naik_series, prob_threshold=0.60):
    """
    Strategi: jika ProbNaik > prob_threshold → BUY hari itu, jika tidak → CASH.
    Return harian diambil dari data['Return'].
    """
    rets = data.loc[test_index, 'Return']
    probs = prob_naik_series.reindex(test_index)

    signals = (probs > prob_threshold).astype(int)  # 1 = BUY, 0 = CASH
    strat_rets = rets * signals

    equity_strat = (1 + strat_rets).cumprod()
    equity_bh = (1 + rets).cumprod()

    total_ret_strat = equity_strat.iloc[-1] - 1
    total_ret_bh = equity_bh.iloc[-1] - 1

    trades = int((signals == 1).sum())
    win_trades = int(((strat_rets > 0) & (signals == 1)).sum())
    win_rate = win_trades / trades if trades > 0 else 0.0

    return {
        "total_ret_strat": float(total_ret_strat),
        "total_ret_bh": float(total_ret_bh),
        "trades": trades,
        "win_rate": float(win_rate),
    }


# ------------------ ANALISIS SATU SAHAM ------------------ #
def analisa_saham_single(ticker, prob_threshold=0.60, start_date='2015-01-01'):
    print(f"\n=== Analisis {ticker} ===")

    # 1. Download data
    try:
        df = yf.download(ticker, start=start_date, progress=False)
    except Exception as e:
        print(f"  Gagal download data: {e}")
        return None

    if df is None or df.empty:
        print("  Data kosong.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        print("  Kolom harga tidak ditemukan.")
        return None

    data = df[[price_col]].copy()
    data.rename(columns={price_col: 'Adj Close'}, inplace=True)

    # 2. Tambah indikator teknikal
    tambah_indikator(data)

    # 3. Target naik/turun
    data['Target'] = np.where(data['Return'] > 0, 1, 0)
    data.dropna(inplace=True)

    if len(data) < 300:
        print(f"  Data terlalu sedikit setelah proses: {len(data)} baris.")
        return None

    # 4. Siapkan fitur untuk model
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
        X, y, test_size=0.15, shuffle=False
    )

    # 5. RandomForest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba_test = model.predict_proba(X_test)[:, 1]  # Prob naik (kelas 1)

    akurasi = accuracy_score(y_test, preds)
    try:
        presisi = precision_score(y_test, preds)
    except ZeroDivisionError:
        presisi = 0.0

    print(f"  Akurasi : {akurasi*100:.2f}% | Presisi Naik : {presisi*100:.2f}%")

    prob_naik_series = pd.Series(proba_test, index=X_test.index)

    # 6. Prediksi hari berikutnya (baris terakhir)
    input_data = data.iloc[[-1]][fitur]
    probabilitas_next = model.predict_proba(input_data)[0]
    prob_naik_next = probabilitas_next[1] * 100
    prob_turun_next = probabilitas_next[0] * 100

    last_date = data.index[-1].strftime('%Y-%m-%d')
    last_price = data['Adj Close'].iloc[-1]
    last_rsi = float(data['RSI'].iloc[-1])

    # 7. Backtest strategi di periode test
    bt = backtest_strategy(
        data=data,
        test_index=X_test.index,
        prob_naik_series=prob_naik_series,
        prob_threshold=prob_threshold
    )

    print(f"  ProbNaik Next: {prob_naik_next:.2f}% | RetStrat Test: {bt['total_ret_strat']*100:.2f}%")

    # 8. Summary
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
        "BT_Trades": bt["trades"],
        "BT_WinRate_%": bt["win_rate"] * 100,
    }

    return summary


# ------------------ MAIN: MULTI-SAHAM SCREENER ------------------ #
if __name__ == "__main__":
    # Contoh daftar saham (bisa Anda ganti dengan daftar LQ45 terbaru)
    daftar_saham = [
        'BBRI.JK', 'BBCA.JK', 'BMRI.JK', 'BBNI.JK',
        'ASII.JK', 'TLKM.JK', 'ICBP.JK', 'INDF.JK',
        'CPIN.JK', 'UNVR.JK', 'KLBF.JK', 'PGAS.JK',
        'PTBA.JK', 'MDKA.JK', 'ADRO.JK', 'ANTM.JK',
        'INCO.JK', 'TINS.JK', 'AKRA.JK', 'UNTR.JK',
        'ITMG.JK', 'TKIM.JK', 'INKP.JK', 'ERAA.JK',
        'ACES.JK', 'AMRT.JK', 'HMSP.JK', 'GGRM.JK',
        'SMGR.JK', 'INTP.JK',
    ]

    PROB_THRESHOLD = 0.60  # threshold BUY untuk backtest strategi

    semua_ringkasan = []
    for kode in daftar_saham:
        try:
            hasil = analisa_saham_single(kode, prob_threshold=PROB_THRESHOLD)
            if hasil is not None:
                semua_ringkasan.append(hasil)
        except Exception as e:
            print(f"  Error tak terduga pada {kode}: {e}")

    if not semua_ringkasan:
        print("\nTidak ada saham yang berhasil dianalisis.")
    else:
        df_summary = pd.DataFrame(semua_ringkasan)

        # CandidateBuy: ProbNaik ≥ 65%, RSI < 70, strategi historis > 0 dan ≥ buy&hold
        df_summary['CandidateBuy'] = (
            (df_summary['ProbNaik_Next_%'] >= 65) &
            (df_summary['RSI_Terakhir'] < 70) &
            (df_summary['BT_TotalRet_Strategy_%'] > 0) &
            (df_summary['BT_TotalRet_Strategy_%'] >= df_summary['BT_TotalRet_BuyHold_%'])
        )

        # Urutkan menurut ProbNaik_Next_% tertinggi
        df_sorted = df_summary.sort_values(by="ProbNaik_Next_%", ascending=False)

        print("\n=== RANGKING SAHAM BERDASARKAN ProbNaik_Next_% (Tertinggi ke Terendah) ===")
        print(df_sorted[['Ticker',
                         'ProbNaik_Next_%',
                         'Akurasi',
                         'BT_TotalRet_Strategy_%',
                         'BT_TotalRet_BuyHold_%',
                         'CandidateBuy']])

        # Simpan ke file
        csv_path = "screener_prediksi_saham_v2.csv"
        df_sorted.to_csv(csv_path, index=False)
        print(f"\n[INFO] Hasil screener disimpan ke: {csv_path}")

        excel_path = "screener_prediksi_saham_v2.xlsx"
        try:
            df_sorted.to_excel(excel_path, index=False)
            print(f"[INFO] Hasil screener juga disimpan ke: {excel_path}")
        except Exception as e:
            print(f"[INFO] Gagal simpan ke Excel (.xlsx), abaikan jika tidak perlu: {e}")

    input("\nTekan Enter untuk keluar...")