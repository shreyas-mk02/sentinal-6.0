
from flask import Flask, render_template, request, jsonify, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings, os, urllib.request, json
warnings.filterwarnings("ignore")

app = Flask(__name__)
CHART_DIR = "stock_charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ── SECTOR MAP ────────────────────────────────────────────
SECTORS = {
    "Banking & Finance": ["HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","BAJFINANCE",
                          "INDUSINDBK","BANDHANBNK","FEDERALBNK","IDFCFIRSTB","PNB","BANKBARODA"],
    "IT & Technology":   ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","MPHASIS","PERSISTENT",
                          "COFORGE","LTTS"],
    "Oil & Energy":      ["RELIANCE","ONGC","IOC","BPCL","GAIL","POWERGRID","NTPC","ADANIGREEN",
                          "TATAPOWER","SUZLON"],
    "Auto":              ["TATAMOTORS","MARUTI","BAJAJ-AUTO","EICHERMOT","HEROMOTOCO","M&M",
                          "ASHOKLEY","TVSMOTOR","BALKRISIND"],
    "Pharma":            ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","APOLLOHOSP","LUPIN",
                          "BIOCON","AUROPHARMA","ALKEM"],
    "FMCG":              ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO",
                          "COLPAL","GODREJCP","EMAMILTD"],
    "Metals & Mining":   ["TATASTEEL","JSWSTEEL","HINDALCO","VEDL","COALINDIA","NMDC",
                          "SAIL","NATIONALUM"],
    "Defence & Railway": ["HAL","BEL","RVNL","IRFC","IRCTC","RAILVIKAS","COCHINSHIP",
                          "MAZDOCK","GRSE"],
    "Real Estate":       ["DLF","GODREJPROP","OBEROIRLTY","PRESTIGE","PHOENIXLTD","BRIGADE"],
    "New Age Tech":      ["ZOMATO","PAYTM","NYKAA","POLICYBZR","DELHIVERY","CARTRADE"],
}

def load_stock_database():
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, timeout=10)
        data = response.read().decode("utf-8")
        lines = data.strip().split("\n")
        db = {}
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                ticker = parts[0].strip().strip('"')
                name   = parts[1].strip().strip('"')
                if ticker and name:
                    db[ticker] = name
        print(f"Loaded {len(db)} NSE stocks")
        return db
    except Exception as e:
        print(f"Using fallback DB: {e}")
        return FALLBACK_DB

FALLBACK_DB = {
    "RELIANCE":"Reliance Industries Ltd","TCS":"Tata Consultancy Services Ltd",
    "HDFCBANK":"HDFC Bank Ltd","ICICIBANK":"ICICI Bank Ltd","INFY":"Infosys Ltd",
    "SBIN":"State Bank of India","WIPRO":"Wipro Ltd","ZOMATO":"Zomato Ltd",
    "TATAMOTORS":"Tata Motors Ltd","TATASTEEL":"Tata Steel Ltd",
    "ADANIENT":"Adani Enterprises Ltd","NTPC":"NTPC Ltd","ONGC":"ONGC Ltd",
    "HCLTECH":"HCL Technologies Ltd","BAJFINANCE":"Bajaj Finance Ltd",
    "TITAN":"Titan Company Ltd","MARUTI":"Maruti Suzuki India Ltd",
    "SUNPHARMA":"Sun Pharmaceutical Industries Ltd","AXISBANK":"Axis Bank Ltd",
    "KOTAKBANK":"Kotak Mahindra Bank Ltd","LT":"Larsen and Toubro Ltd",
    "ITC":"ITC Ltd","HINDUNILVR":"Hindustan Unilever Ltd",
    "BHARTIARTL":"Bharti Airtel Ltd","POWERGRID":"Power Grid Corporation Ltd",
    "HAL":"Hindustan Aeronautics Ltd","BEL":"Bharat Electronics Ltd",
    "SUZLON":"Suzlon Energy Ltd","IRFC":"Indian Railway Finance Corporation Ltd",
    "CDSL":"Central Depository Services India Ltd","RVNL":"Rail Vikas Nigam Ltd",
}

STOCK_DB = load_stock_database()

def get_valid_symbol(ticker):
    for suffix, label in [("NS","NSE"),("BO","BSE")]:
        sym = f"{ticker}.{suffix}"
        try:
            df = yf.Ticker(sym).history(period="5d", interval="1d")
            if not df.empty and df["Volume"].sum() > 0:
                return sym, label
        except:
            continue
    return None, None

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12  = series.ewm(span=12).mean()
    ema26  = series.ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

def compute_bollinger(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + 2*std, sma, sma - 2*std

def compute_risk(df):
    vol = df["Close"].pct_change().dropna().std() * np.sqrt(252) * 100
    if vol < 20:   return "LOW", vol
    elif vol < 40: return "MEDIUM", vol
    else:          return "HIGH", vol

def clean_df(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    return df

def add_features(df):
    df = df.copy()
    df["SMA_10"]          = df["Close"].rolling(10).mean()
    df["SMA_50"]          = df["Close"].rolling(50).mean()
    df["EMA_20"]          = df["Close"].ewm(span=20).mean()
    df["Daily_Return"]    = df["Close"].pct_change()
    df["High_Low_Spread"] = df["High"] - df["Low"]
    df["Vol_Change"]      = df["Volume"].pct_change().replace([np.inf, -np.inf], 0)
    df["RSI"]             = compute_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"] = compute_macd(df["Close"])
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(df["Close"])
    return df

# ── CANDLESTICK CHART ─────────────────────────────────────
def draw_candlestick(ax, df, n=60):
    recent = df.tail(n).copy()
    recent = recent.reset_index()
    GREEN="#00e676"; RED="#ff5252"; SURFACE="#1a2535"
    ax.set_facecolor(SURFACE)

    for i, row in recent.iterrows():
        color = GREEN if row["Close"] >= row["Open"] else RED
        # Candle body
        body_bottom = min(row["Open"], row["Close"])
        body_height = abs(row["Close"] - row["Open"])
        ax.add_patch(mpatches.Rectangle((i-0.3, body_bottom), 0.6,
                     max(body_height, 0.01), color=color, zorder=3))
        # Wick
        ax.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=0.8, zorder=2)

    # X axis labels (every 10 days)
    ticks = list(range(0, len(recent), 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([recent.iloc[t]["Date"].strftime("%d %b") if hasattr(recent.iloc[t]["Date"], "strftime")
                        else str(recent.iloc[t]["Date"])[:10] for t in ticks], fontsize=7)
    ax.set_xlim(-1, len(recent))
    ax.set_ylim(recent["Low"].min()*0.995, recent["High"].max()*1.005)
    ax.tick_params(colors="#90a4ae")
    for spine in ax.spines.values(): spine.set_edgecolor("#263340")
    ax.grid(True, color="#1e2d3d", linewidth=0.5, linestyle="--")

def generate_charts(df, ticker, company, forecast_prices, forecast_dates, target_buy, target_sell):
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0f1923")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3)
    ACCENT="#00e5ff"; GREEN="#00e676"; RED="#ff5252"; YELLOW="#ffd740"
    MUTED="#607d8b"; SURFACE="#1a2535"

    def style_ax(ax, title):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=ACCENT, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors="#90a4ae", labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#263340")
        ax.grid(True, color="#1e2d3d", linewidth=0.5, linestyle="--")

    # ── Chart 1: CANDLESTICK (top left) ──
    ax1 = fig.add_subplot(gs[0, 0])
    draw_candlestick(ax1, df, n=60)
    ax1.set_title(f"{ticker} — Candlestick (60 Days)", color=ACCENT, fontsize=11, fontweight="bold", pad=8)
    ax1.set_ylabel("Price (Rs)", color="#90a4ae")

    # ── Chart 2: Price + BB + Forecast (top right) ──
    ax2 = fig.add_subplot(gs[0, 1])
    recent = df.tail(90)
    ax2.plot(recent.index, recent["Close"],   color=ACCENT,    linewidth=1.8, label="Close")
    ax2.plot(recent.index, recent["SMA_10"],  color=YELLOW,    linewidth=1,   linestyle="--", label="SMA10")
    ax2.plot(recent.index, recent["SMA_50"],  color="#ff9800", linewidth=1,   linestyle="--", label="SMA50")
    ax2.fill_between(recent.index, recent["BB_Upper"], recent["BB_Lower"], alpha=0.08, color=ACCENT)
    ax2.axhline(target_buy,  color=GREEN, linewidth=1.2, linestyle="--", label=f"Buy Rs{target_buy:.0f}")
    ax2.axhline(target_sell, color=RED,   linewidth=1.2, linestyle="--", label=f"Sell Rs{target_sell:.0f}")
    ax2.plot(forecast_dates, forecast_prices, color=GREEN, linewidth=2,
             marker="o", markersize=4, label="Forecast")
    ax2.legend(fontsize=6, facecolor=SURFACE, edgecolor=MUTED, labelcolor="white", loc="upper left")
    style_ax(ax2, "Price + Bollinger Bands + Forecast")

    # ── Chart 3: Volume ──
    ax3 = fig.add_subplot(gs[1, 0])
    colors_vol = [GREEN if c >= o else RED for c, o in zip(recent["Close"], recent["Open"])]
    ax3.bar(recent.index, recent["Volume"], color=colors_vol, alpha=0.8, width=0.8)
    style_ax(ax3, "Volume Analysis")

    # ── Chart 4: RSI ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(recent.index, recent["RSI"], color=YELLOW, linewidth=1.5)
    ax4.axhline(70, color=RED,   linewidth=1, linestyle="--", alpha=0.7)
    ax4.axhline(30, color=GREEN, linewidth=1, linestyle="--", alpha=0.7)
    ax4.fill_between(recent.index, recent["RSI"], 70, where=(recent["RSI"]>=70), alpha=0.2, color=RED)
    ax4.fill_between(recent.index, recent["RSI"], 30, where=(recent["RSI"]<=30), alpha=0.2, color=GREEN)
    ax4.set_ylim(0, 100)
    style_ax(ax4, "RSI — Relative Strength Index")

    # ── Chart 5: MACD ──
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(recent.index, recent["MACD"],        color=ACCENT, linewidth=1.5, label="MACD")
    ax5.plot(recent.index, recent["MACD_Signal"], color=RED,    linewidth=1.2, label="Signal")
    ax5.fill_between(recent.index, recent["MACD"]-recent["MACD_Signal"], 0,
                     where=(recent["MACD"]>=recent["MACD_Signal"]), alpha=0.3, color=GREEN)
    ax5.fill_between(recent.index, recent["MACD"]-recent["MACD_Signal"], 0,
                     where=(recent["MACD"]<recent["MACD_Signal"]),  alpha=0.3, color=RED)
    ax5.legend(fontsize=7, facecolor=SURFACE, edgecolor=MUTED, labelcolor="white")
    style_ax(ax5, "MACD")

    # ── Chart 6: 7-Day Forecast ──
    ax6 = fig.add_subplot(gs[2, 1])
    labels     = [d.strftime("%a %d %b") for d in forecast_dates]
    bar_colors = [GREEN if p >= forecast_prices[0] else RED for p in forecast_prices]
    ax6.bar(range(len(labels)), forecast_prices, color=bar_colors, alpha=0.85, width=0.6)
    ax6.set_xticks(range(len(labels)))
    ax6.set_xticklabels(labels, fontsize=7, rotation=20)
    ax6.axhline(forecast_prices[0], color=YELLOW, linewidth=1.2, linestyle="--")
    for i, v in enumerate(forecast_prices):
        ax6.text(i, v + max(forecast_prices)*0.005, f"Rs{v:.0f}", ha="center", fontsize=7, color="white")
    style_ax(ax6, "7-Day AI Price Forecast")

    fig.suptitle(f"{company}  |  AIML Stock Analytics", color="white", fontsize=14, fontweight="bold", y=0.99)
    filepath = os.path.join(CHART_DIR, f"{ticker}_analysis.png")
    plt.savefig(filepath, dpi=120, bbox_inches="tight", facecolor="#0f1923")
    plt.close()
    return filepath

def run_analysis(ticker):
    symbol, exchange = get_valid_symbol(ticker)
    if symbol is None:
        return None, f"{ticker} not found on NSE or BSE"

    df = yf.Ticker(symbol).history(period="2y", interval="1d")
    if df.empty:
        return None, "No historical data available"

    zero_days = (df["Volume"].tail(10) == 0).sum()
    if zero_days > 5:
        return None, f"Illiquid stock — {zero_days}/10 days had zero volume"

    df = add_features(df)
    df = clean_df(df)

    features = ["Open","High","Low","Close","Volume","SMA_10","SMA_50","EMA_20",
                "Daily_Return","High_Low_Spread","Vol_Change","RSI","MACD","MACD_Signal"]

    pred_row = df.iloc[[-1]].copy()
    train_df = df.iloc[:-1].copy()
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)

    if len(train_df) < 60:
        return None, "Not enough data to train model"

    train_df["Target"] = (train_df["Close"].shift(-5) > train_df["Close"]).astype(int)
    train_df = train_df.dropna(subset=["Target"])

    X_train = np.nan_to_num(train_df[features].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = train_df["Target"].values

    pred_vals = pred_row[features].replace([np.inf, -np.inf], np.nan)
    pred_vals = pred_vals.fillna(train_df[features].median())
    X_pred = np.nan_to_num(pred_vals.values, nan=0.0, posinf=0.0, neginf=0.0)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6, min_samples_leaf=3)
    clf.fit(X_train, y_train)
    prob_up = clf.predict_proba(X_pred)[0][1] * 100

    if prob_up >= 62:   verdict = "BUY"
    elif prob_up <= 38: verdict = "SELL"
    else:               verdict = "HOLD"

    scaler    = MinMaxScaler()
    close_arr = df["Close"].values.reshape(-1, 1)
    scaled    = scaler.fit_transform(close_arr).flatten()
    window    = 30
    X_reg, y_reg = [], []
    for i in range(window, len(scaled)-1):
        X_reg.append(scaled[i-window:i])
        y_reg.append(scaled[i])
    reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    reg.fit(np.array(X_reg), np.array(y_reg))
    cur = list(scaled[-window:])
    forecast_scaled = []
    for _ in range(7):
        pv = reg.predict([cur[-window:]])[0]
        forecast_scaled.append(pv)
        cur.append(pv)
    forecast_prices = scaler.inverse_transform(
        np.array(forecast_scaled).reshape(-1,1)).flatten().tolist()

    last_date = df.index[-1].to_pydatetime()
    forecast_dates = []
    d = last_date
    while len(forecast_dates) < 7:
        d += timedelta(days=1)
        if d.weekday() < 5:
            forecast_dates.append(d)

    risk_level, volatility = compute_risk(df)
    current_price = float(pred_row["Close"].iloc[0])
    atr           = float((df["High"] - df["Low"]).tail(14).mean())
    target_buy    = round(current_price - atr * 0.8, 2)
    target_sell   = round(current_price + atr * 2.0, 2)
    stop_loss     = round(current_price - atr * 1.2, 2)
    rsi_val       = float(pred_row["RSI"].iloc[0])
    macd_val      = float(pred_row["MACD"].iloc[0])
    macd_sig      = float(pred_row["MACD_Signal"].iloc[0])
    rsi_status    = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    macd_status   = "Bullish" if macd_val > macd_sig else "Bearish"
    company       = STOCK_DB.get(ticker, ticker)

    chart_path = generate_charts(df, ticker, company, forecast_prices, forecast_dates, target_buy, target_sell)

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    market_open = (now.weekday() < 5 and
                   (now.hour > 9 or (now.hour==9 and now.minute>=15)) and
                   (now.hour < 15 or (now.hour==15 and now.minute<=30)))

    result = {
        "ticker":        ticker,
        "company":       company,
        "exchange":      exchange,
        "market_status": "LIVE MARKET OPEN" if market_open else "MARKET CLOSED",
        "current_price": current_price,
        "verdict":       verdict,
        "confidence":    round(prob_up, 1),
        "rsi":           round(rsi_val, 1),
        "rsi_status":    rsi_status,
        "macd_status":   macd_status,
        "target_buy":    target_buy,
        "target_sell":   target_sell,
        "stop_loss":     stop_loss,
        "risk_level":    risk_level,
        "volatility":    round(volatility, 1),
        "forecast": [
            {"date": fd.strftime("%a %d %b"), "price": round(fp, 2)}
            for fd, fp in zip(forecast_dates, forecast_prices)
        ],
        "chart_url": f"/charts/{ticker}_analysis.png"
    }
    return result, None

# ── ROUTES ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sectors")
def sectors():
    return jsonify(list(SECTORS.keys()))

@app.route("/sector_stocks")
def sector_stocks():
    sector = request.args.get("sector", "")
    tickers = SECTORS.get(sector, [])
    result = [{"ticker": t, "name": STOCK_DB.get(t, t)} for t in tickers]
    return jsonify(result)

@app.route("/search")
def search():
    q = request.args.get("q", "").strip().upper()
    if not q:
        return jsonify([])
    results = []
    for ticker, name in STOCK_DB.items():
        if ticker.upper() == q:
            results.insert(0, {"ticker": ticker, "name": name})
    for ticker, name in STOCK_DB.items():
        if (ticker.upper().startswith(q) or name.upper().startswith(q)) and \
           not any(r["ticker"]==ticker for r in results):
            results.append({"ticker": ticker, "name": name})
    for ticker, name in STOCK_DB.items():
        if (q in ticker.upper() or q in name.upper()) and \
           not any(r["ticker"]==ticker for r in results):
            results.append({"ticker": ticker, "name": name})
    return jsonify(results[:12])

@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        ticker = request.json.get("ticker", "").strip().upper()
        if not ticker:
            return jsonify({"error": "No ticker provided"}), 400
        result, err = run_analysis(ticker)
        if err:
            return jsonify({"error": err}), 404
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route("/compare", methods=["POST"])
def compare():
    try:
        t1 = request.json.get("ticker1", "").strip().upper()
        t2 = request.json.get("ticker2", "").strip().upper()
        if not t1 or not t2:
            return jsonify({"error": "Provide both tickers"}), 400
        r1, e1 = run_analysis(t1)
        r2, e2 = run_analysis(t2)
        if e1: return jsonify({"error": f"{t1}: {e1}"}), 404
        if e2: return jsonify({"error": f"{t2}: {e2}"}), 404
        return jsonify({"stock1": r1, "stock2": r2})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/charts/<filename>")
def serve_chart(filename):
    return send_from_directory(CHART_DIR, filename)

if __name__ == "__main__":
    print("\n  AIML Stock Terminal v2 running at: http://127.0.0.1:5000")
    print("  Features: Candlestick + Sector Filter + Compare Stocks\n")
    app.run(debug=False, port=5000)
