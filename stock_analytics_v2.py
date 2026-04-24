# ============================================================
#   AIML STOCK ANALYTICS & PREDICTION TERMINAL
#   Features: Past Analysis + 7-Day Forecast + Charts
#             Target Price + Risk Level + BUY/SELL/HOLD
#   Platform: VS Code (Terminal + PNG Charts + Popup)
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pytz
import matplotlib
matplotlib.use("Agg")  # saves PNG without needing a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
warnings.filterwarnings("ignore")

# ── Output folder for charts ─────────────────────────────
CHART_DIR = "stock_charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ── Stock Database ────────────────────────────────────────
STOCK_DATABASE = {
    "RELIANCE": "Reliance Industries Ltd", "TCS": "Tata Consultancy Services Ltd",
    "HDFCBANK": "HDFC Bank Ltd", "ICICIBANK": "ICICI Bank Ltd", "INFY": "Infosys Ltd",
    "HINDUNILVR": "Hindustan Unilever Ltd", "ITC": "ITC Ltd", "SBIN": "State Bank of India",
    "BAJFINANCE": "Bajaj Finance Ltd", "BHARTIARTL": "Bharti Airtel Ltd",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd", "LT": "Larsen and Toubro Ltd",
    "AXISBANK": "Axis Bank Ltd", "ASIANPAINT": "Asian Paints Ltd",
    "MARUTI": "Maruti Suzuki India Ltd", "SUNPHARMA": "Sun Pharmaceutical Industries Ltd",
    "TITAN": "Titan Company Ltd", "WIPRO": "Wipro Ltd", "ULTRACEMCO": "UltraTech Cement Ltd",
    "NESTLEIND": "Nestle India Ltd", "ADANIENT": "Adani Enterprises Ltd",
    "ADANIPORTS": "Adani Ports SEZ Ltd", "POWERGRID": "Power Grid Corporation Ltd",
    "NTPC": "NTPC Ltd", "ONGC": "Oil Natural Gas Corporation Ltd",
    "JSWSTEEL": "JSW Steel Ltd", "TATAMOTORS": "Tata Motors Ltd",
    "TATASTEEL": "Tata Steel Ltd", "HCLTECH": "HCL Technologies Ltd",
    "TECHM": "Tech Mahindra Ltd", "BAJAJ-AUTO": "Bajaj Auto Ltd",
    "HEROMOTOCO": "Hero MotoCorp Ltd", "EICHERMOT": "Eicher Motors Ltd",
    "DIVISLAB": "Divi Laboratories Ltd", "DRREDDY": "Dr Reddys Laboratories Ltd",
    "CIPLA": "Cipla Ltd", "APOLLOHOSP": "Apollo Hospitals Enterprise Ltd",
    "BPCL": "Bharat Petroleum Corporation Ltd", "COALINDIA": "Coal India Ltd",
    "BRITANNIA": "Britannia Industries Ltd", "TATACONSUM": "Tata Consumer Products Ltd",
    "GRASIM": "Grasim Industries Ltd", "INDUSINDBK": "IndusInd Bank Ltd",
    "SBILIFE": "SBI Life Insurance Company Ltd", "HDFCLIFE": "HDFC Life Insurance Company Ltd",
    "BAJAJFINSV": "Bajaj Finserv Ltd", "SIEMENS": "Siemens Ltd",
    "HAVELLS": "Havells India Ltd", "PIDILITIND": "Pidilite Industries Ltd",
    "DABUR": "Dabur India Ltd", "MARICO": "Marico Ltd",
    "GODREJCP": "Godrej Consumer Products Ltd", "MUTHOOTFIN": "Muthoot Finance Ltd",
    "AMBUJACEM": "Ambuja Cements Ltd", "ACC": "ACC Ltd", "SHREECEM": "Shree Cement Ltd",
    "INDIGO": "InterGlobe Aviation Ltd", "HAL": "Hindustan Aeronautics Ltd",
    "BEL": "Bharat Electronics Ltd", "VEDL": "Vedanta Ltd",
    "HINDALCO": "Hindalco Industries Ltd", "NMDC": "NMDC Ltd",
    "SAIL": "Steel Authority of India Ltd", "BANKBARODA": "Bank of Baroda",
    "PNB": "Punjab National Bank", "CANBK": "Canara Bank",
    "UNIONBANK": "Union Bank of India", "IDFCFIRSTB": "IDFC First Bank Ltd",
    "FEDERALBNK": "The Federal Bank Ltd", "BANDHANBNK": "Bandhan Bank Ltd",
    "AUBANK": "AU Small Finance Bank Ltd", "RBLBANK": "RBL Bank Ltd",
    "SBICARD": "SBI Cards and Payment Services Ltd",
    "CHOLAFIN": "Cholamandalam Investment and Finance Ltd",
    "LICHSGFIN": "LIC Housing Finance Ltd", "RECLTD": "REC Ltd",
    "PFC": "Power Finance Corporation Ltd", "IRFC": "Indian Railway Finance Corporation Ltd",
    "HUDCO": "Housing Urban Development Corporation Ltd",
    "ZOMATO": "Zomato Ltd", "NYKAA": "FSN E-Commerce Ventures Nykaa",
    "PAYTM": "One 97 Communications Paytm", "DELHIVERY": "Delhivery Ltd",
    "IRCTC": "Indian Railway Catering Tourism Corporation Ltd",
    "CDSL": "Central Depository Services India Ltd",
    "MCX": "Multi Commodity Exchange India Ltd", "ANGELONE": "Angel One Ltd",
    "SUZLON": "Suzlon Energy Ltd", "RPOWER": "Reliance Power Ltd",
    "ADANIPOWER": "Adani Power Ltd", "ADANIGREEN": "Adani Green Energy Ltd",
    "TATAPOWER": "Tata Power Company Ltd", "NHPC": "NHPC Ltd", "SJVN": "SJVN Ltd",
    "IREDA": "Indian Renewable Energy Dev Agency Ltd", "INOXWIND": "Inox Wind Ltd",
    "LTIM": "LTIMindtree Ltd", "LTTS": "LT Technology Services Ltd",
    "MPHASIS": "Mphasis Ltd", "KPITTECH": "KPIT Technologies Ltd",
    "PERSISTENT": "Persistent Systems Ltd", "COFORGE": "Coforge Ltd",
    "HAPPSTMNDS": "Happiest Minds Technologies Ltd",
    "AUROPHARMA": "Aurobindo Pharma Ltd", "LUPIN": "Lupin Ltd",
    "TORNTPHARM": "Torrent Pharmaceuticals Ltd", "ALKEM": "Alkem Laboratories Ltd",
    "LALPATHLAB": "Dr Lal PathLabs Ltd", "FORTIS": "Fortis Healthcare Ltd",
    "MAXHEALTH": "Max Healthcare Institute Ltd", "BIOCON": "Biocon Ltd",
    "ABBOTINDIA": "Abbott India Ltd", "ASHOKLEY": "Ashok Leyland Ltd",
    "TVSMOTOR": "TVS Motor Company Ltd", "BOSCHLTD": "Bosch Ltd",
    "EXIDEIND": "Exide Industries Ltd", "BALKRISIND": "Balkrishna Industries Ltd",
    "APOLLOTYRE": "Apollo Tyres Ltd", "MRF": "MRF Ltd", "CEATLTD": "CEAT Ltd",
    "COLPAL": "Colgate Palmolive India Ltd", "EMAMILTD": "Emami Ltd",
    "VBLLTD": "Varun Beverages Ltd", "UBL": "United Breweries Ltd",
    "UNITDSPR": "United Spirits Ltd", "GMRINFRA": "GMR Airports Infrastructure Ltd",
    "DALMIA": "Dalmia Bharat Ltd", "HINDZINC": "Hindustan Zinc Ltd",
    "APLAPOLLO": "APL Apollo Tubes Ltd", "JSPL": "Jindal Steel Power Ltd",
    "IOC": "Indian Oil Corporation Ltd", "HINDPETRO": "Hindustan Petroleum Corporation Ltd",
    "GAIL": "GAIL India Ltd", "IGL": "Indraprastha Gas Ltd",
    "PETRONET": "Petronet LNG Ltd", "OIL": "Oil India Ltd",
    "DMART": "Avenue Supermarts Ltd", "TRENT": "Trent Ltd",
    "BATAINDIA": "Bata India Ltd", "IDEA": "Vodafone Idea Ltd",
    "TATACOMM": "Tata Communications Ltd", "DLF": "DLF Ltd",
    "GODREJPROP": "Godrej Properties Ltd", "OBEROIRLTY": "Oberoi Realty Ltd",
    "PRESTIGE": "Prestige Estates Projects Ltd", "PHOENIXLTD": "The Phoenix Mills Ltd",
    "COCHINSHIP": "Cochin Shipyard Ltd", "MAZDOCK": "Mazagon Dock Shipbuilders Ltd",
    "BHEL": "Bharat Heavy Electricals Ltd", "RVNL": "Rail Vikas Nigam Ltd",
    "RAILTEL": "RailTel Corporation of India Ltd", "BLUEDART": "Blue Dart Express Ltd",
    "COROMANDEL": "Coromandel International Ltd", "PIIND": "PI Industries Ltd",
    "UPL": "UPL Ltd", "SRF": "SRF Ltd", "AARTIIND": "Aarti Industries Ltd",
    "DEEPAKNTR": "Deepak Nitrite Ltd", "TATACHEM": "Tata Chemicals Ltd",
    "PAGEIND": "Page Industries Ltd Jockey", "RAYMOND": "Raymond Ltd",
    "INDHOTEL": "The Indian Hotels Company Ltd", "LEMONTREE": "Lemon Tree Hotels Ltd",
    "ICICIGI": "ICICI Lombard General Insurance Ltd",
    "STARHEALTH": "Star Health and Allied Insurance Company Ltd",
    "MAPMYINDIA": "CE Info Systems Ltd MapmyIndia",
    "GLENMARK": "Glenmark Pharmaceuticals Ltd", "PIRAMAL": "Piramal Enterprises Ltd",
    "TORNTPOWER": "Torrent Power Ltd", "CESC": "CESC Ltd",
    "BSE": "BSE Ltd", "NAUKRI": "Info Edge India Ltd Naukri",
    "JUSTDIAL": "Just Dial Ltd", "EASEMYTRIP": "Easy Trip Planners Ltd",
}


# ── Helpers ───────────────────────────────────────────────
def smart_search(query, max_results=8):
    q = query.strip().upper()
    if not q:
        return []
    exact, starts, contains = [], [], []
    for ticker, name in STOCK_DATABASE.items():
        t_up, n_up = ticker.upper(), name.upper()
        if t_up == q:
            exact.append((ticker, name))
        elif t_up.startswith(q) or n_up.startswith(q):
            starts.append((ticker, name))
        elif q in t_up or q in n_up:
            contains.append((ticker, name))
    return (exact + starts + contains)[:max_results]


def get_valid_symbol(ticker):
    for suffix, label in [("NS", "NSE"), ("BO", "BSE")]:
        sym = f"{ticker}.{suffix}"
        try:
            df = yf.Ticker(sym).history(period="5d", interval="1d")
            if not df.empty and df["Volume"].sum() > 0:
                return sym, label
        except Exception:
            continue
    return None, None


def check_market_context():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if (now.weekday() < 5 and
        (now.hour > 9 or (now.hour == 9 and now.minute >= 15)) and
        (now.hour < 15 or (now.hour == 15 and now.minute <= 30))):
        return "LIVE MARKET OPEN"
    return "MARKET CLOSED / HOLIDAY"


def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal


def compute_bollinger(series, window=20):
    sma  = series.rolling(window).mean()
    std  = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, sma, lower


def compute_risk(df):
    daily_returns = df["Close"].pct_change().dropna()
    volatility    = daily_returns.std() * np.sqrt(252) * 100  # annualised %
    if volatility < 20:
        return "LOW", volatility
    elif volatility < 40:
        return "MEDIUM", volatility
    else:
        return "HIGH", volatility


def add_features(df):
    df = df.copy()
    df["SMA_10"]          = df["Close"].rolling(10).mean()
    df["SMA_50"]          = df["Close"].rolling(50).mean()
    df["EMA_20"]          = df["Close"].ewm(span=20).mean()
    df["Daily_Return"]    = df["Close"].pct_change()
    df["High_Low_Spread"] = df["High"] - df["Low"]
    df["Vol_Change"]      = df["Volume"].pct_change()
    df["RSI"]             = compute_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"] = compute_macd(df["Close"])
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(df["Close"])
    return df


# ── Chart Generator ───────────────────────────────────────
def generate_charts(df, ticker, company, forecast_prices, forecast_dates, target_buy, target_sell):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1923")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    ACCENT  = "#00e5ff"
    GREEN   = "#00e676"
    RED     = "#ff5252"
    YELLOW  = "#ffd740"
    MUTED   = "#607d8b"
    BG      = "#0f1923"
    SURFACE = "#1a2535"

    def style_ax(ax, title):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=ACCENT, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors="#90a4ae", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#263340")
        ax.yaxis.label.set_color("#90a4ae")
        ax.xaxis.label.set_color("#90a4ae")
        ax.grid(True, color="#1e2d3d", linewidth=0.5, linestyle="--")

    recent = df.tail(90)

    # --- Chart 1: Price + SMA + Bollinger Bands ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(recent.index, recent["Close"],  color=ACCENT,  linewidth=1.8, label="Close Price")
    ax1.plot(recent.index, recent["SMA_10"], color=YELLOW,  linewidth=1,   linestyle="--", label="SMA 10")
    ax1.plot(recent.index, recent["SMA_50"], color="#ff9800", linewidth=1, linestyle="--", label="SMA 50")
    ax1.fill_between(recent.index, recent["BB_Upper"], recent["BB_Lower"],
                     alpha=0.08, color=ACCENT, label="Bollinger Band")
    ax1.plot(recent.index, recent["BB_Upper"], color=MUTED, linewidth=0.6, linestyle=":")
    ax1.plot(recent.index, recent["BB_Lower"], color=MUTED, linewidth=0.6, linestyle=":")

    # Target lines
    ax1.axhline(target_buy,  color=GREEN, linewidth=1.2, linestyle="--",
                label=f"Buy Target Rs{target_buy:.1f}")
    ax1.axhline(target_sell, color=RED,   linewidth=1.2, linestyle="--",
                label=f"Sell Target Rs{target_sell:.1f}")

    # Forecast
    all_dates  = list(recent.index) + forecast_dates
    all_prices = list(recent["Close"]) + forecast_prices
    ax1.plot(forecast_dates, forecast_prices, color=GREEN, linewidth=2,
             linestyle="-", marker="o", markersize=4, label="7-Day Forecast")

    ax1.legend(fontsize=7, facecolor=SURFACE, edgecolor=MUTED,
               labelcolor="white", loc="upper left")
    style_ax(ax1, f"{ticker} — Price History (90 Days) + 7-Day AI Forecast")
    ax1.set_ylabel("Price (Rs)")

    # --- Chart 2: Volume ---
    ax2 = fig.add_subplot(gs[1, 0])
    colors_vol = [GREEN if c >= o else RED
                  for c, o in zip(recent["Close"], recent["Open"])]
    ax2.bar(recent.index, recent["Volume"], color=colors_vol, alpha=0.8, width=0.8)
    style_ax(ax2, "Volume")
    ax2.set_ylabel("Volume")

    # --- Chart 3: RSI ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(recent.index, recent["RSI"], color=YELLOW, linewidth=1.5)
    ax3.axhline(70, color=RED,   linewidth=1, linestyle="--", alpha=0.7, label="Overbought 70")
    ax3.axhline(30, color=GREEN, linewidth=1, linestyle="--", alpha=0.7, label="Oversold 30")
    ax3.fill_between(recent.index, recent["RSI"], 70,
                     where=(recent["RSI"] >= 70), alpha=0.2, color=RED)
    ax3.fill_between(recent.index, recent["RSI"], 30,
                     where=(recent["RSI"] <= 30), alpha=0.2, color=GREEN)
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=7, facecolor=SURFACE, edgecolor=MUTED, labelcolor="white")
    style_ax(ax3, "RSI (Relative Strength Index)")

    # --- Chart 4: MACD ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(recent.index, recent["MACD"],        color=ACCENT, linewidth=1.5, label="MACD")
    ax4.plot(recent.index, recent["MACD_Signal"], color=RED,    linewidth=1.2, label="Signal")
    ax4.fill_between(recent.index,
                     recent["MACD"] - recent["MACD_Signal"], 0,
                     where=(recent["MACD"] >= recent["MACD_Signal"]),
                     alpha=0.3, color=GREEN)
    ax4.fill_between(recent.index,
                     recent["MACD"] - recent["MACD_Signal"], 0,
                     where=(recent["MACD"] < recent["MACD_Signal"]),
                     alpha=0.3, color=RED)
    ax4.legend(fontsize=7, facecolor=SURFACE, edgecolor=MUTED, labelcolor="white")
    style_ax(ax4, "MACD")

    # --- Chart 5: Forecast Bar ---
    ax5 = fig.add_subplot(gs[2, 1])
    forecast_labels = [d.strftime("%d %b") for d in forecast_dates]
    bar_colors = [GREEN if p >= forecast_prices[0] else RED for p in forecast_prices]
    ax5.bar(forecast_labels, forecast_prices, color=bar_colors, alpha=0.85, width=0.6)
    ax5.axhline(forecast_prices[0], color=YELLOW, linewidth=1.2,
                linestyle="--", label=f"Today Rs{forecast_prices[0]:.1f}")
    for i, (lbl, val) in enumerate(zip(forecast_labels, forecast_prices)):
        ax5.text(i, val + (max(forecast_prices)*0.005), f"Rs{val:.0f}",
                 ha="center", fontsize=7, color="white")
    ax5.legend(fontsize=7, facecolor=SURFACE, edgecolor=MUTED, labelcolor="white")
    style_ax(ax5, "7-Day Price Forecast")
    ax5.set_ylabel("Price (Rs)")

    # Title
    fig.suptitle(f"{company}  |  AIML Stock Analytics Report",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    # Save PNG
    filepath = os.path.join(CHART_DIR, f"{ticker}_analysis.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart saved: {filepath}")

    # Also open as popup
    try:
        import subprocess, sys, platform
        if platform.system() == "Windows":
            os.startfile(filepath)
        elif platform.system() == "Darwin":
            subprocess.call(["open", filepath])
        else:
            subprocess.call(["xdg-open", filepath])
    except Exception:
        pass


# ── Main Prediction Engine ────────────────────────────────
def run_analysis(ticker):
    company = STOCK_DATABASE.get(ticker, ticker)

    print(f"  [1/5] Detecting exchange for {ticker}...")
    symbol, exchange = get_valid_symbol(ticker)
    if symbol is None:
        print(f"  ERROR: {ticker} not found on NSE or BSE. Check spelling.")
        return

    print(f"  [2/5] Downloading data ({symbol})...")
    df = yf.Ticker(symbol).history(period="2y", interval="1d")
    if df.empty:
        print("  ERROR: No data found.")
        return

    zero_days = (df["Volume"].tail(10) == 0).sum()
    if zero_days > 3:
        print(f"  WARNING: Illiquid stock — {zero_days}/10 recent days had zero volume.")
        return

    print("  [3/5] Computing indicators...")
    df = add_features(df)

    features = ["Open","High","Low","Close","Volume","SMA_10","SMA_50","EMA_20",
                "Daily_Return","High_Low_Spread","Vol_Change","RSI","MACD","MACD_Signal"]

    pred_row = df.iloc[[-1]]
    train_df = df.iloc[:-1].dropna(subset=features + ["Close"])

    if len(train_df) < 60:
        print("  ERROR: Not enough data.")
        return

    # ── BUY / SELL / HOLD classifier ──
    train_df["Target"] = (train_df["Close"].shift(-5) > train_df["Close"]).astype(int)
    train_df = train_df.dropna(subset=["Target"])

    clf = RandomForestClassifier(n_estimators=200, random_state=42,
                                  max_depth=6, min_samples_leaf=3)
    clf.fit(train_df[features], train_df["Target"])
    prob_up = clf.predict_proba(pred_row[features])[0][1] * 100

    if prob_up >= 62:
        verdict, badge = "BUY",  "*** BUY  ***"
    elif prob_up <= 38:
        verdict, badge = "SELL", "*** SELL ***"
    else:
        verdict, badge = "HOLD", "--- HOLD ---"

    # ── 7-Day Price Forecast (Regressor) ──
    print("  [4/5] Forecasting next 7 days...")
    scaler    = MinMaxScaler()
    close_arr = df["Close"].values.reshape(-1, 1)
    scaled    = scaler.fit_transform(close_arr).flatten()

    window = 30
    X_reg, y_reg = [], []
    for i in range(window, len(scaled) - 1):
        X_reg.append(scaled[i-window:i])
        y_reg.append(scaled[i])

    X_reg = np.array(X_reg)
    y_reg = np.array(y_reg)

    reg = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)
    reg.fit(X_reg, y_reg)

    last_window  = scaled[-window:]
    forecast_scaled = []
    cur = list(last_window)
    for _ in range(7):
        pred_val = reg.predict([cur[-window:]])[0]
        forecast_scaled.append(pred_val)
        cur.append(pred_val)

    forecast_prices = scaler.inverse_transform(
        np.array(forecast_scaled).reshape(-1, 1)).flatten().tolist()

    last_date      = df.index[-1].to_pydatetime()
    forecast_dates = []
    d = last_date
    while len(forecast_dates) < 7:
        d += timedelta(days=1)
        if d.weekday() < 5:
            forecast_dates.append(d)

    # ── Risk Level ──
    risk_level, volatility = compute_risk(df)

    # ── Target Price ──
    current_price = pred_row["Close"].iloc[0]
    atr           = (df["High"] - df["Low"]).tail(14).mean()
    target_buy    = round(current_price - atr * 0.8, 2)
    target_sell   = round(current_price + atr * 2.0, 2)
    stop_loss     = round(current_price - atr * 1.2, 2)

    # ── RSI & MACD status ──
    rsi_val   = pred_row["RSI"].iloc[0]
    macd_val  = pred_row["MACD"].iloc[0]
    macd_sig  = pred_row["MACD_Signal"].iloc[0]
    rsi_status  = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    macd_status = "Bullish" if macd_val > macd_sig else "Bearish"

    # ── Print Report ──
    print("  [5/5] Generating charts...")
    generate_charts(df, ticker, company, forecast_prices, forecast_dates, target_buy, target_sell)

    print()
    print("=" * 58)
    print("          AIML STOCK ANALYTICS REPORT")
    print("=" * 58)
    print(f"  Company    : {company}")
    print(f"  Ticker     : {ticker}   |   Exchange : {exchange}")
    print(f"  Market     : {check_market_context()}")
    print("-" * 58)
    print(f"  Current Price : Rs {current_price:,.2f}")
    print(f"  RSI           : {rsi_val:.1f}  ({rsi_status})")
    print(f"  MACD          : {macd_status}")
    print("-" * 58)
    print(f"  VERDICT       :  {badge}")
    print(f"  AI Confidence :  {prob_up:.1f}%")
    print("-" * 58)
    print("  TARGET PRICES")
    print(f"    Buy Zone    : Rs {target_buy:,.2f}")
    print(f"    Sell Target : Rs {target_sell:,.2f}")
    print(f"    Stop Loss   : Rs {stop_loss:,.2f}")
    print("-" * 58)
    print("  7-DAY PRICE FORECAST")
    for date, price in zip(forecast_dates, forecast_prices):
        arrow = "^" if price >= current_price else "v"
        diff  = price - current_price
        print(f"    {date.strftime('%a %d %b')} : Rs {price:>8,.2f}  {arrow}  ({diff:+.2f})")
    print("-" * 58)
    print(f"  RISK LEVEL    : {risk_level}  (Volatility: {volatility:.1f}% annualised)")
    print("=" * 58)
    print("  NOTE: For educational purposes only.")
    print("        Not financial advice.")
    print()


# ── Main Loop ─────────────────────────────────────────────
def main():
    print()
    print("=" * 58)
    print("       AIML INTERACTIVE STOCK TERMINAL v2.0")
    print("       NSE + BSE | Auto Exchange Detection")
    print("       AI Forecast + Charts + Risk Analysis")
    print("=" * 58)

    while True:
        print()
        query = input("  Enter company name or ticker (q to quit): ").strip()

        if query.lower() in ("q", "quit", "exit"):
            print("  Goodbye!")
            break

        if not query:
            continue

        results = smart_search(query)

        if not results:
            print(f"  No match found for '{query}'. Try a different name.")
            continue

        if len(results) == 1:
            ticker, name = results[0]
            print(f"  Auto-selected: {ticker} — {name}")
        else:
            print()
            print("  Matching companies found:")
            for i, (t, n) in enumerate(results, 1):
                print(f"    {i}. {t:15s} — {n}")
            print()
            choice = input(f"  Select number (1-{len(results)}) [Enter = first]: ").strip()
            if choice == "":
                ticker, name = results[0]
            elif choice.isdigit() and 1 <= int(choice) <= len(results):
                ticker, name = results[int(choice) - 1]
            else:
                print("  Invalid choice. Try again.")
                continue

        run_analysis(ticker)

        again = input("  Analyse another stock? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("  Goodbye!")
            break


if __name__ == "__main__":
    main()
