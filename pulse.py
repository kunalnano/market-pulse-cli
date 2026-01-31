#!/usr/bin/env python3
"""
Market Pulse - Stock & News Intelligence CLI
Enhanced with multiple indicators and signal scoring.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.text import Text
import time

__version__ = "2.1.1"

console = Console()

# ============== CONFIGURATION ==============
def _user_config_path() -> Path:
    """Return a per-user config path (no external deps)."""
    # Env override wins
    env = os.getenv("MARKET_PULSE_CONFIG")
    if env:
        return Path(os.path.expanduser(env)).resolve()

    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support" / "market-pulse"
    elif os.name == "nt":
        appdata = os.getenv("APPDATA") or str(home / "AppData" / "Roaming")
        base = Path(appdata) / "market-pulse"
    else:
        # XDG default
        base = Path(os.getenv("XDG_CONFIG_HOME", home / ".config")) / "market-pulse"
    return base / "config.json"


def _resolve_config_path() -> Path:
    """Prefer legacy local config.json if present, else per-user path."""
    local_cfg = Path(__file__).parent / "config.json"
    if local_cfg.exists():
        return local_cfg
    return _user_config_path()


CONFIG_PATH = _resolve_config_path()

DEFAULT_CONFIG = {
    "stocks": ["AAPL", "GOOGL"],
    "keywords": ["Google", "Apple", "Port.io", "Anthropic", "OpenAI", "Claude", "Gemini", "AI"],
    "feeds": [
        {"name": "Hacker News", "url": "https://hnrss.org/frontpage"},
        {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
        {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
        {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/technology-lab"},
    ],
    "thresholds": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "pe_max": 35,
        "pe_min": 10,
    },
    "notifications": {
        "email": {"enabled": False, "smtp_server": "smtp.gmail.com", "smtp_port": 587, "sender": "", "password": "", "recipient": ""},
        "sms": {"enabled": False, "phone": "", "carrier": "verizon"}
    },
    "last_run": None,
    "seen_articles": []
}

SMS_GATEWAYS = {"verizon": "@vtext.com", "att": "@txt.att.net", "tmobile": "@tmomail.net", "sprint": "@messaging.sprintpcs.com"}


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
            for key, value in DEFAULT_CONFIG.items():
                if key not in cfg:
                    cfg[key] = value
            return cfg
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    # Ensure directory exists for per-user path; ignore for legacy local file in repo directory
    path = CONFIG_PATH
    cfg_dir = path.parent
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort; if it fails (e.g., package install dir), fallback to user path
        user_path = _user_config_path()
        user_path.parent.mkdir(parents=True, exist_ok=True)
        path = user_path
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)


# ============== INDICATOR CALCULATIONS ==============
def calc_rsi(prices, period=14):
    """RSI: Measures momentum. <30 = oversold (buy), >70 = overbought (sell)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(prices):
    """MACD: Trend momentum. Signal > 0 = bullish, < 0 = bearish."""
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_moving_averages(prices):
    """SMA 50/200: Price vs averages. Golden cross = bullish, Death cross = bearish."""
    sma50 = prices.rolling(window=50).mean()
    sma200 = prices.rolling(window=200).mean()
    return sma50, sma200

def calc_bollinger_bands(prices, period=20):
    """Bollinger Bands: Volatility. Price near lower = oversold, near upper = overbought."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, sma, lower

def calc_volume_trend(volume, period=20):
    """Volume trend: High volume confirms moves, low volume = weak signal."""
    avg_volume = volume.rolling(window=period).mean()
    return avg_volume


def make_sparkline(prices, days=10):
    """Create ASCII sparkline from recent prices."""
    if len(prices) < days:
        return "N/A"
    
    recent = prices.iloc[-days:].values
    min_val, max_val = min(recent), max(recent)
    
    if max_val == min_val:
        return "▅" * days
    
    # Normalize to 0-7 range for 8 levels of spark chars
    sparks = "▁▂▃▄▅▆▇█"
    line = ""
    for val in recent:
        idx = int((val - min_val) / (max_val - min_val) * 7)
        line += sparks[idx]
    
    # Calculate trend direction
    pct_change = ((recent[-1] - recent[0]) / recent[0]) * 100
    if pct_change > 1:
        trend = f"↗ +{pct_change:.1f}%"
        trend_color = "green"
    elif pct_change < -1:
        trend = f"↘ {pct_change:.1f}%"
        trend_color = "red"
    else:
        trend = f"→ {pct_change:.1f}%"
        trend_color = "yellow"
    
    return line, trend, trend_color


# ============== ENHANCED STOCK ANALYSIS ==============
def get_stock_data(ticker: str) -> dict:
    """Fetch stock data with all indicators."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")  # Need 1 year for 200-day MA
        info = stock.info
        
        if hist.empty or len(hist) < 50:
            return {"error": f"Insufficient data for {ticker}"}
        
        close = hist['Close']
        volume = hist['Volume']
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2] if len(close) > 1 else current_price
        day_change = current_price - prev_price
        day_change_pct = (day_change / prev_price * 100) if prev_price else 0.0
        
        # Basic price data
        high_52w = info.get("fiftyTwoWeekHigh", close.max())
        low_52w = info.get("fiftyTwoWeekLow", close.min())
        
        # Position in 52-week range (0% = at low, 100% = at high)
        range_52w = high_52w - low_52w
        position_in_range = ((current_price - low_52w) / range_52w * 100) if range_52w > 0 else 50
        
        # RSI
        rsi_series = calc_rsi(close)
        rsi = rsi_series.iloc[-1]
        
        # MACD
        macd_line, signal_line, histogram = calc_macd(close)
        macd_current = macd_line.iloc[-1]
        macd_signal = signal_line.iloc[-1]
        macd_hist = histogram.iloc[-1]
        macd_crossover = "bullish" if macd_current > macd_signal else "bearish"
        
        # Moving Averages
        sma50, sma200 = calc_moving_averages(close)
        sma50_current = sma50.iloc[-1] if len(sma50.dropna()) > 0 else None
        sma200_current = sma200.iloc[-1] if len(sma200.dropna()) > 0 else None
        
        # Golden/Death Cross detection
        cross_signal = None
        if sma50_current and sma200_current:
            if sma50_current > sma200_current:
                cross_signal = "golden"  # Bullish
            else:
                cross_signal = "death"   # Bearish
        
        # Price vs Moving Averages
        above_sma50 = current_price > sma50_current if sma50_current else None
        above_sma200 = current_price > sma200_current if sma200_current else None
        
        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = calc_bollinger_bands(close)
        bb_position = None
        if bb_upper.iloc[-1] and bb_lower.iloc[-1]:
            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            bb_position = ((current_price - bb_lower.iloc[-1]) / bb_range * 100) if bb_range > 0 else 50
        
        # Volume
        avg_volume = calc_volume_trend(volume)
        current_volume = volume.iloc[-1]
        avg_vol_20 = avg_volume.iloc[-1]
        volume_ratio = (current_volume / avg_vol_20) if avg_vol_20 > 0 else 1
        
        # Sparkline trend (10-day)
        sparkline_result = make_sparkline(close, 10)
        if isinstance(sparkline_result, tuple):
            spark_line, spark_trend, spark_color = sparkline_result
        else:
            spark_line, spark_trend, spark_color = "N/A", "N/A", "yellow"
        
        return {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "price": current_price,
            "change": day_change,
            "change_pct": day_change_pct,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "position_in_range": position_in_range,
            "rsi": rsi,
            "macd": macd_current,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "macd_crossover": macd_crossover,
            "sma50": sma50_current,
            "sma200": sma200_current,
            "cross_signal": cross_signal,
            "above_sma50": above_sma50,
            "above_sma200": above_sma200,
            "bb_position": bb_position,
            "volume_ratio": volume_ratio,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "sparkline": spark_line,
            "spark_trend": spark_trend,
            "spark_color": spark_color,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def score_stock(data: dict) -> dict:
    """
    Score each indicator as green (bullish), yellow (neutral), or red (bearish).
    Returns signals dict with color codes and final verdict.
    """
    if "error" in data:
        return {"error": data["error"], "signals": [], "score": 0, "verdict": "unknown"}
    
    signals = []
    green_count = 0
    red_count = 0
    
    # 1. RSI Signal
    rsi = data.get("rsi")
    if rsi:
        if rsi < 30:
            signals.append(("RSI", f"{rsi:.1f}", "green", "Oversold - potential bounce"))
            green_count += 1
        elif rsi > 70:
            signals.append(("RSI", f"{rsi:.1f}", "red", "Overbought - may pull back"))
            red_count += 1
        else:
            signals.append(("RSI", f"{rsi:.1f}", "yellow", "Neutral zone"))
    
    # 2. MACD Signal
    macd_cross = data.get("macd_crossover")
    macd_hist = data.get("macd_hist", 0)
    if macd_cross:
        if macd_cross == "bullish" and macd_hist > 0:
            signals.append(("MACD", "Bullish", "green", "Momentum rising"))
            green_count += 1
        elif macd_cross == "bearish" and macd_hist < 0:
            signals.append(("MACD", "Bearish", "red", "Momentum falling"))
            red_count += 1
        else:
            signals.append(("MACD", "Mixed", "yellow", "Momentum shifting"))
    
    # 3. Moving Average Cross (Golden/Death)
    cross = data.get("cross_signal")
    if cross:
        if cross == "golden":
            signals.append(("MA Cross", "Golden", "green", "50-day above 200-day"))
            green_count += 1
        else:
            signals.append(("MA Cross", "Death", "red", "50-day below 200-day"))
            red_count += 1
    
    # 4. Price vs SMA200 (long-term trend)
    above_200 = data.get("above_sma200")
    if above_200 is not None:
        if above_200:
            signals.append(("vs SMA200", "Above", "green", "In long-term uptrend"))
            green_count += 1
        else:
            signals.append(("vs SMA200", "Below", "red", "In long-term downtrend"))
            red_count += 1
    
    # 5. Position in 52-week range
    pos = data.get("position_in_range")
    if pos is not None:
        if pos < 25:
            signals.append(("52w Range", f"{pos:.0f}%", "green", "Near 52-week low"))
            green_count += 1
        elif pos > 85:
            signals.append(("52w Range", f"{pos:.0f}%", "red", "Near 52-week high"))
            red_count += 1
        else:
            signals.append(("52w Range", f"{pos:.0f}%", "yellow", "Mid-range"))
    
    # 6. Bollinger Band position
    bb_pos = data.get("bb_position")
    if bb_pos is not None:
        if bb_pos < 20:
            signals.append(("Bollinger", f"{bb_pos:.0f}%", "green", "Near lower band"))
            green_count += 1
        elif bb_pos > 80:
            signals.append(("Bollinger", f"{bb_pos:.0f}%", "red", "Near upper band"))
            red_count += 1
        else:
            signals.append(("Bollinger", f"{bb_pos:.0f}%", "yellow", "Within bands"))
    
    # 7. P/E Ratio
    pe = data.get("pe_ratio")
    if pe:
        if pe < 15:
            signals.append(("P/E Ratio", f"{pe:.1f}", "green", "Undervalued"))
            green_count += 1
        elif pe > 35:
            signals.append(("P/E Ratio", f"{pe:.1f}", "red", "Expensive"))
            red_count += 1
        else:
            signals.append(("P/E Ratio", f"{pe:.1f}", "yellow", "Fair value"))
    
    # 8. Volume confirmation
    vol_ratio = data.get("volume_ratio", 1)
    if vol_ratio > 1.5:
        signals.append(("Volume", f"{vol_ratio:.1f}x avg", "yellow", "High volume - confirms move"))
    elif vol_ratio < 0.5:
        signals.append(("Volume", f"{vol_ratio:.1f}x avg", "yellow", "Low volume - weak signal"))
    else:
        signals.append(("Volume", f"{vol_ratio:.1f}x avg", "yellow", "Normal volume"))
    
    # Final verdict
    total = green_count + red_count
    if total == 0:
        verdict = "HOLD"
        verdict_color = "yellow"
    elif green_count > red_count + 1:
        verdict = "BULLISH"
        verdict_color = "green"
    elif red_count > green_count + 1:
        verdict = "BEARISH"
        verdict_color = "red"
    else:
        verdict = "NEUTRAL"
        verdict_color = "yellow"
    
    return {
        "signals": signals,
        "green": green_count,
        "red": red_count,
        "verdict": verdict,
        "verdict_color": verdict_color,
    }


# ============== RENDER HELPERS (TUI) ==============
def _header_panel(subtitle: str = "") -> Panel:
    title = Text("MARKET PULSE", style="bold cyan")
    subtitle_text = Text(subtitle, style="dim") if subtitle else Text("Stock & News Intelligence", style="dim")
    inner = Align.center(Text.assemble(title, "\n", subtitle_text))
    return Panel(inner, box=box.DOUBLE, border_style="cyan", padding=(1, 2))


def _stock_card(data: dict, score: dict) -> Panel:
    if "error" in data:
        return Panel(f"[red]{data['ticker']}[/red]\n{data['error']}", title=data.get("ticker", "?"), box=box.ROUNDED, border_style="red")

    verdict = score.get("verdict", "NEUTRAL")
    vcolor = score.get("verdict_color", "yellow")
    change = data.get("change", 0) or 0
    arrow = "↗" if change > 0 else ("↘" if change < 0 else "→")
    price_line = f"[bold]${data['price']:.2f}[/bold]  {arrow} {change:+.2f} ({data.get('change_pct', 0.0):+.2f}%)"
    spark = data.get("sparkline", "")
    spark_trend = data.get("spark_trend", "")
    spark_color = data.get("spark_color", "yellow")

    # Short signals summary (max 3)
    sigs = score.get("signals", [])[:3]
    sig_lines = []
    for name, value, color, meaning in sigs:
        dot = {"green": "🟢", "red": "🔴", "yellow": "🟡"}.get(color, "•")
        sig_lines.append(f"{dot} {name}: {value}")
    sig_block = "\n".join(sig_lines)

    body = (
        f"[white]{data['name']}[/white]\n"
        f"{price_line}\n"
        f"{spark}  [{spark_color}]{spark_trend}[/{spark_color}]\n\n"
        f"[bold {vcolor}]{verdict}[/bold {vcolor}]  ([green]{score['green']}[/green]/[red]{score['red']}[/red])\n"
        f"{sig_block}"
    )

    return Panel(body, title=f"[bold]{data['ticker']}[/bold]", box=box.ROUNDED, border_style=vcolor)


# ============== NEWS MONITORING ==============
def fetch_news(feeds: list, keywords: list, seen: list) -> list:
    matches = []
    for feed_info in feeds:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:20]:
                entry_id = entry.get("id") or entry.get("link")
                if entry_id in seen:
                    continue
                title = entry.get("title", "").lower()
                summary = entry.get("summary", "").lower()
                content = title + " " + summary
                matched_keywords = [kw for kw in keywords if kw.lower() in content]
                if matched_keywords:
                    matches.append({
                        "source": feed_info["name"],
                        "title": entry.get("title"),
                        "link": entry.get("link"),
                        "published": entry.get("published", "Unknown"),
                        "summary": entry.get("summary", "")[:300],
                        "keywords": matched_keywords,
                        "id": entry_id,
                    })
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch {feed_info['name']}: {e}[/yellow]")
    return matches

def simple_sentiment(text: str) -> str:
    positive = ["surge", "soar", "gain", "rise", "growth", "profit", "success", "breakthrough", "launch", "partner", "beat"]
    negative = ["drop", "fall", "crash", "loss", "decline", "layoff", "cut", "fail", "concern", "lawsuit", "antitrust", "miss"]
    text_lower = text.lower()
    pos_count = sum(1 for word in positive if word in text_lower)
    neg_count = sum(1 for word in negative if word in text_lower)
    if pos_count > neg_count:
        return "🟢 Positive"
    elif neg_count > pos_count:
        return "🔴 Negative"
    return "⚪ Neutral"

# ============== NOTIFICATIONS ==============
def send_notification(subject: str, body: str, config: dict):
    import smtplib
    from email.mime.text import MIMEText
    notif = config["notifications"]
    if notif["email"]["enabled"]:
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = notif["email"]["sender"]
            msg["To"] = notif["email"]["recipient"]
            with smtplib.SMTP(notif["email"]["smtp_server"], notif["email"]["smtp_port"]) as server:
                server.starttls()
                server.login(notif["email"]["sender"], notif["email"]["password"])
                server.send_message(msg)
            console.print("[green]✓ Email sent[/green]")
        except Exception as e:
            console.print(f"[red]Email failed: {e}[/red]")


# ============== CLI COMMANDS ==============
def cmd_scan(args):
    """Run a full scan of stocks and news with a richer TUI."""
    config = load_config()

    console.print(_header_panel("Full Scan"))
    console.print(Rule(style="dim"))

    stock_cards = []
    with Progress(SpinnerColumn(), TextColumn("[bold]Fetching {task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
        for ticker in config["stocks"]:
            task = progress.add_task(f"{ticker}")
            data = get_stock_data(ticker)
            score = score_stock(data)
            stock_cards.append(_stock_card(data, score))
            progress.update(task, completed=1)

    console.print(Columns(stock_cards, equal=True, expand=True))
    console.print()

    # News section
    console.print("[bold]📰 NEWS ALERTS[/bold]")
    with console.status("Fetching news feeds...", spinner="dots"):
        news = fetch_news(config["feeds"], config["keywords"], config["seen_articles"])

    if news:
        news_table = Table(box=box.ROUNDED)
        news_table.add_column("Source", style="cyan", width=12)
        news_table.add_column("Title", width=50)
        news_table.add_column("Match", style="yellow", width=15)
        news_table.add_column("Sent", width=12)

        for item in news[:8]:
            sentiment = simple_sentiment(item["title"] + item["summary"])
            news_table.add_row(
                item["source"][:12],
                item["title"][:50],
                ", ".join(item["keywords"][:2]),
                sentiment
            )
            config["seen_articles"].append(item["id"])

        console.print(news_table)
        config["seen_articles"] = config["seen_articles"][-500:]
    else:
        console.print("[dim]No new articles matching keywords[/dim]")

    config["last_run"] = datetime.now().isoformat()
    save_config(config)
    console.print()
    console.print(f"[dim]Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")


def cmd_legend(args):
    """Show indicator legend with explanations."""
    console.print(Panel.fit("[bold cyan]📚 INDICATOR LEGEND[/bold cyan]", box=box.DOUBLE))
    console.print()
    
    legend = """
[bold]RSI (Relative Strength Index)[/bold]
  Measures momentum on a 0-100 scale. Like a "exhaustion meter."
  🟢 < 30: Oversold - sellers exhausted, may bounce up
  🟡 30-70: Neutral - normal trading
  🔴 > 70: Overbought - buyers exhausted, may pull back

[bold]MACD (Moving Average Convergence Divergence)[/bold]
  Shows trend momentum by comparing fast vs slow moving averages.
  🟢 Bullish: MACD line above signal line, histogram positive
  🔴 Bearish: MACD line below signal line, histogram negative

[bold]MA Cross (Golden/Death Cross)[/bold]
  Compares 50-day vs 200-day moving average.
  🟢 Golden Cross: 50-day crosses ABOVE 200-day → bullish trend
  🔴 Death Cross: 50-day crosses BELOW 200-day → bearish trend

[bold]vs SMA200 (Price vs 200-day Average)[/bold]
  Is current price above or below the long-term average?
  🟢 Above: Stock in long-term uptrend
  🔴 Below: Stock in long-term downtrend

[bold]52w Range (Position in 52-Week Range)[/bold]
  Where is price between its yearly low (0%) and high (100%)?
  🟢 < 25%: Near yearly low - potential value
  🟡 25-85%: Mid-range
  🔴 > 85%: Near yearly high - expensive

[bold]Bollinger Bands[/bold]
  Volatility bands around a 20-day average.
  🟢 < 20%: Near lower band - oversold
  🟡 20-80%: Within normal range
  🔴 > 80%: Near upper band - overbought

[bold]P/E Ratio (Price to Earnings)[/bold]
  How much you pay per dollar of company earnings.
  🟢 < 15: Cheap/undervalued
  🟡 15-35: Fair value
  🔴 > 35: Expensive

[bold]Volume[/bold]
  Trading activity vs 20-day average.
  High volume confirms price moves. Low volume = weak signal.
"""
    console.print(legend)


def cmd_stocks(args):
    """Detailed stock analysis."""
    config = load_config()
    
    for ticker in config["stocks"]:
        data = get_stock_data(ticker)
        score = score_stock(data)
        
        if "error" in data:
            console.print(f"[red]{ticker}: {data['error']}[/red]")
            continue
        
        verdict_color = score["verdict_color"]
        console.print(Panel.fit(
            f"[bold cyan]{ticker}[/bold cyan] - {data['name']}\n"
            f"[{verdict_color}]● {score['verdict']}[/{verdict_color}] "
            f"([green]{score['green']}[/green] green / [red]{score['red']}[/red] red)",
            box=box.DOUBLE
        ))
        
        # Price info
        price_table = Table(box=box.SIMPLE, show_header=False)
        price_table.add_column("Metric", style="dim", width=15)
        price_table.add_column("Value", justify="right")
        
        price_table.add_row("Current Price", f"${data['price']:.2f}")
        price_table.add_row("52-Week High", f"${data['high_52w']:.2f}")
        price_table.add_row("52-Week Low", f"${data['low_52w']:.2f}")
        price_table.add_row("Position", f"{data['position_in_range']:.0f}% in range")
        if data.get('sma50'):
            price_table.add_row("50-Day MA", f"${data['sma50']:.2f}")
        if data.get('sma200'):
            price_table.add_row("200-Day MA", f"${data['sma200']:.2f}")
        price_table.add_row("P/E Ratio", f"{data['pe_ratio']:.1f}" if data['pe_ratio'] else "N/A")
        price_table.add_row("Analyst", data['recommendation'])
        
        console.print(price_table)
        console.print()
        
        # All signals
        console.print("[bold]Signals:[/bold]")
        for name, value, color, meaning in score["signals"]:
            color_dot = {"green": "🟢", "red": "🔴", "yellow": "🟡"}[color]
            console.print(f"  {color_dot} [bold]{name}[/bold]: {value} - {meaning}")
        console.print()


def cmd_news(args):
    """Show recent news matching keywords."""
    config = load_config()
    news = fetch_news(config["feeds"], config["keywords"], [])
    
    if not news:
        console.print("[dim]No articles found matching keywords[/dim]")
        return
    
    for item in news[:15]:
        sentiment = simple_sentiment(item["title"] + item["summary"])
        console.print(Panel(
            f"[bold]{item['title']}[/bold]\n\n"
            f"[dim]{item['summary'][:200]}...[/dim]\n\n"
            f"Keywords: [yellow]{', '.join(item['keywords'])}[/yellow]  |  {sentiment}\n"
            f"[link={item['link']}]{item['link']}[/link]",
            title=f"[cyan]{item['source']}[/cyan]",
            subtitle=item["published"],
            box=box.ROUNDED
        ))


def cmd_config(args):
    """Show or update configuration."""
    config = load_config()
    
    if args.show:
        console.print_json(data=config)
        return
    
    if args.add_stock:
        ticker = args.add_stock.upper()
        if ticker not in config["stocks"]:
            config["stocks"].append(ticker)
            console.print(f"[green]Added {ticker} to watchlist[/green]")
        else:
            console.print(f"[yellow]{ticker} already in watchlist[/yellow]")
    
    if args.remove_stock:
        ticker = args.remove_stock.upper()
        if ticker in config["stocks"]:
            config["stocks"].remove(ticker)
            console.print(f"[green]Removed {ticker} from watchlist[/green]")
        else:
            console.print(f"[yellow]{ticker} not in watchlist[/yellow]")
    
    if args.add_keyword:
        if args.add_keyword not in config["keywords"]:
            config["keywords"].append(args.add_keyword)
            console.print(f"[green]Added keyword: {args.add_keyword}[/green]")
        else:
            console.print(f"[yellow]Keyword already present: {args.add_keyword}[/yellow]")

    if hasattr(args, 'remove_keyword') and args.remove_keyword:
        if args.remove_keyword in config["keywords"]:
            config["keywords"].remove(args.remove_keyword)
            console.print(f"[green]Removed keyword: {args.remove_keyword}[/green]")
        else:
            console.print(f"[yellow]Keyword not found: {args.remove_keyword}[/yellow]")

    if hasattr(args, 'list') and args.list:
        console.print("[bold]Stocks:[/bold] " + ", ".join(config.get("stocks", [])))
        console.print("[bold]Keywords:[/bold] " + ", ".join(config.get("keywords", [])))
    
    save_config(config)


def cmd_open(args):
    """Open stock chart in browser."""
    import webbrowser
    
    ticker = args.ticker.upper()
    
    urls = {
        "tradingview": f"https://www.tradingview.com/chart/?symbol={ticker}",
        "yahoo": f"https://finance.yahoo.com/quote/{ticker}",
        "google": f"https://www.google.com/finance/quote/{ticker}:NASDAQ",
    }
    
    source = args.source or "tradingview"
    url = urls.get(source, urls["tradingview"])
    
    console.print(f"[cyan]Opening {ticker} on {source.title()}...[/cyan]")
    webbrowser.open(url)


def cmd_watch(args):
    """Live dashboard that auto-refreshes on an interval."""
    config = load_config()
    interval = max(5, args.interval)

    def render_once():
        cards = []
        for t in config["stocks"]:
            data = get_stock_data(t)
            score = score_stock(data)
            cards.append(_stock_card(data, score))
        grid = Columns(cards, equal=True, expand=True)
        header = _header_panel("Live Watch")
        footer = Text(f"Updated {datetime.now().strftime('%H:%M:%S')} • Refresh {interval}s", style="dim")
        return header, grid, footer

    try:
        with Live(refresh_per_second=4, screen=True):
            while True:
                header, body, footer = render_once()
                console.clear()
                console.print(header)
                console.print(body)
                console.print(Align.center(footer))
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watch.[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Market Pulse – Stock & News Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pulse                         Run full scan (default)
  pulse scan                    Same as above - scorecard for all stocks
  pulse stocks                  Detailed analysis with all metrics
  pulse news                    Recent news matching your keywords
  pulse legend                  Explain what each indicator means
  pulse open AAPL               Open TradingView chart in browser
  pulse open MSFT --source yahoo   Open Yahoo Finance instead
  pulse config --show           View current configuration
  pulse config --add-stock NVDA    Add stock to watchlist
  pulse config --remove-stock NVDA Remove stock from watchlist
  pulse config --add-keyword "Tesla" Add keyword to track

Current watchlist: Run 'pulse config --show' to see your stocks and keywords.
Indicator help: Run 'pulse legend' for what RSI, MACD, Bollinger, etc. mean.
"""
    )
    parser.add_argument("--version", action="version", version=f"market-pulse {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Full scan with signal scorecard")
    scan_parser.set_defaults(func=cmd_scan)
    
    # Stocks command
    stocks_parser = subparsers.add_parser("stocks", help="Detailed stock analysis")
    stocks_parser.set_defaults(func=cmd_stocks)
    
    # News command
    news_parser = subparsers.add_parser("news", help="Show matching news")
    news_parser.set_defaults(func=cmd_news)
    
    # Legend command
    legend_parser = subparsers.add_parser("legend", help="Explain all indicators")
    legend_parser.set_defaults(func=cmd_legend)
    
    # Open command
    open_parser = subparsers.add_parser("open", help="Open stock chart in browser")
    open_parser.add_argument("ticker", help="Stock ticker to view")
    open_parser.add_argument("--source", choices=["tradingview", "yahoo", "google"], 
                            default="tradingview", help="Chart source (default: tradingview)")
    open_parser.set_defaults(func=cmd_open)
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("--show", action="store_true", help="Show current config")
    config_parser.add_argument("--add-stock", help="Add stock ticker to watchlist")
    config_parser.add_argument("--remove-stock", help="Remove stock ticker")
    config_parser.add_argument("--add-keyword", help="Add keyword to track")
    config_parser.add_argument("--remove-keyword", help="Remove keyword")
    config_parser.add_argument("--list", action="store_true", help="List stocks and keywords")
    config_parser.set_defaults(func=cmd_config)

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Live dashboard with auto-refresh")
    watch_parser.add_argument("--interval", type=int, default=30, help="Refresh interval seconds (min 5)")
    watch_parser.set_defaults(func=cmd_watch)
    
    args = parser.parse_args()
    
    if args.command is None:
        args.func = cmd_scan
        args.func(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
