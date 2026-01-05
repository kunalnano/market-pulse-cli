<p align="center">
  <img src="assets/header.svg" alt="Market Pulse" width="800">
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"></a>
  <a href="https://github.com/kunalnano/market-pulse/stargazers"><img src="https://img.shields.io/github/stars/kunalnano/market-pulse?style=flat-square&color=yellow" alt="Stars"></a>
  <a href="https://github.com/kunalnano/market-pulse/issues"><img src="https://img.shields.io/github/issues/kunalnano/market-pulse?style=flat-square" alt="Issues"></a>
</p>

<p align="center">
  <b>Your personal stock market radar.</b><br>
  <sub>Technical indicators • News tracking • Signal scoring — all from your terminal.</sub>
</p>

<br>

<p align="center">
  <img src="screenshot.png" alt="Market Pulse Demo" width="680" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
</p>

<br>

---

<br>

## ⚡ Quick Start

```bash
git clone https://github.com/kunalnano/market-pulse.git
cd market-pulse
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python pulse.py
```

<details>
<summary>📦 <b>Add shell alias for quick access</b></summary>

```bash
# Add to ~/.zshrc or ~/.bashrc
alias pulse="/path/to/market-pulse/venv/bin/python /path/to/market-pulse/pulse.py"
```
</details>

<br>

---

<br>

## 🎯 What It Does

<table>
<tr>
<td width="50%">

### 📊 Technical Analysis
- **RSI** — Momentum exhaustion
- **MACD** — Trend direction
- **Bollinger Bands** — Volatility
- **Moving Averages** — 50/200-day trends
- **Golden/Death Cross** — Trend reversals

</td>
<td width="50%">

### 🚦 Signal Scoring
Every indicator gets a traffic light:
- 🟢 **Green** — Bullish signal
- 🟡 **Yellow** — Neutral / Hold
- 🔴 **Red** — Bearish signal

**More green = look closer. More red = wait.**

</td>
</tr>
</table>

<br>

---

<br>

## 📖 Commands

| Command | What it does |
|---------|--------------|
| `pulse` | Full scan with scorecard |
| `pulse stocks` | Detailed analysis |
| `pulse news` | News matching your keywords |
| `pulse legend` | Explain all indicators |
| `pulse open AAPL` | Open chart in browser |
| `pulse config --show` | View your settings |
| `pulse config --add-stock NVDA` | Add to watchlist |

<br>

---

<br>

## 🔍 Signal Reference

<table>
<tr>
<th>Indicator</th>
<th>🟢 Bullish</th>
<th>🔴 Bearish</th>
</tr>
<tr><td><b>RSI</b></td><td>< 30 (oversold)</td><td>> 70 (overbought)</td></tr>
<tr><td><b>MACD</b></td><td>Bullish crossover</td><td>Bearish crossover</td></tr>
<tr><td><b>MA Cross</b></td><td>Golden (50 > 200)</td><td>Death (50 < 200)</td></tr>
<tr><td><b>vs SMA200</b></td><td>Price above</td><td>Price below</td></tr>
<tr><td><b>52w Range</b></td><td>< 25% (near low)</td><td>> 85% (near high)</td></tr>
<tr><td><b>Bollinger</b></td><td>< 20%</td><td>> 80%</td></tr>
<tr><td><b>P/E Ratio</b></td><td>< 15 (cheap)</td><td>> 35 (expensive)</td></tr>
</table>

<br>

> 💡 **Pro tip:** Run `pulse legend` for detailed explanations with real-world analogies.

<br>

---

<br>

## ⚙️ Configuration

On first run, creates `config.json` with:

```
Stocks:    AAPL, GOOGL
Keywords:  Google, Apple, AI, OpenAI, Claude, Anthropic, Gemini
Feeds:     Hacker News, TechCrunch, The Verge, Ars Technica
```

Customize with `pulse config --add-stock TICKER` and `pulse config --add-keyword "term"`.

<br>

---

<br>

## 🛠 Built With

<p>
  <img src="https://img.shields.io/badge/yfinance-Stock%20Data-blue?style=flat-square" alt="yfinance">
  <img src="https://img.shields.io/badge/feedparser-RSS%20Parsing-orange?style=flat-square" alt="feedparser">
  <img src="https://img.shields.io/badge/rich-Terminal%20UI-purple?style=flat-square" alt="rich">
  <img src="https://img.shields.io/badge/pandas-Data%20Analysis-green?style=flat-square" alt="pandas">
</p>

<br>

---

<br>

<p align="center">
  <sub>Built in one conversation with <a href="https://anthropic.com">Claude Opus</a>.<br>
  The kind of tool that used to take a weekend now takes an hour.</sub>
</p>

<p align="center">
  <a href="https://github.com/kunalnano/market-pulse/stargazers">⭐ Star this repo</a> if you found it useful!
</p>
