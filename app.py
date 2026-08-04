import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from CoolProp.HumidAirProp import HAPropsSI

# ============================================================
# 中文字型設定（Streamlit Cloud 為 Linux 環境，需搭配 packages.txt
# 安裝 fonts-noto-cjk，本地端 Windows 則會自動使用內建中文字型）
# ============================================================
def setup_chinese_font():
    candidates = [
        "Noto Sans CJK TC", "Noto Sans TC", "Microsoft JhengHei",
        "PingFang TC", "Heiti TC", "SimHei", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            return name
    return None

FONT_NAME = setup_chinese_font()
matplotlib.rcParams["axes.unicode_minus"] = False

P_ATM = 101325.0  # Pa（標準大氣壓）

st.set_page_config(page_title="空氣線圖查表教學工具", layout="wide")


# ============================================================
# 熱力性質計算（改用 CoolProp HAPropsSI，精度較高、較穩定）
# ============================================================
@st.cache_data
def calc_state(db_c: float, rh_pct: float):
    T = db_c + 273.15
    R = rh_pct / 100.0
    W = HAPropsSI("W", "T", T, "P", P_ATM, "R", R) * 1000.0       # g/kg 乾空氣
    WB = HAPropsSI("Twb", "T", T, "P", P_ATM, "R", R) - 273.15    # °C
    H = HAPropsSI("H", "T", T, "P", P_ATM, "R", R) / 1000.0       # kJ/kg 乾空氣
    V = HAPropsSI("V", "T", T, "P", P_ATM, "R", R)                # m3/kg 乾空氣
    DP = HAPropsSI("D", "T", T, "P", P_ATM, "R", R) - 273.15      # 露點 °C
    return dict(W=W, WB=WB, H=H, V=V, DP=DP)


@st.cache_data
def rh_curve(rh_pct: float, t_min=0.0, t_max=50.0, n=120, w_max=30.0):
    """計算單一 RH% 曲線在 T-W 座標下的資料（供繪圖用）"""
    Ts = np.linspace(t_min, t_max, n)
    Ws = []
    for t in Ts:
        try:
            w = HAPropsSI("W", "T", t + 273.15, "P", P_ATM, "R", rh_pct / 100.0) * 1000.0
        except ValueError:
            w = np.nan
        Ws.append(w)
    Ws = np.array(Ws)
    mask = (~np.isnan(Ws)) & (Ws <= w_max)
    return Ts[mask], Ws[mask]


# ============================================================
# 繪製空氣線圖
# ============================================================
def draw_chart(db, rh, state):
    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=110)

    rh_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for level in rh_levels:
        Ts, Ws = rh_curve(level)
        is_sat = level == 100
        ax.plot(
            Ts, Ws,
            linestyle="-" if is_sat else "--",
            color="#2c3e50" if is_sat else "#95a5a6",
            alpha=0.9 if is_sat else 0.55,
            linewidth=1.4 if is_sat else 0.8,
        )
        if len(Ts) > 3:
            idx = len(Ts) - 1 if not is_sat else int(len(Ts) * 0.9)
            idx = min(idx, len(Ts) - 1)
            ax.text(Ts[idx], Ws[idx], f" {level}%", fontsize=7, color="#7f8c8d",
                     va="center")

    # 狀態點與輔助線
    w = state["W"]
    wb = state["WB"]
    h = state["H"]

    ax.plot(db, w, "o", color="#e74c3c", markersize=8, zorder=5, label="狀態點 P")
    ax.plot([db, db], [0, w], "r--", linewidth=1.4, label="① 乾球溫度 DB")
    ax.plot([db, 50], [w, w], "g--", linewidth=1.4, label="② 濕度比 W")

    # 濕球等焓輔助線（近似畫到飽和曲線上對應濕球點）
    w_sat_wb = HAPropsSI("W", "T", wb + 273.15, "P", P_ATM, "R", 1.0) * 1000.0
    ax.plot([db, wb], [w, w_sat_wb], "b-.", linewidth=1.4, label="③ 濕球溫度 WB")

    ax.annotate(
        f" P ({db:.1f}°C, {rh:.0f}%)\n W={w:.2f} g/kg\n h={h:.1f} kJ/kg",
        xy=(db, w), xytext=(max(db - 16, 1), min(w + 3, 27)),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4", alpha=0.85),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        fontsize=9,
    )

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 30)
    ax.set_xlabel("乾球溫度 Dry-Bulb Temperature, DB (°C)", fontsize=10, fontweight="bold")
    ax.set_ylabel("濕度比 Humidity Ratio, W (g/kg 乾空氣)", fontsize=10, fontweight="bold")
    ax.set_title("ASHRAE 標準空氣線圖 (Psychrometric Chart)", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


# ============================================================
# UI
# ============================================================
st.title("💧 空氣線圖 (Psychrometric Chart) 查表教學與計算工具")
st.caption("輸入乾球溫度與相對濕度，自動計算濕球溫度、濕度比、焓值、比容，並逐步說明查表流程。")

with st.sidebar:
    st.header("輸入參數")
    db = st.number_input("乾球溫度 DB (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
    rh = st.number_input("相對濕度 RH (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    st.markdown("---")
    st.caption("標準大氣壓 P = 101.325 kPa")
    if FONT_NAME is None:
        st.warning("⚠️ 目前環境找不到中文字型，圖表中文可能顯示為方框。部署到 Streamlit Cloud 時請加入 packages.txt（見下方說明）。")

state = calc_state(db, rh)

col_chart, col_info = st.columns([1.55, 1])

with col_chart:
    fig = draw_chart(db, rh, state)
    st.pyplot(fig, width="stretch")

with col_info:
    st.subheader("📊 查表與計算結果")
    m1, m2 = st.columns(2)
    m1.metric("濕球溫度 WB", f"{state['WB']:.2f} °C")
    m2.metric("濕度比 W", f"{state['W']:.2f} g/kg")
    m1.metric("焓值 h", f"{state['H']:.2f} kJ/kg")
    m2.metric("比容 v", f"{state['V']:.4f} m³/kg")
    st.metric("露點溫度 DP", f"{state['DP']:.2f} °C")

    st.subheader("📝 一步步查表教學說明")
    steps_text = f"""
**步驟一：定位狀態點**
1. 在橫軸找到乾球溫度 DB = {db:.1f}°C。
2. 沿垂直線向上延伸，找到與相對濕度 RH = {rh:.0f}% 弧線的交點 P。
3. 點 P 即為目前空氣狀態點。

**步驟二：讀取濕度比 W**
- 從點 P 向右拉水平線至縱軸。
- 讀出濕度比 W ≈ **{state['W']:.2f} g/kg 乾空氣**。

**步驟三：讀取濕球溫度 WB**
- 從點 P 沿左上方等濕球（等焓）斜線移動，至 100% 飽和曲線的交點。
- 對應溫度即為濕球溫度 WB ≈ **{state['WB']:.2f} °C**。

**步驟四：讀取焓值 h**
- 沿與焓線平行的斜線延伸至圖表外側的焓刻度尺。
- 讀出焓值 h ≈ **{state['H']:.2f} kJ/kg 乾空氣**。

**步驟五：讀取比容 v**
- 觀察穿過點 P 附近的比容線（陡峭斜線族）。
- 比例內插得出比容 v ≈ **{state['V']:.4f} m³/kg 乾空氣**。

**補充：露點溫度 DP**
- 從點 P 向左水平延伸至 100% 飽和曲線，對應溫度即為露點 DP ≈ **{state['DP']:.2f} °C**。
"""
    st.markdown(steps_text)

with st.expander("ℹ️ 關於計算方法 / 部署到 Streamlit Cloud 的中文字型設定"):
    st.markdown("""
本工具改用 **CoolProp** 的 `HAPropsSI` 函式計算濕空氣熱力性質，取代原本手刻的
Hyland-Wexler 飽和蒸氣壓公式與濕球溫度二分法搜尋，計算結果與 ASHRAE 圖表基本一致，
且更穩定、不需自行處理收斂問題。

**部署到 Streamlit Cloud 時，若圖表中文顯示為方框，請在 GitHub repo 根目錄新增
一個 `packages.txt` 檔案（沒有副檔名），內容只需一行：**

```
fonts-noto-cjk
```

Streamlit Cloud 會自動用 `apt-get` 安裝這個系統字型套件，重新部署後中文即可正常顯示。
""")
