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

P_ATM = 101325.0   # Pa（標準大氣壓）
P_KPA = P_ATM / 1000.0
R_DA = 0.287055    # kJ/(kg·K) 乾空氣氣體常數
T_MIN, T_MAX = 0.0, 50.0
W_MAX = 30.0        # g/kg

st.set_page_config(page_title="空氣線圖查表教學工具", layout="wide")


# ============================================================
# 熱力性質計算（CoolProp HAPropsSI，精度高、穩定）
# ============================================================
@st.cache_data
def calc_state(db_c: float, rh_pct: float):
    T = db_c + 273.15
    R = rh_pct / 100.0
    W = HAPropsSI("W", "T", T, "P", P_ATM, "R", R) * 1000.0
    WB = HAPropsSI("Twb", "T", T, "P", P_ATM, "R", R) - 273.15
    H = HAPropsSI("H", "T", T, "P", P_ATM, "R", R) / 1000.0
    V = HAPropsSI("V", "T", T, "P", P_ATM, "R", R)
    DP = HAPropsSI("D", "T", T, "P", P_ATM, "R", R) - 273.15
    return dict(W=W, WB=WB, H=H, V=V, DP=DP)


@st.cache_data
def sat_w_g(T_c: float) -> float:
    """飽和濕度比 (g/kg) at given 乾球溫度"""
    return HAPropsSI("W", "T", T_c + 273.15, "P", P_ATM, "R", 1.0) * 1000.0


@st.cache_data
def rh_curve(rh_pct: float, n=100):
    Ts = np.linspace(T_MIN, T_MAX, n)
    Ws = []
    for t in Ts:
        try:
            w = HAPropsSI("W", "T", t + 273.15, "P", P_ATM, "R", rh_pct / 100.0) * 1000.0
        except ValueError:
            w = np.nan
        Ws.append(w)
    Ws = np.array(Ws)
    mask = (~np.isnan(Ws)) & (Ws <= W_MAX)
    return Ts[mask], Ws[mask]


@st.cache_data
def wb_line(wb_c: float, n=40):
    """等濕球溫度線：T_db 從 wb_c(飽和點) 到 T_MAX"""
    Ts = np.linspace(wb_c, T_MAX, n)
    Ws = []
    for t in Ts:
        try:
            w = HAPropsSI("W", "Twb", wb_c + 273.15, "T", t + 273.15, "P", P_ATM) * 1000.0
        except ValueError:
            w = np.nan
        Ws.append(w)
    Ws = np.array(Ws)
    mask = (~np.isnan(Ws)) & (Ws <= W_MAX) & (Ws >= 0)
    return Ts[mask], Ws[mask]


def w_for_h(T_c, h_kjkg):
    """等焓線封閉解（標準 ASHRAE 近似公式，避免大量 CoolProp 呼叫）"""
    return (h_kjkg - 1.006 * T_c) / (2501.0 + 1.805 * T_c) * 1000.0


def sat_w_g_vec(Ts):
    return np.array([sat_w_g(round(float(t), 1)) for t in Ts])


def h_line(h_kjkg: float, n=60):
    Ts = np.linspace(T_MIN, T_MAX, n)
    Ws = w_for_h(Ts, h_kjkg)
    mask = (Ws >= 0) & (Ws <= W_MAX) & (Ws <= sat_w_g_vec(Ts) + 0.3)
    return Ts[mask], Ws[mask]


@st.cache_data
def dew_point_T(w_target: float) -> float:
    """飽和曲線反函數：給定濕度比，反推該濕度下的飽和(露點)溫度。
    用於將背景骨架線裁切成貼合飽和曲線的梯形外觀。"""
    if w_target <= sat_w_g(T_MIN):
        return T_MIN
    lo, hi = T_MIN, T_MAX
    for _ in range(25):
        mid = (lo + hi) / 2.0
        if sat_w_g(round(mid, 2)) < w_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0





def w_for_v(T_c, v_target):
    """等比容線封閉解（理想氣體近似，與原程式 specific_volume 公式互逆）"""
    T_K = T_c + 273.15
    return (v_target * P_KPA / (R_DA * T_K) - 1.0) / 1.6078 * 1000.0


def v_line(v_target: float, n=60):
    Ts = np.linspace(T_MIN, T_MAX, n)
    Ws = w_for_v(Ts, v_target)
    mask = (Ws >= 0) & (Ws <= W_MAX) & (Ws <= sat_w_g_vec(Ts) + 0.3)
    return Ts[mask], Ws[mask]


def to_screen(T, W, skew):
    """座標轉換：斜交座標時 X 會依 W 產生水平偏移，Y 維持濕度比不變"""
    return T + skew * W, W


# ============================================================
# 繪製空氣線圖（skew=0 為直角座標；skew>0 為 ASHRAE 風格斜交座標）
# ============================================================
def draw_chart(db, rh, state, skew=0.0, show_h=True, show_v=True, show_wb=True):
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=110)

    x_max = T_MAX + skew * W_MAX

    # --- 斜交模式：把左上、右下兩個「非圖表範圍」的空白角落改用淺色網底標示 ---
    if skew > 0:
        ax.fill([0, 0, skew * W_MAX], [0, W_MAX, W_MAX],
                 facecolor="#fafafa", edgecolor="#e0e0e0", hatch="////", linewidth=0.4, zorder=0)
        ax.fill([T_MAX, x_max, x_max], [0, 0, W_MAX],
                 facecolor="#fafafa", edgecolor="#e0e0e0", hatch="////", linewidth=0.4, zorder=0)

    # --- 背景骨架格線：貼合飽和曲線裁切，呈現梯形外觀（兩種模式皆適用）---
    for t in range(0, 51, 5):
        w_top = min(W_MAX, sat_w_g(float(t)))
        if w_top <= 0:
            continue
        x0, y0 = to_screen(t, 0, skew)
        x1, y1 = to_screen(t, w_top, skew)
        lw = 0.8 if t % 10 == 0 else 0.4
        ax.plot([x0, x1], [y0, y1], color="#d5d5d5", linewidth=lw, zorder=0)
    for wv in range(0, 31, 2):
        t_left = dew_point_T(float(wv))
        x0, y0 = to_screen(t_left, wv, skew)
        x1, y1 = to_screen(50, wv, skew)
        lw = 0.8 if wv % 10 == 0 else 0.4
        ax.plot([x0, x1], [y0, y1], color="#d5d5d5", linewidth=lw, zorder=0)

    # --- 相對濕度曲線 ---
    rh_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for level in rh_levels:
        Ts, Ws = rh_curve(level)
        Xs, Ys = to_screen(Ts, Ws, skew)
        is_sat = level == 100
        ax.plot(Xs, Ys,
                 linestyle="-" if is_sat else "--",
                 color="#2c3e50" if is_sat else "#95a5a6",
                 alpha=0.9 if is_sat else 0.55,
                 linewidth=1.5 if is_sat else 0.8, zorder=2)
        if len(Ts) > 3:
            # 各 RH 線標籤沿線分散在不同位置，避免在飽和曲線附近全部擠成一團
            frac = 0.45 + 0.42 * (level / 100.0)
            idx = min(int(len(Ts) * frac), len(Ts) - 1)
            ax.text(Xs[idx], Ys[idx], f" {level}%", fontsize=7, color="#7f8c8d", va="center", zorder=3)

    # --- 等焓線：畫在有效區內（飽和曲線以下），並在貼近飽和曲線的端點直接標數字 ---
    y_top = W_MAX
    if show_h:
        for hv in range(10, 121, 10):
            Ts, Ws = h_line(float(hv))
            if len(Ts) < 2:
                continue
            Xs, Ys = to_screen(Ts, Ws, skew)
            ax.plot(Xs, Ys, color="#e67e22", linewidth=0.8, alpha=0.65, zorder=1)

            # 端點在飽和曲線附近（線的起點，T 較低那端）→ 沿線方向外推一小段放標籤
            if len(Xs) >= 2:
                dx, dy = Xs[0] - Xs[1], Ys[0] - Ys[1]
                norm = max((dx**2 + dy**2) ** 0.5, 1e-6)
                lx = Xs[0] + dx / norm * 1.3
                ly = Ys[0] + dy / norm * 1.3
                ax.text(lx, ly, f"{hv}", fontsize=6.5, color="#c0620a",
                         fontweight="bold", ha="center", va="center", zorder=4,
                         bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                    edgecolor="none", alpha=0.7))
        # 圖例外加一行說明焓值單位
        ax.text(0.99, 1.045, "等焓線標籤：kJ/kg 乾空氣", transform=ax.transAxes,
                 fontsize=7.5, color="#c0620a", ha="right", va="bottom")

    # --- 等比容線 ---
    if show_v:
        for vv in np.arange(0.78, 0.97, 0.02):
            Ts, Ws = v_line(float(vv))
            if len(Ts) < 2:
                continue
            Xs, Ys = to_screen(Ts, Ws, skew)
            ax.plot(Xs, Ys, color="#8e44ad", linewidth=0.6, alpha=0.4, zorder=1)

    # --- 等濕球溫度線 ---
    if show_wb:
        for wbv in range(0, 36, 5):
            Ts, Ws = wb_line(float(wbv))
            if len(Ts) < 2:
                continue
            Xs, Ys = to_screen(Ts, Ws, skew)
            ax.plot(Xs, Ys, color="#2980b9", linewidth=0.5, alpha=0.4, zorder=1)

    # --- 狀態點與查表輔助線 ---
    w = state["W"]
    wb = state["WB"]
    h = state["H"]

    Xp, Yp = to_screen(db, w, skew)
    X0, Y0 = to_screen(db, 0, skew)
    Xr, Yr = to_screen(50, w, skew)

    ax.plot(Xp, Yp, "o", color="#e74c3c", markersize=8, zorder=6, label="狀態點 P")
    ax.plot([X0, Xp], [Y0, Yp], "r--", linewidth=1.5, label="① 乾球溫度 DB", zorder=5)
    ax.plot([Xp, Xr], [Yp, Yr], "g--", linewidth=1.5, label="② 濕度比 W", zorder=5)

    w_sat_wb = sat_w_g(round(wb, 1))
    Xwb, Ywb = to_screen(wb, w_sat_wb, skew)
    ax.plot([Xp, Xwb], [Yp, Ywb], "b-.", linewidth=1.5, label="③ 濕球溫度 WB", zorder=5)

    ax.annotate(
        f" P ({db:.1f}°C, {rh:.0f}%)\n W={w:.2f} g/kg\n h={h:.1f} kJ/kg",
        xy=(Xp, Yp), xytext=(max(Xp - 16, 1), min(Yp + 3, 27)),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4", alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        fontsize=9, zorder=7,
    )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_top)
    ax.set_xticks(list(range(0, 51, 10)))
    ax.set_xticklabels([str(t) for t in range(0, 51, 10)])
    ax.set_xlabel("乾球溫度 Dry-Bulb Temperature, DB (°C)", fontsize=10, fontweight="bold")
    ax.set_ylabel("濕度比 Humidity Ratio, W (g/kg 乾空氣)", fontsize=10, fontweight="bold")
    title = "ASHRAE 風格空氣線圖（斜交座標）" if skew > 0 else "空氣線圖（直角座標，簡明版）"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.3)
    legend_loc = "lower right" if skew > 0 else "upper left"
    ax.legend(loc=legend_loc, fontsize=8, framealpha=0.9)
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
    st.header("圖表顯示設定")
    mode = st.radio("座標模式", ["簡明直角座標", "ASHRAE 風格斜交座標"], index=0)
    skew = 1.0 if mode == "ASHRAE 風格斜交座標" else 0.0

    show_h = st.checkbox("顯示等焓線 (h)", value=True)
    show_v = st.checkbox("顯示等比容線 (v)", value=(mode == "ASHRAE 風格斜交座標"))
    show_wb = st.checkbox("顯示等濕球溫度線 (WB)", value=(mode == "ASHRAE 風格斜交座標"))

    st.markdown("---")
    st.caption("標準大氣壓 P = 101.325 kPa")
    if FONT_NAME is None:
        st.warning("⚠️ 目前環境找不到中文字型，圖表中文可能顯示為方框。部署到 Streamlit Cloud 時請加入 packages.txt（見下方說明）。")

state = calc_state(db, rh)

col_chart, col_info = st.columns([1.55, 1])

with col_chart:
    fig = draw_chart(db, rh, state, skew=skew, show_h=show_h, show_v=show_v, show_wb=show_wb)
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

with st.expander("ℹ️ 關於計算方法 / 兩種座標模式 / 部署到 Streamlit Cloud"):
    st.markdown("""
**計算方法**：本工具使用 **CoolProp** 的 `HAPropsSI` 計算濕空氣熱力性質（濕度比、濕球溫度、
焓值、比容、露點），精度與穩定性優於手刻公式。等焓線與等比容線的背景參考線為求繪圖效能，
採用標準 ASHRAE 工程近似封閉解（與 CoolProp 結果誤差在工程可接受範圍內，僅影響背景參考線，
不影響狀態點主要數值）。

**兩種座標模式**：
- **簡明直角座標**：乾球溫度與濕度比互相垂直，閱讀邏輯直覺，適合初學者。
- **ASHRAE 風格斜交座標**：仿照 ASHRAE Psychrometric Chart No.1 的斜交（oblique）繪圖方式，
  將座標框做剪切變形，讓等焓線與等濕球線在視覺上更容易分辨——這是實體工程圖表的標準畫法。
  本工具是依相同物理原理自行繪製，非直接複製 ASHRAE 圖表版面。

**部署到 Streamlit Cloud 時，若圖表中文顯示為方框**，請在 GitHub repo 根目錄新增
一個 `packages.txt` 檔案（沒有副檔名），內容只需一行：

```
fonts-noto-cjk
```

Streamlit Cloud 會自動用 `apt-get` 安裝這個系統字型套件，重新部署後中文即可正常顯示。
""")
