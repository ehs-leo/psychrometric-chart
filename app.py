import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from CoolProp.HumidAirProp import HAPropsSI

# ============================================================
# 中文字型設定（ Streamlit Cloud Linux 環境）
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

# 常數設定
P_ATM = 101325.0   # Pa (1 atm)
P_KPA = P_ATM / 1000.0
R_DA = 0.287055    # kJ/(kg·K)
T_MIN, T_MAX = 0.0, 50.0
W_MAX = 30.0        # g/kg

st.set_page_config(page_title="空氣線圖教學與計算工具", layout="wide")

# ============================================================
# 物性計算核心 (CoolProp)
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
    T_K = T_c + 273.15
    return (v_target * P_KPA / (R_DA * T_K) - 1.0) / 1.6078 * 1000.0

def v_line(v_target: float, n=60):
    Ts = np.linspace(T_MIN, T_MAX, n)
    Ws = w_for_v(Ts, v_target)
    mask = (Ws >= 0) & (Ws <= W_MAX) & (Ws <= sat_w_g_vec(Ts) + 0.3)
    return Ts[mask], Ws[mask]

def to_screen(T, W, skew):
    return T + skew * W, W

# ============================================================
# 圖表繪製 Engine
# ============================================================
def draw_chart(db, rh, state, skew=0.0, show_h=True, show_v=True, show_wb=True):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    x_max = T_MAX + skew * W_MAX

    # 1. 基礎灰網格（擬合飽和線區域）
    for t in range(0, 51, 1):
        w_top = min(W_MAX, sat_w_g(float(t)))
        if w_top <= 0:
            continue
        x0, y0 = to_screen(t, 0, skew)
        x1, y1 = to_screen(t, w_top, skew)
        lw = 0.6 if t % 5 == 0 else 0.25
        alpha = 0.4 if t % 5 == 0 else 0.2
        ax.plot([x0, x1], [y0, y1], color="#888888", linewidth=lw, alpha=alpha, zorder=0)

    for wv in range(0, 31, 1):
        t_left = dew_point_T(float(wv))
        x0, y0 = to_screen(t_left, wv, skew)
        x1, y1 = to_screen(50, wv, skew)
        lw = 0.6 if wv % 2 == 0 else 0.25
        alpha = 0.4 if wv % 2 == 0 else 0.2
        ax.plot([x0, x1], [y0, y1], color="#888888", linewidth=lw, alpha=alpha, zorder=0)

    # 2. 相對濕度線 (RH)
    rh_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for level in rh_levels:
        Ts, Ws = rh_curve(level)
        Xs, Ys = to_screen(Ts, Ws, skew)
        is_sat = (level == 100)
        ax.plot(Xs, Ys,
                linestyle="-" if is_sat else "--",
                color="#111111" if is_sat else "#555555",
                alpha=0.95 if is_sat else 0.6,
                linewidth=1.8 if is_sat else 0.8, zorder=2)
        if len(Ts) > 3:
            idx = min(int(len(Ts) * 0.85), len(Ts) - 1)
            ax.text(Xs[idx], Ys[idx], f"{level}%", fontsize=7, color="#333333", va="center", zorder=3)

    # 3. 等焓線 (Enthalpy)
    if show_h:
        for hv in range(10, 121, 10):
            Ts, Ws = h_line(float(hv))
            if len(Ts) < 2:
                continue
            Xs, Ys = to_screen(Ts, Ws, skew)
            ax.plot(Xs, Ys, color="#d35400", linewidth=0.7, alpha=0.6, zorder=1)
            if len(Xs) >= 2:
                dx, dy = Xs[0] - Xs[1], Ys[0] - Ys[1]
                norm = max((dx**2 + dy**2) ** 0.5, 1e-6)
                lx, ly = Xs[0] + dx / norm * 1.2, Ys[0] + dy / norm * 1.2
                ax.text(lx, ly, f"{hv}", fontsize=6.5, color="#b03a2e", fontweight="bold", ha="center", va="center", zorder=4)

    # 4. 等比容線 (Volume)
    if show_v:
        for vv in np.arange(0.78, 0.98, 0.02):
            Ts, Ws = v_line(float(vv))
            if len(Ts) < 2:
                continue
            Xs, Ys = to_screen(Ts, Ws, skew)
            ax.plot(Xs, Ys, color="#6c3483", linewidth=0.6, linestyle=":", alpha=0.7, zorder=1)

    # 5. 等濕球線 (Wet Bulb)
    if show_wb:
        for wbv in range(0, 36, 5):
            Ts, Ws = wb_line(float(wbv))
            if len(Ts) < 2:
                continue
            Xs, Ys = to_screen(Ts, Ws, skew)
            ax.plot(Xs, Ys, color="#1b4f72", linewidth=0.6, alpha=0.5, zorder=1)

    # 6. 當前狀態點與輔助指示
    w, wb, h = state["W"], state["WB"], state["H"]
    Xp, Yp = to_screen(db, w, skew)
    X0, Y0 = to_screen(db, 0, skew)
    Xr, Yr = to_screen(50, w, skew)

    ax.plot(Xp, Yp, "o", color="#e74c3c", markersize=7, zorder=6, label="狀態點 P")
    ax.plot([X0, Xp], [Y0, Yp], "r--", linewidth=1.2, label="① DB 乾球溫度", zorder=5)
    ax.plot([Xp, Xr], [Yp, Yr], "g--", linewidth=1.2, label="② W 濕度比", zorder=5)

    w_sat_wb = sat_w_g(round(wb, 1))
    Xwb, Ywb = to_screen(wb, w_sat_wb, skew)
    ax.plot([Xp, Xwb], [Yp, Ywb], "b-.", linewidth=1.2, label="③ WB 濕球溫度", zorder=5)

    ax.annotate(
        f" P ({db:.1f}°C, {rh:.0f}%)\n W={w:.2f} g/kg\n h={h:.1f} kJ/kg",
        xy=(Xp, Yp), xytext=(max(Xp - 15, 1), min(Yp + 3, 26)),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffde7", edgecolor="#fbc02d", alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#f57f17"),
        fontsize=8.5, zorder=7,
    )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_top := W_MAX)
    ax.set_xticks(list(range(0, 51, 5)))
    ax.set_yticks(list(range(0, 31, 5)))
    ax.set_xlabel("乾球溫度 Dry-Bulb Temperature, DB (°C)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("濕度比 Humidity Ratio, W (g/kg 乾空氣)", fontsize=9.5, fontweight="bold")
    
    title_str = "ASHRAE 風格空氣線圖（斜交座標）" if skew > 0 else "空氣線圖（直角座標）"
    ax.set_title(title_str, fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left" if skew == 0 else "lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    return fig

# ============================================================
# UI 介面
# ============================================================
st.title("💧 空氣線圖 (Psychrometric Chart) 查表教學工具")

with st.sidebar:
    st.header("1. 輸入狀態點參數")
    db = st.number_input("乾球溫度 DB (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
    rh = st.number_input("相對濕度 RH (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)

    st.markdown("---")
    st.header("2. 圖表模式與層疊設定")
    mode = st.radio("座標模式", ["ASHRAE 風格斜交座標", "簡明直角座標"], index=0)
    skew = 1.6 if mode == "ASHRAE 風格斜交座標" else 0.0

    show_h = st.checkbox("顯示等焓線 (h)", value=True)
    show_v = st.checkbox("顯示等比容線 (v)", value=True)
    show_wb = st.checkbox("顯示等濕球線 (WB)", value=True)

state = calc_state(db, rh)
col_chart, col_info = st.columns([1.6, 1])

with col_chart:
    fig = draw_chart(db, rh, state, skew=skew, show_h=show_h, show_v=show_v, show_wb=show_wb)
    st.pyplot(fig, width="stretch")

with col_info:
    st.subheader("📊 查表計算結果")
    m1, m2 = st.columns(2)
    m1.metric("濕球溫度 WB", f"{state['WB']:.2f} °C")
    m2.metric("濕度比 W", f"{state['W']:.2f} g/kg")
    m1.metric("焓值 h", f"{state['H']:.2f} kJ/kg")
    m2.metric("比容 v", f"{state['V']:.4f} m³/kg")
    st.metric("露點溫度 DP", f"{state['DP']:.2f} °C")

    st.markdown("---")
    st.subheader("📝 步驟說明")
    st.markdown(f"""
1. **定位點 P**：對應 DB = {db:.1f}°C 與 RH = {rh:.0f}% 交點。
2. **濕度比 W**：向右延伸讀取縱軸（W ≈ {state['W']:.2f} g/kg）。
3. **濕球溫度 WB**：沿斜線沿至 100% 飽和線（WB ≈ {state['WB']:.2f} °C）。
4. **焓值 h**：平行於焓線對應刻度（h ≈ {state['H']:.2f} kJ/kg）。
5. **比容 v**：內插穿過該點的比容線（v ≈ {state['V']:.4f} m³/kg）[cite: 1]。
""")