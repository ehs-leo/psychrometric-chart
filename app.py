import streamlit as st
import matplotlib.pyplot as plt
from psychrochart import PsychroChart
import CoolProp.CoolProp as CP

# 設定頁面標題與佈局
st.set_page_config(page_title="ASHRAE 空氣線圖分析工具", layout="wide")

st.title("🌬️ ASHRAE 空氣線圖分析工具 (Psychrometric Chart)")
st.markdown("輸入空氣狀態參數，即時計算熱力性質並在空氣線圖上繪製狀態點。")

# --- 側邊欄：輸入參數 ---
st.sidebar.header("⚙️ 空氣狀態點輸入")

# 大氣壓力 (預設 101.325 kPa，即標準海平面大氣壓)
P_baro = st.sidebar.number_input("大氣壓力 (kPa)", value=101.325, step=0.1, format="%.3f")
P_pa = P_baro * 1000  # 轉成 Pa 供 CoolProp 使用

# 乾球溫度 Dry-Bulb Temperature (°C)
T_db = st.sidebar.slider("乾球溫度 T_db (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)

# 相對濕度 Relative Humidity (%)
RH = st.sidebar.slider("相對濕度 RH (%)", min_value=5.0, max_value=100.0, value=50.0, step=1.0)
RH_frac = RH / 100.0  # 轉成 0~1 的小數

# --- 熱力性質計算 (CoolProp) ---
try:
    # 絕對濕度 / 含濕度 Humidity Ratio W (kg_water / kg_dry_air)
    W = CP.HAPropsSI('W', 'T', T_db + 273.15, 'R', RH_frac, 'P', P_pa)
    # 絕對濕度轉換為 g/kg_da (與圖表 Y 軸單位一致)
    W_g = W * 1000

    # 濕球溫度 Wet-Bulb Temperature (°C)
    T_wb = CP.HAPropsSI('Twb', 'T', T_db + 273.15, 'R', RH_frac, 'P', P_pa) - 273.15

    # 露點溫度 Dew-Point Temperature (°C)
    T_dp = CP.HAPropsSI('Tdp', 'T', T_db + 273.15, 'R', RH_frac, 'P', P_pa) - 273.15

    # 比焓 Enthalpy h (kJ/kg_da)
    H = CP.HAPropsSI('H', 'T', T_db + 273.15, 'R', RH_frac, 'P', P_pa) / 1000.0

    # 比容 Volume v (m³/kg_da)
    V = CP.HAPropsSI('V', 'T', T_db + 273.15, 'R', RH_frac, 'P', P_pa)

    calc_success = True
except Exception as e:
    calc_success = False
    st.error(f"熱力性質計算出錯: {e}")

# --- 主畫面佈局：左側顯示計算結果，右側顯示空氣線圖 ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 狀態點數據評估")
    if calc_success:
        st.metric(label="乾球溫度 (Dry-Bulb Temp)", value=f"{T_db:.1f} °C")
        st.metric(label="相對濕度 (Relative Humidity)", value=f"{RH:.1f} %")
        st.metric(label="濕球溫度 (Wet-Bulb Temp)", value=f"{T_wb:.2f} °C")
        st.metric(label="露點溫度 (Dew-Point Temp)", value=f"{T_dp:.2f} °C")
        st.metric(label="絕對濕度 (Humidity Ratio)", value=f"{W_g:.2f} g/kg_da")
        st.metric(label="焓值 (Enthalpy)", value=f"{H:.2f} kJ/kg_da")
        st.metric(label="比容 (Specific Volume)", value=f"{V:.3f} m³/kg_da")

with col2:
    st.subheader("📈 空氣線圖 (Psychrometric Chart)")
    
    # 1. 載入 psychrochart 預設的 ASHRAE 風格圖表
    chart = PsychroChart.create_default_chart(style="ashrae")
    
    # 設定圖表標題與大小
    fig, ax = plt.subplots(figsize=(10, 7))
    chart.plot(ax=ax)

    # 2. 如果計算成功，把狀態點繪製在空氣線圖上
    if calc_success:
        # psychrochart 繪點的座標為 (乾球溫度, 絕對濕度g/kg)
        points = {
            'Point1': {
                'label': f'狀態點 1 ({T_db}°C, {RH}%)',
                'xy': (T_db, W_g),
                'style': {
                    'color': [0.85, 0.11, 0.38, 0.8],  # 紅色點
                    'marker': 'o',
                    'markersize': 10
                }
            }
        }
        # 繪製點位
        chart.plot_points_d(points, ax=ax)

    # 3. 輸出圖表至 Streamlit 畫面
    st.pyplot(fig)