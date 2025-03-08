import streamlit as st
import os
import tempfile
from motion_capture import process_video
import mediapipe as mp

st.title("🎥 モーションキャプチャアプリ")

uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov"])

# **カラーピッカー（デフォルト: 白）**
marker_color = st.color_picker("関節マーカーの色", "#FFFFFF")
line_color = st.color_picker("骨格ラインの色", "#C8C8C8")

# **サイズ設定（小・中・大）**
size_option = st.radio("マーカーと線の太さを選択", ["小", "中", "大"])

# **サイズ設定を数値に変換**
size_map = {
    "小": {"thickness": 1, "circle_radius": 2},
    "中": {"thickness": 2, "circle_radius": 4},
    "大": {"thickness": 3, "circle_radius": 6},
}
size_config = size_map[size_option]

# **表示する関節の選択（最初は格納）**
with st.expander("🔧 表示する関節の選択（クリックで展開）"):
    selected_joints = {
        "首": st.checkbox("首（頭部）", value=True),
        "上肢": st.checkbox("上肢（肩・肘・手）", value=True),
        "下肢": st.checkbox("下肢（股関節・膝・足）", value=True),
        "体幹": st.checkbox("体幹（胸・腰）", value=True),
    }

if uploaded_file is not None:
    # **一時ファイルに動画を保存**
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input.close()
    
    # **処理を実行（選択された関節のみ解析）**
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    process_video(temp_input.name, temp_output.name, marker_color, line_color, size_config, selected_joints, 
                  min_detection_confidence=0.85, min_tracking_confidence=0.85, visibility_threshold=0.15)
    st.video(temp_output.name)  # ✅ 解析後の動画のみ表示
    
    # **ダウンロードボタン**
    with open(temp_output.name, "rb") as file:
        st.download_button(label="📥 ダウンロード", data=file, file_name="processed_video.mp4", mime="video/mp4")
    
    # **処理後のファイルを削除**
    os.remove(temp_output.name)
    os.remove(temp_input.name)

