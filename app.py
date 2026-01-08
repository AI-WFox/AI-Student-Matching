import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import ast

# --- Cấu hình ---
st.set_page_config(page_title="AI Matching", page_icon="🧩", layout="centered")

# --- Model embedding ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- Load dữ liệu ---
@st.cache_data
def load_data():
    csv_path = "data/user_data.csv"
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Chuyển chuỗi -> list lại cho các cột chứa danh sách
            for col in ["Môn học", "Thời gian rảnh"]:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            print(f"Đã tải dữ liệu từ {csv_path}", file=os.sys.stderr)
            return df
        except Exception as e:
            print(f"Không thể đọc file CSV: {e}. Sử dụng dữ liệu mẫu.", file=os.sys.stderr)
    
    # Dữ liệu mẫu (fallback)
    data = {
        "Tên": ["Ngọc", "Lan", "Nam", "Vy", "Bảo"],
        "Môn học": [["Cơ sở lập trình"], ["Toán rời rạc"], ["Kỹ năng mềm", "Toán rời rạc"], ["Nhập môn CNTT"], ["Kỹ năng mềm"]],
        "Thời gian rảnh": [["Sáng", "Chiều"], ["Chiều"], ["Tối"], ["Sáng"], ["Sáng", "Chiều"]],
        "Giới tính": ["Nữ", "Nữ", "Nam", "Nữ", "Nam"],
        "Sở thích": [
            "Thích code web, đọc sách công nghệ",
            "Yêu thích Toán học và logic",
            "Thích làm việc nhóm, nói chuyện nhiều",
            "Yêu nghệ thuật, thích thiết kế",
            "Thích nghiên cứu AI và công nghệ mới"
        ],
        "Tính cách": [
            "Lập di, kiên nhẫn",
            "Năng động, hướng ngoại",
            "Vui vẻ, thân thiện",
            "Trầm tính, sáng tạo",
            "Phân tích logic, ít nói"
        ]
    }
    return pd.DataFrame(data)

df = load_data()

# --- Tính vector mô tả cá nhân ---
@st.cache_data
def compute_vectors(df):
    df["vector"] = df.apply(lambda row: model.encode(
        row["Sở thích"] + " " + row["Tính cách"]
    ), axis=1)
    return df

df = compute_vectors(df)

# --- UI người dùng ---
st.title("🧩 Gợi ý bạn học phù hợp bằng AI")
st.markdown("### Nhập thông tin của bạn để tìm người học hợp nhất 💡")

user_subjects = st.multiselect(
    "📘 Môn học bạn quan tâm:",
    ["Cơ sở lập trình", "Toán rời rạc", "Kỹ năng mềm", "Nhập môn CNTT"]
)

user_time = st.multiselect(
    "🕒 Thời gian rảnh của bạn:",
    ["Sáng", "Chiều", "Tối"]
)

col1, col2 = st.columns(2)
with col1:
    user_gender = st.selectbox("🚻 Giới tính của bạn:", ["Nam", "Nữ", "Khác"])
with col2:
    target_gender = st.multiselect("🎯 Bạn muốn tìm bạn học giới tính:", ["Nam", "Nữ", "Khác"])

user_hobby = st.text_area("🎨 Sở thích của bạn là gì?")
user_personality = st.text_area("💬 Mô tả tính cách của bạn:")

if st.button("🔍 Tìm bạn học phù hợp", use_container_width=True):
    if not user_subjects or not user_time:
        st.warning("⚠️ Hãy nhập đầy đủ môn học và thời gian rảnh trước khi tìm nhé!")
    else:
        user_vector = model.encode(user_hobby + " " + user_personality)

        filtered_df = df[df["Giới tính"].isin(target_gender)] if target_gender else df

        # --- Áp dụng quy tắc cứng: chỉ giữ người có ít nhất 1 môn và 1 thời gian trùng ---
        def valid_match(row):
            subject_overlap = any(sub in row["Môn học"] for sub in user_subjects)
            time_overlap = any(t in row["Thời gian rảnh"] for t in user_time)
            return subject_overlap and time_overlap

        valid_candidates = filtered_df[filtered_df.apply(valid_match, axis=1)].copy()

        if len(valid_candidates) == 0:
            st.error("😥 Không tìm thấy bạn học nào phù hợp với môn học và thời gian rảnh của bạn.")
        else:
            similarities = cosine_similarity([user_vector], list(valid_candidates["vector"]))
            valid_candidates["Độ hợp (%)"] = (similarities[0] * 100).round(2)

            top_matches = valid_candidates.sort_values(by="Độ hợp (%)", ascending=False).head(3)

            st.markdown("## 🔎 Kết quả gợi ý:")

            for _, row in top_matches.iterrows():
                st.markdown(f"""
                **👤 Tên:** {row['Tên']}  
                **📘 Môn học:** {', '.join(row['Môn học'])}  
                **🕒 Thời gian rảnh:** {', '.join(row['Thời gian rảnh'])}  
                **🚻 Giới tính:** {row['Giới tính']}  
                **🎨 Sở thích:** {row['Sở thích']}  
                **💬 Tính cách:** {row['Tính cách']}  
                **💡 Độ hợp:** `{row['Độ hợp (%)']}%`
                """)
                st.divider()
