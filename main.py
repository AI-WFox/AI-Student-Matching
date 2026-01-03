from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import os
import ast

# --- Load model embedding ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- File lưu dữ liệu ---
FILE_PATH = "data.csv"

# --- Đọc dữ liệu từ CSV (nếu có) ---
if os.path.exists(FILE_PATH):
    df = pd.read_csv(FILE_PATH)
    # Chuyển chuỗi -> list lại cho các cột chứa danh sách
    for col in ["Môn học", "Thời gian rảnh"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
else:
    # Nếu chưa có file thì tạo dữ liệu mẫu
    data = {
        "Tên": ["Ngọc", "Lan", "Nam", "Vy", "Bảo"],
        "Môn học": [["Cơ sở lập trình"], ["Toán rời rạc"], ["Kỹ năng mềm", "Toán rời rạc"],
                    ["Nhập môn CNTT"], ["Kỹ năng mềm"]],
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
            "Điềm tĩnh, kiên nhẫn",
            "Năng động, hướng ngoại",
            "Vui vẻ, thân thiện",
            "Trầm tính, sáng tạo",
            "Phân tích logic, ít nói"
        ]
    }
    df = pd.DataFrame(data)

# --- Tính vector embedding ---
def compute_vectors(df):
    df["vector"] = df.apply(
        lambda row: model.encode(row["Sở thích"] + " " + row["Tính cách"]),
        axis=1
    )
    return df

df = compute_vectors(df)

# --- Kiểm tra điều kiện cứng ---
def valid_match(row, user_subjects, user_times, target_genders):
    subject_overlap = any(sub in row["Môn học"] for sub in user_subjects)
    time_overlap = any(t in row["Thời gian rảnh"] for t in user_times)
    gender_ok = (row["Giới tính"] in target_genders) if target_genders else True
    return subject_overlap and time_overlap and gender_ok

# --- Hàm tìm bạn học ---
def find_best_matches_optimized(user_subjects, user_times, user_gender, target_genders,
                                user_hobby, user_personality, top_n=3):
    user_vector = model.encode(user_hobby + " " + user_personality).reshape(1, -1)
    valid_candidates = df[df.apply(
        lambda row: valid_match(row, user_subjects, user_times, target_genders),
        axis=1
    )].copy()

    if len(valid_candidates) == 0:
        return []

    X = np.array(list(valid_candidates["vector"]))
    X_norm = normalize(X, norm='l2')
    user_vec_norm = normalize(user_vector, norm='l2')

    knn = NearestNeighbors(
        n_neighbors=min(top_n, len(valid_candidates)),
        metric='euclidean',
        algorithm='ball_tree'
    )
    knn.fit(X_norm)
    distances, indices = knn.kneighbors(user_vec_norm)

    cosine_sim = (1 - (distances ** 2) / 2).clip(0, 1)
    top_matches = valid_candidates.iloc[indices[0]].copy()
    top_matches["Độ hợp (%)"] = (cosine_sim[0] * 100).round(2)

    return top_matches[["Tên", "Môn học", "Thời gian rảnh", "Giới tính",
                        "Sở thích", "Tính cách", "Độ hợp (%)"]]

# --- Hàm lưu dữ liệu mới ---
def save_user_data(df, user_data):
    df = pd.concat([df, pd.DataFrame([user_data])], ignore_index=True)
    df.to_csv(FILE_PATH, index=False)
    return df

# --- Chương trình chính ---
if __name__ == "__main__":
    print("=== 💬 HỆ THỐNG GHÉP BẠN HỌC AI ===")

    user_name = input("Nhập tên của bạn: ").strip()
    user_gender = input("Giới tính (Nam/Nữ): ").strip()
    user_subjects = [s.strip() for s in input("Môn học bạn đang học (ngăn cách bằng dấu phẩy): ").split(",")]
    user_times = [t.strip() for t in input("Thời gian rảnh (ví dụ: Sáng, Chiều, Tối): ").split(",")]
    target_gender = input("Muốn tìm bạn học giới tính nào (để trống nếu không giới hạn): ").strip()
    user_hobby = input("Sở thích của bạn: ").strip()
    user_personality = input("Tính cách của bạn: ").strip()

    new_user = {
        "Tên": user_name,
        "Môn học": user_subjects,
        "Thời gian rảnh": user_times,
        "Giới tính": user_gender,
        "Sở thích": user_hobby,
        "Tính cách": user_personality
    }

    # --- Lưu dữ liệu mới ---
    df = save_user_data(df, new_user)
    print("\n✅ Dữ liệu của bạn đã được lưu thành công!")

    # --- Tính vector cho người mới thêm ---
    df = compute_vectors(df)

    # --- Gợi ý bạn học phù hợp ---
    print("\n🔎 Đang tìm bạn học phù hợp nhất cho bạn...\n")

    # Bỏ qua chính người vừa nhập
    df_without_user = df[df["Tên"] != user_name].copy()

    matches = find_best_matches_optimized(
        user_subjects, user_times, user_gender,
        [target_gender] if target_gender else [],
        user_hobby, user_personality
    )

    # Nếu người mới thêm nằm trong df, bỏ qua
    if user_name in matches["Tên"].values:
        matches = matches[matches["Tên"] != user_name]

    if len(matches) == 0:
        print("❌ Không tìm thấy bạn học phù hợp.")
    else:
        print("✅ Gợi ý bạn học phù hợp nhất:\n")
        for i, row in matches.iterrows():
            print(f"- {row['Tên']} ({row['Giới tính']}) — Độ hợp: {row['Độ hợp (%)']}%")
            print(f"  Môn học: {', '.join(row['Môn học'])}")
            print(f"  Thời gian rảnh: {', '.join(row['Thời gian rảnh'])}")
            print(f"  Sở thích: {row['Sở thích']}")
            print(f"  Tính cách: {row['Tính cách']}\n")
