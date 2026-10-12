import os
import pandas as pd
import numpy as np
import gradio as gr
from fastapi import FastAPI # <--- THÊM DÒNG NÀY VÀO ĐÂY
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Đọc file dữ liệu và huấn luyện mô hình ML
file_path = 'Data_BeTongXanh_2026_Ver5.xlsx'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Không tìm thấy file {file_path}. Vui lòng kiểm tra lại!")

df = pd.read_excel(file_path)

# Tập dữ liệu đầu vào (Features) với 5 cỡ sàng đá
X = df[['LoaiXiMang', 'LoaiDa', 'Ximang_kg', 'Trobay_kg', 'Xilocao_kg', 'Nuoc_lit', 'Cat_kg', 'Da_195_125_kg', 'Da_125_95_kg', 'Da_95_475_kg', 'Da_475_236_kg', 'Da_duoi_236_kg', 'Phugia_lit']]
y_7n = df['CuongDoNen_7N_MPa']
y_28n = df['CuongDoNen_28N_MPa']
y_dosut = df['DoSut_cm']

# Tiền xử lý dữ liệu
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['LoaiXiMang', 'LoaiDa'])],
    remainder='passthrough'
)

# Khởi tạo và huấn luyện các mô hình Random Forest
model_7n = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
model_28n = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
model_dosut = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

model_7n.fit(X, y_7n)
model_28n.fit(X, y_28n)
model_dosut.fit(X, y_dosut)

# 2. Hàm dự đoán & Phân tích chuyên môn
def phan_tich_be_tong(loai_xm, loai_da, ximang, trobay, xilocao, nuoc, cat, da_195, da_125, da_95, da_475, da_duoi_236, phugia):
    # Đóng gói dữ liệu đầu vào
    input_data = pd.DataFrame([{
        'LoaiXiMang': loai_xm,
        'LoaiDa': loai_da,
        'Ximang_kg': ximang,
        'Trobay_kg': trobay,
        'Xilocao_kg': xilocao,
        'Nuoc_lit': nuoc,
        'Cat_kg': cat,
        'Da_195_125_kg': da_195,
        'Da_125_95_kg': da_125,
        'Da_95_475_kg': da_95,
        'Da_475_236_kg': da_475,
        'Da_duoi_236_kg': da_duoi_236,
        'Phugia_lit': phugia
    }])
    
    # Dự đoán thông số cơ lý
    pred_7n = model_7n.predict(input_data)[0]
    pred_28n = model_28n.predict(input_data)[0]
    pred_dosut = model_dosut.predict(input_data)[0]
    
    # Tổng hợp cốt liệu lớn
    tong_da_thuc_te = da_195 + da_125 + da_95 + da_475 + da_duoi_236
    
    # Tính toán GWP và Hệ số K (Cuộc thi 2026)
    gwp_co2 = (ximang * 0.9 + xilocao * 0.07 + trobay * 0.02 + nuoc * 0.001 + cat * 0.01 + tong_da_thuc_te * 0.01 + phugia * 0.1)
    he_so_k = gwp_co2 + (1000.0 / pred_7n)
    
    # So sánh mức độ giảm CO2
    tong_ckd = ximang + xilocao + trobay
    co2_tham_chieu = (tong_ckd * 0.9 + nuoc * 0.001 + cat * 0.01 + tong_da_thuc_te * 0.01 + phugia * 0.1)
    co2_giam_pct = ((co2_tham_chieu - gwp_co2) / co2_tham_chieu) * 100
    
    # Phân loại nấc phát thải CO2
    if co2_giam_pct < 10:
        nac_xanh = "Chưa đạt chuẩn xanh (< 10%) - Đề xuất tối ưu lại cấp phối."
    elif 10 <= co2_giam_pct < 30:
        nac_xanh = "Nấc 1: Giảm nhẹ (Cơ bản) 10% - 30%\n• Tối ưu hóa cấp phối (phối hợp phụ gia giảm nước, giảm lượng xi măng).\n• Thay thế 15-30% xi măng bằng tro bay (FA) hoặc xỉ lò cao (GGBS)."
    elif 30 <= co2_giam_pct < 50:
        nac_xanh = "Nấc 2: Giảm trung bình (Tiêu chuẩn) 30% - 50%\n• Thay thế 30-50% xi măng bằng GGBS, tro bay hoặc đất sét nung (LC3).\n• Tối ưu kích thước hạt cốt liệu để giảm thể tích hồ xi măng."
    elif 50 <= co2_giam_pct < 70:
        nac_xanh = "Nấc 3: Giảm cao (Sâu) 50% - 70%\n• Tỷ lệ thay thế xi măng rất cao (50-70% GGBS hoặc hệ phụ gia phối hợp).\n• Sử dụng công nghệ khoáng hóa/bơm CO₂ vào bê tông trong quá trình trộn."
    elif 70 <= co2_giam_pct < 100:
        nac_xanh = "Nấc 4: Giảm rất cao (Đột phá) 70% - 90%\n• Bê tông Polymer vô cơ / Geopolymer (không dùng PC, hoạt hóa bằng kiềm).\n• Dùng chất kết dính phi Portland tiên tiến."
    else:
        nac_xanh = "Nấc 5: Bê tông Trung hòa / Carbon âm 100% (hoặc < 0%)\n• Kết hợp Geopolymer + cốt liệu thu hồi carbon + ứng dụng CCUS trong toàn chuỗi cung ứng."

    # Lời khuyên kỹ thuật chuyên nghiệp
    loi_khuyen = ""
    if tong_ckd > 0 and (xilocao / tong_ckd) > 0.15:
        loi_khuyen += "\n⚠️ LƯU Ý KỸ THUẬT: Hàm lượng Xỉ lò cao (GGBFS) đang vượt mức 15% tổng chất kết dính. Việc sử dụng vượt quá mức này không đóng góp thêm vào cường độ mà gây lãng phí vật tư."
    
    if pred_7n < 30 or pred_28n < 30:
        loi_khuyen += "\n💡 ĐỀ XUẤT TỐI ƯU (Cường độ < 30MPa): Bê tông chưa đạt mác thiết kế mục tiêu, cần xem xét:"
        loi_khuyen += "\n   - Nâng mác xi măng (Sử dụng PC40 thay cho PC30)."
        loi_khuyen += "\n   - Sử dụng cốt liệu Đá vôi để tăng cường lực dính bám cơ học."
        loi_khuyen += "\n   - Tăng tỷ lệ Xi măng portland trong thành phần chất kết dính."
        loi_khuyen += "\n   - Giảm lượng khoáng thay thế (Tro bay/Xỉ lò cao) xuống mức phù hợp."
    else:
        loi_khuyen += "\n✅ KẾT LUẬN: Cấp phối đạt yêu cầu kỹ thuật. Cường độ nén và cấu trúc khung xương cốt liệu ổn định."

    output_text = f"""
📋 KẾT QUẢ ĐÁNH GIÁ MẪU BÊ TÔNG XANH (SVVLXD 2026):
===========================================================
1. Độ Sụt Hỗn Hợp Bê Tông: {pred_dosut:.1f} cm
2. Cường Độ Nén 7 Ngày: {pred_7n:.1f} MPa
3. Cường Độ Nén 28 Ngày: {pred_28n:.1f} MPa
• Tổng Khối lượng Cốt liệu lớn (Đá): {tong_da_thuc_te:.1f} kg/m³
{loi_khuyen}

🌍 ĐÁNH GIÁ PHÁT THẢI CARBON:
===========================================================
• Tổng lượng phát thải (GWP): {gwp_co2:.1f} kg CO₂/m³
• Tỷ lệ cắt giảm CO₂ so với OPC: {co2_giam_pct:.1f}%

📊 PHÂN LOẠI MỨC ĐỘ XANH: 
{nac_xanh}

⚙️ HỆ SỐ ĐÁNH GIÁ TỔNG HỢP (K): {he_so_k:.2f} 
(Lưu ý tiêu chí cuộc thi: K càng thấp càng tốt. Hệ số K phản ánh năng lực tối ưu đồng thời giữa việc giảm phát thải GWP và gia tăng cường độ sớm M7).
"""
    return output_text

# 3. CSS tùy chỉnh: Tone màu sương mù, sườn núi (Trắng/Xám/Xanh nhạt) thanh lịch, chuyên nghiệp
elegant_css = """
body, html {
    background: linear-gradient(135deg, #e6eff5 0%, #ffffff 50%, #dbe4ec 100%) !important;
    color: #1e293b !important;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
}

.gradio-container {
    background: rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(15px) !important;
    -webkit-backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 40px rgba(71, 85, 105, 0.1) !important;
    padding: 30px !important;
    margin-top: 20px !important;
    margin-bottom: 20px !important;
}

h1 {
    color: #0f4c75 !important; /* Xanh than thanh lịch */
    font-weight: 700 !important;
    border-bottom: 2px solid #bbe1fa !important;
    padding-bottom: 10px;
}

h2, h3, label, span {
    color: #334155 !important; /* Xám đậm chuyên nghiệp */
    font-weight: 600 !important;
}

/* Tùy chỉnh Nút bấm */
button.primary {
    background: linear-gradient(135deg, #3282b8 0%, #0f4c75 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(15, 76, 117, 0.25) !important;
    transition: all 0.3s ease !important;
    border-radius: 6px !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #0f4c75 0%, #1b262c 100%) !important;
    box-shadow: 0 6px 20px rgba(15, 76, 117, 0.4) !important;
    transform: translateY(-1px);
}

/* Khung TextBox kết quả */
textarea {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    font-family: 'Consolas', monospace !important;
}
"""

# Định nghĩa giao diện Gradio
with gr.Blocks(css=elegant_css) as demo:
    gr.Markdown("""
    # 🏢 HỆ THỐNG DỰ ĐOÁN & ĐÁNH GIÁ BÊ TÔNG XANH 
    *Công cụ ứng dụng Trí tuệ Nhân tạo (Machine Learning) hỗ trợ thiết kế, phân tích cường độ cơ lý và tối ưu hóa hệ số phát thải CO₂ (GWP) cho vật liệu Bê tông cốt liệu.*
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🛠️ 1. Thành Phần Chất Kết Dính")
            phap_mon = gr.Dropdown(['PC30', 'PC40'], label="Phân loại Xi Măng", value="PC30")
            ximang = gr.Number(value=150.0, label="Khối lượng Xi Măng Portland (kg/m³)")
            trobay = gr.Number(value=180.0, label="Khối lượng Tro Bay - FA [30%-70%] (kg/m³)")
            xilocao = gr.Number(value=20.0, label="Khối lượng Xỉ Lò Cao - GGBFS [Max 15%] (kg/m³)")
            
            gr.Markdown("### 💧 2. Nước & Phụ Gia")
            nuoc = gr.Number(value=175.0, label="Khối lượng Nước (lít/m³)")
            phugia = gr.Number(value=5.0, label="Hàm lượng Phụ Gia Hóa Học (lít/m³)")
            cat = gr.Number(value=700.0, label="Khối lượng Cốt Liệu Nhỏ - Cát (kg/m³)")
            
        with gr.Column():
            gr.Markdown("### 🪨 3. Cấp Phối Khung Xương (Cốt Liệu Lớn)")
            linh_thach_da = gr.Dropdown(['Đá vôi', 'Đá cuội', 'Đá hỗn hợp (Vôi+Cuội)'], label="Phân loại Đá", value="Đá vôi")
            
            gr.Markdown("*(Phân bổ các cỡ hạt cho chuẩn Dmax 19.5mm)*")
            da_195 = gr.Number(value=470.0, label="Cỡ hạt từ 19.5 đến 12.5 mm (kg/m³)")
            da_125 = gr.Number(value=315.0, label="Cỡ hạt từ 12.5 đến 9.5 mm (kg/m³)")
            da_95 = gr.Number(value=160.0, label="Cỡ hạt từ 9.5 đến 4.75 mm (kg/m³)")
            da_475 = gr.Number(value=85.0, label="Cỡ hạt từ 4.75 đến 2.36 mm (kg/m³)")
            da_duoi_236 = gr.Number(value=20.0, label="Lọt sàng dưới 2.36 mm (kg/m³)")
            
    nut_chay = gr.Button("📊 THỰC HIỆN PHÂN TÍCH & DỰ ĐOÁN", variant="primary")

    gr.Markdown("### 📑 4. Kết Quả Báo Cáo Chuyên Sâu")
    ket_qua = gr.Textbox(label="Bảng Đánh Giá Thông Số Kỹ Thuật", lines=20)

    nut_chay.click(
        fn=phan_tich_be_tong,
        inputs=[phap_mon, linh_thach_da, ximang, trobay, xilocao, nuoc, cat, da_195, da_125, da_95, da_475, da_duoi_236, phugia],
        outputs=ket_qua
    )

# --- QUAN TRỌNG: Khởi tạo biến `app` cho Vercel ---
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")
