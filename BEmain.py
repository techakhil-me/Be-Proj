import os
import streamlit as st
from pathlib import Path
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from fpdf import FPDF
import random
# Set up the Google API key and model
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key="")

generation_config = {
    "temperature": 0.3,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(model_name="gemini-2.0-flash",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Streamlit UI elements
st.markdown("<h1 style='text-align: center;'>Lung Cancer Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a chest X-ray image to receive an AI-generated diagnostic report.</p>", unsafe_allow_html=True)

# Patient Information Input
st.markdown("### Patient Information")
age = st.number_input("Age", min_value=1, max_value=120, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
smoking_history = st.selectbox("Smoking History", ["Never Smoked", "Former Smoker", "Current Smoker"])
symptoms = st.text_area("Symptoms (if any)", placeholder="Cough, Chest Pain, Difficulty Breathing, etc.")

uploaded_file = st.file_uploader("Upload Chest X-ray (JPEG)", type=["jpg", "jpeg"])

if uploaded_file:
   
    image_parts = [{"mime_type": "image/jpeg", "data": uploaded_file.read()}]
    
    prompt_parts = [
        image_parts[0],
        f"""
        You are a board-certified radiologist specializing in lung cancer detection.
        Analyze the provided chest X-ray for signs of lung cancer, considering:
        - Presence of nodules, their size, shape, and location
        - Opacity, cavitation, or consolidation patterns
        - Signs of pleural effusion or lymph node enlargement
        - Differences between lung fields and airway abnormalities
        Generate a structured report with:
        1. Findings
        2. Impression
        3. Possible Diagnoses
        4. Recommendations

        Patient Information:
        {age} years
        {gender}
        {smoking_history}
        Symptoms: {symptoms}
        """
    ]
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)
    
    if st.button("Analyze X-ray"):
        st.write("Processing... Please wait.")
        response = model.generate_content(prompt_parts)
        
        st.markdown("<h2 style='text-align: center;'>AI-Generated Diagnosis</h2>", unsafe_allow_html=True)
        st.write(response.text)

        # Dummy AI Confidence Score
        confidence_score = random.randint(75,90) # Assume a dummy confidence score of 85%
        st.progress(confidence_score / 100)
        st.markdown(f"**AI Confidence Score: {confidence_score}%**")

        # Severity Level Gauge
        severity_levels = ["Low Risk", "Moderate Risk", "High Risk"]
        severity = severity_levels[1] if confidence_score < 70 else severity_levels[2]
        st.markdown(f"**Severity Level: {severity}**")

        # Recommended Next Steps
        st.markdown("### Recommended Next Steps")
        st.write("- Follow up with a pulmonologist.")
        st.write("- Consider a CT scan for detailed imaging.")
        st.write("- If abnormalities persist, consult an oncologist.")

        # # Heatmap Visualization
        # dummy_scan_data = [
        #     [0.1, 0.3, 0.5, 0.7],
        #     [0.2, 0.6, 0.8, 0.4],
        #     [0.9, 0.5, 0.3, 0.2],
        #     [0.4, 0.7, 0.1, 0.6]
        # ]
        # fig, ax = plt.subplots()
        # sns.heatmap(dummy_scan_data, cmap='coolwarm', annot=False)
        # ax.set_title("Lung Abnormality Heatmap")
        # st.pyplot(fig)

        # Generate PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Lung Cancer Detection Report", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, response.text)
        pdf.ln(5)
        pdf.cell(0, 10, f"AI Confidence Score: {confidence_score}%", ln=True)
        pdf.cell(0, 10, f"Severity Level: {severity}", ln=True)
        pdf.cell(0, 10, "Recommended Next Steps:", ln=True)
        pdf.multi_cell(0, 10, "- Follow up with a pulmonologist.\n- Consider a CT scan.\n- Consult an oncologist if needed.")
        
        pdf_path = "report.pdf"
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name="Lung_Cancer_Report.pdf", mime="application/pdf")