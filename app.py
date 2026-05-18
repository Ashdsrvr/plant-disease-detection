import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time
from datetime import datetime

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)

# --------------------------------
# Mixed Precision (same as training)
# --------------------------------
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# --------------------------------
# Streamlit Config
# --------------------------------
st.set_page_config(
    page_title="Agriculture AI - Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------
# Professional Dark Theme CSS Styling
# --------------------------------
st.markdown("""
    <style>
        /* Root color scheme - Dark Theme with Nature inspired accents */
        :root {
            --primary-green: #4caf50;
            --accent-green: #66bb6a;
            --light-green: #1b5e20;
            --sky-blue: #29b6f6;
            --light-blue: #0288d1;
            --dark-bg: #121212;
            --card-bg: #1e1e1e;
            --border-color: #2d2d2d;
            --text-light: #e0e0e0;
            --text-primary: #ffffff;
            --danger-red: #ef5350;
            --warning-orange: #ffb74d;
        }
        
        /* Main background */
        .main {
            background: linear-gradient(135deg, #121212 0%, #1a1a1a 100%);
        }
        
        .stApp {
            background: linear-gradient(135deg, #121212 0%, #1a1a1a 100%);
        }
        
        /* Header Section */
        .header-section {
            background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
            color: white;
            padding: 40px 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(76, 175, 80, 0.3);
            text-align: center;
        }
        
        .header-section h1 {
            font-size: 2.8em;
            margin: 0 0 10px 0;
            font-weight: 800;
            letter-spacing: 0.5px;
        }
        
        .header-section p {
            font-size: 1.1em;
            margin: 0;
            opacity: 0.95;
            font-weight: 300;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #e0e0e0;
            font-size: 1.05em;
            padding: 20px;
            background: #1e1e1e;
            border-left: 4px solid #4caf50;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.1);
        }
        
        /* Upload Card */
        .upload-card {
            background: #1e1e1e;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
            border-top: 4px solid #29b6f6;
        }
        
        .upload-card h3 {
            color: #4caf50;
            font-size: 1.3em;
            margin-top: 0;
            margin-bottom: 10px;
        }
        
        .upload-card p {
            color: #b0b0b0;
            font-size: 0.95em;
            margin: 8px 0;
        }
        
        /* Image Display Container */
        .image-display {
            background: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            border: 2px solid #1b5e20;
        }
        
        /* Result Cards */
        .result-card {
            background: #1e1e1e;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            margin-bottom: 15px;
            border-left: 5px solid #29b6f6;
        }
        
        .result-card.success {
            border-left-color: #4caf50;
        }
        
        .result-card.warning {
            border-left-color: #ffb74d;
        }
        
        .result-title {
            color: #66bb6a;
            font-weight: 700;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 0 0 12px 0;
        }
        
        .result-value {
            font-size: 1.8em;
            font-weight: 800;
            color: #4caf50;
            margin: 10px 0;
        }
        
        .confidence-percentage {
            font-size: 2.5em;
            font-weight: 900;
            background: linear-gradient(135deg, #29b6f6 0%, #0288d1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 10px 0;
        }
        
        .confidence-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.95em;
            margin-top: 10px;
        }
        
        .confidence-high {
            background: #1b5e20;
            color: #4caf50;
            border: 2px solid #4caf50;
        }
        
        .confidence-medium {
            background: #3e2723;
            color: #ffb74d;
            border: 2px solid #ffb74d;
        }
        
        .confidence-low {
            background: #4c2c2c;
            color: #ef5350;
            border: 2px solid #ef5350;
        }

        /* Top 3 Predictions Cards */
        .top3-card {
            background: #1e1e1e;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            margin-bottom: 12px;
            border-left: 5px solid #29b6f6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .top3-card.rank-1 {
            border-left-color: #4caf50;
            background: linear-gradient(90deg, #1e1e1e 0%, #173319 100%);
        }
        .top3-card.rank-2 {
            border-left-color: #ffb74d;
        }
        .top3-card.rank-3 {
            border-left-color: #ef5350;
        }
        .top3-name {
            color: #e0e0e0;
            font-weight: 600;
            font-size: 1.1em;
        }
        .top3-conf {
            color: #ffffff;
            font-weight: 700;
            font-size: 1.1em;
            background: rgba(255,255,255,0.1);
            padding: 4px 12px;
            border-radius: 20px;
        }
        
        /* Disease Info Panel */
        .disease-info-panel {
            background: #1e1e1e;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            border-top: 4px solid #4caf50;
        }
        .info-section {
            margin-bottom: 20px;
        }
        .info-section:last-child {
            margin-bottom: 0;
        }
        .info-section h4 {
            color: #4caf50;
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 1.1em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .info-section p {
            color: #b0b0b0;
            margin: 0;
            line-height: 1.6;
            font-size: 0.95em;
        }
        
        /* Info Box */
        .info-box {
            background: linear-gradient(135deg, #1b5e20 0%, #0d3819 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #29b6f6;
            margin-top: 30px;
            color: #c8e6c9;
        }
        
        .info-box p {
            margin: 8px 0;
            font-size: 0.95em;
        }
        
        /* Placeholder */
        .placeholder {
            text-align: center;
            padding: 80px 20px;
            background: #1e1e1e;
            border-radius: 12px;
            border: 3px dashed #404040;
            margin-top: 40px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .placeholder p {
            font-size: 1.1em;
            color: #b0b0b0;
            margin: 0;
        }
        
        /* Divider */
        hr {
            margin: 30px 0;
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, #404040 50%, transparent 100%);
        }
        
        /* Spinner text */
        .stSpinner {
            color: #29b6f6;
        }
        /* Recommendation Cards */
        .rec-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .rec-card {
            background: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            border-top: 4px solid;
            transition: transform 0.2s ease-in-out;
        }
        
        .rec-card:hover {
            transform: translateY(-5px);
        }
        
        .rec-card h4 {
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 1.1em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .rec-card p {
            color: #b0b0b0;
            margin: 0;
            font-size: 0.95em;
            line-height: 1.5;
        }
        
        .rec-pesticide { border-top-color: #f44336; }
        .rec-pesticide h4 { color: #f44336; }
        
        .rec-fertilizer { border-top-color: #4caf50; }
        .rec-fertilizer h4 { color: #4caf50; }
        
        .rec-irrigation { border-top-color: #29b6f6; }
        .rec-irrigation h4 { color: #29b6f6; }
        
        .rec-weather { border-top-color: #ff9800; }
        .rec-weather h4 { color: #ff9800; }
        
        .rec-organic { border-top-color: #8bc34a; }
        .rec-organic h4 { color: #8bc34a; }
        
        .rec-best-practices { border-top-color: #9c27b0; }
        .rec-best-practices h4 { color: #9c27b0; }
        
        .rec-emergency { 
            border-top-color: #d32f2f;
            background: linear-gradient(135deg, #1e1e1e 0%, #311313 100%);
        }
        .rec-emergency h4 { color: #ff5252; }
    </style>
""", unsafe_allow_html=True)

NUM_CLASSES = 38

# --------------------------------
# Build EXACT Training Architecture
# --------------------------------
@st.cache_resource
def load_model():

    base_model = DenseNet121(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = Sequential([
        base_model,

        GlobalAveragePooling2D(),

        BatchNormalization(),

        Dropout(0.5),

        Dense(
            512,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),

        BatchNormalization(),

        Dropout(0.5),

        Dense(
            NUM_CLASSES,
            activation='softmax',
            dtype='float32'
        )
    ])

    # IMPORTANT:
    # load the FULL model weights
    model.load_weights("densenet_best.h5")

    return model


model = load_model()

# --------------------------------
# Class Names
# --------------------------------
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# --------------------------------
# Disease Information Dictionary & Helper
# --------------------------------
def get_disease_info(class_name):
    """Returns a structured dictionary with disease information based on class name."""
    if 'healthy' in class_name.lower():
        return {
            "cause": "None. The plant is currently healthy.",
            "symptoms": "Leaves appear green, firm, and free of spots, discoloration, or abnormal growth.",
            "treatment": "No treatment required. Maintain current care routine.",
            "prevention": "Continue regular watering, proper nutrition, and routine monitoring to keep the plant healthy."
        }
    
    info = {
        "cause": "Fungal, bacterial, or viral infection depending on the specific pathogen.",
        "symptoms": "Discoloration, spots, wilting, or abnormal growth on leaves and stems.",
        "treatment": "Remove affected plant parts. Apply appropriate fungicide or bactericide.",
        "prevention": "Ensure good air circulation, avoid overhead watering, and practice crop rotation."
    }
    
    if 'blight' in class_name.lower():
        info["cause"] = "Often caused by fungal-like organisms (Oomycetes) like Phytophthora or true fungi like Alternaria."
        info["symptoms"] = "Rapid yellowing, browning, and dying of leaves. Dark, concentric, water-soaked spots."
        info["treatment"] = "Apply appropriate fungicides (e.g., copper-based). Remove and destroy infected plants immediately."
        info["prevention"] = "Ensure proper spacing for air circulation. Avoid watering leaves directly. Use resistant varieties."
    elif 'mildew' in class_name.lower():
        info["cause"] = "Various fungal species (e.g., Podosphaera, Erysiphe) that thrive in high humidity and moderate temperatures."
        info["symptoms"] = "White or gray powdery spots on leaves, stems, and sometimes flowers. Leaves may curl or twist."
        info["treatment"] = "Use sulfur, neem oil, or potassium bicarbonate based fungicides. Prune affected areas."
        info["prevention"] = "Improve air circulation, prune crowded areas, and keep leaves dry by watering at the base."
    elif 'rust' in class_name.lower():
        info["cause"] = "Fungal pathogens from the order Pucciniales."
        info["symptoms"] = "Yellow, orange, red, or brown powdery pustules mostly on the undersides of leaves."
        info["treatment"] = "Apply copper-based fungicides or sulfur powders. Remove infected leaves."
        info["prevention"] = "Remove alternate hosts if applicable. Water at the base of the plant to keep foliage dry."
    elif 'spot' in class_name.lower() or 'scab' in class_name.lower():
        info["cause"] = "Often caused by bacterial pathogens (e.g., Xanthomonas) or fungal pathogens (e.g., Venturia inaequalis)."
        info["symptoms"] = "Small, dark, water-soaked spots on leaves that may enlarge and turn brown, black, or have a yellow halo."
        info["treatment"] = "Use copper-based bactericides or fungicides. Remove severely affected leaves and debris."
        info["prevention"] = "Avoid working with wet plants. Practice crop rotation and use disease-free seeds. Clear autumn debris."
    elif 'virus' in class_name.lower() or 'mosaic' in class_name.lower():
        info["cause"] = "Viral pathogens (e.g., TMV, TYLCV), often transmitted by insects like aphids, thrips, or whiteflies."
        info["symptoms"] = "Mottling, yellowing (chlorosis), curling, wrinkling, and stunted growth of leaves."
        info["treatment"] = "There is no cure for viral infections. Infected plants must be immediately removed and destroyed."
        info["prevention"] = "Control insect vectors, keep tools sanitized, weed the area, and use resistant plant varieties."
    elif 'rot' in class_name.lower():
        info["cause"] = "Fungi or bacteria that thrive in excessively wet, warm conditions."
        info["symptoms"] = "Dark, mushy, and decaying areas on leaves, fruits, stems, or roots."
        info["treatment"] = "Prune infected parts well below the rotting area. Apply appropriate fungicidal sprays."
        info["prevention"] = "Ensure well-draining soil, avoid overwatering, and maintain proper plant spacing."
    elif 'mite' in class_name.lower():
        info["cause"] = "Tetranychus urticae (Two-spotted spider mite) or other mite species."
        info["symptoms"] = "Tiny yellow or white speckles on leaves. Fine webbing may be visible under leaves or on stems."
        info["treatment"] = "Use insecticidal soap, neem oil, or horticultural oils. Introduce natural predators like ladybugs."
        info["prevention"] = "Keep plants well-watered, as mites prefer dry, dusty conditions. Periodically wash down foliage."
    elif 'greening' in class_name.lower() or 'haunglongbing' in class_name.lower():
        info["cause"] = "Candidatus Liberibacter asiaticus bacteria, spread by the Asian citrus psyllid."
        info["symptoms"] = "Yellow shoots, mottled leaves, stunted growth, and misshapen, bitter fruit."
        info["treatment"] = "No cure exists. Infected trees must be removed and destroyed to prevent spread."
        info["prevention"] = "Control psyllid populations with insecticides. Use certified disease-free trees."
    elif 'mold' in class_name.lower():
        info["cause"] = "Fungal pathogens like Passalora fulva (Tomato Leaf Mold) that favor high relative humidity."
        info["symptoms"] = "Pale greenish-yellow spots on upper leaf surfaces. Olive-green to brown velvety mold on undersides."
        info["treatment"] = "Apply appropriate fungicides. Improve ventilation and reduce humidity."
        info["prevention"] = "Space plants adequately. Avoid overhead watering. Use resistant seed varieties."

    return info

# --------------------------------
# Farmer Recommendation System
# --------------------------------
def get_farmer_recommendations(class_name):
    """Returns structured farmer recommendations based on predicted disease."""
    if 'healthy' in class_name.lower():
        return {
            "pesticide": "None required.",
            "fertilizer": "Maintain balanced N-P-K fertilizer based on soil test.",
            "irrigation": "Regular deep watering. Avoid waterlogging.",
            "weather": "Monitor for extreme temperature shifts or prolonged rain.",
            "organic": "Apply compost tea or seaweed extract for immunity.",
            "best_practices": "Keep field clear of weeds, practice crop rotation, and inspect plants weekly.",
            "emergency": "Not applicable for healthy plants."
        }
        
    rec = {
        "pesticide": "Apply appropriate broad-spectrum fungicide/bactericide/insecticide.",
        "fertilizer": "Pause heavy nitrogen fertilizer, it can encourage soft growth susceptible to infection.",
        "irrigation": "Water at the base of the plant early in the day. Avoid overhead irrigation.",
        "weather": "Protect plants from excessive rain or humidity if possible.",
        "organic": "Use neem oil, Bacillus subtilis, or copper fungicides as appropriate.",
        "best_practices": "Sanitize all farming tools. Remove and destroy infected plant debris.",
        "emergency": "Quarantine the affected area. If spread is rapid, remove heavily infected plants immediately."
    }
    
    if 'blight' in class_name.lower():
        rec["pesticide"] = "Chlorothalonil, Mancozeb, or Copper-based fungicides."
        rec["fertilizer"] = "Ensure adequate Potassium (K) to boost cell wall strength."
        rec["irrigation"] = "Strict drip irrigation. Keep foliage completely dry."
        rec["weather"] = "Blight thrives in wet, cool-to-warm weather. Spray before forecasted rain."
        rec["organic"] = "Copper soap, Trichoderma harzianum, and Bio-fungicides."
        rec["best_practices"] = "Increase spacing for airflow, stake trailing plants."
        rec["emergency"] = "Immediately uproot and burn/dispose of plants with severe stem/leaf blight."
    elif 'mildew' in class_name.lower():
        rec["pesticide"] = "Myclobutanil, Sulfur-based sprays, or Potassium bicarbonate."
        rec["fertilizer"] = "Avoid high Nitrogen which promotes leafy growth that mildew loves."
        rec["irrigation"] = "Water early in the morning so plants dry quickly during the day."
        rec["weather"] = "High humidity and warm days/cool nights favor mildew."
        rec["organic"] = "Diluted milk spray (1:10), Neem oil, or Sulfur dust."
        rec["best_practices"] = "Prune lower leaves and thin out the canopy for better sunlight penetration."
        rec["emergency"] = "Prune and destroy heavily coated leaves. Do not compost them."
    elif 'rust' in class_name.lower():
        rec["pesticide"] = "Tebuconazole, Propiconazole, or Mancozeb."
        rec["fertilizer"] = "Maintain balanced nutrition. Phosphorus (P) can help root vigor."
        rec["irrigation"] = "Avoid wetting the leaves, as spores require moisture to germinate."
        rec["weather"] = "Spores spread easily in wind and rain. Treat proactively if humid."
        rec["organic"] = "Copper fungicides, Neem oil, or Sulfur."
        rec["best_practices"] = "Remove alternate hosts (weeds) nearby. Clean up fallen leaves."
        rec["emergency"] = "Remove heavily infected plants to save the rest of the crop."
    elif 'spot' in class_name.lower() or 'scab' in class_name.lower():
        rec["pesticide"] = "Copper octanoate or Mancozeb for bacterial/fungal spots."
        rec["fertilizer"] = "Avoid foliar feeding until the spots are controlled."
        rec["irrigation"] = "Use soaker hoses or drip tape. Splashing water spreads the disease."
        rec["weather"] = "Frequent rainfall exacerbates spots. Apply preventative sprays."
        rec["organic"] = "Baking soda solution, Neem oil, or Copper fungicides."
        rec["best_practices"] = "Practice 2-3 year crop rotation. Do not save seeds from infected plants."
        rec["emergency"] = "Defoliate severely spotted leaves to lower the spore/bacteria count."
    elif 'virus' in class_name.lower() or 'mosaic' in class_name.lower():
        rec["pesticide"] = "Insecticides (Imidacloprid, Pyrethrins) to control insect vectors like aphids."
        rec["fertilizer"] = "Normal feeding, but it will not cure the virus."
        rec["irrigation"] = "Standard practices, but avoid handling wet plants."
        rec["weather"] = "Warm weather increases insect vector activity. Monitor closely."
        rec["organic"] = "Insecticidal soaps, Neem oil, and introducing beneficial insects (Ladybugs)."
        rec["best_practices"] = "Control weeds, manage insects, and wash hands/tools frequently."
        rec["emergency"] = "IMMEDIATE REMOVAL AND DESTRUCTION of the infected plant. Viruses cannot be cured."
    elif 'rot' in class_name.lower():
        rec["pesticide"] = "Fosetyl-al, Metalaxyl (for root rots), or Captan."
        rec["fertilizer"] = "Stop fertilizing until rot is controlled. Avoid excess nitrogen."
        rec["irrigation"] = "Reduce watering immediately. Allow the soil to dry out between waterings."
        rec["weather"] = "Heavy rains or prolonged damp periods are critical risk factors."
        rec["organic"] = "Improve soil drainage, apply Trichoderma to the soil."
        rec["best_practices"] = "Plant on raised beds. Ensure excellent field drainage."
        rec["emergency"] = "If roots/crown are completely rotted, pull the plant to stop spread to neighbors."
    elif 'mite' in class_name.lower():
        rec["pesticide"] = "Abamectin, Spiromesifen, or specific Miticides."
        rec["fertilizer"] = "Adequate watering and feeding helps plant withstand mite damage."
        rec["irrigation"] = "Mites love dry, dusty conditions. Occasional overhead sprinkling can suppress them."
        rec["weather"] = "Hot, dry weather leads to rapid mite population explosions."
        rec["organic"] = "Horticultural oils, Neem oil, and releasing predatory mites (Phytoseiulus persimilis)."
        rec["best_practices"] = "Keep the area dust-free. Inspect undersides of leaves weekly."
        rec["emergency"] = "Cut off heavily webbed/infested leaves or branches and bag them."
    elif 'greening' in class_name.lower() or 'haunglongbing' in class_name.lower():
        rec["pesticide"] = "Systemic insecticides (e.g., Imidacloprid) to control the Asian citrus psyllid."
        rec["fertilizer"] = "Enhanced foliar nutritional sprays can temporarily mask symptoms and prolong life."
        rec["irrigation"] = "Maintain optimal soil moisture to reduce stress on the root system."
        rec["weather"] = "Psyllid populations peak during warm flush periods."
        rec["organic"] = "Horticultural oils to deter psyllids, reflective mulches."
        rec["best_practices"] = "Scout frequently for psyllids. Buy certified disease-free nursery stock."
        rec["emergency"] = "Uproot and chip/burn the entire tree. Do not transport any plant material."
    elif 'mold' in class_name.lower():
        rec["pesticide"] = "Chlorothalonil, Copper sprays, or Mancozeb."
        rec["fertilizer"] = "Avoid over-fertilizing with nitrogen."
        rec["irrigation"] = "Keep relative humidity low by using drip irrigation only."
        rec["weather"] = "Cloudy, humid weather is the prime condition for leaf molds."
        rec["organic"] = "Potassium bicarbonate sprays, Neem oil."
        rec["best_practices"] = "Prune to maximize airflow. Use fans if in a greenhouse."
        rec["emergency"] = "Remove lower leaves entirely to improve air circulation at the soil level."

    return rec

# --------------------------------
# Preprocess Image
# --------------------------------
def preprocess_image(image):

    image = image.resize((224, 224))

    img_array = np.array(image)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array.astype(np.float32),
        axis=0
    )

    return img_array

# --------------------------------
# Generate PDF Report
# --------------------------------
def generate_pdf_report(image, disease_name, confidence_percent):
    """Generate a professional PDF report with analysis results"""
    
    # Create PDF in memory
    pdf_buffer = io.BytesIO()
    
    # Create PDF document
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.lib import colors
    
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#4caf50'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#29b6f6'),
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#e0e0e0'),
        spaceAfter=6,
        alignment=TA_LEFT
    )
    
    # Title
    story.append(Paragraph("🌿 Agriculture AI - Plant Disease Detection", title_style))
    story.append(Paragraph("Analysis Report", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Date and time
    current_date = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    story.append(Paragraph(f"<b>Report Generated:</b> {current_date}", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Save image to buffer
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Add image
    story.append(Paragraph("<b>Uploaded Leaf Image:</b>", heading_style))
    img = RLImage(img_buffer, width=2.5*inch, height=2.5*inch)
    story.append(img)
    story.append(Spacer(1, 0.3*inch))
    
    # Analysis Results
    story.append(Paragraph("<b>Analysis Results:</b>", heading_style))
    
    # Confidence level status
    if confidence_percent >= 80:
        confidence_status = "🟢 HIGHLY CONFIDENT"
        confidence_color = "#4caf50"
    elif confidence_percent >= 60:
        confidence_status = "🟡 MODERATELY CONFIDENT"
        confidence_color = "#ffb74d"
    else:
        confidence_status = "🔴 LOW CONFIDENCE"
        confidence_color = "#ef5350"
    
    # Create results table
    results_data = [
        ['Detected Condition', disease_name],
        ['Confidence Score', f'{confidence_percent:.1f}%'],
        ['Confidence Level', confidence_status],
    ]
    
    results_table = Table(results_data, colWidths=[2*inch, 3.5*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#1b5e20')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4caf50')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#404040')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#1e1e1e'), colors.HexColor('#262626')]),
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    story.append(Paragraph("<b>Recommendations:</b>", heading_style))
    recommendations = [
        "• Consult with agricultural experts for treatment guidance",
        "• For best accuracy, use clear, well-lit images of affected areas",
        "• Ensure the leaf is in focus and fills most of the frame",
        "• Multiple predictions can help verify diagnosis reliability",
        "• Monitor the plant regularly for disease progression",
    ]
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#757575'),
        alignment=TA_CENTER
    )
    story.append(Paragraph("This report was generated by AI-Powered Plant Disease Classifier", footer_style))
    story.append(Paragraph("For professional agricultural advice, consult certified experts", footer_style))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    
    return pdf_buffer

# --------------------------------
# UI - Professional Plant Disease Detection System
# --------------------------------

# Header
st.markdown("""
    <div class="header-section">
        <h1>🌿 Agriculture AI - Plant Disease Detection</h1>
        <p>AI-Powered Leaf Health Detection & Analysis System</p>
    </div>
""", unsafe_allow_html=True)

# Subtitle
st.markdown("""
    <div class="subtitle">
        🔬 Upload a clear photo of a plant leaf to instantly detect diseases and receive AI-powered analysis. 
        Trained on 87,000+ crop images to identify 38 different plant conditions with high accuracy.
    </div>
""", unsafe_allow_html=True)

# Main Layout
col_upload, col_preview = st.columns([1, 1.3], gap="large")

with col_upload:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("### 📤 UPLOAD IMAGE")
    st.markdown("<p>Select a plant leaf image</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #999; font-size: 0.85em;'>Supported: JPG, JPEG, PNG, JFIF</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose image file",
        type=["jpg", "jpeg", "png", "jfif"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with col_preview:
        st.markdown('<div class="image-display">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing
    st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.spinner("🔍 Analyzing plant image with AI model..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0]
        
        # Get top 3 indices
        top_3_indices = np.argsort(prediction)[-3:][::-1]
        
        top_1_index = top_3_indices[0]
        confidence = prediction[top_1_index]
        disease_name = class_names[top_1_index].replace("_", " ").replace("including ", "").replace("including", "")
        confidence_percent = confidence * 100
    
    # Primary Result Section
    st.markdown("### 📊 PRIMARY DIAGNOSIS")
    
    # Animated Progress Bar
    progress_text = "Calculating confidence..."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(int(confidence_percent)):
        time.sleep(0.005)
        my_bar.progress(percent_complete + 1, text=f"Confidence: {percent_complete + 1}%")
    my_bar.progress(int(confidence_percent), text=f"Final Confidence: {confidence_percent:.1f}%")
    
    result_col1, result_col2 = st.columns(2, gap="large")
    
    with result_col1:
        st.markdown('<div class="result-card success">', unsafe_allow_html=True)
        st.markdown('<p class="result-title">🧬 Detected Condition</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="result-value">{disease_name}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with result_col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<p class="result-title">📈 Confidence Score</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="confidence-percentage">{confidence_percent:.1f}%</p>', unsafe_allow_html=True)
        
        # Confidence level badge
        if confidence_percent >= 80:
            badge_class = "confidence-high"
            status_text = "🟢 HIGHLY CONFIDENT"
        elif confidence_percent >= 60:
            badge_class = "confidence-medium"
            status_text = "🟡 MODERATELY CONFIDENT"
        else:
            badge_class = "confidence-low"
            status_text = "🔴 LOW CONFIDENCE"
        
        st.markdown(f'<p class="confidence-badge {badge_class}">{status_text}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Top 3 and Disease Info
    st.markdown("<hr>", unsafe_allow_html=True)
    col_top3, col_info = st.columns([1, 1.3], gap="large")
    
    with col_top3:
        st.markdown("### 🏆 TOP 3 PREDICTIONS")
        for i, idx in enumerate(top_3_indices):
            rank = i + 1
            name = class_names[idx].replace("_", " ").replace("including ", "").replace("including", "")
            conf = prediction[idx] * 100
            
            card_class = f"top3-card rank-{rank}"
            st.markdown(f"""
                <div class="{card_class}">
                    <div class="top3-name">#{rank} {name}</div>
                    <div class="top3-conf">{conf:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
    with col_info:
        st.markdown("### 📖 DISEASE INFORMATION")
        info = get_disease_info(class_names[top_1_index])
        
        st.markdown(f"""
            <div class="disease-info-panel">
                <div class="info-section">
                    <h4>🦠 Cause</h4>
                    <p>{info['cause']}</p>
                </div>
                <div class="info-section">
                    <h4>🤒 Symptoms</h4>
                    <p>{info['symptoms']}</p>
                </div>
                <div class="info-section">
                    <h4>💊 Treatment</h4>
                    <p>{info['treatment']}</p>
                </div>
                <div class="info-section">
                    <h4>🛡️ Prevention</h4>
                    <p>{info['prevention']}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # --------------------------------
    # Farmer Recommendation System
    # --------------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🚜 FARMER ACTION PLAN")
    st.markdown("<p style='color: #b0b0b0; margin-bottom: 20px;'>Professional agricultural recommendations based on the predicted disease. Always verify with local agricultural extension offices before applying chemicals.</p>", unsafe_allow_html=True)
    
    recs = get_farmer_recommendations(class_names[top_1_index])
    
    st.markdown(f"""
        <div class="rec-grid">
            <div class="rec-card rec-pesticide">
                <h4>🧪 Recommended Pesticide</h4>
                <p>{recs['pesticide']}</p>
            </div>
            <div class="rec-card rec-fertilizer">
                <h4>🌱 Fertilizer Advice</h4>
                <p>{recs['fertilizer']}</p>
            </div>
            <div class="rec-card rec-irrigation">
                <h4>💧 Irrigation Strategy</h4>
                <p>{recs['irrigation']}</p>
            </div>
            <div class="rec-card rec-weather">
                <h4>🌤️ Weather Precautions</h4>
                <p>{recs['weather']}</p>
            </div>
            <div class="rec-card rec-organic">
                <h4>🍃 Organic Treatments</h4>
                <p>{recs['organic']}</p>
            </div>
            <div class="rec-card rec-best-practices">
                <h4>📋 Best Practices</h4>
                <p>{recs['best_practices']}</p>
            </div>
            <div class="rec-card rec-emergency" style="grid-column: 1 / -1;">
                <h4>🚨 Emergency Action</h4>
                <p><strong>{recs['emergency']}</strong></p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Export PDF Section
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("### 📥 EXPORT RESULTS")
    
    col_pdf, col_space = st.columns([1, 2])
    
    with col_pdf:
        # Generate PDF
        pdf_buffer = generate_pdf_report(image, disease_name, confidence_percent)
        
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"Plant_Disease_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    # Info Box
    st.markdown("""
        <div class="info-box">
            <p><strong>💡 RECOMMENDATIONS:</strong></p>
            <p>• For best accuracy, use clear, well-lit images of affected areas</p>
            <p>• Ensure the leaf is in focus and fills most of the frame</p>
            <p>• Multiple predictions can help verify diagnosis reliability</p>
            <p>• Consult agricultural experts for treatment guidance</p>
        </div>
    """, unsafe_allow_html=True)

else:
    # Placeholder state
    st.markdown("""
        <div class="placeholder">
            <p>📸 👆 Upload a plant leaf image to begin analysis</p>
        </div>
    """, unsafe_allow_html=True)