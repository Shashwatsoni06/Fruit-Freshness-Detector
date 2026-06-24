# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Fruit Freshness Detector",
    page_icon="🍎",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.main {
    background: linear-gradient(
        135deg,
        #fff7e6,
        #fff2cc,
        #ffe6cc
    );
}

.title {
    text-align:center;
    font-size:55px;
    font-weight:800;
    color:#2e7d32;
}

.subtitle {
    text-align:center;
    color:#555;
    font-size:20px;
    margin-bottom:20px;
}

.result-card {
    padding:20px;
    border-radius:20px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

.fresh {
    background:#e8f5e9;
    color:#1b5e20;
}

.rotten {
    background:#ffebee;
    color:#b71c1c;
}

.footer {
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# MODEL LOADING
# ==========================

@st.cache_resource
def load_model():

    if os.path.exists("fruits_classification_model.keras"):
        return tf.keras.models.load_model(
            "fruits_classification_model.keras"
        )

    return tf.keras.models.load_model(
        "fruits_classification_model.h5"
    )

model = load_model()

# ==========================
# HEADER
# ==========================

st.markdown(
    "<div class='title'>🍎 Fruit Freshness Detector </div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-Powered Fresh vs Rotten Fruit Classification</div>",
    unsafe_allow_html=True
)

st.warning(
    "Upload only fruit images. "
    "The model was trained on fruit datasets."
)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.header("📊 Model Information")

    st.info("""
    **Model Type**
    - CNN

    **Input Size**
    - 224 × 224

    **Output Classes**
    - Fresh Fruit
    - Rotten Fruit

    **Framework**
    - TensorFlow / Keras
    """)

    st.success(
        "Upload a fruit image and click Analyze."
    )

# ==========================
# IMAGE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "📤 Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    # ==========================
    # PREPROCESSING
    # ==========================

    img = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    if st.button(
        "🔍 Analyze Fruit",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            prediction = model.predict(
                img_array,
                verbose=0
            )[0][0]

        fresh_prob = (1 - prediction) * 100
        rotten_prob = prediction * 100

        confidence = max(
            fresh_prob,
            rotten_prob
        )

        with col2:

            st.subheader("📈 Prediction Result")

            if prediction >= 0.5:

                st.markdown(
                    """
                    <div class='result-card rotten'>
                    🍂 ROTTEN FRUIT
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class='result-card fresh'>
                    🍏 FRESH FRUIT
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            st.progress(
                min(int(confidence), 100)
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            # ==========================
            # PROBABILITY CHART
            # ==========================

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=[
                            "Fresh",
                            "Rotten"
                        ],
                        values=[
                            fresh_prob,
                            rotten_prob
                        ],
                        hole=0.55
                    )
                ]
            )

            fig.update_layout(
                height=350,
                margin=dict(
                    t=20,
                    b=20,
                    l=20,
                    r=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Fresh %",
                    f"{fresh_prob:.2f}"
                )

            with c2:

                st.metric(
                    "Rotten %",
                    f"{rotten_prob:.2f}"
                )

            # Debug value
            with st.expander(
                "Developer Information"
            ):
                st.write(
                    "Raw Prediction:",
                    float(prediction)
                )

st.markdown("---")

st.markdown(
"""
<div class='footer'>
Fruit Freshness Detection System<br>
Built using TensorFlow + Streamlit
</div>
""",
unsafe_allow_html=True
)

