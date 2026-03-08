import streamlit as st

st.set_page_config(
    page_title="SmartDream AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0d1b35; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🏠 SmartDream AI Platform")
st.subheader("Your intelligent smart home ecosystem")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🌐 Smart Home Digital Twin
    Monitor, simulate, and optimise every device in your home using a live digital twin.
    
    **Features:**
    - Real-time device monitoring across rooms
    - AI-powered energy optimisation (PuLP solver)
    - Conflict detection & resolution
    - Predictive energy forecasting (LSTM)
    - Natural language home assistant
    
    👈 Navigate using **Digital Twin** in the sidebar
    """)

with col2:
    st.markdown("""
    ### 🏡 Dreamhouse AI Blueprint Generator
    Chat with an AI architect to design your custom floor plan — rendered as a professional blueprint.
    
    **Features:**
    - Conversational requirement gathering
    - Intelligent room layout engine
    - Canvas-rendered architectural blueprints
    - One-click layout regeneration
    
    👈 Navigate using **Dreamhouse AI** in the sidebar
    """)

st.markdown("---")
st.caption("SmartDream AI Platform ·  Built with Streamlit")
