import os
import sys
import pickle
import streamlit as st
import numpy as np
import time
from typing import List, Tuple

# --- Mock/Actual Module Imports ---
# This block allows the app to run even without the full backend pipeline.
try:
    from books_recommendation_system.exception.exception_handler import AppException
    from books_recommendation_system.logger.log import logging
    from books_recommendation_system.config.configuration import AppConfiguration
    from books_recommendation_system.pipeline.training_pipeline import TrainingPipeline
except ImportError:
    # --- Mock classes for standalone demonstration ---
    class AppConfiguration:
        def get_recommendation_config(self):
            return {
                "book_pivot_serialized_objects": 'artifacts/book_pivot.pkl', 
                "final_rating_serialized_objects": 'artifacts/final_rating.pkl', 
                "trained_model_path": 'artifacts/model.pkl' 
            }
    
    class RecommendationConfig:
        def __init__(self, data):
            self.book_pivot_serialized_objects = data['book_pivot_serialized_objects']
            self.final_rating_serialized_objects = data['final_rating_serialized_objects']
            self.trained_model_path = data['trained_model_path']

    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    
    logging = MockLogger()
    
    class MockPipeline:
        def start_training_pipeline(self): 
            logging.info("Mock training pipeline started")
            time.sleep(2)
    
    TrainingPipeline = MockPipeline
    
    class AppException(Exception):
        def __init__(self, message, error_details=None):
            super().__init__(message)
            self.error_details = error_details

# --- Page Configuration ---
st.set_page_config(
    page_title="Book Discovery Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced CSS for a Modern Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Hide Streamlit's default elements including deploy button */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    .stDeployButton { display: none !important; }
    
    /* Hide sidebar toggle button completely */
    [data-testid="collapsedControl"] { display: none !important; }
    button[data-testid="collapsedControl"] { display: none !important; }
    
    /* Force sidebar to be permanently open and visible */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        position: relative !important;
        width: 21rem !important;
        min-width: 21rem !important;
        max-width: 21rem !important;
        transform: translateX(0) !important;
        transition: none !important;
    }
    
    section[data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
        width: 21rem !important;
        min-width: 21rem !important;
        max-width: 21rem !important;
        margin-left: 0 !important;
        transform: translateX(0) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 21rem !important;
        transform: translateX(0) !important;
    }
    
    /* Adjust main content to account for permanent sidebar */
    .main .block-container {
        padding-left: 2rem !important;
        max-width: calc(100% - 21rem) !important;
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #10172A 0%, #1E1B4B 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1300px !important;
    }
    
    /* --- Sidebar Styling --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        margin: 1rem 0 2rem 0;
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    section[data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2) !important;
    }
    
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4) !important;
    }
    
    .sidebar-section {
        margin-top: 2.5rem;
    }
    
    .sidebar-section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e0e7ff;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(139, 92, 246, 0.3);
    }
    
    .stat-card {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    
    .stat-number { font-size: 2rem; font-weight: 800; color: #a78bfa; }
    .stat-label { font-size: 0.8rem; color: #cbd5e1; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    
    .feature-item {
        display: flex;
        align-items: center;
        margin: 0.8rem 0;
        padding: 0.8rem;
        background: rgba(139, 92, 246, 0.08);
        border-radius: 10px;
        border-left: 3px solid #a78bfa;
    }
    .feature-icon { font-size: 1.2rem; margin-right: 0.8rem; min-width: 25px; }
    .feature-text { color: #e0e7ff; font-size: 0.85rem; font-weight: 500; }
    
    /* --- Main Content Styling --- */
    .content-card {
        background: rgba(30, 27, 75, 0.5);
        border: 1px solid rgba(138, 92, 246, 0.3);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 50px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(15px);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #c4b5fd 0%, #f9a8d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.2rem; text-align: center; color: #c4b5fd;
        margin: 1rem 0 1.5rem 0; line-height: 1.6;
    }
    
    .stSelectbox > div > div {
        background: rgba(45, 27, 78, 0.9) !important;
        border: 2px solid rgba(138, 92, 246, 0.5) !important;
        border-radius: 12px !important; color: #e0e7ff !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #a78bfa !important;
        box-shadow: 0 0 20px rgba(167, 139, 250, 0.3) !important;
    }
    
    .stSelectbox label { display: none !important; }
    
    .stButton button {
        background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%) !important;
        font-size: 1.1rem !important; padding: 1rem 2.5rem !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.4) !important;
        border: none !important; color: white !important;
        transition: all 0.3s ease !important; font-weight: 600 !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(139, 92, 246, 0.6) !important;
    }
    
    /* --- Recommendation Cards --- */
    .rec-title {
        font-size: 2rem; font-weight: 800; color: #e0e7ff;
        margin: 2.5rem 0 2rem 0; text-align: center;
    }
    
    .book-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px; padding: 1.5rem; text-align: center;
        transition: all 0.3s ease; height: 100%; backdrop-filter: blur(10px);
    }
    
    .book-card:hover {
        transform: translateY(-10px); background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .book-card img {
        border-radius: 8px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        max-height: 220px; object-fit: cover;
    }
    
    .book-title {
        font-weight: 600; font-size: 1rem; color: #f0f2f6;
        margin-top: 1rem; min-height: 50px; display: -webkit-box;
        -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    
    /* Advanced Features */
    .streamlit-expanderHeader {
        background: rgba(139, 92, 246, 0.2) !important;
        color: #e0e7ff !important; border-radius: 12px !important;
        font-weight: 600 !important; border: 1px solid rgba(139, 92, 246, 0.3) !important;
    }
    .streamlit-expanderContent {
        background: rgba(139, 92, 246, 0.08);
        border-radius: 0 0 12px 12px;
        border: 1px solid rgba(138, 92, 246, 0.2);
        border-top: none !important;
    }
    .training-list { list-style: none; padding: 0; }
    .training-list li {
        background: rgba(139, 92, 246, 0.12); padding: 0.8rem 1.2rem;
        margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #a78bfa;
        color: #e0e7ff; font-weight: 500;
    }
    
    .custom-msg {
        color: white; padding: 1.2rem; border-radius: 12px;
        text-align: center; font-weight: 600; margin: 1.5rem 0;
    }
    .error-msg { background: linear-gradient(135deg, #ef4444, #dc2626); }
    
    .footer {
        text-align: center; color: #a78bfa; padding: 2rem; margin-top: 2rem;
        border-top: 1px solid rgba(138, 92, 246, 0.2);
    }
</style>

<script>
    // Force sidebar to stay permanently open
    function forceSidebarOpen() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.setAttribute('aria-expanded', 'true');
            sidebar.style.width = '21rem';
            sidebar.style.minWidth = '21rem';
            sidebar.style.transform = 'translateX(0)';
            sidebar.style.display = 'block';
            sidebar.style.visibility = 'visible';
        }
        
        // Hide the collapse button
        const collapseBtn = document.querySelector('[data-testid="collapsedControl"]');
        if (collapseBtn) {
            collapseBtn.style.display = 'none';
        }
    }
    
    // Run immediately and repeatedly
    forceSidebarOpen();
    setInterval(forceSidebarOpen, 100);
    
    // Observe DOM changes
    const observer = new MutationObserver(forceSidebarOpen);
    observer.observe(document.body, { childList: true, subtree: true, attributes: true });
</script>
""", unsafe_allow_html=True)

# --- Core Recommendation Logic ---
class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        self.is_mock = 'MockPipeline' in globals()
        try:
            if self.is_mock:
                self.recommendation_config = RecommendationConfig(app_config.get_recommendation_config())
            else:
                self.recommendation_config = app_config.get_recommendation_config()
            os.makedirs('artifacts', exist_ok=True)
            os.makedirs('templates', exist_ok=True)
        except Exception as e:
            logging.error(f"Configuration error: {e}")
            self.recommendation_config = None

    def fetch_data(self, suggestion) -> Tuple[List[str], List[str]]:
        if self.is_mock:
            titles = [f"Recommended Book {j}" for j in range(1, 7)]
            posters = [f"https://picsum.photos/id/{10+j}/200/300" for j in range(6)]
            return titles, posters
        try:
            with open(self.recommendation_config.book_pivot_serialized_objects, 'rb') as f:
                book_pivot = pickle.load(f)
            with open(self.recommendation_config.final_rating_serialized_objects, 'rb') as f:
                final_rating = pickle.load(f)
            
            titles, posters = [], []
            for name in book_pivot.index[suggestion[0]]:
                book_data = final_rating[final_rating['title'] == name]
                if not book_data.empty:
                    titles.append(book_data.iloc[0]['title'])
                    posters.append(book_data.iloc[0]['image_url'])
            return titles, posters
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return [], []

    def recommend_book(self, book_name: str) -> Tuple[List[str], List[str]]:
        if self.is_mock:
            return self.fetch_data(np.array([[0, 1, 2, 3, 4, 5]]))
        try:
            with open(self.recommendation_config.trained_model_path, 'rb') as f:
                model = pickle.load(f)
            with open(self.recommendation_config.book_pivot_serialized_objects, 'rb') as f:
                book_pivot = pickle.load(f)
            
            book_id_array = np.where(book_pivot.index == book_name)[0]
            if len(book_id_array) == 0:
                raise AppException(f"Book '{book_name}' not found in the dataset.")
            
            distances, suggestions = model.kneighbors(
                book_pivot.iloc[book_id_array[0], :].values.reshape(1, -1), n_neighbors=6)
            return self.fetch_data(suggestions)
        except Exception as e:
            logging.error(f"Recommendation error for '{book_name}': {e}")
            raise AppException("Failed to get recommendations. Please try another book.")

    def render_recommendations(self, selected_book: str):
        try:
            with st.spinner('✨ Conjuring your personalized book list...'):
                titles, urls = self.recommend_book(selected_book)
            
            st.markdown(f'<div class="rec-title">For Fans of "{selected_book}"</div>', unsafe_allow_html=True)
            
            recs = list(zip(titles, urls))[1:]
            if not recs:
                st.warning("Could not find recommendations. Try a different book!")
                return

            cols = st.columns(len(recs))
            for i, (title, url) in enumerate(recs):
                with cols[i]:
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    st.image(url, use_container_width=True)
                    st.markdown(f'<div class="book-title">{title}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        except AppException as e:
            st.markdown(f'<div class="custom-msg error-msg">❌ {e}</div>', unsafe_allow_html=True)

    def train_engine_ui(self):
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="hero-title" style="font-size: 2.5rem;">🚀 Model Training In Progress</div>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Please wait while the AI model is being updated with the latest data...</p>', unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            for percent_complete in range(100):
                if percent_complete < 30:
                    status_text.info("Step 1/3: Analyzing and preprocessing data...")
                elif percent_complete < 80:
                    status_text.info("Step 2/3: Training collaborative filtering model...")
                else:
                    status_text.info("Step 3/3: Finalizing and saving the model...")
                
                time.sleep(0.04)
                progress_bar.progress(percent_complete + 1)
            
            if not self.is_mock:
                TrainingPipeline().start_training_pipeline()
            
            status_text.success("🎉 Training Completed Successfully!")
            logging.info("Training completed successfully!")
            st.balloons()
            time.sleep(3)

        except Exception as e:
            status_text.error(f"❌ Training Failed: {e}")
            logging.error(f"Training failed: {e}")
            time.sleep(5)
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- Utility Functions ---
@st.cache_data
def load_book_names() -> List[str]:
    try:
        with open(os.path.join('templates', 'book_names.pkl'), 'rb') as f:
            data = pickle.load(f)
        return sorted(data.tolist() if hasattr(data, 'tolist') else data)
    except FileNotFoundError:
        logging.warning("book_names.pkl not found. Using default list.")
        return sorted(["The Great Gatsby", "1984", "The Hobbit", "The Da Vinci Code", "Dune"])

# --- Main Application ---
def main():
    obj = Recommendation()
    book_names = load_book_names()

    if 'run_training' not in st.session_state:
        st.session_state.run_training = False

    # --- Sidebar ---
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Book Discovery Engine</div>', unsafe_allow_html=True)
        
        if st.button('🧠 Train AI Model', use_container_width=True):
            st.session_state.run_training = True
            st.rerun()
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section-title">Quick Stats</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{len(book_names)}</div><div class="stat-label">Books</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="stat-card"><div class="stat-number">5</div><div class="stat-label">Per Rec</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section-title">How It Works</p>', unsafe_allow_html=True)
        features = [
            ("✨", "Collaborative Filtering analyzes reading patterns"),
            ("🤖", "Machine Learning powers personalized results"),
            ("🎯", "Select a book to discover similar titles"),
            ("⚡", "Instant recommendations with high accuracy")
        ]
        for icon, text in features:
            st.markdown(f'<div class="feature-item"><div class="feature-icon">{icon}</div><div class="feature-text">{text}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Main Content ---
    if st.session_state.run_training:
        obj.train_engine_ui()
        st.session_state.run_training = False
        st.rerun()
    else:
        st.markdown("""
        <div class="content-card">
            <div class="hero-title">Find Your Next Great Read</div>
            <div class="hero-subtitle">Select a book you enjoyed, and our AI will find similar titles you'll love.</div>
        </div>
        """, unsafe_allow_html=True)
        
        selected_book = st.selectbox(
            "book_select", book_names,
            index=book_names.index("The Great Gatsby") if "The Great Gatsby" in book_names else 0,
            label_visibility="collapsed"
        )
        if st.button('✨ Get Recommendations', use_container_width=True):
            obj.render_recommendations(selected_book)

        st.markdown('<div class="content-card" style="margin-top: 2rem;">', unsafe_allow_html=True)
        with st.expander("🔧 Model Training & Maintenance", expanded=False):
            st.markdown("""
            <p style="color: #e0e7ff; font-size: 1.1rem; font-weight: 600;">Why Retrain the Model?</p>
            <p style="color: #cbd5e1;">Regular training ensures recommendations stay fresh and accurate by incorporating:</p>
            <ul class="training-list">
                <li>📝 New user ratings and reviews</li>
                <li>📚 Recently added books to the database</li>
                <li>🎯 Improved algorithm performance</li>
            </ul>
            """, unsafe_allow_html=True)
            
            if st.button('🔄 Optimize & Retrain Model', key='train_main'):
                st.session_state.run_training = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Powered by Advanced Machine Learning & Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()