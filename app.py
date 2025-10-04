import os
import sys
import pickle
import streamlit as st
import numpy as np
import time

# --- Mock/Actual Module Imports ---
try:
    from books_recommendation_system.exception.exception_handler import AppException
    from books_recommendation_system.logger.log import logging
    from books_recommendation_system.config.configuration import AppConfiguration
    from books_recommendation_system.pipeline.training_pipeline import TrainingPipeline
except ImportError:
    class AppConfiguration:
        def get_recommendation_config(self):
            return {
                "book_pivot_serialized_objects": 'book_pivot.pkl', 
                "final_rating_serialized_objects": 'final_rating.pkl', 
                "trained_model_path": 'model.pkl' 
            }
    class RecommendationConfig:
        def __init__(self, data):
            self.book_pivot_serialized_objects = data['book_pivot_serialized_objects']
            self.final_rating_serialized_objects = data['final_rating_serialized_objects']
            self.trained_model_path = data['trained_model_path']

    class MockLogger:
        def info(self, msg): pass
    logging = MockLogger()
    class MockPipeline:
        def start_training_pipeline(self): pass
    TrainingPipeline = MockPipeline
    class AppException(Exception):
        def __init__(self, message, sys):
            super().__init__(message)

# --- Application Setup ---

# Page configuration
st.set_page_config(
    page_title="Book Recommender 📖",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern Styling (Enhanced Hover Effect) ---
st.markdown("""
<style>
    /* Global Overrides */
    .stApp {
        background-color: #f7f9fc; 
        color: #1f2937; 
    }
    
    /* Dark Mode Compatibility */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #1e1e1e; 
            color: #f0f0f0;
        }
        .recommendation-card {
            background: #2b2b2b; 
            box-shadow: 0 4px 12px rgba(255, 255, 255, 0.08);
        }
        .book-title {
            color: #f0f0f0 !important; 
        }
        .selectbox-container {
            background: #2b2b2b;
            border: 1px solid #444;
        }
        .liked-book-text {
            color: #f0f0f0 !important;
        }
        .liked-book-banner {
            background-color: #2b2b2b !important;
            border-left-color: #f0f0f0 !important; 
        }
    }

    /* Main Header */
    .main-header {
        font-size: 4rem; 
        font-weight: 800; 
        background: linear-gradient(135deg, #4c66f0 0%, #764ba2 100%); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 1rem; 
        margin-bottom: 0.5rem;
        letter-spacing: -1px; 
    }
    .sub-header {
        font-size: 1.3rem;
        color: #6b7280; 
        text-align: center;
        margin-bottom: 3rem;
    }

    /* Recommendation Card Styling */
    .recommendation-card {
        background: white;
        border-radius: 12px; 
        padding: 15px; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); 
        border: none; 
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin-bottom: 15px; 
    }
    /* ENHANCED HOVER EFFECT */
    .recommendation-card:hover {
        transform: translateY(-10px); /* Increased lift */
        box-shadow: 0 15px 30px rgba(76, 102, 240, 0.4); /* Stronger shadow, hint of blue gradient color */
    }
    
    .book-title { 
        font-weight: 600;
        font-size: 1rem; 
        color: #333; 
        margin-top: 10px;
        min-height: 40px; 
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        line-height: 1.3;
        word-break: break-word; 
    }
    .book-image {
        border-radius: 8px;
        width: 100%;
        max-width: 130px; 
        height: 180px; 
        object-fit: cover; 
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); 
    }

    /* Custom style for the "Because you liked" banner */
    .liked-book-banner {
        background-color: white; 
        padding: 15px 25px; 
        border-radius: 10px; 
        margin-bottom: 25px; 
        border-left: 5px solid #4c66f0; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    .liked-book-text {
        font-size: 1.1rem; 
        color: #333; 
        font-weight: 500;
    }
    .liked-book-title {
        font-size: 1.1rem; 
        color: #4c66f0; 
        font-weight: 700;
        display: block; 
        margin-top: 5px;
    }
    
    /* Other styles for buttons, containers, etc. remain the same */
    .stButton button {
        background: linear-gradient(135deg, #4c66f0 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 14px 30px; 
        border-radius: 30px; 
        font-weight: 700; 
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(76, 102, 240, 0.3); 
    }
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(76, 102, 240, 0.5); 
    }
    .selectbox-container {
        background: white;
        padding: 30px; 
        border-radius: 18px; 
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.05);
        margin: 30px 0;
        border: 1px solid #e5e7eb; 
    }
</style>
""", unsafe_allow_html=True)

# --- Recommendation Logic Classes (FIXED recommendation_engine) ---

class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            if 'MockPipeline' in globals() and AppConfiguration is globals().get('AppConfiguration'):
                self.recommendation_config = RecommendationConfig(app_config.get_recommendation_config())
            else:
                self.recommendation_config = app_config.get_recommendation_config()
        except Exception:
            pass 

    def fetch_poster(self, suggestion):
        if 'MockPipeline' in globals():
            return [f"https://via.placeholder.com/130x180/4c66f0/ffffff?text={b.replace(' ', '+')}" for i, b in enumerate([f"Book {j+1}" for j in range(6)])]
        
        try:
            # Original Logic
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            final_rating = pickle.load(open(self.recommendation_config.final_rating_serialized_objects, 'rb'))

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]: 
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)

            return poster_url
        
        except Exception as e:
            return [f"https://via.placeholder.com/130x180/4c66f0/ffffff?text=Error" for _ in range(6)]

    def recommend_book(self, book_name):
        if 'MockPipeline' in globals():
            mock_books = [book_name] + [f"Recommended Book Title {i+1}" for i in range(5)]
            mock_suggestions = np.array([[0, 1, 2, 3, 4, 5]])
            poster_url = self.fetch_poster(mock_suggestions)
            return mock_books, poster_url
            
        try:
            # Original Logic
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path, 'rb'))
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

            poster_url = self.fetch_poster(suggestion)
            
            for i in range(len(suggestion)):
                books = book_pivot.index[suggestion[i]]
                for j in books:
                    books_list.append(j)
            return books_list, poster_url   
        
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            with st.spinner('Training the recommendation model... This may take a while.'):
                if 'MockPipeline' in globals():
                    time.sleep(2) 
                else:
                    obj = TrainingPipeline()
                    obj.start_training_pipeline()

            st.markdown('<div class="success-message">🎉 Training Completed Successfully!</div>', unsafe_allow_html=True)
            logging.info("Training completed successfully!")
        except Exception as e:
            st.error("Training failed! Check system logs.")
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_books):
        try:
            recommended_books, poster_url = self.recommend_book(selected_books)
            
            # Enhanced UI for the "Because you liked" banner
            st.markdown(
                f'<div class="liked-book-banner">'
                f'<span class="liked-book-text">Because you liked:</span> '
                f'<strong class="liked-book-title">{selected_books}</strong>'
                '</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("### 📖 Recommended Books")
            
            with st.container():
                cols = st.columns(5)
                
                start_index = 1
                end_index = min(len(recommended_books), start_index + 5)
                
                for i in range(start_index, end_index):
                    with cols[i - start_index]:
                        book_title = recommended_books[i]
                        image_url = poster_url[i]
                        
                        # --- FINAL FIX: Use a single markdown block to contain all card elements ---
                        # This resolves the issue of the floating white boxes and the missing images.
                        st.markdown(f"""
                            <div class="recommendation-card">
                                <img src="{image_url}" class="book-image" alt="{book_title}">
                                <div class="book-title">{book_title}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        # --------------------------------------------------------------------------
                        
        except Exception as e:
            st.error("An error occurred during recommendation. Please check if the model files are present.")
            raise AppException(e, sys) from e

# --- Main Application Execution ---

if __name__ == "__main__":
    
    obj = Recommendation()
    
    book_names = []
    try:
        loaded_book_names = pickle.load(open(os.path.join('templates', 'book_names.pkl'), 'rb'))
        
        if hasattr(loaded_book_names, 'tolist'):
            book_names = loaded_book_names.tolist()
        else:
            book_names = loaded_book_names
        
    except FileNotFoundError:
        book_names = ["1984", "The Da Vinci Code", "Harry Potter and the Sorcerer's Stone", "To Kill a Mockingbird", "The Great Gatsby", "The Catcher in the Rye", "And Then You Die"]
        os.makedirs('templates', exist_ok=True)
        try:
            with open(os.path.join('templates', 'book_names.pkl'), 'wb') as f:
                pickle.dump(book_names, f)
        except Exception:
            pass


    # Header Section
    st.markdown('<h1 class="main-header">📚 Book Discovery Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover your next favorite book using a sophisticated collaborative filtering model.</p>', unsafe_allow_html=True)
    
    # --- Sidebar for Controls and Information ---
    with st.sidebar:
        st.markdown("### ⚙️ System Controls")
        st.markdown("---")
        
        if st.button('🚀 Train Recommender System', key='train_sidebar', use_container_width=True):
            obj.train_engine()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        <p style="font-size: 0.95rem; color: #4b5563;">
        This system uses **collaborative filtering** to recommend books based on user preferences and reading patterns.
        </p>
        <h4 style="font-size: 1rem; color: #4c66f0; margin-top: 15px;">How it works:</h4>
        <ul style="font-size: 0.9rem; padding-left: 20px;">
            <li>1. Select a book you like</li>
            <li>2. Click 'Get Recommendations'</li>
            <li>3. Discover similar books!</li>
        </ul>
        """, unsafe_allow_html=True)


    # --- Main Content Area ---
    
    with st.container(border=True):
        st.markdown("### Find Your Next Read")
        
        initial_index = book_names.index("And Then You Die") if "And Then You Die" in book_names else 0
        
        selected_books = st.selectbox(
            "🔍 Start typing to search for a book:",
            book_names,
            index=initial_index,
            help="Select one of the books used in the model's training data."
        )
        
        st.markdown("<br>", unsafe_allow_html=True) 
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button('🎯 Get Recommendations', key='main_recommend_btn', use_container_width=True):
                if selected_books:
                    try:
                        obj.recommendations_engine(selected_books)
                    except AppException as e:
                        st.error(f"Error during recommendation: {e}")
                else:
                    st.warning("Please select a book first!")

    # --- Training Section in Main Area ---
    st.markdown("---")
    st.markdown("### 🛠️ Maintenance & Retraining")
    
    with st.expander("Why Retrain?", expanded=False):
        st.markdown("""
        Retraining the model incorporates the latest data, which is crucial for maintaining recommendation accuracy and relevance over time. Use this when new user ratings or books have been added to your dataset.
        """)

    if st.button('🚀 Retrain Model Now', use_container_width=False, key='train_main', type='secondary'): 
        st.markdown(
            """
            <script>
            var buttons = window.parent.document.querySelectorAll('button[key="train_main"]');
            buttons.forEach(function(button) {
                button.classList.add('training-button');
            });
            </script>
            """,
            unsafe_allow_html=True
        )
        obj.train_engine()