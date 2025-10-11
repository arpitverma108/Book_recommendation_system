import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommendation_system.logger.log import logging
from books_recommendation_system.config.configuration import AppConfiguration
from books_recommendation_system.pipeline.training_pipeline import TrainingPipeline
from books_recommendation_system.exception.exception_handler import AppException


class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
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
            raise AppException(e, sys) from e

    def recommend_book(self, book_name):
        try:
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
            # Create placeholder for animation
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Show animated training message
            progress_placeholder.markdown("""
                <div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.15); border-radius: 20px; margin: 2rem 0;'>
                    <div class='training-animation'>
                        <div class='spinner'></div>
                    </div>
                    <h3 style='color: #ffffff; margin-top: 1rem;'>🤖 AI Model Training in Progress...</h3>
                    <p style='color: #ffffff; opacity: 0.9; font-size: 1.1rem;'>Please wait while we train the recommendation engine</p>
                </div>
                <style>
                .training-animation {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .spinner {
                    width: 80px;
                    height: 80px;
                    border: 8px solid rgba(255, 255, 255, 0.3);
                    border-top: 8px solid #ffffff;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Simulated progress updates
            progress_bar = status_placeholder.progress(0)
            status_text = st.empty()
            
            status_text.markdown("<p style='text-align: center; color: #ffffff; font-weight: 600;'>Loading data...</p>", unsafe_allow_html=True)
            progress_bar.progress(20)
            
            # Start actual training
            obj = TrainingPipeline()
            
            status_text.markdown("<p style='text-align: center; color: #ffffff; font-weight: 600;'>Processing features...</p>", unsafe_allow_html=True)
            progress_bar.progress(40)
            
            status_text.markdown("<p style='text-align: center; color: #ffffff; font-weight: 600;'>Training model...</p>", unsafe_allow_html=True)
            progress_bar.progress(60)
            
            obj.start_training_pipeline()
            
            status_text.markdown("<p style='text-align: center; color: #ffffff; font-weight: 600;'>Saving model...</p>", unsafe_allow_html=True)
            progress_bar.progress(90)
            
            progress_bar.progress(100)
            status_text.markdown("<p style='text-align: center; color: #ffffff; font-weight: 600;'>Complete!</p>", unsafe_allow_html=True)
            
            # Clear animation and show success
            progress_placeholder.empty()
            status_placeholder.empty()
            status_text.empty()
            
            st.success("✅ Training Completed Successfully!")
            st.balloons()
            logging.info(f"Training completed successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_books):
        try:
            with st.spinner('🔍 Finding perfect recommendations for you...'):
                recommended_books, poster_url = self.recommend_book(selected_books)

            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #ffffff; font-size: 2rem; font-weight: 700; margin-bottom: 30px;'>📚 Recommended Books For You</h2>", unsafe_allow_html=True)

            cols = st.columns(5)
            for idx, col in enumerate(cols):
                with col:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 15px; transition: all 0.3s;'>
                            <img src='{poster_url[idx + 1]}' style='width: 100%; border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.3); margin-bottom: 15px;'/>
                            <p style='color: #ffffff; font-weight: 600; font-size: 0.95rem; line-height: 1.4; margin: 0;'>{recommended_books[idx + 1]}</p>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error generating recommendations: {str(e)}")
            raise AppException(e, sys) from e


if __name__ == "__main__":
    # Page configuration
    st.set_page_config(
        page_title="BookWise - AI Recommender",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for enhanced styling and visible selectbox text
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
        * { font-family: 'Poppins', sans-serif; }

        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        }

        h1 { color: #ffffff !important; font-size: 4rem !important; font-weight: 800 !important; text-align: center; margin-bottom: 0.5rem !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); letter-spacing: -1px; }
        .subtitle { text-align: center; color: #ffffff; font-size: 1.3rem; margin-bottom: 3rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }

        .stButton > button {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            padding: 0.85rem 2.5rem;
            font-size: 1.1rem;
            font-weight: 700;
            border-radius: 50px;
            box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
            transition: all 0.3s ease;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(245, 87, 108, 0.6);
            background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        }

        /* Fixed selectbox text visibility */
        .stSelectbox > div > div input {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        .stSelectbox [role="button"] {
            color: #000000 !important;
        }
        .stSelectbox > div > div div[data-baseweb="select"] {
            background-color: #ffffff !important;
        }
        .stSelectbox > div > div label { color: #ffffff !important; }

        /* Feature cards styling */
        .feature-card { background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(10px); padding: 2rem; border-radius: 20px; text-align: center; margin: 1rem 0; border: 2px solid rgba(255, 255, 255, 0.2); transition: all 0.4s ease; }
        .feature-card:hover { transform: translateY(-10px) scale(1.02); box-shadow: 0 12px 40px rgba(0,0,0,0.2); border-color: rgba(255, 255, 255, 0.4); background: rgba(255, 255, 255, 0.2); }
        .feature-icon { font-size: 3.5rem; margin-bottom: 1rem; filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.2)); }
        .feature-card h3 { color: #ffffff !important; font-size: 1.5rem !important; margin: 0.8rem 0 !important; font-weight: 700 !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }
        .feature-card p { color: #ffffff !important; font-size: 1rem !important; margin: 0 !important; opacity: 0.95; line-height: 1.6; }

        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
        .stDeployButton {display: none;} button[kind="header"] {display: none;} [data-testid="stToolbar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

    # Header section
    st.markdown("<h1>📚 BookWise</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Discover Your Next Favorite Book with AI-Powered Recommendations</p>", unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🤖</div>
                <h3>AI-Powered</h3>
                <p>Advanced machine learning algorithms analyze your preferences</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>⚡</div>
                <h3>Lightning Fast</h3>
                <p>Get personalized recommendations in seconds</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🎯</div>
                <h3>Highly Accurate</h3>
                <p>Collaborative filtering ensures relevant suggestions</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

    obj = Recommendation()

    # Training section
    with st.expander("⚙️ System Training (Admin Only)", expanded=False):
        st.markdown("#### Train the Recommendation Model")
        st.info("Click below to train the recommendation system with the latest data. This may take a few minutes.")
        if st.button('🚀 Start Training'):
            obj.train_engine()

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    # Recommendation section
    st.markdown("<div style='background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 2.5rem; border-radius: 25px; border: 2px solid rgba(255, 255, 255, 0.2);'>", unsafe_allow_html=True)
    st.markdown("<h3>🔍 Find Your Next Great Read</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; margin-bottom: 1.5rem;'>Select a book you enjoyed, and we'll recommend similar titles you'll love!</p>", unsafe_allow_html=True)

    book_names = pickle.load(open(os.path.join('templates', 'book_names.pkl'), 'rb'))

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_books = st.selectbox(
            "Search for a book",
            book_names,
            help="Start typing or select from the dropdown"
        )
    with col2:
        st.markdown("<div style='margin-top: 1.85rem;'></div>", unsafe_allow_html=True)
        recommend_button = st.button('✨ Get Recommendations', use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if recommend_button:
        if selected_books:
            obj.recommendations_engine(selected_books)
        else:
            st.warning("⚠️ Please select a book first!")

    # Footer
    st.markdown("<div style='margin: 4rem 0 2rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <p style='font-size: 1.1rem; font-weight: 500;'>Made with ❤️ using Streamlit | Powered by Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)