import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Churn Analysis Dashboard", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS STYLING =====
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
        color : #000000
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .prediction-card h2 {
        font-size: 2rem;
        margin: 0;
        font-weight: 600;
    }
    
    .prediction-card p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Form styling */
    .form-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Navigation styling */
    .nav-link-selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Dark theme adjustments */
    .dark-theme .form-section {
        background: #1a202c;
        border-color: #2d3748;
    }
    
    .dark-theme .metric-card {
        background: #2d3748;
        color: white;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        border-radius: 8px;
        border-left: 4px solid #48bb78;
    }
    
    .stError {
        border-radius: 8px;
        border-left: 4px solid #f56565;
    }
    
    .stWarning {
        border-radius: 8px;
        border-left: 4px solid #ed8936;
    }
    
    .stInfo {
        border-radius: 8px;
        border-left: 4px solid #4299e1;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== THEME CONFIGURATION =====
def configure_theme():
    """Configure theme settings with improved styling"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        theme = st.selectbox('🎨 Theme', ['Light', 'Dark'], key='theme_selector')
        
        if theme == 'Dark':
            st.markdown("""
                <style>
                .stApp {
                    background-color: #0E1117;
                    color: white;
                }
                .dark-theme { display: block; }
                
                /* Dark theme option menu adjustments */
                .nav-link {
                    color: #ffffff !important;
                }
                .nav-link:hover {
                    color: #667eea !important;
                    background-color: rgba(102, 126, 234, 0.1) !important;
                }
                </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <style>
                /* Light theme option menu adjustments */
                .nav-link {
                    color: #333333 !important;
                }
                .nav-link:hover {
                    color: #667eea !important;
                    background-color: rgba(102, 126, 234, 0.1) !important;
                }
                </style>
            """, unsafe_allow_html=True)
        
        return theme

# ===== NAVIGATION MENU =====
def create_navigation():
    """Enhanced navigation with custom CSS for better icon control"""
    
    # Custom CSS for better icon and text control
    st.markdown("""
        <style>
        /* Custom option menu styling */
        div[data-testid="stHorizontalBlock"] [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        div[data-testid="stHorizontalBlock"] [data-baseweb="tab"] {
            height: auto;
            white-space: pre-wrap;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            border: 1px solid rgba(102, 126, 234, 0.2);
            padding: 12px 20px;
            font-weight: 500;
            color: #333333;
            transition: all 0.3s ease;
        }
        
        div[data-testid="stHorizontalBlock"] [data-baseweb="tab"]:hover {
            background: rgba(102, 126, 234, 0.1);
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        }
        
        div[data-testid="stHorizontalBlock"] [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transform: translateY(-2px);
        }
        
        /* Ensure icons are visible in selected state */
        div[data-testid="stHorizontalBlock"] [data-baseweb="tab"][aria-selected="true"] svg {
            color: white !important;
            fill: white !important;
        }
        
        /* Icon styling */
        div[data-testid="stHorizontalBlock"] [data-baseweb="tab"] svg {
            color: #667eea !important;
            fill: #667eea !important;
            margin-right: 8px;
        }
        
        /* Container styling */
        div[data-testid="stHorizontalBlock"] [data-baseweb="tab-list"] {
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            padding: 1rem;
            border-radius: 15px;
            border: 1px solid rgba(102, 126, 234, 0.1);
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    return option_menu(
        menu_title=None,
        options=["🔮 Prediction", "💬 Text2SQL", "🔍 RAG"],
        icons=["graph-up-arrow", "chat-dots-fill", "search-heart"],
        menu_icon=None,
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0",
                "background": "transparent"
            },
            "icon": {
                "color": "#667eea", 
                "font-size": "18px"
            }, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "padding": "0px",
                "background": "transparent",
                "border": "none"
            },
            "nav-link-selected": {
                "background": "transparent"
            }
        }
    )

# ===== PREDICTION PAGE =====
def render_prediction_page():
    """Render the customer churn prediction page"""
    # Page header
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Predict customer attrition</h1>
        <p>Predict customer churn probability using machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create prediction form
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown('<div class="section-title">📋 Customer Information</div>', unsafe_allow_html=True)
        
        # Customer Demographics Section
        st.markdown("##### 👤 Demographics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            senior_citizen = st.selectbox(
                "Senior Citizen", 
                ["No", "Yes"], 
                help="Is the customer a senior citizen?"
            )
            partner = st.selectbox(
                "Partner", 
                ["No", "Yes"],
                help="Does the customer have a partner?"
            )
            dependents = st.selectbox(
                "Dependents", 
                ["No", "Yes"],
                help="Does the customer have dependents?"
            )
        
        with col2:
            tenure = st.number_input(
                "Tenure (months)", 
                min_value=0, 
                max_value=100,
                help="Number of months the customer has stayed with the company"
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)", 
                min_value=0.0, 
                max_value=200.0,
                format="%.2f",
                help="The amount charged to the customer monthly"
            )
            total_charges = st.number_input(
                "Total Charges ($)", 
                min_value=0.0,
                format="%.2f",
                help="The total amount charged to the customer"
            )
        
        with col3:
            contract = st.selectbox(
                "Contract Type", 
                ["Month-to-month", "One year", "Two year"],
                help="The contract term of the customer"
            )
            paperless_billing = st.selectbox(
                "Paperless Billing", 
                ["No", "Yes"],
                help="Whether the customer has paperless billing"
            )
            payment_method = st.selectbox(
                "Payment Method", 
                ["Bank transfer (automatic)", "Credit card (automatic)", 
                 "Electronic check", "Mailed check"],
                help="The customer's payment method"
            )
        
        # Services Section
        st.markdown("##### 🛡️ Services")
        col4, col5, col6, col7 = st.columns(4)
        
        with col4:
            online_security = st.selectbox(
                "Online Security", 
                ["No", "No internet service", "Yes"],
                help="Whether the customer has online security"
            )
        with col5:
            online_backup = st.selectbox(
                "Online Backup", 
                ["No", "No internet service", "Yes"],
                help="Whether the customer has online backup"
            )
        with col6:
            device_protection = st.selectbox(
                "Device Protection", 
                ["No", "No internet service", "Yes"],
                help="Whether the customer has device protection"
            )
        with col7:
            tech_support = st.selectbox(
                "Tech Support", 
                ["No", "No internet service", "Yes"],
                help="Whether the customer has tech support"
            )
        
        # Submit button
        st.markdown("---")
        submitted = st.form_submit_button(
            "🚀 Predict Churn Probability", 
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            with st.spinner('🤖 Analyzing customer data...'):
                # Process form data
                processed_data = process_prediction_data(
                    senior_citizen, partner, dependents, tenure, online_security,
                    online_backup, device_protection, tech_support, contract,
                    paperless_billing, payment_method, monthly_charges, total_charges
                )
                
                # Make prediction
                make_prediction(processed_data)

def process_prediction_data(senior_citizen, partner, dependents, tenure, online_security,
                          online_backup, device_protection, tech_support, contract,
                          paperless_billing, payment_method, monthly_charges, total_charges):
    """Process and encode form data for prediction"""
    
    # Convert categorical variables to numeric
    senior_citizen_encoded = 1 if senior_citizen == "Yes" else 0
    partner_encoded = 1 if partner == "Yes" else 0
    dependents_encoded = 1 if dependents == "Yes" else 0
    
    # Service encodings
    service_encoding = {"No": 0, "No internet service": 1, "Yes": 2}
    online_security_encoded = service_encoding[online_security]
    online_backup_encoded = service_encoding[online_backup]
    device_protection_encoded = service_encoding[device_protection]
    tech_support_encoded = service_encoding[tech_support]
    
    # Contract encoding
    contract_encoding = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    contract_encoded = contract_encoding[contract]
    
    # Other encodings
    paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0
    payment_method_encoding = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }
    payment_method_encoded = payment_method_encoding[payment_method]
    
    return [
        senior_citizen_encoded, partner_encoded, dependents_encoded, tenure,
        online_security_encoded, online_backup_encoded, device_protection_encoded,
        tech_support_encoded, contract_encoded, paperless_billing_encoded,
        payment_method_encoded, monthly_charges, total_charges
    ]

def make_prediction(features):
    """Make prediction API call and display results"""
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"features": features},
            timeout=10
        )
        
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            probability = prediction * 100
            
            # Display prediction result with custom styling
            if probability > 70:
                risk_level = "High Risk"
                risk_color = "#f56565"
                risk_icon = "🚨"
            elif probability > 40:
                risk_level = "Medium Risk"
                risk_color = "#ed8936"
                risk_icon = "⚠️"
            else:
                risk_level = "Low Risk"
                risk_color = "#48bb78"
                risk_icon = "✅"
            
            st.markdown(f"""
            <div class="prediction-card" style="background: linear-gradient(135deg, {risk_color}88 0%, {risk_color}aa 100%);">
                <h2>{risk_icon} {risk_level}</h2>
                <p>Churn Probability: {probability:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Churn Probability", f"{probability:.1f}%")
            with col2:
                st.metric("Risk Level", risk_level)
            with col3:
                retention_score = 100 - probability
                st.metric("Retention Score", f"{retention_score:.1f}%")
                
        else:
            st.error("❌ Error getting prediction from the server")
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection error: {str(e)}")
        st.info("💡 Make sure the prediction server is running on localhost:8000")

# ===== TEXT2SQL PAGE =====
def render_text2sql_page():
    """Render the Text2SQL page"""
    st.markdown("""
    <div class="main-header">
        <h1>💬 Natural Language to SQL</h1>
        <p>Convert your questions into SQL queries automatically</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Query input section
    st.markdown("### 🤔 Ask Your Question")
    
    # Example queries
    with st.expander("💡 Example Questions"):
        examples = [
            "How many customers churned last month?",
            "What is the average monthly charge for customers with fiber optic?",
            "Show me customers with more than 50 months tenure",
            "Which payment method has the highest churn rate?",
            "List top 10 customers by total charges"
        ]
        for example in examples:
            if st.button(f"📝 {example}", key=f"example_{example}", help="Click to use this example"):
                st.session_state.sql_query = example
    
    # Query input
    query = st.text_area(
        "Enter your question about the data:",
        value=st.session_state.get("sql_query", ""),
        height=100,
        placeholder="e.g., How many customers have churned in the last 6 months?",
        help="Ask any question about your customer data in plain English"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_btn = st.button("🚀 Generate SQL", type="primary", use_container_width=True)
    
    if generate_btn and query:
        with st.spinner('🧠 Generating SQL query...'):
            execute_text2sql(query)
    elif generate_btn and not query:
        st.warning("⚠️ Please enter a question first")

def execute_text2sql(query):
    """Execute Text2SQL query"""
    try:
        response = requests.post(
            "http://localhost:8000/generate_sql",
            json={"query": query},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        # Display SQL query
        st.success("✅ SQL Generated Successfully!")
        st.markdown("### 📝 Generated SQL Query")
        st.code(result["sql"], language="sql")
        
        # Display results if available
        if result.get("results"):
            st.markdown("### 📊 Query Results")
            df = pd.DataFrame(result["results"])
            
            # Display dataframe with better formatting
            st.dataframe(
                df, 
                use_container_width=True,
                hide_index=True
            )
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if st.button("📈 Visualize Data", use_container_width=True):
                    create_quick_visualization(df)
        else:
            st.info("ℹ️ Query executed successfully but returned no results")
        
        # Raw response in expander
        with st.expander("🔍 View Technical Details"):
            st.json(result)
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                st.json(e.response.json())
            except:
                st.text(e.response.text)

def create_quick_visualization(df):
    """Create a quick visualization of the query results"""
    st.markdown("### 📊 Quick Visualization")
    
    if len(df.columns) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-axis", df.columns, key="viz_x")
        with col2:
            y_axis = st.selectbox("Y-axis", df.columns, key="viz_y")
        
        if x_axis and y_axis:
            try:
                # Simple bar chart
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                fig.update_layout(
                    title_font_size=16,
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create visualization: {str(e)}")

# ===== RAG PAGE =====
def render_rag_page():
    """Render the RAG (Retrieval Augmented Generation) page"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Knowledge Base Query</h1>
        <p>Search and get answers from your knowledge base using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Query input section
    st.markdown("### 🔎 Search Knowledge Base")
    
    # Recent queries (if any)
    if 'rag_history' not in st.session_state:
        st.session_state.rag_history = []
    
    if st.session_state.rag_history:
        with st.expander("🕒 Recent Queries"):
            for i, past_query in enumerate(reversed(st.session_state.rag_history[-5:])):
                if st.button(f"🔄 {past_query}", key=f"rag_history_{i}"):
                    st.session_state.current_rag_query = past_query
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        value=st.session_state.get("current_rag_query", ""),
        height=100,
        placeholder="e.g., What factors contribute most to customer churn?",
        help="Ask questions about your business domain and get AI-powered answers"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_btn and query:
        # Add to history
        if query not in st.session_state.rag_history:
            st.session_state.rag_history.append(query)
        
        with st.spinner('🤖 Searching knowledge base...'):
            execute_rag_query(query)
    elif search_btn and not query:
        st.warning("⚠️ Please enter a question first")

def execute_rag_query(query):
    """Execute RAG query"""
    try:
        response = requests.post(
            "http://localhost:8000/rag_query",
            json={"query": query},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        st.success("✅ Query Processed Successfully!")
        
        # Display the answer with better formatting
        st.markdown("### 💡 Answer")
        st.markdown(f"""
        <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #0ea5e9;">
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{result["answer"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display confidence score if available
        if "confidence" in result:
            confidence = result["confidence"]
            confidence_color = "#48bb78" if confidence > 0.8 else "#ed8936" if confidence > 0.5 else "#f56565"
            
            st.markdown(f"""
            <div style="margin: 1rem 0;">
                <span style="background: {confidence_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.9rem;">
                    🎯 Confidence: {confidence:.1%}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        # Display relevant sources/documents
        if result.get("sources"):
            st.markdown("### 📚 Source Documents")
            
            for idx, source in enumerate(result["sources"], 1):
                with st.expander(f"📄 Source {idx}", expanded=idx == 1):
                    st.markdown(f"""
                    <div style="background: #fafafa; padding: 1rem; border-radius: 8px; font-family: 'Georgia', serif; line-height: 1.6;">
                        {source}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Technical details
        with st.expander("🔧 Technical Details"):
            st.json(result)
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                st.json(e.response.json())
            except:
                st.text(e.response.text)

# ===== MAIN APPLICATION =====
def main():
    """Main application function"""
    # Load custom CSS
    load_custom_css()
    
    # Configure theme
    theme = configure_theme()
    
    # Add sidebar info
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Dashboard Info")
        st.info("This dashboard provides customer churn analysis capabilities including prediction, SQL generation, and knowledge base querying.")
        
        st.markdown("### 🔗 Quick Links")
        st.markdown("- 🔮 **Prediction**: Analyze customer churn risk")
        st.markdown("- 💬 **Text2SQL**: Generate SQL from natural language")
        st.markdown("- 🔍 **RAG**: Query knowledge base with AI")
    
    # Create navigation
    selected = create_navigation()
    
    # Route to appropriate page
    if selected == "🔮 Prediction":
        render_prediction_page()
    elif selected == "💬 Text2SQL":
        render_text2sql_page()
    elif selected == "🔍 RAG":
        render_rag_page()

if __name__ == "__main__":
    main()