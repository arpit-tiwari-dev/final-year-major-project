import streamlit as st
import base64
import pandas as pd
import lightgbm as lgb
import joblib
from datetime import datetime
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objs as go
import os

# Load pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'lightgbm_model.pkl')
model = joblib.load(model_path)

# Page configuration
st.set_page_config(
    page_title="Online Payment Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.95;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background
bg_path = os.path.join(os.path.dirname(__file__), 'fbg2.jpg')
set_background(bg_path)

# Custom CSS for the entire app
st.markdown("""
    <style>
    /* Main theme colors and styles */
    :root {
        --primary-color: #1e3d59;
        --secondary-color: #ff6e40;
        --background-color: rgba(255, 255, 255, 0.3);
        --text-color: #ffffff;
        --card-bg: rgba(255, 255, 255, 0.3);
    }

    /* Title styles */
    .main-title {
        color: var(--text-color);
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }

    /* Navigation menu styling */
    .stOptionMenu {
        background-color: var(--card-bg) !important;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .li {
        font-size: 1.2rem;
   }
    .nav-link {
        color: var(--text-color) !important;
        font-weight: 500;
        transition: all 0.3s ease;
        margin: 0 5px;
        border-radius: 5px;
    }

    .nav-link:hover {
        background-color: var(--secondary-color) !important;
        transform: translateY(-2px);
    }

    .nav-link-selected {
        background-color: var(--secondary-color) !important;
        color: white !important;
    }
            
    /* Card styling */
    .custom-card {
        font-size: 1.2rem;
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .custom-card:hover {
        transform: translateY(-5px);
    }

    /* Alert styling */
    .fraud-alert {
        font-size: 1.2rem;
        background-color: rgba(220, 53, 69, 0.9);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: slideIn 0.5s ease-out;
    }

    .safe-alert {
        font-size: 1.2rem;
        background-color: rgba(40, 167, 69, 0.9);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: slideIn 0.5s ease-out;
    }

    /* Animation keyframes */
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }


    /* Input fields styling */
    .stNumberInput, .stSelectbox {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 5px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }

    .stButton > button {
        background-color: var(--secondary-color) !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 0.5rem 2rem !important;
        border-radius: 5px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }
    
    /* Global font size for all elements */
    * {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Title (above navigation)
st.markdown('<h1 class="main-title">💳 Online Payment Fraud Detection System</h1>', unsafe_allow_html=True)

# Navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "Prediction", "History", "About"],
    icons=['house','search','clock-history','file-person'],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0", "background-color": "rgba(255, 255, 255, 0.1)"},
        "icon": {"color": "white", "font-size": "1rem"},
        "nav-link": {
            "font-size": "1rem",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#ff6e40"
        },
        "nav-link-selected": {"background-color": "#ff6e40"},
    }
)

def home():
    # Welcome card
    st.markdown("""
        <div class="custom-card">
            <h2 style="color: white; text-align: center; margin-bottom: 1.5rem;">
                Welcome to the Online Payment Fraud Detection System
            </h2>
            <p style="color: white; text-align: justify;font-size: 1.25rem;">
                Our system leverages advanced machine learning models to detect fraudulent online transactions 
                in real-time, helping businesses and individuals prevent financial losses due to fraud.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Key Features Carousel
    features = [
        {
            "title": "Individual Transaction Prediction",
            "description": "Analyze one transaction at a time and receive instant results.",
            "icon": "🔍"
        },
        {
            "title": "Batch Prediction",
            "description": "Upload a CSV file containing multiple transactions for bulk fraud detection.",
            "icon": "📊"
        },
        {
            "title": "History Log",
            "description": "Track past predictions, view detailed logs, and download them for further analysis.",
            "icon": "📜"
        },
        {
            "title": "Informative Resources",
            "description": "Learn about online payment fraud detection, tips on how to protect yourself from fraud, and prevention strategies.",
            "icon": "📚"
        }
    ]

    # Display features in a carousel-like layout
    st.markdown('<div class="features-carousel">', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, feature in enumerate(features):
        with cols[idx % 2]:
            st.markdown(f"""
                <div class="custom-card" style="height: 200px;">
                    <h3 style="color: white; text-align: center;">{feature['icon']} {feature['title']}</h3>
                    <p style="color: white; text-align: center; padding: 1rem;font-size: 1.2rem;">
                        {feature['description']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
     # Fraud Statistics Cards
    st.markdown("<h2 style='color: white; text-align: center; margin: 2rem 0;'>Online Fraud Statistics</h2>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: white; text-align: center;">💰 Global Impact</h3>
                <p style="color: white; text-align: center; font-size: 24px; font-weight: bold;">$32.39 Billion</p>
                <p style="color: white; text-align: center;">Expected card fraud losses globally by 2024</p>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        
            <div class="custom-card">
                <h3 style="color: white; text-align: center;">📈 Fraud Increase</h3>
                <p style="color: white; text-align: center; font-size: 24px; font-weight: bold;">44.7%</p>
                <p style="color: white; text-align: center;">Rise in online payment fraud since 2020</p>
            </div> 
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: white; text-align: center;">🔒 Prevention</h3>
                <p style="color: white; text-align: center; font-size: 24px; font-weight: bold;">80%</p>
                <p style="color: white; text-align: center;">Of fraud attempts can be prevented with proper detection</p>
            </div>
        """, unsafe_allow_html=True)

    # Fraud Prevention Tips
    st.markdown("""
        <div class="custom-card">
            <h3 style="color: white;">🛡️ Protect Yourself from Online Fraud</h3>
            <ul style="color: white; font-size: 1.2rem;">
                <li>Always use strong, unique passwords for different accounts</li>
                <li>Enable two-factor authentication whenever possible</li>
                <li>Monitor your accounts regularly for suspicious activity</li>
                <li>Never share sensitive information through unsecured channels</li>
                <li>Be cautious of unsolicited emails and messages</li>
                <li>Use secure and updated payment platforms</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Information cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
                    <a href="https://www.datavisor.com/wiki/real-time-monitoring/" target="_blank" style="text-decoration: none;">
            <div class="custom-card">
                <h3 style="color: white;">Real-time Detection</h3>
                <img src="data:image/png;base64,{}" style="width: 100%; border-radius: 5px; margin: 1rem 0;">
                <p style="color: white;font-size: 1.2rem;">
                    Our system processes transactions in real-time, providing instant fraud detection and alerts.
                    <br><u><b>Click me to learn more</b></u>
                </p>
            </div> </a>
        """.format(get_base64(os.path.join(os.path.dirname(__file__), 'scam.jpg'))), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
                    <a href="https://www.fraud.com/post/5-fraud-detection-methods-for-every-organization" target="_blank" style="text-decoration: none;">
            <div class="custom-card">
                <h3 style="color: white;">Strategies & Technologies</h3>
                <img src="data:image/png;base64,{}" style="width: 100%; border-radius: 5px; margin: 1rem 0;">
                <p style="color: white;font-size: 1.2rem;">
                    Include Machine Learning, Artificial Intelligence, Blockchain-based secure payment processing.
                    <br><u><b>Click me to learn more</b></u>
                </p>
            </div> </a>
        """.format(get_base64(os.path.join(os.path.dirname(__file__), 'detect.jpg'))), unsafe_allow_html=True)

    with col3:
        st.markdown("""
                    <a href="https://www.fraud.com/post/fraud-prevention" target="_blank" style="text-decoration: none;">
            <div class="custom-card">
                <h3 style="color: white;">Fraud Prevention</h3>
                <img src="data:image/png;base64,{}" style="width: 100%; border-radius: 5px; margin: 1rem 0;">
                <p style="color: white;font-size: 1.2rem;">
                    Involves measures using Artificial Intelligence, Machine Learning, biometrics to detect and prevent fraudulent activities.
                    <br><u><b>Click me to learn more</b></u>
                </p>
            </div> </a>
        """.format(get_base64(os.path.join(os.path.dirname(__file__), 'fr.jpg'))), unsafe_allow_html=True)
    with col4:
        st.markdown("""
                    <a href="https://www.fraud.com/post/fraud-data-analytics" target="_blank" style="text-decoration: none;">
            <div class="custom-card">
                <h3 style="color: white;">Advanced Analytics</h3>
                <img src="data:image/png;base64,{}" style="width: 100%; border-radius: 5px; margin: 1rem 0;">
                <p style="color: white;font-size: 1.2rem;">
                    Utilizing ML algorithms to analyze transaction patterns and detect fraudulent activities.
                    <br><u><b>Click me to learn more</b></u>
                </p>
            </div> </a>
        """.format(get_base64(os.path.join(os.path.dirname(__file__), 'fraud_bg.jpg'))), unsafe_allow_html=True)

    with col5:
        st.markdown("""
                    <a href="https://www.fraud.com/post/strong-customer-authentication" target="_blank" style="text-decoration: none;">
            <div class="custom-card">
                <h3 style="color: white;">Customer Authentication</h3>
                <img src="data:image/png;base64,{}" style="width: 100%; border-radius: 5px; margin: 1rem 0;">
                <p style="color: white;font-size: 1.2rem;">
                    Methods like passwords, two-factor authentication, OTPs and AI-powered risk-based authentication.
                    <br><u><b>Click me to learn more</b></u>
                </p>
            </div> </a>
        """.format(get_base64(os.path.join(os.path.dirname(__file__), 'fraud_detect.jpg'))), unsafe_allow_html=True)

def prediction():
    st.title("🔍 Fraud Prediction")
    option = st.selectbox("Choose Prediction Type", ["Individual Transaction", "Batch File Upload"])

    if "show_fraud_report_form" not in st.session_state:
        st.session_state.show_fraud_report_form = False

    if option == "Individual Transaction":
        st.subheader("Individual Transaction Prediction")

        # Initialize session state for form values if not exists
        if 'form_values' not in st.session_state:
            st.session_state.form_values = {
                'transaction_type': 'CASH_IN',
                'amount': 0.0,
                'old_balance': 0.0,
                'new_balance': 0.0,
            }
        if 'form_submitted' not in st.session_state:
            st.session_state.form_submitted = False

        # Reset function
        def reset_form():
            st.session_state.form_values = {
                'transaction_type': 'CASH_IN',
                'amount': 0.0,
                'old_balance': 0.0,
                'new_balance': 0.0,
            }
            st.session_state.transaction_type = 'CASH_IN'
            st.session_state.amount = 0.0
            st.session_state.old_balance = 0.0
            st.session_state.new_balance = 0.0
            st.session_state.form_submitted = False

        # Form container
        with st.form("prediction_form", clear_on_submit=False):
            transaction_type = st.selectbox(
                "Transaction Type",
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                key='transaction_type',
                index=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"].index(st.session_state.form_values['transaction_type']),
            )
            amount = st.number_input(
                "Transaction Amount",
                 
                
                value=st.session_state.form_values['amount'],
                key='amount',
                help="Enter transaction amount",
            )
            old_balance_orig = st.number_input(
                "Old Balance (Origin)",
                
                
                value=st.session_state.form_values['old_balance'],
                key='old_balance',
                help="Enter the account's old balance",
            )
            new_balance_orig = st.number_input(
                "New Balance (Origin)",
                
                
                value=st.session_state.form_values['new_balance'],
                key='new_balance',
                help="Enter the account's new balance",
            )
            # Buttons
            col1, col2 = st.columns(2)

            with col1:
                predict = st.form_submit_button("Predict", use_container_width=True, type="primary")

                if  predict:
                    if amount < 0 or old_balance_orig < 0 or new_balance_orig < 0:
                        st.warning("Values must not be negative. Please correct the inputs.")
                    else:
                        data = pd.DataFrame([[transaction_type, amount, old_balance_orig, new_balance_orig]],
                                            columns=["transaction_type", "amount", "old_balance_orig", "new_balance_orig"])
                        data['transaction_type'] = data['transaction_type'].map({
                            'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4
                        })
                        prediction = model.predict(data)[0]
                        is_fraud = "Fraudulent" if prediction == 1 else "Not Fraudulent"

                        # Save to history
                        st.session_state.history.append({
                            "timestamp": datetime.now(),
                            "type": "Individual",
                            "transaction_type": transaction_type,
                            "amount": amount,
                            "old_balance_orig": old_balance_orig,
                            "new_balance_orig": new_balance_orig,
                            "prediction": is_fraud
                        })

                        # Enhanced alert display
                        if is_fraud == 'Not Fraudulent':
                            st.markdown("""
                                
                                <div class="safe-alert">
                                    <i class="fas fa-check-circle"></i>
                                    <div>
                                        <h3>Safe Transaction</h3>
                                        <p>This transaction appears to be legitimate.</p>
                                    </div>
                                </div>
                                        
                            """, unsafe_allow_html=True)                            
                            st.session_state.show_fraud_report_form = False
                        else:
                            st.markdown("""
                                <div class="fraud-alert">
                                    <i class="fas fa-exclamation-circle"></i>
                                    <div>
                                        <h3>Fraudulent Transaction Detected</h3>
                                        <p>This transaction has been flagged as potentially fraudulent.</p>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                            st.session_state.show_fraud_report_form = True
            with col2:
                reset = st.form_submit_button("Reset Form", use_container_width=True, type="secondary", on_click=reset_form)
        # Fraud Report Form (Rendered Conditionally)
        if st.session_state.show_fraud_report_form:
            st.subheader("📋 Report Fraudulent Transaction")
            name = st.text_input("Your Name")
            account_number = st.text_input("Account Number")
            report_message = st.text_area("Detailed Description of Suspicious Activity")
            submit_report = st.button("Submit Report")
            if submit_report:
                if not name or not account_number or not report_message:
                    st.error("❌ Please fill in all fields to submit the report.")
                else:
                    # Mock report submission
                    st.success("✅ Report submitted successfully! Thank you for reporting. Our team will investigate this transaction.")
                    st.info(f"**Report Summary:**\n- **Name:** {name}\n- **Account Number:** {account_number}\n- **Message:** {report_message}")
                    st.session_state.show_fraud_report_form = False       

    elif option == "Batch File Upload":
        st.subheader("Batch File Upload Prediction")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            # Check if required columns exist
            required_columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceDest']
            if not all(col in df.columns for col in required_columns):
                st.error("Upload Error: Dataset must contain columns: type, amount, oldbalanceOrg, newbalanceDest")
                return None
            # Map transaction types
            type_mapping = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4, 0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1: 4}
            df['type'] = df['type'].map(type_mapping)

            # Predict
            predictions = model.predict(df[['type', 'amount', 'oldbalanceOrg', 'newbalanceDest']])
            df['isFraud'] = ["Fraudulent" if pred == 1 else "Not Fraudulent" for pred in predictions]

            # Fraud Statistics
            total_transactions = len(df)
            fraud_transactions = df[df['isFraud'] == 'Fraudulent']
            fraud_count = len(fraud_transactions)
            
            # Create two columns
            col1, col2 = st.columns(2)

            # Display in columns
            with col1:
                st.markdown(f"""<div class="custom-card"><h2>Total Transactions:\n</h2><h3>{total_transactions}</h3></div>""", unsafe_allow_html=True)

            with col2:
                st.markdown(f"""<div class="custom-card"><h2>Fraudulent Transactions:\n</h2><h3>{fraud_count}</h3></div>""", unsafe_allow_html=True)
            
            # Create columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(df, use_container_width= True, height= 450)
            with col2:
                # Pie Chart
                fig = px.pie(
                    values=[fraud_count, total_transactions - fraud_count], 
                    names=['Fraudulent', 'Legitimate'],
                    title='Transaction Fraud Distribution'
                )
                fig.update_layout(title_x=0.25)
                st.plotly_chart(fig)

            # Download buttons
            csv = df.to_csv(index=False).encode()
            st.download_button("Download Predicted CSV", data=csv, file_name="predictions.csv", mime="text/csv")

            st.session_state.history.append({
                "timestamp": datetime.now(),
                "type": "Batch",
                "file_name": uploaded_file.name,
                "num_records": len(df),
                "fraud_count": fraud_count
            })

    st.markdown('</div>', unsafe_allow_html=True)

def history():
    st.title("📜 Prediction History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Styling for history
        def color_fraud(val):
            if isinstance(val, str):
                if val == 'Not Fraudulent':
                    return 'background-color: green' 
                elif val == 'Fraudulent': 
                    return 'background-color: red'
            return ''
        
        styled_df = df.style.applymap(color_fraud)
        st.dataframe(styled_df)
        
        csv = df.to_csv(index=False).encode()
        st.download_button("Download History", csv, "history.csv", "text/csv")
    else:
        st.info("No history available.")
    st.markdown('</div>', unsafe_allow_html=True)

def about():
    st.markdown('<h1 style="color: white; text-align: center;">📘 About</h1>'
        '<h2 class="section-title">About Online Payment Fraud Detection</h2>', unsafe_allow_html=True)
    st.markdown("""
    ##### Fraudulent activities in online payments are on the rise due to the rapid growth of digital transactions. Cybercriminals use various techniques, such as phishing, identity theft, and malware, to exploit vulnerabilities in online payment systems. Protecting users and businesses from such fraud requires advanced tools and awareness.
    """)

    st.markdown("""### Features of Fraud Detection:""")

    # Create two columns
    col1, col2 = st.columns([1, 1])
    with col1:
        # Add text in the second column
        st.markdown("""
                    <div class="custom-card">
        •    Real-Time Analysis:<br>
            Fraudulent transactions are identified as they happen, minimizing potential losses. Real-time monitoring uses algorithms to detect anomalies in payment patterns.<br>
        •  Machine Learning:<br>
            Leveraging historical data, machine learning models can identify patterns and predict fraudulent behavior. These systems improve over time, adapting to new threats.<br>
        •  Multi-Layer Security:<br>  
            Modern fraud detection systems combine multiple security protocols, including encryption, tokenization, and biometric authentication, to ensure robust protection.<br>
        •  Risk Scoring:<br>
            Transactions are assigned risk scores based on factors like geolocation, transaction amount, and device used, helping to identify suspicious activities.<br>
        •  Behavioral Analytics:  <br>
            Monitoring user behavior, such as login patterns and spending habits, to flag deviations that may indicate fraud.</div>
         """,unsafe_allow_html=True)
    with col2:
        # Add an image related to fraud detection
        st.markdown('<img class="img" style="margin-left:50px;margin-right:40px ;height:500px;width:600px" src="https://5logistics.com/wp-content/uploads/Fraud-1.jpg" alt="Fraud Detection">', unsafe_allow_html=True)    

    # Create a new section for tips for staying protected during online transactions
    st.markdown("""### Tips for Staying Protected During Online Transactions:""")

    # Create two more columns for this section
    col3, col4 = st.columns([1, 1])

    with col3:
        st.markdown("""
                    <div class="custom-card">
        •  Verify Website Security: <br>
            Only transact on websites with HTTPS protocols. Look for a padlock icon in the browser's address bar.
            Avoid using public Wi-Fi for online transactions unless you're connected to a trusted VPN.<br>
        •  Enable Multi-Factor Authentication (MFA):<br>
            Add an extra layer of security by requiring a one-time password (OTP) or biometric authentication alongside your login credentials.<br>
        •  Keep Devices Updated:<br>
            Regularly update your operating system, browser, and payment apps to patch known vulnerabilities.<br>
        •  Monitor Bank Statements: <br>
            Review your bank statements and transaction history regularly to detect unauthorized activities early.<br>
        •  Avoid Phishing Scams: <br>
            Be cautious of emails, messages, or calls requesting sensitive information. Cybercriminals often pose as legitimate entities.</div>
         """,unsafe_allow_html=True)

    with col4:
        # Add an image for machine learning or analytics
        st.markdown('<img style="margin-left:50px;height:530px;width:600px" src="https://www.digipay.guru/static/24fb1b1f75d3f9ddb1373c2e1cebbd75/16546/online-payment-security-Image_04.png" alt="Fraud Detection">', unsafe_allow_html=True)
    
    # Raising awareness section
    st.markdown("""### Raising Awareness:""")
    
    col7, col8 = st.columns([1, 1])

    with col7:
        st.markdown("""
                <div class="custom-card">
        • Report Suspicious Activity: <br>
            Inform your bank or payment service provider immediately if you notice any unusual activity.<br>
        • Leverage Fraud Detection Tools: <br>
            Use services or tools that proactively monitor transactions and send alerts for suspicious activities.<br>
            By adopting these practices and leveraging advanced fraud detection tools, you can significantly minimize the risk of falling victim to online payment fraud.</div>
        """,unsafe_allow_html=True)

    with col8:
        st.markdown('<img style="margin-left:50px;height:300px;width:600px" src="https://blogimage.vantagefit.io/vfitimages/2021/06/MENTEL-HEALTH-AWARNESS-CELEBRATION--1.png" alt="Fraud Detection">', unsafe_allow_html=True)
    
    # Initialize session state for form
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    
    # Create form using Streamlit's form API
    with st.form(key='feedback_form'):
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: white; text-align: center;">Feedback Form</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Form inputs
        name = st.text_input("Name")
        email = st.text_input("Email")
        feedback = st.text_area("Feedback", height=100)
        
        submitted = st.form_submit_button(label="Submit Feedback")

    # Form validation and submission handling
    if submitted:
        if not name or not name.strip():
            st.error("Please enter your name.")
        elif not email or not email.strip():
            st.error("Please enter your email address.")
        elif "@" not in email or "." not in email:
            st.error("Please enter a valid email address.")
        elif not feedback or not feedback.strip():
            st.error("Please enter your feedback.")
        else:
            # Here you can add code to save the feedback to a database or send it via email
            st.success("Thank you for your feedback! We appreciate your input.")

# Main app routing
if selected == "Home":
    home()
elif selected == "Prediction":
    prediction()
elif selected == "History":
    history()
elif selected == "About":
    about()