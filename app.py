
import pandas as pd
import streamlit as st
import pickle
from pre_process import pre_process

suggested_description = [
    'OUTGOING WIRE 2108370 757118',
    'WEB TFR FR 000275674445       FEB 24 WEB PORTAL FEE           120604009529',
    'PREAUTHORIZED ACH DEBIT',
    'PREAUTHORIZED ACH CREDIT'
]
suggested_description = sorted(suggested_description)

with open("model.pkl", 'rb') as file:
    model_pkl = pickle.load(file)

with open("vectorizer.pkl", 'rb') as file:
    vectorizer_pkl = pickle.load(file)

st.image('ss.png', use_column_width=True)

menu = ['Description Classification', 'About Us']
choice = st.sidebar.selectbox('Menu', menu)

# Initialize session state
if 'tab' not in st.session_state:
    st.session_state.tab = 'Input one description'

if choice == 'Description Classification':
    st.subheader("Description Classification")
    type = st.radio("", options=["Input one description", "Input multiple descriptions", "Upload description file"], key='tab')

    if type == "Input one description":
        st.markdown("### Input description to text area ###")
        suggestion = st.selectbox('Choose a suggested description or "Manual Input"',
                                  options=suggested_description + ["Manual Input"])

        if suggestion == "Manual Input":
            customer_description = st.text_input('Description')
        else:
            customer_description = suggestion

        if st.button('Predict'):
            customer_description = pre_process(customer_description)
            customer_description = vectorizer_pkl.transform([customer_description]).toarray()
            pred = model_pkl.predict(customer_description)
            prediction_result = 'Credit' if pred[0] == 1 else 'Debit'
            st.markdown(f'**Prediction:** {prediction_result}')

    elif type == "Input multiple descriptions":
        st.markdown("### Input descriptions to text area ###")
        description_df = pd.DataFrame(columns=["Description"])

        descriptions = []
        for i in range(5):
            description = st.text_area(f"Description {i+1}:")
            if description:
                descriptions.append({"Description": description})

        if descriptions:
            description_df = pd.concat([description_df, pd.DataFrame(descriptions)], ignore_index=True)
            description_df = description_df[description_df["Description"].str.len() >= 2]

            if st.button('Predict'):
                description_df['clean_input'] = description_df['Description'].apply(pre_process)
                test_vec = vectorizer_pkl.transform(description_df['clean_input']).toarray()
                pred = model_pkl.predict(test_vec)
                description_df['Predict'] = pred
                description_df['Predict'] = description_df['Predict'].replace({1: 'Credit', 0: 'Debit'})
                description_df = description_df[['Description', 'Predict']]
                st.write(description_df)

    elif type == "Upload description file":
        st.markdown("### Upload description file ###")
        uploaded_file = st.file_uploader("Please upload 'csv' or 'xlsx' file", type=["csv", "xlsx"])

        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                description_df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                description_df = pd.read_excel(uploaded_file)

            first_column_name = description_df.columns[0]
            description_df.rename(columns={first_column_name: "Description"}, inplace=True)

            description_df['clean_input'] = description_df['Description'].apply(pre_process)
            test_vec = vectorizer_pkl.transform(description_df['clean_input']).toarray()
            pred = model_pkl.predict(test_vec)
            description_df['Predict'] = pred
            description_df['Predict'] = description_df['Predict'].replace({1: 'Credit', 0: 'Debit'})
            description_df = description_df[['Description', 'Predict']]
            st.write(description_df)

elif choice == 'About Us':
    st.subheader("About Us")

    st.write("<br>", unsafe_allow_html=True)
    
    st.markdown('''
    **Silversea Analytics**

    At Silversea Analytics, we specialize in providing cutting-edge data analysis and machine learning solutions tailored to meet the unique needs of our clients.

    **Our Projects**
    Our team has successfully completed numerous projects, leveraging the latest in data science and machine learning technologies to deliver actionable insights and innovative solutions.

    **Our Mission**
    Our mission is to empower businesses with the tools and knowledge they need to harness the power of their data and achieve their strategic objectives.

    **Contact Us**
    For more information about our services, please visit our [website](https://silversea-analytics.com/) or get in touch with us directly via:
    - **Phone:** +1 361-425-6123
    - **Email:** info@silversea-analytics.com


    ''')

    st.write("<br><br><br>", unsafe_allow_html=True)
