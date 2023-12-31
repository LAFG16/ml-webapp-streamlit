import streamlit as st
from pickle import load
import pickle

st.markdown('<style>h1 { color: brown; }</style>', unsafe_allow_html=True)
# Website title
st.title("INSURANCE CALCULATOR")

# Value selection
sex_val = st.selectbox("Sex:",
                     ("Male", "Female")
                     )

age_val = st.slider("Age:",
                        min_value = 18,
                        max_value = 64
                        )

bmi_val = st.slider("Body mass index:",
                        min_value = 15.96,
                        max_value = 53.13
                        )

children_val = st.number_input("Number of childrens:",
                     min_value = 0,
                     max_value = 5
                     )

smoker_val = st.selectbox("Are you smoker?:",
                        ("Yes", "No")
                        )

region_val = st.selectbox("From what region are you?:",
                        ("Southwest", "Southeast", "Northwest", "Northeast")
                        )

# load factorized values
fact_val = load(open("/workspaces/ml-webapp-streamlit/data/processed/fact_val.pk", "rb"))

# prediction buttom
if st.button("prediction"):
    row = [age_val, 
           bmi_val,
           children_val, 
           (fact_val["smoker"][smoker_val.lower()]),
           fact_val["region"][region_val.lower()],
           fact_val["sex"][sex_val.lower()]
           ]
    

    # load scaler
    scaler = load(open("/workspaces/ml-webapp-streamlit/models/normalized_scaler.pk", "rb"))
    scal_row = scaler.transform([row])

    # load model
    model = load(open("/workspaces/ml-webapp-streamlit/models/linear_model.pk", "rb"))
    y_pred = model.predict([row])

    st.text("The cost of your insurance is: " + str(round(y_pred[0, 0], 2)))