from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import runnable
import time
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import shap

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash",
                             google_api_key=st.secrets["google_api_key"])

model = pickle.load(open("model.pkl", "rb"))

explainer = shap.TreeExplainer(model)
import streamlit as st
st.set_page_config(
    layout="wide"
)

st.markdown("""
<style>
.block-container{
    max-width: 1100px;
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title(":blue[Medical Insurance] Charges Prediction :blue-background[AI]")
st.write("Calculate estimated insurance costs based on personal parameters.")

col_left, col_right = st.columns([1.5,1], gap="large", border = True)
with col_left:
    bmi = st.number_input("📊 :blue[**BMI Index**]", min_value=16, max_value=50, value=18)
    age = st.number_input("🗓️ :blue[**Age (Years)**]", min_value=18, max_value=65, value=25, step=1)
    sex = st.pills("👤 :gray[Gender]", ["👨 Male", "👩 Female"], selection_mode="single")
    sex1 = sex
    smoker = st.pills(" :gray[Smoking Status]", ["🚭 No", "🚬 Yes"], selection_mode="single")
    smoker1 = smoker
    sex = 1 if sex == "👨 Male" else 0
    smoker = 1 if smoker == "🚬 Yes" else 0

    df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "smoker_numeric": smoker,
        "sex_numeric": sex
    }])

    with open("prompt_template.txt", "r") as f:
        template_text = f.read()

    template = PromptTemplate(
        template=template_text,
        input_variables=["shap_dict"]
    )

    if st.button("Predict"):

        # Prediction
        prediction = np.expm1(model.predict(df)[0])

        # st.metric(label="Estimated Insurance Charges",value=f"${prediction:,.2f}")
        with st.container(border=True):
            st.subheader("📊 :blue[Prediction Summary]")

            with st.container(border=True):
                c1, c2 = st.columns(2)
                c1.metric("Age", f"{age} yrs")
                c2.metric("BMI", f"{bmi}")

                st.divider()

                c1.caption(f"**Gender:** `{sex1}`")
                c2.caption(f"**Smoker:** `{smoker1}`")

            st.metric("💳 Estimated Insurance Charges", f"${prediction:,.2f}")

        # SHAP
        shap_values = explainer(df)

        shap_dict = dict(
            zip(df.columns, shap_values.values[0])
        )

        prompt = template.invoke({
            "shap_dict": shap_dict
        })

        response = llm.invoke(prompt)


        def generate_text(response):
            for word in response.text.split(" "):
                yield word + " "
                time.sleep(0.02)


        st.write_stream(generate_text(response))


with col_right:
    st.markdown("### ✨ Leo")
    st.caption("Your personal medical insurance assistant")
    with st.container(border=True):
        st.chat_message("Assistant").write(
            "👋 Hi! I'm Leo. Ask me anything about your insurance policy."
        )
        query = st.chat_input("Ask Assistant about your policy...")
        if query:
            with st.spinner("Searching..."):
                result = runnable.invoke(query)
            st.write(result)


