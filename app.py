import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
from sklearn.preprocessing import OrdinalEncoder


@st.cache_resource
# Load models
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)
    
def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
         .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-repeat: repeat;
            background-attachment: fixed;
            }}     
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local(r"12.png")

model_car=load_model("final_carmodel.pkl")

encoder_city=load_model("encoder_city.pkl")
encoder_Insurance_Validity=load_model("encoder_Insurance_Validity.pkl")
encoder_bt=load_model("encoder_bt.pkl")
encoder_ft=load_model("encoder_ft.pkl")
encoder_oem=load_model("encoder_oem.pkl")
encoder_model=load_model("encoder_model.pkl")
encoder_transmission=load_model("encoder_transmission.pkl")
encoder_variantName=load_model("encoder_variantName.pkl")

ml_df=pd.read_excel("ml_dl.xlsx")
st.title("Car Price Prediction App")

categorical_features = ["city", "ft", "bt", "transmission", "oem", "model", "variantName", "Insurance Validity"]
dropdown_options = {feature: ml_df[feature].unique().tolist() for feature in categorical_features}

tab1, tab2 = st.tabs(["Home", "Predict"])
with tab1:
    st.markdown("""
                **CarDekho Price Prediction App**
                
                *🚗 Welcome to the CarDekho Price Prediction App! 🚗

                This app helps users predict the selling price of a used car based on various factors such as year of purchase, present price, fuel type, transmission type, and more.

                🔍 How It Works:
                Enter car details like brand, model, year, kilometers driven, fuel type, and more.
                Click on Predict Price to get an estimated resale value.
                Make informed decisions before buying or selling a car!
                📊 Features:
                ✅ Machine Learning-based price prediction
                ✅ Interactive user-friendly interface
                ✅ Insights on car resale trends
                
                🚀 Try it now and get an instant price estimate!*
                """)
with tab2:
    a1,a2,a3=st.columns(3)
    a4,a5,a6=st.columns(3)
    a7,a8,a9=st.columns(3)
    a10,a11,a12=st.columns(3)
    a13,a14=st.columns(2)
    
    with a1:
        city_select=st.selectbox("Select City",dropdown_options["city"])
        city=encoder_city.transform([[city_select]])[0][0]
    with a2:
        ft_select=st.selectbox("Select fuel Type",dropdown_options["ft"])
        ft=encoder_ft.transform([[ft_select]])[0][0]
    with a3:
        bt_select=st.selectbox("Select Body Type",dropdown_options["bt"])
        bt=encoder_bt.transform([[bt_select]])[0][0]
    with a4:
        km=st.number_input("Enter KM driven",min_value=10)
    with a5:
        transmission_select=st.selectbox("Select Transmission",dropdown_options["transmission"])
        transmission=encoder_transmission.transform([[transmission_select]])[0][0]
    with a6:
        ownerNo=st.number_input("Enter no. of Owner's",min_value=1)
    with a7:
        oem_list=ml_df[ml_df["ft"]==ft_select]["oem"]
        oem_filtered=oem_list.unique().tolist()
        oem_select=st.selectbox("Select car manufacture name",oem_filtered)
        oem=encoder_oem.transform([[oem_select]])[0][0]
    with a8:
        model_list=ml_df[ml_df["oem"]==oem_select]["model"]
        model_filtered=model_list.unique().tolist()
        model_select=st.selectbox("Select car Model name",model_filtered)
        model=encoder_model.transform([[model_select]])[0][0]
    with a9:
        modelYear=st.number_input("Enter car manufacture year",min_value=1900)
    with a10:
        variantName_list=ml_df[ml_df["model"]==model_select]["variantName"]
        variantName_filtered=variantName_list.unique().tolist()
        variantName_select=st.selectbox("Select Model variant Name",variantName_filtered)
        variantName=encoder_variantName.transform([[variantName_select]])[0][0]
    with a11:
        Registration_Year=st.number_input("Enter car registration year",min_value=1900)
    with a12:
        InsuranceValidity_select=st.selectbox("Select Insurance Type",dropdown_options["Insurance Validity"])
        InsuranceValidity=encoder_Insurance_Validity.transform([[InsuranceValidity_select]])[0][0]
    with a13:
        Seats=st.number_input("Enter seat capacity",min_value=4)
    with a14:
        EngineDisplacement=st.number_input("Enter Engine CC",min_value=799)
        
    if st.button('Predict'):
        input_data = pd.DataFrame([city,ft,bt,km,transmission,ownerNo,oem,model,modelYear,variantName,Registration_Year,InsuranceValidity,Seats,EngineDisplacement])

        prediction = model_car.predict(input_data.values.reshape(1, -1))
                
        st.subheader("Predicted Car Price")
        st.markdown(f"### :green[₹ {prediction[0]:,.2f}]")
with tab3:
    @st.cache_data
    def load_car_data():
        return pd.read_excel(r"ml_dl.xlsx")
    def get_car_details_by_brand_or_model(name, df):
        df = df.dropna(subset=['oem', 'model'])  # Ensure 'oem' and 'model' have no NaN values

        # Normalize case for comparison
        name_lower = name.lower()
        df['oem'] = df['oem'].str.lower()
        df['model'] = df['model'].str.lower()

        # Check if name matches an OEM (brand)
        if name_lower in df['oem'].unique():
            filtered_cars = df[df['oem'] == name_lower]
            if filtered_cars.empty:
                return [{"message": f"No cars found for brand: {name}"}]
            return filtered_cars.head(5)[['oem', 'model', 'price', 'ft', 'transmission']].to_dict('records')

        # Check if name matches a model
        elif name_lower in df['model'].unique():
            filtered_cars = df[df['model'] == name_lower]
            if filtered_cars.empty:
                return [{"message": f"No cars found for model: {name}"}]
            return filtered_cars.head(5)[['oem', 'model', 'price', 'ft', 'transmission']].to_dict('records')

        return [{"message": f"No cars found for brand or model: {name}"}]

    st.header("Car Chatbot Assistant 💬")
    df = load_car_data()
            
    user_query = st.text_input("Ask me about cars!", "")

    if user_query:
            if "tell me about" in user_query.lower():
                brand_name = user_query.lower().replace("tell me about", "").strip()
                details = get_car_details_by_brand_or_model(brand_name, df)
                st.write("### Car Details")
                st.write(d for d in details)
            else:
                st.write("I'm still learning to answer more queries!")
