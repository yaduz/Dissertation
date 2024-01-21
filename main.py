import streamlit as st
import pandas as pd
import joblib
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import requests

# Function to load and display information about Alzheimer's Disease
def about_alzheimers():
    st.title("About Alzheimer's Disease")
    
    st.write("Alzheimer's disease is a progressive brain disorder that affects memory, thinking, and behavior.")
    
    st.write("Key facts about Alzheimer's Disease:")
    
    st.write("1. Alzheimer's is the most common cause of dementia, a general term for a decline in cognitive ability severe enough to interfere with daily life.")
    
    st.write("2. The disease is named after Dr. Alois Alzheimer, who first described it in 1906.")
    
    st.write("3. Alzheimer's disease typically progresses slowly in three general stages - mild, moderate, and severe.")
    
    st.write("4. Common symptoms include memory loss, confusion, difficulty in completing familiar tasks, and changes in personality.")
    
    st.write("5. While there is currently no cure for Alzheimer's disease, early diagnosis and management can help improve the quality of life for affected individuals.")
    
    st.write("For more information and support, you can visit organizations and resources dedicated to Alzheimer's disease.")

    st.write("Resources:")
    st.write("- [Alzheimer's Association](https://www.alz.org/)")
    st.write("- [National Institute on Aging - Alzheimer's Disease Education and Referral Center](https://www.nia.nih.gov/)")

# Function to handle data collection
def data_collection():
    st.write("### Dataset Description")
    st.write("The dataset utilized for this research is sourced from the UCI Machine Learning Repository at [https://archive.ics.uci.edu/dataset/732/darwin](https://archive.ics.uci.edu/dataset/732/darwin).")
    st.write("Named the DARWIN (Diagnosis AlzheimeR WIth handwriting) dataset, it is tailored for the purpose of distinguishing Alzheimer’s disease patients from healthy individuals through handwriting analysis.")
    st.write("It contains a total of 25 sets of attributes, each corresponding to a different handwriting sample. These attributes include various measurements and features related to handwriting characteristics. Here’s a breakdown of some of the attributes:")

    st.write("• air_time1 to air_time25: Time spent in the air while writing.")
    st.write("• disp_index1 to disp_index25: Displacement index values.")
    st.write("• gmrt_in_air1 to gmrt_in_air25: Gesture motion ratio time values in the air.")
    st.write("• gmrt_on_paper1 to gmrt_on_paper25: Gesture motion ratio time values on paper.")
    st.write("• max_x_extension1 to max_x_extension25: Maximum extension in the X-axis.")
    st.write("• max_y_extension1 to max_y_extension25: Maximum extension in the Y-axis.")
    st.write("• mean_acc_in_air1 to mean_acc_in_air25: Mean acceleration in the air.")
    st.write("• mean_acc_on_paper1 to mean_acc_on_paper25: Mean acceleration on paper.")
    st.write("• mean_gmrt1 to mean_gmrt25: Mean gesture motion ratio time.")
    st.write("• mean_jerk_in_air1 to mean_jerk_in_air25: Mean jerk in the air.")
    st.write("• mean_jerk_on_paper1 to mean_jerk_on_paper25: Mean jerk on paper.")
    st.write("• mean_speed_in_air1 to mean_speed_in_air25: Mean speed in the air.")
    st.write("• mean_speed_on_paper1 to mean_speed_on_paper25: Mean speed on paper.")
    st.write("• num_of_pendown1 to num_of_pendown25: Number of pendown events.")
    st.write("• paper_time1 to paper_time25: Time spent on paper.")
    st.write("• pressure_mean1 to pressure_mean25: Mean pressure.")
    st.write("• pressure_var1 to pressure_var25: Pressure variance.")
    st.write("• total_time1 to total_time25: Total time taken (air_time + paper_time).")

    st.write("The dataset also includes a “class” column, which likely indicates the class or label associated with each handwriting sample, possibly indicating whether it’s related to Alzheimer’s disease or not.")
    st.write("Each set of attributes (e.g., air_time1 to air_time25) represents measurements taken during different instances of handwriting, which can be used for analyzing and potentially diagnosing Alzheimer’s disease based on handwriting patterns.")

    st.write("### Dataset Characteristics")
    st.write("The DARWIN dataset is structured in tabular form, with each row representing a distinct participant. It is designed for classification tasks, aiming to categorize individuals into two groups: Alzheimer’s disease patients and healthy individuals.")

    st.write("### Dataset Specifications")
    st.write("Notably, the dataset encompasses a total of 174 instances, each corresponding to a participant in the study. It comprises a substantial 451 features, essential for the analysis of handwriting patterns. Remarkably, the dataset exhibits no missing values, ensuring data integrity for subsequent analysis.")
    st.write("The DARWIN dataset was thoughtfully created to facilitate research aimed at enhancing machine learning methodologies for Alzheimer’s disease prediction via handwriting analysis. Its inception is driven by the urgent need for improved diagnostic tools, particularly for early detection. The absence of missing values further underscores its suitability for research, minimizing data imputation challenges and bolstering dataset reliability.")

# @st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('stacked_lr_model_1.pkl')
    return model

model = load_model()

# Function for making predictions
def prediction():
    st.title("Alzheimer's Disease Prediction")

    # Input fields with default values
    input_data = {
        'air_time17': st.number_input('Air Time', value=0.0, format="%.7f"),
        'disp_index17' : st.number_input('Dispersion Index', value=0.0, format="%.7f"),
        'gmrt_in_air17' : st.number_input('GMRT in Air', value=0.0, format="%.7f"),
        'gmrt_on_paper17' : st.number_input('GMRT on Paper', value=0.0, format="%.7f"),
        'max_x_extension17' : st.number_input('Max X Extension', value=0.0, format="%.7f"),
        'max_y_extension17' : st.number_input('Max Y Extension', value=0.0, format="%.7f"),
        'mean_acc_in_air17' : st.number_input('Mean Acceleration in Air', value=0.0, format="%.7f"),
        'mean_acc_on_paper17' : st.number_input('Mean Acceleration on Paper', value=0.0, format="%.7f"),
        'mean_gmrt17' : st.number_input('Mean GMRT1', value=0.0, format="%.7f"),
        'mean_jerk_in_air17' : st.number_input('Mean Jerk in Air', value=0.0, format="%.7f"),
        'mean_jerk_on_paper17' : st.number_input('Mean Jerk on Paper', value=0.0, format="%.7f"),
        'mean_speed_in_air17' : st.number_input('Mean Speed in Air', value=0.0, format="%.7f"),
        'mean_speed_on_paper17' : st.number_input('Mean Speed on Paper', value=0.0, format="%.7f"),
        'num_of_pendown17' : st.number_input('Number of Pen Downs', value=0.0, format="%.7f"),
        'paper_time17' : st.number_input('Paper Time', value=0.0, format="%.7f"),
        'pressure_mean17' : st.number_input('Pressure Mean', value=0.0, format="%.7f"),
        'pressure_var17': st.number_input('Pressure Variance', value=0.0, format="%.7f"),
        'total_time17' : st.number_input('Total Time', value=0.0, format="%.7f"),
    }

    # Creating columns for buttons
    col1, col2, col3 = st.columns(3)

    # Placing each button in a column
    with col1:
        predict_button = st.button("Predict")

    with col2:
        clear_button = st.button("Clear")

    with col3:
        cancel_button = st.button("Cancel")

    if predict_button:
        if all(value != 0.0 for value in input_data.values()):
            # Convert input data to DataFrame or required format
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df) # Replace with your prediction code
            probabilities = model.predict_proba(input_df)  # Get class probabilities
            classes = model.classes_  # Get class labels

            # Create a bar chart to visualize the probabilities
            fig, ax = plt.subplots()
            ax.pie(probabilities[0], labels=classes, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            # Display the chart in Streamlit
            st.pyplot(fig)
        
            # Display the prediction result
            prediction = model.predict(input_df)
            probability_present = probabilities[0][1]
            probability_healthy = probabilities[0][0]

            # Generate a response sentence based on the probabilities
            if probability_present > probability_healthy:
                response = "Based on the prediction, there is a higher likelihood of Alzheimer's disease presence."
            else:
                response = "Based on the prediction, the individual is more likely to be healthy."
            st.write(f"Prediction: {prediction[0]}")  # This assumes prediction is a single value, adjust as needed
            st.write(f"Probability of Alzheimer's Disease: {probability_present:.4f}")
            st.write(f"Probability of being Healthy: {probability_healthy:.4f}")
            st.write(response)
        else:
            st.error("Please fill in all the fields.")  

    if clear_button:
        # Code to clear all input fields
        for key in input_data.keys():
            input_data[key] = 0.0
        st.experimental_rerun()

    if cancel_button:
        # Redirect to the homepage
        st.experimental_rerun()

# Function for help or documentation
def help_section():
    st.title("Help & Documentation")

    st.write("### Welcome to the Alzheimer's Disease Prediction App Help Section")
    st.write("This section provides guidance and documentation to help you navigate and use the app effectively.")

    st.write("### App Overview")
    st.write("The Alzheimer's Disease Prediction App is designed to predict the likelihood of Alzheimer's disease based on handwriting analysis. It uses a machine learning model trained on the DARWIN dataset to make predictions.")
    
    st.write("### Navigation")
    st.write("The app has a sidebar on the left that allows you to navigate to different sections:")
    st.write("1. **Home**: Provides an introduction to the app.")
    st.write("2. **About Alzheimer Disease**: Offers information about Alzheimer's disease.")
    st.write("3. **Data Collection**: Explains the dataset used for prediction.")
    st.write("4. **Prediction**: Allows you to input handwriting data and receive predictions.")
    st.write("5. **Help**: You are here! This section provides assistance.")

    st.write("### Using the Prediction Section")
    st.write("To use the prediction section:")
    st.write("1. Fill in the input fields with handwriting data.")
    st.write("2. Click the 'Predict' button to get the prediction.")
    st.write("3. The app will display the prediction, probabilities, and a sentence based on the result.")
    
    st.write("### Clearing and Canceling")
    st.write("If you want to clear the input fields, click the 'Clear' button. To cancel and return to the homepage, click the 'Cancel' button.")
    
    st.write("### About the Dataset")
    st.write("The 'Data Collection' section provides detailed information about the dataset used for prediction. You can find information about dataset attributes, characteristics, and specifications.")
    
    st.write("### About Alzheimer's Disease")
    st.write("The 'About Alzheimer Disease' section offers insights into Alzheimer's disease and its importance in early detection.")

    st.write("### Additional Help")
    st.write("If you need further assistance or have questions, please feel free to reach out to our support team.")
    
    st.write("Thank you for using the Alzheimer's Disease Prediction App!")

# Main app function
def main():
    # Sidebar
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Select a page below", 
                            ["Home", 
                             "About Alzheimer Disease", 
                             "Data Collection", 
                             "Prediction", 
                             "Help"],
                            format_func=lambda x: {"Home": "🏠 Home",
                                                   "About Alzheimer Disease": "ℹ️ About Alzheimer Disease",
                                                   "Data Collection": "📊 Data Collection",
                                                   "Prediction": "🔮 Prediction",
                                                   "Help": "❓ Help"}[x])

    # Page routing
    if menu == "Home":
        st.title("Welcome to Alzheimer's Disease Prediction App")
        st.write("Description of the app...")
        st.write("This app is designed to predict the likelihood of Alzheimer's disease based on input data related to various factors.")
        st.write("You can use this app to enter relevant information and obtain predictions about the presence of Alzheimer's disease or the individual's health status.")
        st.write("Please navigate through the menu options on the left to learn more about Alzheimer's disease, collect data, make predictions, or access help and documentation.")
    elif menu == "About Alzheimer Disease":
        about_alzheimers()
    elif menu == "Data Collection":
        data_collection()
    elif menu == "Prediction":
        prediction()
    elif menu == "Help":
        help_section()

if __name__ == "__main__":
    main()
