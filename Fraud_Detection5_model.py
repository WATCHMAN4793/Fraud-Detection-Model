import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the saved model
loaded_model = pickle.load(open("fraud_detection5_model.sav", "rb"))

# LabelEncoder for the 'type' column (assuming it's categorical like 'CASH_IN', 'CASH_OUT', etc.)
le = LabelEncoder()
transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']  # list the types in your data
le.fit(transaction_types)  # Fit the encoder to known types

# Title for the Streamlit app
st.title("Fraud Detection Web App")

# Description of the app
st.write("This web application predicts whether a transaction is fraudulent or not based on provided details.")

# Input fields for user to enter transaction details
transaction_type = st.selectbox("Transaction Type", transaction_types)  # Dropdown for transaction type
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.1)
oldbalanceOrg = st.number_input("Old Balance of Origin Account", min_value=0.0, step=0.1)
oldbalanceDest = st.number_input("Old Balance of Destination Account", min_value=0.0, step=0.1)

# Convert the input data into a format suitable for the model
input_data = [
    le.transform([transaction_type])[0],  # Encode the 'type' field
    amount,                               # Amount
    oldbalanceOrg,                        # Old balance of the origin account
    oldbalanceDest                        # Old balance of the destination account
]

# Display the input data for confirmation
# st.write("Input Data:", input_data)

# When the user presses the 'Predict' button, make a prediction
if st.button('Predict'):
    # Convert input data into numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the data to match the model's expected input
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make the prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Output the prediction result
    if prediction[0] == 0:
        st.write("Result: No Fraudulent Activity")
    else:
        st.write("Result: Fraudulent Activity Detected")