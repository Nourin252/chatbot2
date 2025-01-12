import torch
print(torch.__version__)
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import dateparser  # For date parsing
from datetime import datetime

# Load the pre-trained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize session state variables
if 'appointment_info' not in st.session_state:
    st.session_state.appointment_info = {"doctor": None, "date": None}

# Function to validate and parse dates
def parse_date(user_input):
    parsed_date = dateparser.parse(user_input)
    if parsed_date and parsed_date >= datetime.now():
        return parsed_date.strftime("%Y-%m-%d")
    return None

# Healthcare chatbot function for Streamlit app
def healthcare_chatbot():
    st.title("MedAI Chatbot: Your Virtual Healthcare Assistant")

    # Introduction
    st.write("Hello! I am your virtual assistant. How can I help you today?")

    # Input for user queries
    user_input = st.text_input("You:", key="user_input")

    if user_input:
        # Appointment Booking Intent
        if re.search(r"appointment|book|schedule", user_input, re.IGNORECASE):
            if st.session_state.appointment_info["doctor"] and st.session_state.appointment_info["date"]:
                st.write(f"Your appointment with Dr. {st.session_state.appointment_info['doctor']} is scheduled for {st.session_state.appointment_info['date']}.")
            elif not st.session_state.appointment_info["doctor"]:
                st.write("Sure, I can help you book an appointment. Please provide the doctor's name.")
                doctor_name = st.text_input("Doctor's Name:", key="doctor_name")
                if doctor_name:
                    st.session_state.appointment_info["doctor"] = doctor_name
                    st.write(f"Got it. Dr. {doctor_name}. What date would you like to book?")
            elif not st.session_state.appointment_info["date"]:
                st.write("Please provide the preferred date for the appointment.")
                date_input = st.text_input("Preferred Date:", key="date_input")
                if date_input:
                    parsed_date = parse_date(date_input)
                    if parsed_date:
                        st.session_state.appointment_info["date"] = parsed_date
                        st.write(f"Your appointment with Dr. {st.session_state.appointment_info['doctor']} has been scheduled for {st.session_state.appointment_info['date']}.")
                    else:
                        st.write("I couldn't understand the date. Please provide a valid date (e.g., 'next Monday' or '2025-01-10').")

        # Symptoms or Health Advice Intent
        elif re.search(r"symptoms|advice|feeling unwell|cold|fever", user_input, re.IGNORECASE):
            st.write("I'm here to provide general advice, but please consult a doctor for a diagnosis.")
            st.write("What are your symptoms?")
            symptoms = st.text_input("Symptoms:", key="symptoms")
            if symptoms:
                st.write(f"Based on your symptoms ({symptoms}), I recommend resting, staying hydrated, and consulting a healthcare provider. If it's severe, please seek immediate medical attention.")

        # General Service Information Intent
        elif re.search(r"services|offerings|information", user_input, re.IGNORECASE):
            st.write("MedAI offers the following services:")
            st.write("1. Appointment scheduling")
            st.write("2. Teleconsultations")
            st.write("3. Health education resources")
            st.write("How else can I assist you?")

        # Fallback to AI Model for Freeform Responses
        else:
            input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
            bot_output = model.generate(
                input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            st.write(f"MedAI Chatbot: {response}")

if __name__ == "__main__":
    healthcare_chatbot()
