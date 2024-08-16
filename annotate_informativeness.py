import argparse
import datetime
import json
import os
import pandas as pd
from pprint import pprint
import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_source_json", default=os.environ['data_source_json'] if 'data_source_json' in os.environ else None, type=str, help="Path to .json file with source data for annotation.")
parser.add_argument("--n_annotators", default=os.environ['n_annotators'] if 'n_annotators' in os.environ else None, type=int, help="Number of annotators to split source file across.")
args = parser.parse_args()

# Ensure arguments are provided
assert args.data_source_json is not None and args.n_annotators is not None

# Get source data
source_data = json.load(open(args.data_source_json, "r"))
data_name = args.data_source_json.split('/')[-1].replace(".json", "")
output_dir = f"output_{data_name}"
os.makedirs(output_dir, exist_ok=True)

# Initialize session state for annotator ID
if "annotator_idx" not in st.session_state:
    st.session_state.annotator_idx = None

def next_page():
    st.session_state.page += 1

# Annotator ID Entry and Validation
if st.session_state.annotator_idx is None:
    annotator_idx = st.number_input(
        "Enter your annotator ID:",
        min_value=0,
        max_value=args.n_annotators - 1,
        step=1
    )
    if st.button("Next"):
        assert len(source_data) % args.n_annotators == 0, "Length of annotated examples should be evenly divisible by number of annotators."
        samples_per_annotator = len(source_data) // args.n_annotators
        samples = source_data[samples_per_annotator * annotator_idx: samples_per_annotator * (annotator_idx + 1)]
        if len(samples) == 0:
            st.error(f"Did not retrieve any samples for annotator ID {annotator_idx}. Please check your input.")
        else:
            st.session_state.annotator_idx = annotator_idx
            st.session_state.samples = samples
else:
    st.number_input(
        "Enter your annotator ID:",
        min_value=0,
        max_value=args.n_annotators - 1,
        step=1,
        value=st.session_state.annotator_idx,
        disabled=True
    )

if "samples" in st.session_state:
    st.title("Annotation Task")

    st.write("""
    *Imagine you just had eye surgery, and are currently unable to see. You're performing a task you're familiar with, but need help to determine whether you successfully completed it. You video call a friend (who is unfamiliar with the task) and show them what you're working on. You then ask them some yes/no questions to figure out whether you successfully completed the task.*
    """)

    st.write("""
    For each annotation task, you will be given the following information:
    - A **sentence** describing the procedure you're trying to perform.
    - An list of **questions** you asked your friend, and their **answers**.
    """)

    st.write("""
    You must rate how **informative** the questions and answers are. By informative, we mean: **based on all the information you have, how sure are you about whether you succeeded?**
    """)

    st.write("""
    You can also choose to mark "Instructions Unclear", which means that the sentence itself is not clear, so you're not sure how to determine whether the procedure is successful. This should only be used in rare cases.
    """)

    st.write("""
    *Some tips:*
    - Your task is to rate how sure you are, NOT whether you believe the procedure is successfully completed or not.
    - Consider all questions and answers as a whole; if you have contradictory information, this may reduce your sureness.
    - The instructional text and questions may refer to "someone" or a "person"; always assume this is referring to yourself (the person performing the task).
    - The questions may refer to a "photo" or "image"; always assume this is referring to the video feed your friend would see through the video call.
    """)

    ratings = []
    for sample_idx, sample in enumerate(st.session_state.samples):
        st.write("---")
        st.write(f"### Annotation {sample_idx + 1}")
        st.write(f"**Sentence:** *{sample['procedure']}*")
        
        st.write("**Previous questions and answers:**")
        if len(sample['previous_questions_answers']) == 0:
            st.write("None")
        else:
            for q_idx, (q, a) in enumerate(sample['previous_questions_answers']):
                st.write(f"{q_idx+1}. *{q}*     (Answer: *{a}*)")

        st.write(f"**Last question:** *{sample['question']}*")
        st.write(f"**Last answer:** *{sample['answer']}*")

        rating = st.radio(
            "**Your rating (select one):**",
            options=[
                "1 (very uninformative/unsure)", 
                "2 (slightly uninformative/unsure)", 
                "3 (neutral; may or may not be informative)", 
                "4 (slightly informative)", 
                "5 (very informative)", 
                "Instructions Unclear"
            ],
            index=2,
            key=str(sample_idx)
        )

        ratings.append({
            "annotator_index": st.session_state.annotator_idx,
            "annotation_index": sample_idx,
            "procedure": sample['procedure'],
            "last_question": sample['question'],
            "last_answer": sample['answer'],
            "rating": rating
        })

    st.write("---")

    # Function to send email with attachment
    def send_email_with_attachment(to_email, from_email, subject, body, attachment):
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="annotation_results.csv"')

        msg.attach(part)

        # Set up the SMTP server
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)  # Example for Gmail, adjust as necessary
            server.starttls()
            server.login(from_email, st.secrets["email_password"])  # st.secrets["email_password"] should be set in Streamlit's secrets management
            text = msg.as_string()
            server.sendmail(from_email, to_email, text)
            server.quit()
            return True
        except Exception as e:
            st.error(f"Failed to send email: {e}")
            return False

    if st.button("Submit"):
        results_df = pd.DataFrame(ratings)
        csv_data = results_df.to_csv(index=False).encode()

        # Retrieve email details from Streamlit secrets
        to_email = st.secrets["mailto_address"]
        from_email = st.secrets["from_email"]

        if send_email_with_attachment(
            to_email=to_email,
            from_email=from_email,
            subject=f"TRAVEl Annotation Results from Annotator {st.session_state.annotator_idx}",
            body="Please find the attached annotation results.",
            attachment=csv_data
        ):
            st.success("Results submitted successfully!")
        else:
            st.error("Failed to submit results. Please press Submit again.")