import argparse
import datetime
import json
import os
import pandas as pd
import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_source_relevance", type=str, help="Path to .json file with source data for relevance annotation.")
parser.add_argument("--data_source_informativeness", type=str, help="Path to .json file with source data for informativeness annotation.")
parser.add_argument("--n_annotators_per_file", type=int, help="Number of annotators to split source file across.")
args = parser.parse_args()

# Ensure arguments are provided
assert args.data_source_relevance is not None, "Relevance data source must be provided."
assert args.data_source_informativeness is not None, "Informativeness data source must be provided."
assert args.n_annotators_per_file is not None, "Number of annotators per file must be provided."

# Initialize session state for navigation and annotator ID
if "page" not in st.session_state:
    st.session_state.page = 0

if "annotator_idx" not in st.session_state:
    st.session_state.annotator_idx = None

if "task_type" not in st.session_state:
    st.session_state.task_type = None

def next_page():
    st.session_state.page += 1

# Page 1: Select Task Type and Enter Annotator ID
if st.session_state.page == 0:
    st.title("Select Task Type and Enter Annotator ID")

    task_type = st.selectbox(
        "Select the type of annotation task:",
        options=["Relevance", "Informativeness"]
    )
    st.session_state.task_type = task_type

    if st.session_state.annotator_idx is None:
        annotator_idx = st.number_input(
            "Enter your annotator ID:",
            min_value=0,
            max_value=args.n_annotators_per_file - 1,
            step=1
        )
        if st.button("Next"):
            # Get source data
            if task_type == "Relevance":
                data_path = args.data_source_relevance
            else:
                data_path = args.data_source_informativeness

            try:
                with open(data_path, "r") as f:
                    source_data = json.load(f)
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                st.stop()

            data_name = os.path.basename(data_path).replace(".json", "")
            output_dir = f"output_{data_name}"
            os.makedirs(output_dir, exist_ok=True)

            samples_per_annotator = len(source_data) // args.n_annotators_per_file
            samples = source_data[samples_per_annotator * annotator_idx: samples_per_annotator * (annotator_idx + 1)]
            if len(samples) == 0:
                st.error(f"Did not retrieve any samples for annotator ID {annotator_idx}. Please check your input.")
            else:
                st.session_state.annotator_idx = annotator_idx
                st.session_state.samples = samples
                next_page()
    else:
        st.number_input(
            "Enter your annotator ID:",
            min_value=0,
            max_value=args.n_annotators_per_file - 1,
            step=1,
            value=st.session_state.annotator_idx,
            disabled=True
        )

# Page 2: Annotation Task
if st.session_state.page == 1:
    st.title(f"{st.session_state.task_type} Annotation Task")

    if st.session_state.task_type == "Relevance":
        task_intro = (
            "You must rate how **relevant** the potential next question is. By relevant, we mean: "
            "**given the previous questions and answers, how helpful could an answer to this question "
            "be in determining whether you successfully completed the task?**"
        )
    else:
        task_intro = (
            "You must rate how **informative** the questions and answers are. By informative, we mean: "
            "**based on all the information you have, how sure are you about whether you succeeded?**"
        )

    st.write(task_intro)

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

        if st.session_state.task_type == "Relevance":
            st.write(f"**Potential next question:** *{sample['question']}*")
            rating_options = [
                "1 (very irrelevant)",
                "2 (slightly irrelevant)",
                "3 (neutral; may or may not be relevant)",
                "4 (slightly relevant)",
                "5 (very relevant)",
                "Instructions Unclear"
            ]
        else:
            st.write(f"**Last question:** *{sample['question']}*")
            st.write(f"**Last answer:** *{sample['answer']}*")
            rating_options = [
                "1 (very uninformative/unsure)", 
                "2 (slightly uninformative/unsure)", 
                "3 (neutral; may or may not be informative)", 
                "4 (slightly informative)", 
                "5 (very informative)", 
                "Instructions Unclear"
            ]

        rating = st.radio(
            "**Your rating (select one):**",
            options=rating_options,
            index=2,
            key=str(sample_idx)
        )

        ratings.append({
            "annotator_index": st.session_state.annotator_idx,
            "annotation_index": sample_idx,
            "procedure": sample['procedure'],
            "rating": rating
        })

        if st.session_state.task_type == "Relevance":
            ratings[-1]["potential_question"] = sample['question']
        else:
            ratings[-1]["last_question"] = sample['question']
            ratings[-1]["last_answer"] = sample['answer']

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
            subject=f"TRAVEl Annotation Results from Task {st.session_state.task_type}, Annotator {st.session_state.annotator_idx}",
            body="Please find the attached annotation results.",
            attachment=csv_data
        ):
            st.success("Results submitted successfully!")
        else:
            st.error("Failed to submit results. Please press Submit again.")
