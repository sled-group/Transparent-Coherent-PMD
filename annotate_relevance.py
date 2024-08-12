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
parser.add_argument("--data_source_json", default=os.environ['data_source_json'] if 'data_source_json' in os.environ else None, type=str, help="Path to .json file with source data for annotation.")
parser.add_argument("--n_annotators", default=os.environ['n_annotators'] if 'n_annotators' in os.environ else None, type=int, help="Number of annotators to split source file across.")
parser.add_argument("--annotator_idx", default=os.environ['annotator_idx'] if 'annotator_idx' in os.environ else None, type=int, help="Index of annotator this form will be for.")
args = parser.parse_args()

# Ensure arguments are provided
assert args.data_source_json is not None and args.n_annotators is not None and args.annotator_idx is not None

# Get source data
source_data = json.load(open(args.data_source_json, "r"))
samples = source_data[args.n_annotators * args.annotator_idx: args.n_annotators * (args.annotator_idx + 1)]
if len(samples) == 0:
    raise ValueError(f"Did not retrieve any samples: {str(samples)} loaded from {args.data_source_json}")
data_name = args.data_source_json.split('/')[-1].replace(".json", "")
output_dir = f"output_{data_name}_annotator{args.annotator_idx+1}of{args.n_annotators}"
os.makedirs(output_dir, exist_ok=True)

# Streamlit app
st.title("Annotation Task")

st.write("""
*Imagine you just had eye surgery, and are unable to see. You're performing a task you're familiar with, but need help to determine whether you successfully completed it. You video call a friend (who is unfamiliar with the task) and show them what you're working on. You then ask them some yes/no questions to figure out whether you successfully completed the task.*
""")

st.write("""
For each annotation task, you will be given the following information:
- A **sentence** describing the procedure you're trying to perform.
- An optional list of **previous questions** you already asked, and their answers.
- A **potential next question** you could ask your friend.
""")

st.write("""
You must rate how **relevant** the potential next question is. By relevant, we mean: given the previous questions and answers, how helpful could an answer to this question be in determining whether the procedure has been completed?
""")

st.write("""
You can also choose to mark "Instructions Unclear", which means that the instructional text itself is not clear, so you're not sure how to determine whether it's successful. This should only be used in rare cases.
""")

st.write("""
*Some tips:*
- Only judge the relevance of the potential next question, not the previous questions (which may or may not be relevant).
- A question may seem relevant to the task at hand, but you should consider it irrelevant if it doesn't provide essential information to judge the success of the task.
- If a seemingly relevant question is redundant with previous questions, you may consider it less relevant.
- Assume that the answer to the question won't contradict the information you have from previous questions and answers. If previous questions and answers already contradict each other, consider whether this question could sway you one way or another.
- The instructional text and questions may refer to "someone" or a "person"; always assume this is referring to yourself (the person performing the task).
- The questions may refer to a "photo" or "image"; always assume this is referring to the video feed your friend would see through the video call.
""")

ratings = []
for sample_idx, sample in enumerate(samples):
    st.write("---")
    st.write(f"### Annotation {sample_idx + 1}")
    st.write(f"**Sentence:** *{sample['procedure']}*")
    
    st.write("**Previous questions and answers:**")
    if len(sample['previous_questions_answers']) == 0:
        st.write("None")
    else:
        for q_idx, (q, a) in enumerate(sample['previous_questions_answers']):
            st.write(f"{q_idx+1}. *{q}*     (Answer: *{a}*)")

    st.write(f"**Potential next question:** *{sample['question']}*")

    rating = st.radio(
        "**Your rating (select one):**",
        options=[
            "1 (very irrelevant)", 
            "2 (slightly irrelevant)", 
            "3 (neutral)", 
            "4 (slightly relevant)", 
            "5 (very relevant)", 
            "Instructions Unclear"
        ],
        index=2,
        key=str(sample_idx)
    )

    ratings.append({
        "annotator_index": args.annotator_idx,
        "annotation_index": sample_idx,
        "procedure": sample['procedure'],
        "potential_question": sample['question'],
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
        subject=f"TRAVEl Annotation Results from Annotator {args.annotator_idx + 1}",
        body="Please find the attached annotation results.",
        attachment=csv_data
    ):
        st.success("Results submitted successfully!")
    else:
        st.error("Failed to submit results. Please press Submit again.")