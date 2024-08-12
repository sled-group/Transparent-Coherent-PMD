import argparse
import datetime
import json
import os
import pandas as pd
from pprint import pprint
import streamlit as st

parser = argparse.ArgumentParser()
parser.add_argument("--data_source_json", default=os.environ['data_source_json'] if 'data_source_json' in os.environ else None, type=str, help="Path to .json file with source data for annotation.")
parser.add_argument("--n_annotators", default=os.environ['n_annotators'] if 'n_annotators' in os.environ else None, type=int, help="Number of annotators to split source file across.")
parser.add_argument("--annotator_idx", default=os.environ['annotator_idx'] if 'annotator_idx' in os.environ else None, type=int, help="Index of annotator this form will be for.")
args = parser.parse_args()

pprint(args)

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
- An optional list of **previous questions** you already asked, and **their answers**.
- The **last question** you just asked your friend, and **its answer**.
""")

st.write("""
You must rate how **informative** the last question and its answer are. By informative, we mean: compared to what you knew from the previous questions and answers, how much more sure would the last question and answer make you about whether you succeeded?
""")

st.write("""
You can also choose to mark "Instructions Unclear", which means that the instructional text itself is not clear, so you're not sure how to determine whether it's successful. This should only be used in rare cases.
""")

st.write("""
*Some tips:*
- Only judge the informativeness of the last question and answer, not the previous questions and answers (which may or may not be informative).
- A question may seem relevant to the task at hand, but you should consider it uninformative if it doesn't provide essential information to judge the success of the task.
- If a seemingly informative question is redundant with previous questions, you may consider it less informative.
- If the last answer contradicts critical information you had from previous questions and answers, you may consider it more informative.
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

    st.write(f"**Last question:** *{sample['question']}*")
    st.write(f"**Last answer:** *{sample['answer']}*")

    rating = st.radio(
        "**Your rating (select one):**",
        options=[
            "1 (very informative)", 
            "2 (slightly informative)", 
            "3 (neutral)", 
            "4 (slightly uninformative)", 
            "5 (very uninformative)", 
            "Instructions Unclear"
        ],
        index=2,
        key=str(sample_idx)
    )

    ratings.append({
        "annotator_index": args.annotator_idx,
        "annotation_index": sample_idx,
        "procedure": sample['procedure'],
        "last_question": sample['question'],
        "last_answer": sample['answer'],
        "rating": rating
    })

st.write("---")

# Save the results
if st.button("Submit"):
    results_df = pd.DataFrame(ratings)
    timestamp = datetime.datetime.now()
    output_file = os.path.join(output_dir, f"response_{timestamp.strftime('%Y%m%d%H%M%S')}.csv")
    results_df.to_csv(output_file, index=False)
    st.success(f"Annotations saved to {output_file}")
