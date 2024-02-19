from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import torch.nn as nn

from travel.data.mistake_detection import MistakeDetectionExample
from travel.model.vqa import VQAResponse

@dataclass_json
@dataclass
class VQGOutputs:
    """Dataclass to hold all LM outputs from visual question generation (VQG)."""
    procedure_id: int
    procedure_description: str
    target_object: str
    questions: list[str]
    answers_str: list[str]
    answers: list[VQAResponse] = field(default_factory=list)
    
    def __post_init__(self):
        """Validation steps to ensure every QA-pair is valid and every question has an answer."""
        for answer in self.answers_str:
            try: 
                self.answers.append(VQAResponse[answer])
            except:
                raise ValueError(f"Unrecognized VQA answer could not be accepted by VQAResponse class: {answer}")
            
        assert len(self.questions) == len(self.answers), "VQGOutputs received mismatched number of questions and answers."
        
        for question in self.questions:
            if not question.strip().endswith("?"):
                print(f"Warning: Question '{question}' doesn't appear to be a question.")

# List of examples to use for in-context learning for VQG
# TODO: add more?
VQG_DEMONSTRATIONS = [
    VQGOutputs(
        procedure_id=-1,
        procedure_description='Remove pears from syrup and cool.',
        target_object='pears',
        questions=[
            "Are any pears in the syrup?",
            "Are the pears on a cool surface?"
        ],
        answers_str=[
            "No",
            "Yes"
        ]
    ),
    VQGOutputs(
        procedure_id=-1,
        procedure_description='Cut each tortilla with 2 1/2-inch cutter into 3 rounds, making 4 dozen mini tortillas.',
        target_object='tortillas',
        questions=[
            "Are there 4 dozen small tortillas?",
            "Are the tortillas about 2 1/2 inches?"
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),
    VQGOutputs(
        procedure_id=-1,
        procedure_description='In a medium bowl combine all ingredients excluding the cheese.',
        target_object='bowl',
        questions=[
            "Are there several ingredients in the bowl?",
            "Is there cheese in the bowl?"
        ],
        answers_str=[
            "Yes",
            "No"
        ]
    ),    
    VQGOutputs(
        procedure_id=-1,
        procedure_description='Place hash brown patties in a single layer in a greased 9 x 13 inch baking dish',
        target_object='baking dish',
        questions=[
            "Are there any hash brown patties stacked on top of each other in the baking dish?",
            "Is the baking dish greased?"
        ],
        answers_str=[
            "No",
            "Yes"
        ]   
]

VQG_PROMPT_TEMPLATE = 'The instructions say to "{instruction_step}". To visually verify that this procedure is complete, what are {n_questions} questions we could ask about an image of a target object and their expected answers?\n'
VQG_EXAMPLE_TEMPLATE = VQG_PROMPT_TEMPLATE + \
                       "Target object: {target_object}\n" + \
                       "{question_list}"
VQG_QUESTION_TEMPLATE = "{question_number}. {question} {answer}"

def generate_vqg_prompt(instruction_step: str) -> str:
    """
    Returns a prompt for VQG, i.e., for zero-shot inference or to come after several in-context demonstrations.

    :param instruction_step: Recipe or instruction step to generate instructions for. Should usually be a sentence in imperative form.
    :param n_questions: Number of questions expected for generation.
    :return: String including a prompt to generate `n_questions` questions to verify the success of `instruction_step`.
    """
    n_questions = len(VQG_DEMONSTRATIONS[0].questions) # TODO: maybe have a few possible options for different sizes of question sets?
    return VQG_PROMPT_TEMPLATE.format(instruction_step=instruction_step,
                                        n_questions=str(n_questions))

def generate_vqg_example(vqg_output: VQGOutputs) -> str:
    """
    Returns a full VQG prompt example for in-context learning.

    :param vqg_output: VQGOutputs object for VQG example.
    :return: String including a full demonstration of a prompt and several questions and expected answers for generating visual verification questions.
    """
    return VQG_EXAMPLE_TEMPLATE.format(instruction_step=vqg_output.procedure_description,
                                       n_questions = len(vqg_output.questions),
                                       target_object=vqg_output.target_object,
                                       question_list="\n".join([VQG_QUESTION_TEMPLATE.format(
                                            question_number=question_idx + 1,
                                            question=question,
                                            answer=answer.name
                                       ) for question_idx, (question, answer) in enumerate(zip(vqg_output.questions, vqg_output.answers))]))

# TODO: later may need to account for chat-based prompts
def generate_vqg_prompt_icl(procedure_description: str, n_demonstrations: int=3) -> str:
    """
    Returns a prompt for VQG including in-context demonstrations.

    :param procedure_description: String description of a procedure (e.g., recipe step) to generate visual questions for.
    :param n_demonstrations: Number of in-context demonstrations to include from `VQG_DEMONSTRATIONS`.
    """
    assert n_demonstrations <= len(VQG_DEMONSTRATIONS), f"Requested {n_demonstrations} in-context demonstrations for VQG, but only {len(VQG_DEMONSTRATIONS)} are available in travel.model.vqg.VQG_DEMONSTRATIONS."

    examples = [generate_vqg_example(demo) for demo in VQG_DEMONSTRATIONS[:n_demonstrations]]
    examples += [generate_vqg_prompt(procedure_description)]
    return "\n\n".join(examples)
