from dataclasses import dataclass, field, asdict
import os
import json
import random
from typing import Any, Optional

from travel.data.vqa import VQAResponse

@dataclass
class VQGInputs:
    """Dataclass to hold all LM inputs for visual question generation (VQG)."""
    procedure_id: int
    procedure_description: str
    prompt: str

    def to_dict(self):
        return asdict(self)
    
    @staticmethod
    def from_dict(data):
        return VQGInputs(**data)

@dataclass
class VQGOutputs:
    """Dataclass to hold all LM outputs from visual question generation (VQG)."""
    procedure_id: int
    procedure_description: str
    questions: list[str]
    answers_str: list[str]
    answers: list[VQAResponse] = field(default_factory=list)
    target_object: Optional[str] = None
    
    def __post_init__(self):
        """Validation steps to ensure every QA-pair is valid and every question has an answer."""
        # Clear answers (they may already populated if loading from a file, but need to double check that answers can be parsed to VQAResponse)
        self.answers = []

        for answer in self.answers_str:
            try: 
                self.answers.append(VQAResponse[answer])
            except:
                raise ValueError(f"Unrecognized VQA answer could not be accepted by VQAResponse class: {answer}")
            
        assert len(self.questions) == len(self.answers), "VQGOutputs received mismatched number of questions and answers."
        
        for question in self.questions:
            if not question.strip().endswith("?"):
                print(f"Warning: Question '{question}' doesn't appear to be a question.")
    def to_dict(self):
        return asdict(self)

# List of examples to use for in-context learning for VQG
# Examples are arbitrarily selected from Ego4D training data, but intend to cover the state change space in simulators, e.g., those annotated in PIGLeT data: https://aclanthology.org/2021.acl-long.159.pdf
# - Receptacle/location
# - Temperature
# - Broken
# - Cooked
# - Dirty
# - Filled with liquid / used up
# - Open
# - Picked up
# - Sliced
# - Toggled
VQG_DEMONSTRATIONS = [
    VQGOutputs(
        procedure_id=540088,
        procedure_description="Soak the sponge in a soapy water with your hands",
        questions=[
            "Is the sponge in the water?",
            "Is the water soapy?",
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),
    VQGOutputs(
        procedure_id=600088,
        procedure_description="Open the bottle",
        questions=[
            "Is the bottle open?",
            "Does the bottle have a lid on it?",
        ],
        answers_str=[
            "Yes",
            "No"
        ]
    ),
    VQGOutputs(
        procedure_id=107275,
        procedure_description="Take the baking tray away from the table",
        questions=[
            "Is the baking tray on the table?",
            "Is the baking tray somewhere not on the table?",
        ],
        answers_str=[
            "No",
            "Yes"
        ]
    ),
    VQGOutputs(
        procedure_id=357010,
        procedure_description="Turn on a torch light",
        questions=[
            "Is the torch light powered on?",
            "Is the torch light lit up?"
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),
    VQGOutputs(
        procedure_id=426130,
        procedure_description="Fold the right edge of the wrapper",
        questions=[
            "Is the wrapper completely flat?",
            "Is the right edge of the wrapper folded?",
        ],
        answers_str=[
            "No",
            "Yes"
        ]
    ),
    VQGOutputs(
        procedure_id=730006,
        procedure_description="Pour the water into the blue container",
        questions=[
            "Is there water in the blue container?",
            "Is the blue container empty?",
        ],
        answers_str=[
            "Yes",
            "No"
        ]
    ),
    VQGOutputs(
        procedure_id=83057,
        procedure_description="Paint the patio with the paint brush",
        questions=[
            "Is the patio painted?",
            "Is someone holding a paint brush?",
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),
    VQGOutputs(
        procedure_id=412036,
        procedure_description="Spread the black peas on the salad with the spoon in your hand",
        questions=[
            "Are the black peas on the salad?",
            "Is there a spoon in someone's hand?",
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),
    VQGOutputs(
        procedure_id=80062,
        procedure_description="Scoop paint from the pallet on the table with the paint brush",
        questions=[
            "Is there paint on the paint brush?",
            "Is the paint brush in someone's hand?",
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),   
    VQGOutputs(
        procedure_id=540138,
        procedure_description="Wash the car with a sponge in your hand",
        questions=[
            "Is the car clean?",
            "Is the sponge being held?",
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),
    VQGOutputs(
        procedure_id=599053,
        procedure_description="Pick the scrubber from the sink",
        questions=[
            "Is the scrubber in the sink?",
            "Is the scrubber in someone's hand?",
        ],
        answers_str=[
            "No",
            "Yes"
        ]
    ), 
    VQGOutputs(
        procedure_id=404040,
        procedure_description="Peel the onion",
        questions=[
            "Is the onion's skin removed?",
            "Is the onion peeled?",
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),       
    VQGOutputs(
        procedure_id=496081,
        procedure_description="Put the dirt in the dust bin",
        questions=[
            "Is there dirt in the dust bin?",
            "Is there any dirt that is not in the dust bin?",
        ],
        answers_str=[
            "Yes",
            "No"
        ]
    ),
    VQGOutputs(
        procedure_id=581090,
        procedure_description="Cut dough into two",
        questions=[
            "Is the dough in two pieces?",
            "Is the dough whole?",
        ],
        answers_str=[
            "Yes",
            "No"
        ]
    ),   
    VQGOutputs(
        procedure_id=513269,
        procedure_description="Break the walnut with the nutcracker in your hand",
        questions=[
            "Is there a cracked nut?",
            "Is the nut cracker being held in someone's hand?",
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),   
    # VQGOutputs(
    #     procedure_id=168002,
    #     procedure_description="Drop the brush in your hand on the oven",
    #     questions=[
    #         "Is the brush on the oven?",
    #         "Is the brush in a hand?"
    #     ],
    #     answers_str=[
    #         "Yes",
    #         "No"
    #     ]
    # ),
    VQGOutputs(
        procedure_id=188064,
        procedure_description="Turn off the tap",
        questions=[
            "Is the water running?",
            "Is the faucet switched off?",
        ],
        answers_str=[
            "No",
            "Yes"
        ]
    ),   
    VQGOutputs(
        procedure_id=523017,
        procedure_description="Heat the edge of the bag with the lighter",
        questions=[
            "Is there a flame coming from the lighter?",
            "Is the lighter near the bag?",
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),   
    VQGOutputs(
        procedure_id=321040,
        procedure_description="Close the fridge",
        questions=[
            "Is the fridge open?",
            "Can you see inside the fridge?",
        ],
        answers_str=[
            "No",
            "No"
        ]
    ),       
    VQGOutputs(
        procedure_id=25084,
        procedure_description="Chop green beans with a knife on the chopping board",
        questions=[
            "Are the green beans on the cutting board sliced?",
            "Is someone using a knife?",
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),       
    VQGOutputs(
        procedure_id=349047,
        procedure_description="Arrange the dumplings on the tray",
        questions=[
            "Are there dumplings on the tray?",
            "Are there any dumplings that are not on the tray?",
        ],
        answers_str=[
            "Yes",
            "No"
        ]
    ),
]

VQG_DEMONSTRATIONS_OLD = [
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
    ),
    VQGOutputs(
        procedure_id=449062,
        procedure_description="Loosen the soil in the flower pot with your hand",
        questions=[
            "Is the soil in the flower pot loose?",
            "Is the hand dirty?",
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    ),
    VQGOutputs(
        procedure_id=-1,
        procedure_description='In a bowl, beat eggs, milk, salt and mustard.',
        target_object='bowl',
        questions=[
            "Does the bowl contain eggs, milk, salt, and mustard?",
            "Is the bowl mixed?"
        ],
        answers_str=[
            "Yes",
            "Yes"
        ]
    )    
]

N_GENERATED_QUESTIONS = len(VQG_DEMONSTRATIONS[0].questions)

VQG_PROMPT_TEMPLATE = 'The instructions say "{instruction_step}". To visually verify that this procedure is complete, what are {n_questions} questions we could ask about an image of the scene and their expected answers?\n'
VQG_EXAMPLE_TEMPLATE = VQG_PROMPT_TEMPLATE + \
                       "{question_list}"
VQG_QUESTION_TEMPLATE = "{question_number}. {question} {answer}"

def generate_vqg_prompt(instruction_step: str) -> str:
    """
    Returns a prompt for VQG, i.e., for zero-shot inference or to come after several in-context demonstrations.

    :param instruction_step: Recipe or instruction step to generate instructions for. Should usually be a sentence in imperative form.
    :return: String including a prompt to generate `n_questions` questions to verify the success of `instruction_step`.
    """
    return VQG_PROMPT_TEMPLATE.format(instruction_step=instruction_step,
                                      n_questions=str(N_GENERATED_QUESTIONS))

def generate_vqg_example(vqg_output: VQGOutputs) -> str:
    """
    Returns a full VQG prompt example for in-context learning.

    :param vqg_output: VQGOutputs object for in-context VQG example.
    :return: String including a full demonstration of a prompt and several questions and expected answers for generating visual verification questions.
    """
    return VQG_EXAMPLE_TEMPLATE.format(instruction_step=vqg_output.procedure_description,
                                       n_questions = len(vqg_output.questions),
                                       question_list="\n".join([VQG_QUESTION_TEMPLATE.format(
                                            question_number=question_idx + 1,
                                            question=question,
                                            answer=answer.name
                                       ) for question_idx, (question, answer) in enumerate(zip(vqg_output.questions, vqg_output.answers))]))

def save_vqg_inputs(inputs: list[VQGInputs], path: str):
    """
    Saves generated VQG inputs to a json file.
    
    :param inputs: List of VQGInputs holding prompts for VQG.
    :param path: Filename ending with .json to save inputs at.
    """
    assert path.endswith(".json"), "save_vqg_inputs expects a json filename!"
    json.dump(
        [inp.to_dict() for inp in inputs],
        open(path, "w"),
        indent=4
    )

def load_vqg_inputs(path: str) -> list[VQGInputs]:
    """
    Loads generated VQG inputs from a json file.

    :param path: json filename to load from.
    """
    return [VQGInputs.from_dict(inp) for inp in json.load(open(path, "r"))]

def generate_vqg_prompt_icl(procedure_description: str, n_demonstrations: int=3) -> str:
    """
    Returns a prompt for VQG including in-context demonstrations.

    :param procedure_description: String description of a procedure (e.g., recipe step) to generate visual questions for.
    :param n_demonstrations: Number of in-context demonstrations to include from `VQG_DEMONSTRATIONS`.
    :return: Prompt for VQG including in-context demonstrations.
    """
    assert n_demonstrations <= len(VQG_DEMONSTRATIONS), f"Requested {n_demonstrations} in-context demonstrations for VQG, but only {len(VQG_DEMONSTRATIONS)} are available in travel.model.vqg.VQG_DEMONSTRATIONS."
    demonstrations = VQG_DEMONSTRATIONS
    random.shuffle(demonstrations) # Shuffle demonstrations for each prompt to ensure the ordering is not sub-optimal
    demonstrations = demonstrations[:n_demonstrations] # We'll randomly select n_demonstrations of the available demonstrations
    examples = [generate_vqg_example(demo) for demo in demonstrations]
    examples += [generate_vqg_prompt(procedure_description)]
    return "\n\n".join(examples)

def parse_vqg_outputs(generated_language: str, procedure_id: int, procedure_description: str, expected_n_questions: int=2) -> VQGOutputs:
    """
    Converts generated questions and answers into a VQGOutputs object.

    :param generated_language: Text generated by an LM which matches the format of prompt templates in this file.
    :param procedure_id: Int identifier for procedure (e.g., recipe or instruction step).
    :param procedure_description: Procedure description in text (e.g., recipe or instruction step text).
    :return: VQGOutputs parsed from generated language.
    """
    questions_answers = [(q_a.split("?")[0].strip() + "?", q_a.split("?")[1].strip()) for q_a in generated_language.split("\n")[0:expected_n_questions]] # NOTE: only extract k=2 questions and answers; can adjust this as needed later
    questions = [q[2:].strip() for q, _ in questions_answers]          
    answers = [a for _, a in questions_answers]
    output = VQGOutputs(procedure_id=procedure_id,
                        procedure_description=procedure_description,
                        questions=questions,
                        answers_str=answers)
    return output

def save_vqg_outputs(vqg_outputs: dict[Any, VQGOutputs], path: str):
    """
    Saves dict of VQGOutputs created by `run_vqg.py`.
    
    :param vqg_outputs: Dictionary mapping procedure ID int to VQGOutputs.
    :param path: Path to save json file (directory).
    """
    if not path.endswith(".json"):
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, "vqg_outputs.json")
    else:
        if not os.path.exists("/".join(path.split("/")[:-1])):
            os.makedirs("/".join(path.split("/")[:-1]))
    json.dump({k: v.to_dict() if v is not None else v for k, v in vqg_outputs.items()}, 
              open(path, "w"),
              indent=4)    

def load_vqg_outputs(path: str) -> dict[Any, VQGOutputs]:
    """
    Loads dict of VQGOutputs created by `run_vqg.py`.
    
    :param path: Path to directory to load json file from (a directory that includes a vqg_outputs.json in it).
    """
    if not path.endswith(".json"):
        path = os.path.join(path, "vqg_outputs.json")
    if os.path.exists(path):
        vqg_outputs = json.load(open(path, "r"))
        try:
            vqg_outputs = {int(k): VQGOutputs(**v) for k, v in vqg_outputs.items()}
        except:
            vqg_outputs = {str(k): (VQGOutputs(**v) if v is not None else None) for k, v in vqg_outputs.items()}
        return vqg_outputs
    else:
        return {}
