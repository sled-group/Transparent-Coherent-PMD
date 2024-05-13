from dataclasses import dataclass, field, asdict
import os
import json
import torch
from tqdm import tqdm
from transformers import TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset
from typing import Any, Optional

from travel.constants import CACHE_FREQUENCY
from travel.model.vqa import VQAResponse

@dataclass
class VQGInputs:
    """Dataclass to hold all LM inputs for visual question generation (VQG)."""
    procedure_id: int
    procedure_description: str
    prompt: str

    def to_dict(self):
        return asdict(self)
    
    @staticmethod
    def from_dict(self, data):
        return VQGInputs(**data)

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

VQG_PROMPT_TEMPLATE = 'The instructions say to "{instruction_step}". To visually verify that this procedure is complete, what are {n_questions} yes/no questions we could ask about an image of a target object and their expected answers?\n'
VQG_EXAMPLE_TEMPLATE = VQG_PROMPT_TEMPLATE + \
                       "Target object: {target_object}\n" + \
                       "{question_list}"
VQG_QUESTION_TEMPLATE = "{question_number}. {question} (yes/no) {answer}"

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

def save_vqg_inputs(inputs: list[VQGInputs], path: str):
    """
    Saves generated VQG inputs to a json file.
    
    :param inputs: List of VQGInputs holding prompts for VQG.
    :param path: Filename ending with .json to save inputs at.
    """
    assert path.endswith(".json"), "save_vqg_inputs expects a json filename!"
    json.dump(
        [inp.to_dict() for inp in inputs],
        open(path, "w")
    )

def load_vqg_inputs(path: str) -> list[VQGInputs]:
    """
    Loads generated VQG inputs from a json file.

    :param path: json filename to load from.
    """
    return [VQGInputs.from_dict(inp) for inp in json.load(open(path, "r"))]

# TODO: later may need to account for chat-based prompts
def generate_vqg_prompt_icl(procedure_description: str, n_demonstrations: int=3) -> str:
    """
    Returns a prompt for VQG including in-context demonstrations.

    :param procedure_description: String description of a procedure (e.g., recipe step) to generate visual questions for.
    :param n_demonstrations: Number of in-context demonstrations to include from `VQG_DEMONSTRATIONS`.
    :return: Prompt for VQG including in-context demonstrations.
    """
    assert n_demonstrations <= len(VQG_DEMONSTRATIONS), f"Requested {n_demonstrations} in-context demonstrations for VQG, but only {len(VQG_DEMONSTRATIONS)} are available in travel.model.vqg.VQG_DEMONSTRATIONS."

    examples = [generate_vqg_example(demo) for demo in VQG_DEMONSTRATIONS[:n_demonstrations]]
    examples += [generate_vqg_prompt(procedure_description)]
    return "\n\n".join(examples)

# TODO: include Ruixuan's improvements here... some of which are below
# TODO: may need to reform prompts for recipe steps to include more information from the recipe - previous steps, ingredients, or recipe name?
# TODO: does there need to be a single target object for VQG?
# TODO: increase number of questions to 3? or use a variable number
def run_vqg(lm: TextGenerationPipeline, inputs: list[VQGInputs], input_ids: list[str], save_path: Optional[str]=None, vqg_outputs: dict[str, VQGOutputs]={}) -> dict[str, VQGOutputs]:
    """
    Runs VQG with a given LM text generation pipeline and list of VQG inputs.

    :param lm: TextGenerationPipeline from `transformers` library.
    :param inputs: List of VQG inputs, including procedures, prompts, etc.
    :param input_ids: Unique string identifiers for inputs. These may characterize a specific prompt or run of a prompt (e.g., at a different temperature).
    :param save_path: Optional path to save VQG outputs during and after running. Must either be a json filename or path to a directory.
    :param vqg_outputs: Partly filled dictionary of VQG outputs to start from; only pass this if starting from a partially completed run of VQG, and make sure complete/incomplete prompts are managed appropriately outside of this method.
    """
    assert len(inputs) == len(input_ids), "run_vqg expected the same number of inputs and input IDs!"
    prompt_idx = 0
    with torch.no_grad():
        # TODO: implement data parallelism over multiple GPUs?
        for inp, inp_id, out in tqdm(zip(inputs,
                                    input_ids,
                                    lm(KeyDataset([inp.to_dict() for inp in inputs], "prompt"), 
                                        batch_size=8, 
                                        max_new_tokens=128, 
                                        return_full_text=False, 
                                        truncation="do_not_truncate")),
                            desc="running VQG",
                            total=len(inputs)):

            procedure_id = int(inp.procedure_id)
            step = inp.procedure_description

            text = out[0]['generated_text']
            
            # Hack: sometimes output from LLaMA 2 starts with Љ and whitespace characters, and sometimes Љ replaces the first "T" in "Target object:"
            text_fixed = text.replace("Љ", "").strip() 
            if not text_fixed.startswith("Target object:") and ":" in text_fixed:
                text_fixed = "Target object: " + ":".join(text_fixed.split(":")[1:]).strip()
            
            # Parse reported target object and questions and answers
            try:
                try:
                    output = parse_vqg_outputs(text_fixed, procedure_id, step)
                except:
                    print("Warning: failed to parse a VQG output.")
                    continue
            except:
                print("Error parsing VQG outputs:")
                print(text)
                print('======')
                print(text_fixed)
                raise

            vqg_outputs[inp_id] = output

            if prompt_idx % CACHE_FREQUENCY == 0 and save_path is not None:
                print("Saving progress...")
                save_vqg_outputs(vqg_outputs, save_path)

            prompt_idx += 1

    # Save progress one last time after completion
    save_vqg_outputs(vqg_outputs, save_path)
    return vqg_outputs

def parse_vqg_outputs(generated_language: str, procedure_id: int, procedure_description: str) -> VQGOutputs:
    """
    Converts generated questions and answers into a VQGOutputs object.

    :param generated_language: Text generated by an LM which matches the format of prompt templates in this file.
    :param procedure_id: Int identifier for procedure (e.g., recipe or instruction step).
    :param procedure_description: Procedure description in text (e.g., recipe or instruction step text).
    :return: VQGOutputs parsed from generated language.
    """
    target_object = generated_language.split("\n")[0].split("Target object: ")[1].strip()
    questions_answers = [(q_a.split("? (yes/no)")[0].strip() + "?", q_a.split("(yes/no)")[1].strip()) for q_a in generated_language.split("\n")[1:3]] # NOTE: only extract k=2 questions and answers; can adjust this as needed later
    questions = [q[2:].strip() for q, _ in questions_answers]          
    answers = [a for _, a in questions_answers]
    output = VQGOutputs(procedure_id,
                        procedure_description,
                        target_object,
                        questions,
                        answers)
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

    json.dump({k: v.to_dict() for k, v in vqg_outputs.items()}, 
              open(path, "w"),
              indent=4)    

def load_vqg_outputs(path: str) -> dict[Any, VQGOutputs]:
    """
    Loads dict of VQGOutputs created by `run_vqg.py`.
    
    :param path: Path to directory to load json file from (a directory that includes a vqg_outputs.json in it).
    """
    if not path.endswith(".json"):
        path = os.path.join(path, "vqg_outputs.json")
    vqg_outputs = json.load(open(path, "r"))
    try:
        vqg_outputs = {int(k): VQGOutputs(**v) for k, v in vqg_outputs.items()}
    except:
        vqg_outputs = {str(k): VQGOutputs(**v) for k, v in vqg_outputs.items()}
    return vqg_outputs
