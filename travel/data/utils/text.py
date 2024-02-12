import spacy
from spacy.tokens import Token

def simple_present_to_imperative(nlp: spacy.Language, sentence: str) -> str:
    """
    Converts a sentence input_sentence in simple present tense (e.g., "Someone eats the cookie") into imperative form (e.g., "Eat the cookie").
    
    :param nlp: NLP object from spaCy. Example: `nlp = spacy.load('en_core_web_sm')`
    :param sentence: Input sentence in simple present tense.
    :return: New sentence in imperative (command) form.
    """
    
    # Process the sentence using spaCy
    doc = nlp(sentence)

    # Initialize variables
    verb = None
    subj_idx = -1
    
    # Iterate over the tokens
    for token in doc:
        # Find the subject and verb
        if "subj" in token.dep_:
            subj_idx = token.i
        if token.pos_ == "VERB" and verb is None:  # Get the first verb
            verb = token
    
    # If a subject and verb are found, and the subject precedes the verb
    if subj_idx != -1 and verb is not None and subj_idx < verb.i:
        # Reconstruct the sentence without the subject and use the lemma of the verb
        imperative_tokens = [verb.lemma_] + [token.text for token in doc if token.i != subj_idx and token.i != verb.i]
        imperative_tokens = [token.replace("his", "your").replace("her", "your").replace("their", "your") for token in imperative_tokens]
        imperative_sentence = ' '.join(imperative_tokens)
        if imperative_sentence.endswith("."):
            imperative_sentence = imperative_sentence[:-1].strip()
        return imperative_sentence.capitalize()
    
    # If the sentence doesn't match the expected pattern, return it as is
    return sentence