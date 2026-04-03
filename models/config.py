import pathlib, os
parent_path = pathlib.Path(__file__).parent.absolute().parent

class Config:

    RULE_FILE_PATH = os.path.join(parent_path, 'data', 'rule.json')
    PROMPT_FILE_PATH = os.path.join(parent_path, 'data', 'system_prompt.json')
