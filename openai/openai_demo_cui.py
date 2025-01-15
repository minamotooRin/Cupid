import os 

import fire
import pickle
import time
import re
import copy
import random

from pathlib import Path
from openai import OpenAI

class Prompt:
    def __init__(self, instr, query = None):
        self.instr = instr
        self.demons = []
        self.query = query

    def add_demon(self, user, assi):
        self.demons.append({'user': user, 'assi': assi})

    def get_prompt(self):
        prompt = [ 
            {"role": "system", "content": self.instr},
        ]
        for demon in self.demons:
            prompt.append({"role": "user", "content": demon['user']})
            prompt.append({"role": "system", "content": demon['assi']})
        if self.query:
            prompt.append({"role": "user", "content": self.query})

        return prompt


MY_API_KEY = ""

client = OpenAI(
    api_key=MY_API_KEY
)

def api_call(msgs, model_engine,):
    chat_completion = client.chat.completions.create(
        messages = msgs,
        model = model_engine,
    )
    return chat_completion



model_engine = "gpt-4o"
PRICE = {
    "input": 2.5/1000000,
    "output": 10/1000000,
}
output_path:str = "./output"


instruction = """
You are to embody a character defined by specific configurations of the Big Five personality traits. For each trait, specify a precise position on the spectrum and maintain consistent expression of these traits through responses:

Openness to Experience [Scale 1-5]:


Define exact position between inventive/curious vs consistent/cautious
Specify how this manifests in:

Approach to novel situations
Information processing style
Creative problem-solving tendencies




Conscientiousness [Scale 1-5]:


Define exact position between efficient/organized vs extravagant/careless
Detail expression through:

Task management approach
Decision-making process
Planning and execution style




Extraversion [Scale 1-5]:


Define exact position between outgoing/energetic vs solitary/reserved
Demonstrate through:

Social interaction patterns
Energy management
Communication style




Agreeableness [Scale 1-5]:


Define exact position between friendly/compassionate vs critical/judgmental
Express through:

Conflict handling
Collaboration approach
Response to others' needs




Neuroticism [Scale 1-5]:


Define exact position between sensitive/nervous vs resilient/confident
Manifest in:

Stress response patterns
Emotional processing
Adaptation to change
"""

query = """
Generate 50 diverse daily scenarios spanning work, social interactions, and personal life. For each scenario, provide 10 unique thought-perspective pairs that demonstrate Openness at scale 2, Conscientiousness at scale 5. Each thought-perspective pair should:

Be expressed as detailed declarative sentences that reveal deeper personality traits
Show the natural progression from initial observation to creative insight
Demonstrate how the thought process influences the resulting perspective
Reflect authentic personality traits through consistent patterns of thinking
Include both emotional and analytical components where relevant
"""

prompt_generator = Prompt(instruction, query)
prompt = prompt_generator.get_prompt()
print(prompt)

response = api_call(prompt, model_engine)


prompt_tokens = response.usage.prompt_tokens
completion_tokens = response.usage.completion_tokens
cost = prompt_tokens * PRICE["input"] + completion_tokens * PRICE["output"]
print(f"Cost: {cost}")


print(f"Number of choices: {len(response.choices)}")
msg = response.choices[0].message.content
print(msg)
