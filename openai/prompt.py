class Prompt:
    def __init__(self, instr, query = None):
        self.instr = instr
        self.demons = []
        self.query = query

    def add_demon(self, user, assi):
        self.demons.append({'user': user, 'assi': assi})

    def get_prompt(self):
        prompt = [ 
            {"role": "assistant", "content": self.instr},
        ]
        for demon in self.demons:
            prompt.append({"role": "user", "content": demon['user']})
            prompt.append({"role": "assistant", "content": demon['assi']})
        if self.query:
            prompt.append({"role": "user", "content": self.query})

        return prompt

