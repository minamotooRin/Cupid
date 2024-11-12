class Agent:
    def __init__(self, llm, instruction = ""):
        
        self.llm = llm

        self.instruction = instruction
        self.msg_history = []
        self.reset_msg_history()

        self.controller_info = {
            "pn_pairs": [], # [ [pos1, neg1], [pos2, neg2], ...]
            "layers_id": [], # [0, 1, 2, ...]
            "coeff": 0.5,
            "batch_size": 16
            }
        self.controller_activations = {}

    @property
    def instruction(self):
        return self._instruction 
    
    @instruction.setter
    def instruction(self, instruction):
        self._instruction = instruction
        self.reset_msg_history()
    
    @property
    def msgs(self):
        return self.msg_history 
    
    @property
    def controller_texts(self):
        return self.controller_texts 

    def set_llm(self, llm):
        self.llm = llm
        if self.controller_activations:
            self.set_controller_text(self.controller_info["pn_pairs"], self.controller_info["layers_id"], self.controller_info["coeff"], self.controller_info["batch_size"])
    
    def set_controller_text(self, pn_pairs, layers_id = [], coeff = 0.5, batch_size=16):
        
        if layers_id == []:
            layers_id = list(range(self.llm.model.config.num_hidden_layers))
            layers_id = layers_id[1:] # exclude the first layer

        self.controller_info["pn_pairs"] = pn_pairs
        self.controller_info["layers_id"] = layers_id
        self.controller_info["coeff"] = coeff
        self.controller_info["batch_size"] = batch_size

        self.controller_activations = self.llm.get_controller_activations(pn_pairs, layers_id, coeff, batch_size)
        
    def reset_msg_history(self):
        self.msg_history = [{
                "system_msg": self.instruction
            }]
        
    def get_response(self, msg, reply_prefix  = "", max_length = 50):
        self.msg_history.append({
                "user_msg": msg
            })
        
        formatted_prompt = self.llm.format_prompt(self.msg_history, reply_prefix)
        texts = self.llm.get_response(formatted_prompt, self.controller_activations, max_length = max_length)
        response = self.llm.extract_response(texts, reply_prefix)

        self.msg_history.append({
                "assistant_msg": response
            })

        return texts, response
    
    def __str__(self):
        return f'Agent(instruction={self.instruction}, llm={self.llm})'

    def __repr__(self):
        return str(self)