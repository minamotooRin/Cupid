

class Agent:
    def __init__(self, llm, instruction = ""):
        
        self.llm = llm
        self.instruction = instruction
        self.msg_history = []

        self.reset()

    @property
    def instruction(self):
        return self._instruction 
    
    @instruction.setter
    def instruction(self, instruction):
        self._instruction = instruction
        self.reset()
    
    @property
    def msgs(self):
        return self.msg_history 
    
    def set_controller(self, controller):
        #     pos_instrs = warp_config["pos_instrs"]
        #     neg_instrs = warp_config["neg_instrs"]
        #     activation_demons = warp_config["activation_demons"]
        #     pn_pairs = list(zip(pos_instrs, neg_instrs))

        #     layer_id = warp_config["layer_id"]
        #     coeff = warp_config["coeff"]
        #     model = get_wrapped_model(model, tokenizer, pn_pairs, layer_id, coeff)
        pass
        
    def reset(self):
        self.msg_history = [{
                "system_msg": self.instruction
            }]
        
    def get_response(self, msg, reply_prefix  = ""):
        self.msg_history.append({
                "user_msg": msg
            })
        
        formatted_prompt = self.llm.format_prompt(self.msg_history, reply_prefix)
        texts = self.llm.get_response(formatted_prompt)
        response = self.llm.extract_response(texts, reply_prefix)

        self.msg_history.append({
                "assistant_msg": response
            })

        return response
    
    def __str__(self):
        return f'Agent(instruction={self.instruction}, llm={self.llm})'

    def __repr__(self):
        return str(self)