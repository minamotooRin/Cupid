import fire
import json
import logging

from utils import seed_everything
from Agent import Agent
from LLM import ModelPool

from flask import Flask, render_template, request, jsonify

class InferenceApp:
    def __init__(self, config_path):
        seed_everything(42)

        self.app = Flask(__name__)

        config = json.load(open(config_path, "r"))
        self.model_pool = ModelPool()
        self.model_pool.add_model(config["template_type"], config["model_name"], config["device"])

        self.agent = Agent(self.model_pool[config["model_name"]])
        
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/', methods=['GET'])
        def index():
            return render_template('index.html')
        
        @self.app.route('/inference', methods=['POST', 'OPTIONS'])
        def inference():

            if request.method == 'OPTIONS':
                # 处理预检请求
                response = self.app.make_default_options_response()
                return response
            
            data = request.get_json()
            sys = data.get('sys', '')
            user = data.get('user', '')
            positive_case = data.get('PC', '')
            negetive_case = data.get('NC', '')
            coe = float(data.get('coe', 1))
                
            self.agent.instruction = sys
            self.agent.set_controller_text([[positive_case, negetive_case]], coeff=coe)

            response = self.agent.get_response(user, reply_prefix="")

            return jsonify({'output': response})

    def run(self, host='0.0.0.0', port=5000, debug=False):
        self.app.run(host=host, port=port, debug=debug)

def main(
        config_path: str = 'configs/inference_conf0.json',
):
    
    inference_app = InferenceApp(config_path)
    logging.info(f"Starting Inference App...")
    inference_app.run(debug=True)

if __name__ == '__main__':
    fire.Fire(main)
