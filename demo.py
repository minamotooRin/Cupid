# %%
import json
import logging
import pandas as pd 
import fire

from tqdm import tqdm
from pathlib import Path

from Questionnaire.IPIPtest import IPIPTest
from AgentFactory import AgentFactory
from utils import clear_json, abbr2lang

def main(
    config_path: str = "configs/conf_GPT.json",
    excel_dir: str = "IPIP/",
    langs: str = ["DE", "ZH", "JA", "MN"],
):

    logging.basicConfig(level=logging.WARNING)

    config = json.load(open(config_path, "r"))
    config = config["model"]

    agent_factory = AgentFactory()
    agent = agent_factory.create_agent(config)

    for lang in langs:

        print(f"Processing language: {lang}")

        excel_path = f"{excel_dir}/{lang}.xlsx"

        multi_list = pd.read_excel(excel_path)
        instuments = multi_list['instrument'].unique()

        ProgressBar = tqdm(total=len(instuments), desc=f"Processing all instuments in {lang}")

        for instrument in tqdm(instuments):
            ProgressBar.update(1)
            print(f"Processing instrument: {instrument}")
            
            save_path = f"output/{config['template_type']}/{lang}/{instrument}.jsonl"
            if Path(save_path).exists():
                print(f"File {save_path} already exists, skipping.")
                continue
            if not Path(save_path).parent.exists():
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            qn = IPIPTest(multi_list, instrument)

            for no, q in enumerate(qn.questions):

                agent.reset_msg_history()
                agent.instruction = config["system_msg"].format(lang=abbr2lang[lang])
                prompt = [
                    {"type": "text", "text": q},
                ]
                try:
                    response, text = agent.get_response(prompt)
                    dict = json.loads(clear_json(text))
                    qn.record_answer(no, dict)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"text: {text}")
                    continue

            qn.save_to_jsonl(save_path)
            print(agent.llm.total_cost())

if __name__ == "__main__":
    fire.Fire(main)