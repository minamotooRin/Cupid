import json
import pandas as pd
from pathlib import Path

class IPIPTest():
    def __init__(self, multiple_item_table: pd.DataFrame, instrument: str):
        self.multiple_item_table = multiple_item_table
        self.instrument = instrument
        self.item_table = self.multiple_item_table[self.multiple_item_table["instrument"] == self.instrument]
        self.questions = self.item_table['text'].tolist()
        self.trait_label = self.item_table["label"].tolist()
        self.key_label = self.item_table["key"].tolist()
        self.answers = {}

    def __iter__(self):
        return iter(enumerate(self.questions, start=0))

    def record_answer(self, question_no: int, answer: dict):
        if question_no < 0 or question_no >= len(self.questions) or "score" not in answer:
            return
        self.answers[question_no] = answer

    def analyze(self):
        trait_scores = {}
        for q_no, ans in self.answers.items():
            try:
                score = int(ans["score"])
            except:
                print(f"Invalid score for question {q_no}: {ans['score']}")
                continue
            if self.key_label[q_no] < 0:
                score = 6 - score
            trait = self.trait_label[q_no]
            if trait is None:
                continue
            trait_scores.setdefault(trait, []).append(score)
        return {trait: sum(scores)/len(scores) for trait, scores in trait_scores.items()}
    
    def save_to_jsonl(self, filename: str):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w', encoding="utf-8") as f:
            for q_no, ans in self.answers.items():
                data = {
                    "question_no": q_no,
                    "question": self.questions[q_no],
                    "answer": ans
                }
                f.write(json.dumps(data) + '\n')

    def load_from_jsonl(self, filename: str):
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                q_no = data["question_no"]
                question = data["question"]
                answer = data["answer"]
                self.record_answer(q_no, answer)
    