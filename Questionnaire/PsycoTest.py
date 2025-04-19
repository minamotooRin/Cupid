from abc import ABC, abstractmethod

# universal_trait_mappings = {
#     {
#         'universal_trait': 'Openness',
#         'BigFive':            'Openness to Experience',
#         'MBTI':               'Intuition (N)',
#         'DISC':               None,
#         'Enneagram':          ['Type 4', 'Type 5', 'Type 7'],
#         'HEXACO':             'Openness to Experience'
#     },
#     {
#         'universal_trait': 'Conscientiousness',
#         'BigFive':            'Conscientiousness',
#         'MBTI':               'Judging (J)',
#         'DISC':               'Conscientiousness',
#         'Enneagram':          ['Type 1', 'Type 3'],
#         'HEXACO':             'Conscientiousness'
#     },
#     {
#         'universal_trait': 'Extraversion',
#         'BigFive':            'Extraversion',
#         'MBTI':               'Extraversion (E)',
#         'DISC':               ['Dominance (D)', 'Influence (I)'],
#         'Enneagram':          ['Type 3', 'Type 7', 'Type 8'],
#         'HEXACO':             'Extraversion'
#     },
#     {
#         'universal_trait': 'Agreeableness',
#         'BigFive':            'Agreeableness',
#         'MBTI':               'Feeling (F)',
#         'DISC':               'Steadiness (S)',
#         'Enneagram':          ['Type 2', 'Type 9'],
#         'HEXACO':             'Agreeableness'
#     },
#     {
#         'universal_trait': 'Emotional Stability',
#         'BigFive':            'Neuroticism (inverse)',
#         'MBTI':               None,
#         'DISC':               None,
#         'Enneagram':          ['Type 6 (low stability)', 
#                                 'Type 4 (low stability)', 
#                                 'Type 9 (high stability)'],
#         'HEXACO':             'Emotionality (inverse)'
#     },
#     {
#         'universal_trait': 'Honesty-Humility',
#         'BigFive':            None,
#         'MBTI':               None,
#         'DISC':               None,
#         'Enneagram':          ['Type 1'],
#         'HEXACO':             'Honesty-Humility'
#     }
# }

class PsycoTest(ABC):
    def __init__(self):
        self.questions = self.get_questions()
        self.traits_map = self.get_traits_map()
        self.answers = {}

    def __iter__(self):
        return iter(enumerate(self.questions, start=0))

    def record_answer(self, question_no: int, answer):
        if question_no < 0 or question_no >= len(self.questions):
            return
        self.answers[question_no] = answer

    def analyze(self):
        trait_scores = {}
        for q_no, ans in self.answers.items():
            trait = self.traits_map.get(q_no)
            if trait is None:
                continue
            trait_scores.setdefault(trait, []).append(ans)
        return {trait: sum(scores)/len(scores) for trait, scores in trait_scores.items()}

    @abstractmethod
    def get_questions(self) -> list:
        pass

    @abstractmethod
    def get_traits_map(self) -> dict:
        pass