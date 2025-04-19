from Questionnaire.PsycoTest import PsycoTest

BFI_ITEM_LIST = [
    "Talks a lot",
    "Notices other people’s weak points",
    "Does things carefully and completely",
    "Is sad, depressed",
    "Is original, comes up with new ideas",
    "Keeps their thoughts to themselves",
    "Is helpful and not selfish with others",
    "Can be kind of careless",
    "Is relaxed, handles stress well",
    "Is curious about lots of different things",
    "Has a lot of energy",
    "Starts arguments with others",
    "Is a good, hard worker",
    "Can be tense; not always easy going",
    "Clever; thinks a lot",
    "Makes things exciting",
    "Forgives others easily",
    "Isn’t very organized",
    "Worries a lot",
    "Has a good, active imagination",
    "Tends to be quiet",
    "Usually trusts people",
    "Tends to be lazy",
    "Doesn’t get upset easily; steady",
    "Is creative and inventive",
    "Has a good, strong personality",
    "Can be cold and distant with others",
    "Keeps working until things are done",
    "Can be moody",
    "Likes artistic and creative experiences",
    "Is kind of shy",
    "Kind and considerate to almost everyone",
    "Does things quickly and carefully",
    "Stays calm in difficult situations",
    "Likes work that is the same every time",
    "Is outgoing; likes to be with people",
    "Is sometimes rude to others",
    "Makes plans and sticks to them",
    "Get nervous easily",
    "Likes to think and play with ideas",
    "Doesn’t like artistic things (plays, music)",
    "Likes to cooperate; goes along with others",
    "Has trouble paying attention",
    "Knows a lot about art, music and books",
]

BFI_traits = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "Openness",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
]

UNIVERSAL_TRAITS = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Emotional Stability",
    "Openness",
    "Honesty Humility",
]

class BFI(PsycoTest):
    def __init__(self):
        super().__init__()
        # self.TRAITS_2_UNI = {
        #     "Extraversion":"Extraversion",
        #     "Agreeableness":"Agreeableness",
        #     "Conscientiousness":"Conscientiousness",
        #     "Neuroticism":"Emotional Stability",
        #     "Openness":"Openness",
        # }

    def get_questions(self) -> list:
        return BFI_ITEM_LIST

    def get_traits_map(self) -> dict:
        TRAITS_2_UNI = {
            "Extraversion":"Extraversion",
            "Agreeableness":"Agreeableness",
            "Conscientiousness":"Conscientiousness",
            "Neuroticism":"Emotional Stability",
            "Openness":"Openness",
        }
        return {k: TRAITS_2_UNI[v] for k, v in zip(range(len(BFI_ITEM_LIST)), BFI_traits)}
        # return {k: v for k, v in zip(range(len(BFI_ITEM_LIST)), BFI_traits)}
