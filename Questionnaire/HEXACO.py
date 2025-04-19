from Questionnaire.PsycoTest import PsycoTest

HEXACO_ITEM_LIST = [
    "I would be quite bored by a visit to an art gallery.",
    "I plan ahead and organize things, to avoid scrambling at the last minute.",
    "I rarely hold a grudge, even against people who have badly wronged me.",
    "I feel reasonably satisfied with myself overall.",
    "I wouldn’t use flattery to get a raise or promotion at work, even if I thought it would succeed.",
    "I sometimes can't help worrying about little things.",
    "I prefer jobs that involve active social interaction to those that involve working alone.",
    "I tend to be lenient in judging other people.",
    "I often push myself very hard when trying to achieve a goal.",
    "I worry a lot less than most people do.",
    "I would feel afraid if I had to travel in bad weather conditions.",
    "I wouldn’t mind wearing clothes that are a little out of the ordinary.",
    "If I knew that I could never get caught, I would be willing to steal a million dollars.",
    "People sometimes tell me that I am too critical of others.",
    "I rarely express my opinions in group meetings.",
    "I sometimes can't help thinking about little things that upset me.",
    "I sometimes feel that I am a worthless person.",
    "I sometimes try to get others to do my duties.",
    "I like people who have unconventional views.",
    "I make decisions based on the feeling of the moment rather than on careful thought.",
    "I rarely hold a grudge, even against people who have badly wronged me.",
    "I feel like crying when I see other people crying.",
    "I often express my admiration for others.",
    "I would be very bored by a job that involved only routine tasks.",
    "I would be tempted to buy stolen property if I were financially tight.",
    "When people make mistakes, I am usually very forgiving.",
    "I feel comfortable around people in social situations.",
    "I often worry about things that turn out to be unimportant.",
    "I sometimes feel that I am a worthless person.",
    "I would be very bored by a job that involved only routine tasks.",
    "I often push myself very hard when trying to achieve a goal.",
    "I usually see the good side of people.",
    "I find it difficult to avoid certain thoughts that are intrusive.",
    "I feel reasonably satisfied with myself overall.",
    "I wouldn’t pretend to like someone just to get that person to do favors for me.",
    "I often check my work over repeatedly to find any mistakes.",
    "I make decisions based on the feeling of the moment rather than on careful thought.",
    "I don't mind being the center of attention.",
    "I would be afraid to travel in a small airplane.",
    "I would feel afraid if I had to travel in bad weather conditions.",
    "I wouldn’t use flattery to get a raise or promotion at work, even if I thought it would succeed.",
    "I wouldn’t mind wearing clothes that are a little out of the ordinary.",
    "I often check my work over repeatedly to find any mistakes.",
    "I would be tempted to buy stolen property if I were financially tight.",
    "I like people who have unconventional views.",
    "I rarely express my opinions in group meetings.",
    "I make decisions based on the feeling of the moment rather than on careful thought.",
    "I usually see the good side of people.",
    "I wouldn’t pretend to like someone just to get that person to do favors for me.",
    "I would never accept a bribe, even if it were very large.",
    "I tend to be lenient in judging other people.",
    "I feel like crying when I see other people crying.",
    "I rarely hold a grudge, even against people who have badly wronged me.",
    "I like to plan things carefully ahead of time.",
    "People sometimes tell me that I am too critical of others.",
    "I am an ordinary person who is no better than others.",
    "I sometimes can't help worrying about little things.",
    "I prefer jobs that involve active social interaction to those that involve working alone.",
    "I often express my admiration for others.",
    "I worry a lot less than most people do.",
    "I would never accept a bribe, even if it were very large."
]

HEXACO_traits = [
    "O", "C", "A", "E", "H", "E", "X", "A", "C", "E",
    "E", "O", "H", "A", "X", "E", "E", "C", "O", "C",
    "A", "E", "X", "O", "H", "A", "X", "E", "E", "O",
    "C", "A", "E", "E", "H", "C", "C", "X", "E", "E",
    "H", "O", "C", "H", "O", "X", "C", "A", "H", "H",
    "A", "E", "A", "C", "A", "H", "E", "X", "X", "E",
    "H"
]

class HEXACO(PsycoTest):
    def __init__(self):
        super().__init__()

    def get_questions(self) -> list:
        return HEXACO_ITEM_LIST

    def get_traits_map(self) -> dict:
        TRAITS_2_UNI = {
            "H":"Honesty Humility",
            "E":"Emotional Stability",
            "X":"Extraversion",
            "A":"Agreeableness",
            "C":"Conscientiousness",
            "O":"Openness",
        } 
        return {k: TRAITS_2_UNI[v] for k, v in zip(range(len(HEXACO_ITEM_LIST)), HEXACO_traits)}



