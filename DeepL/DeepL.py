# %%
import logging
import deepl
import pandas as pd 
import fire

from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)

class Translator:
    def __init__(self, auth_key):
        self.auth_key = auth_key
        self.translator = deepl.Translator(auth_key)
        self.billed_characters = 0

    def translate_text(self, text, source_lang, target_lang):
        try:
            result = self.translator.translate_text(text, source_lang=source_lang, target_lang=target_lang)
            self.billed_characters += result.billed_characters
            return result.text
        except deepl.exceptions.DeepLException as e:
            logging.error(f"Error translating text: {e}")
            return text
    
    def __str__(self):
        return f"Translator(auth_key={self.auth_key}), billed_characters={self.billed_characters})"

auth_key = '508e43b0-f363-479b-b8d4-eca3d752e761:fx'
translator = Translator(auth_key)

def translate_text(text, source_lang="EN", target_lang="DE"):
    # Translate the text using the DeepL API
    result = translator.translate_text(text, source_lang, target_lang)
    return result

def translate_column(df, column_name, source_lang="EN", target_lang="DE"):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Translate the specified column
    # df_copy[column_name] = df_copy[column_name].apply(lambda x: translate_text(x, source_lang, target_lang) if isinstance(x, str) else x)

    # Translate the specified column with tqdm
    for index, row in tqdm(df_copy.iterrows(), total=df_copy.shape[0], desc=f"Translating {column_name}"):
        text = row[column_name]
        if isinstance(text, str):
            translated_text = translate_text(text, source_lang, target_lang)
            df_copy.at[index, column_name] = translated_text
        else:
            df_copy.at[index, column_name] = text

    return df_copy

def main(      
    target_lang = 'mn_MN',
):
    multi_list = pd.read_excel('/home/youyuan/Cupid/TedoneItemAssignmentTable30APR21.xlsx')
    multi_list_translated = translate_column(multi_list, 'text', source_lang="EN", target_lang=target_lang)
    output_file = f"/home/youyuan/Cupid/TedoneItemAssignmentTable30APR21_{target_lang}.xlsx"
    multi_list_translated.to_excel(output_file, index=False)

    print(f"Saving translated DataFrame to {output_file}, billed characters: {translator.billed_characters}")

if __name__ == "__main__":
    fire.Fire(main)

