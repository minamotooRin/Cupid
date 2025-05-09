def clear_json(text):
    # result = text.replace("\n", ", ")
    result = text.replace("json", "")
    result = result.replace("`", "")
    result = result.replace("{,  \"","{\"")
    result = result.replace(", }", "}")
    result = result.replace(",,", ",")
    result = result.strip(", ")
    result = result.strip()
    return result

abbr2lang = {
    "DE": "German",
    "ZH": "Chinese",
    "JA": "Japanese",
    "MN": "Mongolian",
    "EN": "English",
}