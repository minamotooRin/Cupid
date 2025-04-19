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