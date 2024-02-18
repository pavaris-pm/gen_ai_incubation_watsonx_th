def detect_language(text):
    thai_chars = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะัาำิีึืุูเแโใไ็่้๊๋์")
    if any(char in thai_chars for char in text):
        return "th"
    else:
        return "en"

sentence = "สวัสดีครับ how are you?"
language = detect_language(sentence)
print(language)  # Output: th

sentence = "Hello, how are you?"
language = detect_language(sentence)
print(language)  # Output: en
