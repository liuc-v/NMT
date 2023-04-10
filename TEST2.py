from langdetect import detect

text = "你好，世界"
lang = detect(text) # 'zh-cn'
print(lang)