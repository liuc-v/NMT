# 导入NLTK库
import nltk

# 定义一个英文句子
sentence = "The first time I heard that song was in Hawaii on radio. I was just a kid, and loved it very much! What a fantastic song!"

# 使用NLTK的word_tokenize方法进行分词
words = nltk.word_tokenize(sentence)
print("分词结果：")
print(words)

# 使用NLTK的PorterStemmer类进行词干提取
stemmer = nltk.PorterStemmer()
stems = [stemmer.stem(word) for word in words]
print("词干提取结果：")
print(stems)

# 使用NLTK的WordNetLemmatizer类进行词性还原
lemmatizer = nltk.WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in words]
print("词性还原结果：")
print(lemmas)
