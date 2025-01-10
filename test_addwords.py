import random
from sklearn.metrics import accuracy_score
from nltk import FreqDist
from nltk.corpus import stopwords, brown
from collections import Counter
import nltk
#第一次运行需要取消下面的注释，下载
# nltk.download('stopwords')
# nltk.download('brown')


# 样本数据
texts = [
    "This is a very interesting example of machine learning application.",
    "Text classification is a key task in natural language processing.",
    "Robustness testing is essential for evaluating model performance.",
    "Adding words to sentences can help test robustness of classifiers."
]
labels = [1, 1, 0, 0]

# 模拟分类模型
def mock_model_predict(texts):
    return [random.choice([0, 1]) for _ in texts]


# # 构建非停用词的高频词库
# def build_candidate_highfrequency_word_pool(texts, source="localgraph", top_n=100):
#     stop_words = set(stopwords.words("english"))
#     if source == "localgraph":
#         # 使用图中的词汇
#         all_words = []
#         for text in texts:
#             for word in nltk.word_tokenize(text):
#             all_words.extend([word.lower() for word in nltk.word_tokenize(text) if word.isalpha() and word.lower() not in stop_words])
#         # word_freq = Counter(all_words)
#         word_freq = FreqDist(all_words)
#         return [word for word, _ in word_freq.most_common(top_n)]
#     elif source == "brown":
#         # 使用brown词库，categories=['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']
#         all_words = brown.words() # brown.words(categories="")
#         filtered_words = [word.lower() for word in all_words if word.lower() not in stop_words and word.isalpha()]
#         word_freq = FreqDist(filtered_words)
#         return [word for word, _ in word_freq.most_common(top_n)]
#     else:
#         return texts
    
# 构建非停用词的所有词库
def build_candidate_all_word_pool(texts, source="localgraph"):
    stop_words = set(stopwords.words("english"))
    if source == "localgraph":
        # 使用图中的词汇
        all_words = []
        for text in texts:
            all_words.extend([word.lower() for word in nltk.word_tokenize(text) if word.isalpha() and word.lower() not in stop_words])
        return all_words
    
    elif source == "brown":
        # 使用brown词库，categories=['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']
        all_words = brown.words() # brown.words(categories="")
        filtered_words = [word.lower() for word in all_words if word.lower() not in stop_words and word.isalpha()]
        return filtered_words
    
    else:
        print(f"无候选词")
        return [""]

# 添加词
def perturb_add_words(text, candidate_words, add_word_num=1):
    words = text.split()
    for _ in range(add_word_num):
        insert_position = random.randint(0, len(words))  # 随机插入位置
        insert_word = random.choice(candidate_words)  # 随机选一个词
        words.insert(insert_position, insert_word)
    return " ".join(words)

# 扰动函数
def perturb_text(texts, candidate_words, node_p, perturb_word_p): 
    node_num = len(texts) #节点数量
    perturb_node_num = int(node_num * node_p) # 扰动节点数量
    samples = list(range(0, node_num))
    selected_samples = random.sample(samples, perturb_node_num) # 随机抽样指定数量的节点
    print(f"node_num:{node_num}, perturb_node_num:{perturb_node_num}, selected_samples:{selected_samples}")
    
    perturbed_texts = texts.copy()
    for i in selected_samples:
        text = texts[i]
        word_num = len(text.split())
        perturb_word_num = int(perturb_word_p * word_num)
        print(f"word_num:{word_num}, perturb_word_num: {perturb_word_num}\n")
        if perturb_word_num < 0:
            # 随机删除词
            perturb_word_num = - perturb_word_num
            words = text.split()
            for _ in range(perturb_word_num):
                if words: words.pop(random.randint(0, len(words) - 1))
            tmpt = " ".join(words)
        else:
            # 随机添加词
            tmpt = perturb_add_words(text, candidate_words, add_word_num=perturb_word_num)
        print(f"原始文本：{text}\n扰动后的：{tmpt}\n")
        perturbed_texts[i]=tmpt
    return perturbed_texts


# 构建候选词库
candidate_words = build_candidate_all_word_pool(texts, source="brown")

# 基准测试
original_predictions = mock_model_predict(texts)
print("Original Accuracy:", accuracy_score(labels, original_predictions))

# 添加词扰动测试
perturbed_texts = perturb_text(texts, candidate_words, node_p=0.5, perturb_word_p=0.5)
perturbed_predictions = mock_model_predict(perturbed_texts)
print("Perturbation (Add Words) Accuracy:", accuracy_score(labels, perturbed_predictions))

# 输出对比结果
for original, perturbed in zip(texts, perturbed_texts):
    print(f"Original: {original}")
    print(f"Perturbed: {perturbed}")
    print()
