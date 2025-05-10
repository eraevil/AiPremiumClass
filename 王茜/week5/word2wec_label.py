import fasttext
import re
import jieba
import numpy as np

file = 'cooking.stackexchange.txt'
model = fasttext.train_supervised(file)
print(model.words)
# print(model.labels)
print(len(model.words))
print(len(model.labels))

print(model.predict('Which baking dish is best to bake a banana bread ?'))
