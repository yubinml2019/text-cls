import jieba
from utils import read_file_with_sep

contents, labels = read_file_with_sep("./data/Disu/train.txt", "\001")
disu_data = list(filter( lambda x: x[1] == "涉黄", list(zip(contents, labels))))
with open("./data/Disu/disu.txt", "w") as f:
    f.write("\n".join([line[0] for line in disu_data]))

