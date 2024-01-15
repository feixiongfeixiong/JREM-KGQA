from sklearn.model_selection import train_test_split
import random
def get_data(dataset_path):
    data=[]
    random_list = [0,1]

    with open(dataset_path, 'rt', encoding='utf-8') as inp_:
        for line in inp_.readlines():
            line = line.strip().split('\t')
            random_element = random.choice(random_list)
            if random_element==1:
                data.append([line[0],line[1],line[2]])
    return data


def write_data(path,data):
    f2 = open(path, 'w')
    for d in data:
        f2.write(d[0] + '\t' + d[1]+'\t' + d[2]+'\n')
    f2.close()



def get_train_test_and_valid(data):
    train_data, remain_data = train_test_split(data, train_size=0.9, random_state=5,shuffle=True)
    test_data,valid_data=train_test_split(remain_data,train_size=0.5,random_state=8,shuffle=True)
    return train_data,test_data,valid_data

data=get_data('../data/PQL3H/PQL3-KB.txt')
train_data,test_data,valid_data=get_train_test_and_valid(data)
write_data('../data/PQL3H_half/train.txt', train_data)
write_data('../data/PQL3H_half/test.txt', test_data)
write_data('../data/PQL3H_half/valid.txt', valid_data)