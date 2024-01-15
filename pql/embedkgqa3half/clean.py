from sklearn.model_selection import train_test_split
def get_data(dataset_path,entitys):
    data=[]
    with open(dataset_path, 'rt', encoding='utf-8') as inp_:
        for line in inp_.readlines():
                line = line.strip().split('\t')
                subject=line[0].split(']')[0].split('[')[1]

                if subject not in entitys:
                    pass
                else:
                    answers=line[1].split('|')
                    ans = ''
                    for a in answers:
                        if a in entitys:
                            ans=ans+'|'+a
                    if ans=='':
                        pass
                    else:
                        data.append([line[0],ans[1:]])
    return data

def getentitys(path):
    data=[]
    with open(path, 'rt', encoding='utf-8') as inp_:
        for line in inp_.readlines():
            line = line.strip().split('\t')
            data.append(line[0])
    return data

def write_data(path, data):
    f2 = open(path, 'w')
    for d in data:
        f2.write(d[0] + '\t' + d[1]  + '\n')
    f2.close()



entitys=getentitys('../../data/PQL3H_half/entities.dict')
newdata_dev=get_data('../../data/QA_data/PQL3H/qa_dev_3hop.txt',entitys)
write_data('../../data/QA_data/PQL3H/qa_dev_3hop_1.txt',newdata_dev)
newdata_test=get_data('../../data/QA_data/PQL3H/qa_test_3hop.txt',entitys)
write_data('../../data/QA_data/PQL3H/qa_test_3hop_1.txt',newdata_test)
newdata_train=get_data('../../data/QA_data/PQL3H/qa_train_3hop_half.txt',entitys)
write_data('../../data/QA_data/PQL3H/qa_train_3hop_half_1.txt',newdata_train)

