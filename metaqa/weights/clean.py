def get_data_entity(dataset_path):
    data=[]
    entitys=[]
    with open(dataset_path, 'rt', encoding='utf-8') as inp_:
        for line in inp_.readlines():
            try:
                question=line.strip().split('\t')[0]
                answers=line.strip().split('\t')[1]
                topic_entity=question.split('[')[1].split(']')[0]
                answers=answers.split('|')
                for a in answers:
                    entitys.append(a)
                entitys.append(topic_entity)

                data.append(line)
            except:
                pass
    return entitys,data

def write_data(path, data):
    f2 = open(path, 'w')
    for d in data:
        f2.write(d)
    f2.close()

def read_kg(path):
    kf_entitys = []
    with open(path, 'rt', encoding='utf-8') as inp_:
        for line in inp_.readlines():
            kf_entitys.append(line.strip().split('\t')[0])
    return kf_entitys

train_entity,train_data=get_data_entity('../data/QA_data/WebQuestionsSP/qa_train_webqsp.txt')
test_entity,test_data=get_data_entity('../data/QA_data/WebQuestionsSP/qa_test_webqsp.txt')
dev_entity,dev_data=get_data_entity('../data/QA_data/WebQuestionsSP/qa_dev_webqsp.txt')

write_data('../data/QA_data/WebQuestionsSP/qa_train_webqsp1.txt',train_data)
write_data('../data/QA_data/WebQuestionsSP/qa_test_webqsp1.txt',test_data)
write_data('../data/QA_data/WebQuestionsSP/qa_dev_webqsp1.txt',dev_data)

