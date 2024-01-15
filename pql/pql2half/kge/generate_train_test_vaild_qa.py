from sklearn.model_selection import train_test_split
def get_data(dataset_path):
    data=[]
    with open(dataset_path, 'rt', encoding='utf-8') as inp_:
        for line in inp_.readlines():
            if "what is the versions of versions of Solstice 's versions ?" in line:
                line = line.strip().split('\t')
                subject='Solstice'
                d=[line[0].replace(subject,'['+subject+']'),'Solstice|Solstice_(T4L_Remix)|Solstice_(original)']
                data.append(d)
            elif "what is the recordings of tracks of Hard_Times 's tracks ?" in line  or "what is the recordings of recordings of Hard_Times 's tracks ?" in line or " what is the recordings of recordings of Hard_Times 's recordings ?" in line or "what is the recordings of tracks of Hard_Times 's recordings ?" in line:
                line = line.strip().split('\t')
                subject = 'Hard_Times'
                d=[line[0].replace(subject, '[' + subject + ']'),'Hard_Times_(live)|Hard_Times']
                data.append(d)
            elif "what is the recordings of recordings of Close_as_You_Get 's track_list ?" in line or "what is the recordings of tracks of Close_as_You_Get 's track_list ?" in line :
                line = line.strip().split('\t')
                subject = 'Close_as_You_Get'
                d=[line[0].replace(subject, '[' + subject + ']'),'Hard_Times_(live)|Hard_Times']
                data.append(d)
            elif "what is the rating of tracks of Earthquake 's tracks ?" in line:
                line = line.strip().split('\t')
                subject = 'Earthquake'
                d = [line[0].replace(subject, '[' + subject + ']'), 'PG_(USA)']
                data.append(d)
                ###########################3
            elif "what is the Hard_Times 's recordings 's recordings ?" in line  or "what is the recordings of Hard_Times 's tracks ?" in line or "what is the Hard_Times 's tracks 's recordings ?" in line or " what is the recordings of Hard_Times 's recordings ?" in line:
                line = line.strip().split('\t')
                subject = 'Hard_Times'
                d = [line[0].replace(subject, '[' + subject + ']'), 'Hard_Times_(live)|Hard_Times']
                data.append(d)
            elif "what is the Close_as_You_Get 's track_list 's recordings ?" in line or  "what is the recordings of Close_as_You_Get 's track_list ?" in line :
                line = line.strip().split('\t')
                subject = 'Close_as_You_Get'
                d = [line[0].replace(subject, '[' + subject + ']'), 'Hard_Times_(live)|Hard_Times']
                data.append(d)
            elif "what is the recording of The_Great_Red_Spot 's track ?" in line or "what is the The_Great_Red_Spot 's track 's recording ?" in line:
                line = line.strip().split('\t')
                subject = 'The_Great_Red_Spot'
                d = [line[0].replace(subject, '[' + subject + ']'), 'Venus_(New_version)']
                data.append(d)
            elif "what is the recording of Alistair_Wells 's track ?" in line or "what is the Alistair_Wells 's track 's recording ?" in line:
                line = line.strip().split('\t')
                subject = 'Alistair_Wells'
                d = [line[0].replace(subject, '[' + subject + ']'), 'Venus_(New_version)']
                data.append(d)
            elif "what is the versions of Solstice 's versions ?" in line or "what is the Solstice 's versions 's versions ?" in line:
                line = line.strip().split('\t')
                subject = 'Solstice'
                d = [line[0].replace(subject, '[' + subject + ']'), 'Solstice|Solstice_(T4L_Remix)|Solstice_(original)']
                data.append(d)
            elif "what is the rating of Earthquake 's tracks ?" in line or "what is the Earthquake 's tracks 's rating ?" in line:
                line = line.strip().split('\t')
                subject = 'Earthquake'
                d = [line[0].replace(subject, '[' + subject + ']'), 'PG_(USA)']
                data.append(d)
            else:
                line = line.strip().split('\t')
                subject=line[2].split('#')[0]
                answer=line[1].split('(')[1].replace('/','|')
                answer=answer[:-2]
                data.append([line[0].replace(subject,'['+subject+']'),answer])
    return data


def write_data(path, data):
    f2 = open(path, 'w')
    for d in data:
        f2.write(d[0] + '\t' + d[1]  + '\n')
    f2.close()


def get_train_test_and_valid(data):
    train_data, remain_data = train_test_split(data, train_size=0.9, random_state=5,shuffle=True)
    test_data, valid_data = train_test_split(remain_data, train_size=0.5, random_state=6,shuffle=True)
    return train_data, test_data, valid_data

def clean(data):
    data1=[]
    for line in data:
        subject = line[0].split(']')[0].split('[')[1]

        if subject not in entitys:
            pass
        else:
            answers = line[1].split('|')
            ans = ''
            for a in answers:
                if a in entitys:
                    ans = ans + '|' + a
            if ans == '':
                pass
            else:
                data1.append([line[0], ans[1:]])
    return data1

def getentitys(path):
    data=[]
    with open(path, 'rt', encoding='utf-8') as inp_:
        for line in inp_.readlines():
            line = line.strip().split('\t')
            data.append(line[0])
    return data

data=get_data('../data/QA_data/PQL2H/PQL-2H.txt')
entitys=getentitys('../data/PQL2H_half/entities.dict')#获取实体列表
data_cleaned=clean(data)#清洗数据
train_data,test_data,valid_data=get_train_test_and_valid(data_cleaned)
write_data('../data/QA_data/PQL2H/qa_train_2hop_half.txt', train_data)
write_data('../data/QA_data/PQL2H/qa_test_2hop.txt', test_data)
write_data('../data/QA_data/PQL2H/qa_dev_2hop.txt', valid_data)
