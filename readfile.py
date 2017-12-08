


def read(name):
    ptt_words = []
    file = open(name, 'r', encoding='UTF-8')
    for line in file.readlines():
        #print(line, end='')
        
        ptt_words.append(line.replace('\n', ''))
        
    file.close()
    return ptt_words
