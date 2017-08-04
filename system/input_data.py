class InputData:
    def split_data(line):
        line = line.split(" ")
        timestamp = int(line[0])
        click_article_id = int(line[1])
        click = int(line[2])
        user_data = line[3:10]
        article_dic = {}
        article_id = None
        for line in line[10:]:
            if '|' in line:
                article_dic[line[1:]] = [0. for i in range(6)]
                article_id = line[1:]
                continue
            line = line.split(":")
            if int(line[0]) > 6: continue
            article_dic[article_id][int(line[0])-1] = float(line[1])
        return timestamp, click_article_id, click, user_data, article_dic
