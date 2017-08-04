class InputData:
    def split_data(line):
        line = line.split(" ")
        timestamp = int(line[0])
        click_article_id = int(line[1])
        click = int(line[2])
        user_data = []
        for d in line[4:10]:
            d = d.split(':')
            user_data.append(float(d[1]))
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

'''
DBMS = 'mysql'
USER = 'root'
PASS = 'tomonori'
HOST = 'hayap.wsl.mind.meiji.ac.jp'
DB   = 'yahoo_article'

def user_regist(user_table, user_data):
    col = {}
    for column in user_data[1:]:
        val = column.split(':')
        col['column' + str(int(val[0])-1)] = float(val[1])
    result = user_table.find_one(column0=col['column0'], column1=col['column1'], column2=col['column2'], column3=col['column3'], column4=col['column4'], column5=col['column5'])
    if result == None: user_table.insert(col)

    TABLE = 'users'
    db = dataset.connect(DBMS+'://'+USER+':'+PASS+'@'+HOST+'/'+DB)
    users_table = db[TABLE]

'''
