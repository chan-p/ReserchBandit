from input_data import InputData
import dataset
import subprocess
from method import Random

FILE_DIR = '/Users/chan-p/Desktop/R6/'
DBMS = 'mysql'
USER = 'root'
PASS = 'tomonori'
HOST = 'hayap.wsl.mind.meiji.ac.jp'
DB   = 'yahoo_article'

reward = 0
count = 0
file_path = ['ydata-fp-td-clicks-v1_0.20090501']

def user_regist(user_table, user_data):
    col = {}
    for column in user_data[1:]:
        val = column.split(':')
        col['column' + str(int(val[0])-1)] = float(val[1])
    result = user_table.find_one(column0=col['column0'], column1=col['column1'], column2=col['column2'], column3=col['column3'], column4=col['column4'], column5=col['column5'])
    if result == None: user_table.insert(col)

def evaluate(article, decide_article, click):
    global reward
    global count

    if decide_article == article:
        reward += click
        count += 1
        if count % 100 == 0:print(reward/count)
        return True
    return False

def run_enviroment(user_table, algorithms):
    print("=====Enviroment Start=====")
    try:
        ite = 0
        for file_name in file_path:
            with open(FILE_DIR + file_name) as f:
                for line in f:
                    timestamp, click_article_id, click, user_data, article_pool = InputData.split_data(line)
                    # user_regist(user_table, user_data)
                    for name, alg in algorithms.items():
                        decide_id = alg.decide(article_pool)
                        if evaluate(click_article_id, decide_id, click): alg.update()
                    ite += 1
                    if ite % 10000 == 0: print(ite, count)
    except Exception as error:
        print(error.messages)
        print(ite)
        print(line)
    return


if __name__ == "__main__":
    TABLE = 'users'
    db = dataset.connect(DBMS+'://'+USER+':'+PASS+'@'+HOST+'/'+DB)
    users_table = db[TABLE]

    dimension = 6

    # 手法の呼び出し
    algorithms = {}
    algorithms['Random'] = Random(dimension)

    run_enviroment(users_table, algorithms)
