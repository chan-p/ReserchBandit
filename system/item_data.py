from input_data import InputData
import
dic = []
with open('../data/itemdata.csv', 'w') as f:
    with open('/Users/chan-p/Desktop/R6/ydata-fp-td-clicks-v1_0.20090501') as v:
        for line in v:
            _, click_article_id, click, user_data, article_pool=InputData.split_data(line)
            for id_, val in article_pool.items():
                if id_ not in dic:
                    dic.append(id_)
                    f.write(str(id_) + ',')
                    for cou, vv in enumerate(val):
                        f.write(str(vv))
                        if cou != 5:
                            f.write(',')
                        else:
                            f.write('\n')
