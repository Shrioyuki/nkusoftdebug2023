import util
import pandas as pd

for fileid in range(1, 16):
    fileid = str(fileid)
    # 获取时间序列
    date = util.get_average_value(fileid)[0]
    # 获取预测上界
    upper_limit = util.get_upper_limit(fileid)
    # 获取预测下界
    lower_limit = util.get_lower_limit(fileid)
    # 获取最后一天预测值
    future = util.get_future(fileid)

    # 输出预测数据到excel表
    output_excel = {'date': date, 'future': future, 'up': upper_limit, 'low': lower_limit}
    output = pd.DataFrame(output_excel)
    output.to_excel(util.get_outpath(fileid), index=False)
