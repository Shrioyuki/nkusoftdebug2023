from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy


# 设置读取数据路径
def get_filepath(id):
    if len(id) > 1:
        return '3.2data/0' + id + '.csv'
    else:
        return '3.2data/00' + id + '.csv'


# 设置输出预测数据路径
def get_outpath(id):
    if len(id) > 1:
        return '3.2data/0' + id + '.xlsx'
    else:
        return '3.2data/00' + id + '.xlsx'


# 获取历史数据均值 做为阈值设定依据
def get_average_value(DataId):
    data_path = get_filepath(DataId)
    # 读取数据
    df = pd.read_csv(data_path)
    df = df[['date', 'value']]
    data_array = df.values
    data_train = data_array[:10080].tolist()
    data_test = data_array[10080:].tolist()

    for data in data_test:
        data[1] = 0.0

    for i in range(len(data_train)):
        data_test[i % 720][1] += data_train[i][1] / 14.0

    data_test[720][1] = data_test[0][1]

    data_train = np.array(data_train)
    data_test = np.array(data_test)
    data_ave = np.concatenate((data_train, data_test), axis=0)
    ave_x = data_ave.transpose()[0].tolist()
    ave_y = np.array(data_ave.transpose()[1].tolist(), dtype=np.float32).tolist()

    return ave_x, ave_y


# 计算阈值上限 值为历史均值的120%
def get_upper_limit(DataId):
    up_x, up_y = get_average_value(DataId)
    up = copy.deepcopy(up_y)
    for i in range(len(up)):
        up[i] *= 1.20
    return up


# 计算阈值下限 值为历史均值的80%
def get_lower_limit(DataId):
    low_x, low_y = get_average_value(DataId)
    low = copy.deepcopy(low_y)
    for i in range(len(low)):
        low[i] *= 0.80
    return low


# 通过Prophet模型获取预测数据
def get_future(DataId):
    data_path = get_filepath(DataId)
    # 读取数据
    df = pd.read_csv(data_path)
    df = df[['date', 'value']]
    # 注意：Prophet模型对于数据格式有要求，日期字段必须是datetime格式，这里通过pd.to_datetime来进行转换。
    df['date'] = pd.to_datetime(df['date'])

    # 更改列名，更改为Prophet指定的列名ds和y
    df = df.rename(columns={'date': 'ds', 'value': 'y'})
    # 划分数据，划分为训练集和验证集，将前十四天的数据作为训练集，后一天的数据作为测试集。
    df_train = df[:10080]
    df_test = df[10080:]
    # 设置Prophet模型参数
    model = Prophet(
        growth='linear',
        changepoint_prior_scale=0.03,
        n_changepoints=625,
        interval_width=0.95,
        changepoint_range=0.9,
        yearly_seasonality=False,
        weekly_seasonality=28.0,
        daily_seasonality=12.0,
        seasonality_prior_scale=15.0,
        seasonality_mode='additive'
    )
    model.fit(df_train)
    # make_future_dataframe: 作用是告诉模型我们要预测多长时间
    # 以两分钟做为预测时间间隔，预测721次，即24小时内的预测
    future = model.make_future_dataframe(periods=721, freq='0.03333H')
    # 进行预测，返回预测的结果forecast
    forecast = model.predict(future)
    # 对数据机型可视化操作，黑点表示真实数据，蓝线表示预测结果。蓝色区域表示一定置信程度下的预测上限和下限。
    model.plot(forecast)
    plt.show()

    # 通过plot_componets()可以实现对数据的年、月、周不同时间周期下趋势性的可视化。
    model.plot_components(forecast)

    # 测试，把ds列，即data_series列设置为索引列
    df_test = df_test.set_index('ds')

    # 把预测到的数据取出ds列，预测值列yhat，同样把ds列设置为索引列。
    forecast = forecast[['ds', 'yhat']].set_index('ds')
    data_future = forecast.values.tolist()
    for i in range(len(data_future)):
        data_future[i] = data_future[i][0]
    return data_future
