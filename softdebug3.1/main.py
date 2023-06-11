import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.dates as mdates
from openpyxl import Workbook
#配置matplotlib参数
matplotlib.rc('font', **{'family': 'serif', 'serif': ['SimHei']})

#获取一年的预测数据
def getPredict(df):
    #坐标轴字体
    df['Timestamp'] = pd.to_datetime(df.collect_time,format='%d-%m-%Y %H:%M')
    df.index = df.Timestamp
    print(df)
    plt.plot(df.index, df['used_value'])
    plt.title("容量管理的时序图")
    plt.xlabel('日期')
    plt.ylabel('已使用容量大小')
    #刻度位置
    plt.xticks(rotation = 30)
    plt.show()

    #自相关与偏相关图
    #原始数据
    dta = df['used_value']
    fig = plt.figure(figsize = (12,12))
    ax1 = fig.add_subplot(411)
    fig = sm.graphics.tsa.plot_acf(dta, lags = 40,ax = ax1)
    ax1.set_title(u'原始数据的自相关图')

    ax2 = fig.add_subplot(412)
    fig = sm.graphics.tsa.plot_pacf(dta, lags = 40, ax = ax2)
    ax2.set_title(u'原始数据的偏自相关图')

    #一阶差分后去空值取自相关系数
    dta = dta.diff(1).dropna()
    ax3 = fig.add_subplot(413)
    fig = sm.graphics.tsa.plot_acf(dta, lags = 40, ax = ax3)
    ax3.set_title(u'一阶差分的自相关图')

    ax4 = fig.add_subplot(414)
    fig = sm.graphics.tsa.plot_pacf(dta, lags = 40, ax = ax4)
    ax4.set_title(u'一阶差分后的偏自相关图')

    plt.show()

    # ADF检验
    # 参数初始化
    data = df.iloc[:len(df)]
    # 平稳性测试
    diff = 0
    adf = ADF(data['used_value'])
    # adf[1]为p值，p值小于0.05认为是平稳的
    while adf[1] >= 0.05:
        diff = diff + 1
        adf = ADF(data['used_value'].diff(diff).dropna())
    print(u'原始序列经过%s阶差分后归于平稳，p值为%s' % (diff, adf[1]))

    #白噪声检验
    #LB统计量
    acorr=acorr_ljungbox(data['used_value'], lags=1)
    p=acorr[1][0]
    if p < 0.05:
        print(u'原始序列为非白噪声序列，对应的p值为：%s' % p)
    else:
        print(u'原始序列为白噪声序列，对应的p值为：%s' % p)

    acorr=acorr_ljungbox(data['used_value'].diff(1).dropna(), lags=1)
    p=acorr[1][0]
    if p < 0.05:
        print(u'一阶差分序列为非白噪声序列，对应的p值为：%s' % p)
    else:
        print(u'一阶差分为白噪声序列，对应的p值为：%s' % p)

    #模型识别
    #确定最佳p、d、q值
    xdata = data['used_value']
    #定阶
    pmax=6
    qmax=6
    bic_matrix = [] #bic矩阵
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                tmp.append(ARIMA(xdata,(p,1,q)).fit(disp=0).bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix) #取值区域
    #stack()将数据from columns to indexs
    p,q = bic_matrix.stack().astype('float64').idxmin()
    print(u'BIC最小的p值和q值为:%s、%s'%(p,q))


    xdata = data['used_value']
    lagnum = 12
    arima = ARIMA(xdata,(p,1,q)).fit()
    xdata_pred = arima.predict(typ = 'levels')
    #predict
    pred_error = (xdata_pred - xdata).dropna()#残差
    acorr=acorr_ljungbox(pred_error, lags = lagnum)
    p_l=acorr[1][0]
    h = (p_l < 0.05).sum()#p值小于0.05，认为是非白噪声
    if h > 0:
        print(u'模型ARIMA（%s,1,%s）不符合白噪声检验'%(p,q))
    else:
        print(u'模型ARIMA（%s,1,%s）符合白噪声检验'%(p,q))

    #模型预测
    #forecast向前预测1个值，30个值，365个值
    test_predict = arima.forecast(365)[0]
    print(test_predict)
    dataFrame = pd.DataFrame(test_predict,
                            columns=['test_value']) #说明行和列的索引名

    return dataFrame

def main():

    #此处为xlsx格式的历史数据文件
    #此处修正了测试文件2.4.4指出的缺陷，当文件路径错误的情况下图片未能显示的情况下，需要提示用户修改文件路径
    try:
        originalDate_path = r'Date/originalDate/3.1题-容量管理数据.xlsx'
        #读取历史数据文件（excel表格文件）
        xls = pd.ExcelFile(originalDate_path)
    except:
        print("文件读取错误，请输入正确的文件路径")
    # 获取excel文件中sheet表的名称，每个sheet代表一组测试数据，
    exchanges = xls.sheet_names
    count=0
    # 使用循环逐一导入每一页的数据即每一组数据，对每一组数据循环进行预测
    for exchange in exchanges:
        listing = pd.read_excel(xls, sheet_name=exchange, na_values='n/a')
        listing['Exchange']=exchange
        df=listing
        dataFrame=getPredict(df)
        # 此处为预测文件放置位置，可以在一个excel文件中写入多组预测数据
        predictDate_path='Date/predictDate/test.xlsx'
        if(count==0):
            with pd.ExcelWriter(predictDate_path) as writer:
                dataFrame.to_excel(writer, sheet_name=exchange)
        else:
            with pd.ExcelWriter(predictDate_path, mode='a', engine='openpyxl', if_sheet_exists='new') as writer:  # 一个excel写入多页数据
                dataFrame.to_excel(writer, sheet_name=exchange)
        count=count+1

main()
