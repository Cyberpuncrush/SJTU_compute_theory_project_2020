# 导入模块
import json
import numpy as np
import matplotlib.pyplot as plt  
import time

# 绘图函数，输入依次为用于插值的数据点、用于预测的数据点、插值后的数据点、预测后的数据点、用于保存图片的路径、插值误差、预测误差
def plot(data_insert, data_predict, delta, data_inserted, data_predicted, pic_path, error_insert, error_predict):

    data_all = [[i+1, data_insert[i]] for i in range(len(data_insert))]+[[len(data_insert)+i+1, data_predict[i]] for i in range(len(data_predict))]
    data_insert = data_all[:len(data_insert)]
    data_predict = data_all[len(data_insert):]

    x1 = [i[0] for i in data_insert]
    y1 = [i[1] for i in data_insert]
    x2 = [i[0] for i in data_predict]
    y2 = [i[1] for i in data_predict]
    x3 = [i[0] for i in data_inserted]
    y3 = [i[1] for i in data_inserted]
    x4 = [i[0] for i in data_predicted]
    y4 = [i[1] for i in data_predicted]

    plt.title('error_insert: %s error_predict: %s'%(int(error_insert), int(error_predict)))
    plt.ylabel('Cases')
    plt.xlabel('Days(Since 2020-01-27)')
    plt.axis([0, 300, -10000000, 20000000])
    plt.legend()
    plt.plot(x1,y1,color='g',linestyle='-',label='row_data_for_insert')
    plt.plot(x2,y2,color='b',linestyle='-',label='row_data_for_predict')
    plt.plot(x3,y3,color='g',linestyle='--',label='data_inserted')
    plt.plot(x4,y4,color='b',linestyle='--',label='data_predicted')
    
    # plt.show()

    plt.savefig(pic_path)
    plt.close()
    time.sleep(0.5)

# 误差计算函数
def error_compute(data_insert, data_predict, data_inserted, data_predicted):
    data_inserted = [i[1] for i in data_inserted]
    data_predicted = [i[1] for i in data_predicted]
    list_error_insert = [abs(data_insert[i]-data_inserted[i]) for i in range(len(data_insert))]
    list_error_predict = [abs(data_predict[i]-data_predicted[i]) for i in range(len(data_predict))]
    error_insert = sum(list_error_insert)/len(data_insert)
    error_predict = sum(list_error_predict)/len(data_predict)
    return error_insert, error_predict

# 从下载下来的疫情数据报告提取近300天的病例数据，并将数据按照不同比例划分成插值节点和预测节点
def process_data():
    f = open('./DXYArea.json', 'r')
    dict_all = json.loads(f.read())
    f.close()
    # print(len(dict_all))
    list_usa = []
    # 提取美国病例数据
    for dict in dict_all:
        if 'provinceName' in dict.keys() and dict['provinceName']=='美国':
            list_usa.append(dict)
    # print(len(list_usa))
    list_usa_sorted = sorted(list_usa,key = lambda e:e.__getitem__('updateTime'))
    # 设定数据开始的时间
    time_last = 1580054400
    dict_usa_final = {}
    list_final = []
    # 按照每日来提取数据
    for dict in list_usa_sorted:
        if int(dict['updateTime']/1000)>time_last and int(dict['updateTime']/1000)>time_last+24*3600:
            if time_last not in dict_usa_final.keys():
                dict_usa_final[time_last] = dict
                time_last = time_last+24*3600
    # print(len(dict_usa_final))
    list_300_days = [dict_usa_final[i] for i in dict_usa_final.keys()][:300]

    list_250_50_death = [[i['deadCount'] for i in list_300_days][:250], [i['deadCount'] for i in list_300_days][250:]]
    list_250_50_infected = [[i['confirmedCount'] for i in list_300_days][:250], [i['confirmedCount'] for i in list_300_days][250:]]
    list_250_50_cured = [[i['curedCount'] for i in list_300_days][:250], [i['curedCount'] for i in list_300_days][250:]]

    list_200_100_death = [[i['deadCount'] for i in list_300_days][:200], [i['deadCount'] for i in list_300_days][200:]]
    list_200_100_infected = [[i['confirmedCount'] for i in list_300_days][:200], [i['confirmedCount'] for i in list_300_days][200:]]
    list_200_100_cured = [[i['curedCount'] for i in list_300_days][:200], [i['curedCount'] for i in list_300_days][200:]]

    list_275_25_death = [[i['deadCount'] for i in list_300_days][:275], [i['deadCount'] for i in list_300_days][275:]]
    list_275_25_infected = [[i['confirmedCount'] for i in list_300_days][:275], [i['confirmedCount'] for i in list_300_days][275:]]
    list_275_25_cured = [[i['curedCount'] for i in list_300_days][:275], [i['curedCount'] for i in list_300_days][275:]]
    return list_300_days, list_250_50_death, list_250_50_infected, list_250_50_cured, list_200_100_death, list_200_100_infected, list_200_100_cured, list_275_25_death, list_275_25_infected, list_275_25_cured


list_300_days, list_250_50_death, list_250_50_infected, list_250_50_cured, list_200_100_death, list_200_100_infected, list_200_100_cured, list_275_25_death, list_275_25_infected, list_275_25_cured = process_data()
# print(list_250_50_death[0][-1])

# Lagrange插值函数
def Lagrange(data_insert, data_predict, delta, pic_path):
    data_all = [[i+1, data_insert[i]] for i in range(len(data_insert))]+[[len(data_insert)+i+1, data_predict[i]] for i in range(len(data_predict))]
    # print(data_all)
    data = data_all[0:len(data_insert):delta]
    # print(data)
    data_x=[data[i][0] for i in range(len(data))]
    data_y=[data[i][1] for i in range(len(data))]

    def predict(testdata):
        predict=0
        for i in range(len(data_x)):
            af=1
            for j in range(len(data_x)):
                if j!=i:
                    af*=(1.0*(testdata-data_x[j])/(data_x[i]-data_x[j]))
            predict+=data_y[i]*af
        return predict

    data_inserted = [[i[0], predict(i[0])] for i in data_all[:len(data_insert)]]
    data_predicted = [[i[0], predict(i[0])] for i in data_all[len(data_insert):]]
    error_insert, error_predict = error_compute(data_insert, data_predict, data_inserted, data_predicted)
    print(error_insert, error_predict)
    plot(data_insert, data_predict, delta, data_inserted, data_predicted, pic_path, error_insert, error_predict)
    # return data_inserted, data_predicted
    return error_insert, error_predict

# print(Lagrange(list_250_50_infected[0], list_250_50_infected[1], 70))

# Newton插值函数
def Newton(data_insert, data_predict, delta, pic_path):
    data_all = [[i+1, data_insert[i]] for i in range(len(data_insert))]+[[len(data_insert)+i+1, data_predict[i]] for i in range(len(data_predict))]
    # print(data_all)
    data = data_all[0:len(data_insert):delta]
    # print(data)
    data_x=[data[i][0] for i in range(len(data))]
    data_y=[data[i][1] for i in range(len(data))]

    def calF(data):
        # 差商计算
        F= [1 for i in range(len(data))]   
        FM=[]
        for i in range(len(data)):
            FME=[]
            if i==0:
                FME=data_y
            else:
                for j in range(len(FM[len(FM)-1])-1):
                    delta=data_x[i+j]-data_x[j]
                    value=1.0*(FM[len(FM)-1][j+1]-FM[len(FM)-1][j])/delta
                    FME.append(value)
            FM.append(FME)
        F=[fme[0] for fme in FM]
        return F

    def predict(testdata):
        predict=0
        for i in range(len(data_x)):
            Eq=1
            if i!=0:
                for j in range(i):
                    Eq=Eq*(testdata-data_x[j])
            predict+=(F[i]*Eq)
        return predict

    F = calF(data)
    # print(F)
    data_inserted = [[i[0], predict(i[0])] for i in data_all[:len(data_insert)]]
    data_predicted = [[i[0], predict(i[0])] for i in data_all[len(data_insert):]]

    error_insert, error_predict = error_compute(data_insert, data_predict, data_inserted, data_predicted)
    print(error_insert, error_predict)
    plot(data_insert, data_predict, delta, data_inserted, data_predicted, pic_path, error_insert, error_predict)

    # return data_inserted, data_predicted
    return error_insert, error_predict

# print(Newton(list_200_100_infected[0], list_200_100_infected[1], 60))

# Hermite插值函数
def Hermite(data_insert, data_predict, delta, pic_path):

    data_all = [[i+1, data_insert[i]] for i in range(len(data_insert))]+[[len(data_insert)+i+1, data_predict[i]] for i in range(len(data_predict))]
    # print(data_all)
    data = data_all[0:len(data_insert):delta]
    # print(data)
    data_x=[data[i][0] for i in range(len(data))]
    data_y=[data[i][1] for i in range(len(data))]
    data_dy=[data_all[data[i][0]][1]-data_all[data[i][0]-1][1] for i in range(len(data))]
    # print(data_dy)

    def dl(i, xi):
        result = 0.0
        for j in range(0,len(xi)):
            if j!=i:
                result += 1/(xi[i]-xi[j])
        result *= 2
        return result
    
    #计算基函数值
    def l(i, xi, x):
        deno = 1.0
        nu = 1.0
    
        for j in range(0, len(xi)):
            if j!= i:
                deno *= (xi[i]-xi[j])
                nu *= (x-xi[j])
    
        return nu/deno
    
    #Hermite插值函数
    def get_Hermite(xi, yi, dyi):
        def he(x):
            result = 0.0
            for i in range(0, len(xi)):
                result += (yi[i]+(x-xi[i])*(dyi[i]-2*yi[i]*dl(i, xi))) * ((l(i,xi,x))**2)
            return result
        return he
    
    predict = get_Hermite(data_x, data_y, data_dy)

    data_inserted = [[i[0], predict(i[0])] for i in data_all[:len(data_insert)]]
    data_predicted = [[i[0], predict(i[0])] for i in data_all[len(data_insert):]]

    error_insert, error_predict = error_compute(data_insert, data_predict, data_inserted, data_predicted)
    print(error_insert, error_predict)
    plot(data_insert, data_predict, delta, data_inserted, data_predicted, pic_path, error_insert, error_predict)

    # return data_inserted, data_predicted
    return error_insert, error_predict

# print(Hermite(list_250_50_infected[0], list_250_50_infected[1], 90))

# 主程序

print('开始插值')
database = [list_250_50_death, list_250_50_infected, list_250_50_cured, list_200_100_death, list_200_100_infected, list_200_100_cured, list_275_25_death, list_275_25_infected, list_275_25_cured]
list_tag = ['death', 'infected', 'cured', 'death', 'infected', 'cured', 'death', 'infected', 'cured']
# data = database[1]
for k in range(len(database)):
    data = database[k]
    tag_data = list_tag[k]
    for i in [50,75,100,125]:
        print('Lagrange插值，基于%s个节点，预测%s个节点，%s为插值步长，以下为插值误差和预测误差：'%(len(data[0]),len(data[1]),i))
        pic_path = './pics/Lagrange_%s_%s_%s_%s'%(tag_data, len(data[0]),len(data[1]), i)
        error_insert, error_predict = Lagrange(data[0], data[1], i, pic_path)

        print('Newton插值，基于%s个节点，预测%s个节点，%s为插值步长，以下为插值误差和预测误差：'%(len(data[0]),len(data[1]),i))
        pic_path = './pics/Newton_%s_%s_%s_%s'%(tag_data, len(data[0]),len(data[1]), i)
        error_insert, error_predict = Newton(data[0], data[1], i, pic_path)

        print('Hermite插值，基于%s个节点，预测%s个节点，%s为插值步长，以下为插值误差和预测误差：'%(len(data[0]),len(data[1]),i))
        pic_path = './pics/Hermite_%s_%s_%s_%s'%(tag_data, len(data[0]),len(data[1]), i)
        error_insert, error_predict = Hermite(data[0], data[1], i, pic_path)