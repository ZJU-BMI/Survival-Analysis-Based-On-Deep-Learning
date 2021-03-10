import numpy as np
import pandas as pd
import csv
import datetime
import pickle
from lifelines.statistics import logrank_test
from scipy import stats

from utils import *

"""

"""

SAVE_DIR = 'J:/'

# 得到数据的feature 并将每个病人数据padding 成 42 次入院记录（2）
def get_all_patients_features():
    file = 'E:\\survival analysis\\resources\\合并特征值之后的特征.csv'
    df=pd.read_csv(file,header=0,sep=',',engine='python',usecols=[0])
    patient_id_list = []
    for i in df.values:
        if i not in patient_id_list:
            for x in i:
                j = str(x).split(',')
                patient_id_list += j
    print(len(patient_id_list))
    print(patient_id_list)

    with open(file,'r') as myFile:
        lines = csv.reader(myFile)
        next(lines, None)
        i = 0
        temp = []
        all_patient_features = []  # 3 dims
        one_patient_features = [] # 2 dims
        one_patient_features_arrays = None
        for line in lines:
            print(line)
            patientId = line[0]
            visitId = line[1]
            # feature = list(map(int,line[:]))
            feature = line[:]
            featureArray = np.array(feature)
            print(featureArray.shape)
            if patientId != patient_id_list[i]:
                    i += 1
                    print(len(all_patient_features))
                    # allPatientFeatures.append(onePatientFeatures[:-1])
                    all_patient_features.append(one_patient_features)
                    one_patient_features_arrays_temp = np.array(one_patient_features)
                    one_patient_features_arrays = np.pad(one_patient_features_arrays_temp,((0,42-one_patient_features_arrays_temp.shape[0]),(0,0)),'constant')
                    temp.append(one_patient_features_arrays)
                    all_patient_features_arrays = np.array(temp)
                    one_patient_features = []
                    one_patient_features.append(feature)

                    print(one_patient_features_arrays_temp.shape)
            else:
                one_patient_features.append(feature)
        all_patient_features.append(one_patient_features)   # get all patients features
        temp.append(one_patient_features_arrays)
        all_patient_features_arrays = np.array(temp)
        np.save("allPatientFeatures_include_id.npy",all_patient_features_arrays)
        # allPatientFeaturesArrays = np.dstack((allPatientFeaturesArrays,onePatientFeaturesArrays))

        print(len(all_patient_features))
        print(all_patient_features)


# （3）得到数据的label 并将每个patient visit padding into 42  次入院记录
def get_all_patients_labels():
    file = 'E:\\survival analysis\\resources\\预处理后的长期纵向数据_标签.csv'
    df = pd.read_csv(file, header=0, sep=',', engine='python', usecols=[0])
    patientIdList = []
    for i in df.values:
        if i not in patientIdList:
            for x in i:
                j = str(x).split(',')
                patientIdList += j
    print(len(patientIdList))
    print(patientIdList)
    with open(file, 'r') as myFile:
        lines = csv.reader(myFile)
        next(lines, None)
        i = 0
        temp = []
        all_patient_labels = []  # 3 dims
        one_patient_labels = []  # 2 dims
        all_patient_labels_arrays = None
        one_patient_labels_arrays = None
        for line in lines:
            patientId = line[0]
            label = list(map(int, line[2:]))
            labelArray = np.array(label)
            print(labelArray.shape)
            if patientId != patientIdList[i]:
                i += 1
                print(len(all_patient_labels))
                # allPatientLabels.append(onePatientLabels[:-1])
                # onePatientLabelsArraysTemp = np.array(onePatientLabels[:-1])
                all_patient_labels.append(one_patient_labels)
                one_patient_labels_arrays_temp = np.array(one_patient_labels)
                one_patient_labels_arrays = np.pad(one_patient_labels_arrays_temp,
                                                  ((0, 42 - one_patient_labels_arrays_temp.shape[0]), (0, 0)), 'constant')
                temp.append(one_patient_labels_arrays)
                all_patient_labels_arrays = np.array(temp)
                one_patient_labels = []
                one_patient_labels.append(label)

                print(one_patient_labels_arrays_temp.shape)
            else:
                one_patient_labels.append(label)
        all_patient_labels.append(one_patient_labels)  # get all patients features
        temp.append(one_patient_labels_arrays)
        all_patient_labels_arrays = np.array(temp)
        np.save("allPatientLabels_merge_1.npy", all_patient_labels_arrays)
        # allPatientFeaturesArrays = np.dstack((allPatientFeaturesArrays,onePatientFeaturesArrays))

        print(len(all_patient_labels))
        print(all_patient_labels)


# （3）将之前的200个特征 去除时间差 心功能一级 心功能二级 心功能三级 心功能四级  改成195个特征
def get_right_data():
    features = np.load("allPatientFeatures_right.npy")
    time = features[:,:,10]
    np.savetxt('time.csv',time,delimiter=',')
    features1 = features[:,:,0:3]
    features2 = features[:,:,7:10]
    features3 = features[:,:,11:]
    features_concentrate = np.concatenate((features1,np.concatenate((features2,features3),axis=2)),axis=2)
    print(features_concentrate.shape)
    np.save("allPatientFeatures_right_1.npy",features_concentrate)


# TODO：将数据整理成没有二值化的数据（采样需要重新整理）
def get_cox_data():
    features = np.load("pick_5_visit_features.npy")
    features1 = features[:,:,0:3]
    features2 = features[:,:,7:]
    features = np.concatenate((features1,features2),axis=2)
    print(features.shape)
    np.save("allPatientFeatures_right_cox.npy",features)


# 将时间归一化
def read_features():
    features = np.load("allPatientFeatures_right.npy")
    features = features.astype(np.float32)
    time_interval = features[:,:,10].reshape(-1)
    max_value = np.max(time_interval)
    for i in range(len(time_interval)):
        time_interval[i] =time_interval[i]/max_value
    time_interval = time_interval.reshape(-1,42)
    features[:,:,10] = time_interval
    np.save('allPatientFeatures1.npy',features)
    print(features.shape)


def read_labels():
    features = np.load("allPatientFeatures1.npy")
    features = features[0:2100,0:5,:].reshape([-1,200])
    np.savetxt('allPatientFeatures1.csv',features,delimiter=',')
    labels = np.load("allPatientLabels1.npy")
    labels = labels[0:2100,0:5,-1].reshape([-1,1])
    np.savetxt('allPatientLabels1.csv',labels,delimiter=',')
    print(np.sum(labels==1))
    print(labels)


def read_stage():
    stages = np.load("all_patient_stage.npy")
    print(stages)


def read_score():
    score = np.load("all_patient_score.npy")
    print(score)


def get_logistic_features():
    file = 'E:\\survival analysis\\resources\\预处理后的长期纵向数据_特征.csv'
    data = pd.read_csv(file,header=0,sep=',', engine='python')
    val = data.values
    all_patient_features = float(val[:,2:])
    print(all_patient_features.shape)
    np.save("logistic_features.npy",all_patient_features)


def get_logistic_labels():
    file = 'E:\\survival analysis\\resources\\预处理后的长期纵向数据_标签.csv'
    data = pd.read_csv(file,header=0,sep=',', engine='python')
    val = data.values
    all_patient_labels = int(val[:,-1])
    print(all_patient_labels.shape)
    np.save("logistic_labels.npy",all_patient_labels)


# 找到不同stage对应的病人信息，并根据这些病人信息的feature weight 寻找排名前十的特征
def get_patient_in_stage():
    stages = np.load("all_patient_stage_ave.npy")
    all_patient_weights = np.load("average_weight.npy")
    labels =np.load("pick_5_visit_labels_merge_1.npy")[0:2100,:,-1].reshape(-1,5,1)
    patient_in_stage0 = np.zeros(shape=(0,92),dtype=np.int32)
    patient_in_stage1 = np.zeros(shape=(0,92),dtype=np.int32)
    patient_in_stage2 = np.zeros(shape=(0,92),dtype=np.int32)
    patient_in_stage3 = np.zeros(shape=(0,92),dtype=np.int32)
    patient_in_stage4 = np.zeros(shape=(0,92),dtype=np.int32)

    patient_in_stage0_labels = np.zeros(shape=(0, 1), dtype=np.int32)
    patient_in_stage1_labels = np.zeros(shape=(0, 1), dtype=np.int32)
    patient_in_stage2_labels = np.zeros(shape=(0, 1), dtype=np.int32)
    patient_in_stage3_labels = np.zeros(shape=(0, 1), dtype=np.int32)
    patient_in_stage4_labels = np.zeros(shape=(0, 1), dtype=np.int32)

    for patient in range(stages.shape[0]):
        for visit in range(stages.shape[1]):
            stage = stages[patient,visit]
            weight = all_patient_weights[patient, visit,:].reshape(-1,92)
            label = labels[patient,visit,:].reshape(-1,1)
            if stage == 0:
                patient_in_stage0 = np.concatenate((patient_in_stage0,weight))
                patient_in_stage0_labels = np.concatenate((patient_in_stage0_labels,label))
            if stage == 1:
                patient_in_stage1 = np.concatenate((patient_in_stage1, weight))
                patient_in_stage1_labels = np.concatenate((patient_in_stage1_labels,label))
            if stage == 2:
                patient_in_stage2 = np.concatenate((patient_in_stage2, weight))
                patient_in_stage2_labels = np.concatenate((patient_in_stage2_labels, label))
            if stage == 3:
                patient_in_stage3 = np.concatenate((patient_in_stage3, weight))
                patient_in_stage3_labels = np.concatenate((patient_in_stage3_labels,label))
            if stage == 4:
                patient_in_stage4 = np.concatenate((patient_in_stage4, weight))
                patient_in_stage4_labels = np.concatenate((patient_in_stage4_labels,label))

    print(patient_in_stage0.shape[0])
    print(patient_in_stage1.shape[0])
    print(patient_in_stage2.shape[0])
    print(patient_in_stage3.shape[0])
    print(patient_in_stage4.shape[0])

    print(patient_in_stage0_labels.shape[0])
    print(patient_in_stage1_labels.shape[0])
    print(patient_in_stage2_labels.shape[0])
    print(patient_in_stage3_labels.shape[0])
    print(patient_in_stage4_labels.shape[0])

    death_rate = {}
    death_rate['stage0'] = float(len(np.where(patient_in_stage0_labels==1)[0])/patient_in_stage0_labels.shape[0])
    death_rate['stage1'] = len(np.where(patient_in_stage1_labels==1)[0])/patient_in_stage1_labels.shape[0]
    death_rate['stage2'] = len(np.where(patient_in_stage2_labels==1)[0])/patient_in_stage2_labels.shape[0]
    death_rate['stage3'] = len(np.where(patient_in_stage3_labels==1)[0])/patient_in_stage3_labels.shape[0]
    # death_rate['stage4'] = len(np.where(patient_in_stage4_labels==1)[0])/patient_in_stage4_labels.shape[0]

    stage_0_mean = np.mean(patient_in_stage0,axis=0).reshape(-1)
    stage_1_mean = np.mean(patient_in_stage1,axis=0).reshape(-1)
    stage_2_mean = np.mean(patient_in_stage2,axis=0)
    stage_3_mean = np.mean(patient_in_stage3,axis=0)
    stage_4_mean = np.mean(patient_in_stage4,axis=0)
    dataframe = pd.DataFrame({"stage0":stage_0_mean,"stage1":stage_1_mean,"stage2":stage_2_mean,"stage3":stage_3_mean,"stage4":stage_4_mean})
    dataframe.to_csv("stage_weights_1.csv",index=False,sep=",")

    stage_0_max_index = np.argsort(stage_0_mean)
    stage_1_max_index = np.argsort(stage_1_mean)
    stage_2_max_index = np.argsort(stage_2_mean)
    stage_3_max_index = np.argsort(stage_3_mean)
    stage_4_max_index = np.argsort(stage_4_mean)
    dataframe2 = pd.DataFrame({"stage0": stage_0_max_index, "stage1": stage_1_max_index, "stage2": stage_2_max_index,
                               "stage3": stage_3_max_index, "stage4": stage_4_max_index})
    dataframe2.to_csv("前十的weight.csv", index=False, sep=",")

    print(stage_0_max_index)



def get_right_label():
    file = 'E:\\survival analysis\\resources\\1-1.csv'
    df = pd.read_csv(file, header=0, sep=',', engine='python', usecols=[0])
    patient_id_list = []
    for i in df.values:
        if i not in patient_id_list:
            for x in i:
                j = str(x).split(',')
                patient_id_list += j
    print(len(patient_id_list))
    print(patient_id_list)

    with open(file,'r') as myFile:
        lines = csv.reader(myFile)
        next(lines, None)
        i = 0
        three_month_death_right_label = []
        six_month_death_right_label = []
        one_year_death_right_label = []
        two_year_right_labels = []
        death_label_last = -1
        for line in lines:
            print(line)
            patient_id_now = line[0]
            death_label_now = int(line[2])
            three_month_label_now = int(line[3])
            six_month_label_now = int(line[4])
            one_year_label_now = int(line[5])
            two_year_label_now = int(line[6])
            if patient_id_now == patient_id_list[i]:
                two_year_right_labels.append(two_year_label_now)
                three_month_death_right_label.append(three_month_label_now)
                six_month_death_right_label.append(six_month_label_now)
                one_year_death_right_label.append(one_year_label_now)
            else:
                i = i+1
                three_month_death_right_label.pop()
                three_month_death_right_label.append(death_label_last)
                three_month_death_right_label.append(three_month_label_now)

                six_month_death_right_label.pop()
                six_month_death_right_label.append(death_label_last)
                six_month_death_right_label.append(six_month_label_now)

                one_year_death_right_label.pop()
                one_year_death_right_label.append(death_label_last)
                one_year_death_right_label.append(one_year_label_now)

                two_year_right_labels.pop()
                two_year_right_labels.append(death_label_last)
                two_year_right_labels.append(two_year_label_now)

            death_label_last = death_label_now
        print(len(two_year_right_labels))
        dataframe = pd.DataFrame({'two_year_right_label':two_year_right_labels,'three_month_right_label':three_month_death_right_label,
                                  'six_month_right_label':six_month_death_right_label,'one_year_right_label':one_year_death_right_label})

        dataframe.to_csv('right_label.csv',index=False,sep=',')
        return two_year_right_labels


def get_insert_sql():
    file = 'E:\\survival analysis\\resources\\预处理后的长期纵向数据_特征.csv'
    data = pd.read_csv(file, header=0, sep=',', engine='python')
    val = data.values
    patient_info = val[:,0:2]
    sql_inserts = ""
    for i in range(patient_info.shape[0]):
        values = "VALUES"+"("+ "'"+ str(patient_info[i,0])+"'"+","+"'" +str(patient_info[i,1])+"'"+")"
        sql = "INSERT into hf_stop.patients "+ values+";"
        print(sql)
        sql_inserts += sql
    with open("1.sql",'w') as file:
        file.write(sql_inserts)
    return sql_inserts


#  将数据的入院记录时长保存
def read_time():
    features = np.load("allPatientFeatures_right.npy")
    mask = np.sign(np.max(np.abs(features),2))
    length = np.sum(mask,1)
    time = []
    for patient in range(features.shape[0]):
        patient_last_time = features[patient,length[patient]-1,10]
        time.append(patient_last_time)
    np.savetxt("length.csv",length, delimiter=",")


def read_features1():
    features= np.load("allPatientFeatures_merge.npy")
    print(features.shape)


def get_feature_selection():
    feature_weight = np.load("average_weight.npy")
    feature_weight_new = feature_weight.reshape(-1, feature_weight.shape[2])
    weight = np.mean(feature_weight_new,axis=0)
    np.savetxt("feature_selection.csv",weight, delimiter=',')


def get_logistic_log_rank():
    time_real = np.load("pick_5_visit_features_merge_1.npy")[0:2100, :, 0].reshape(-1)
    logistic_file = "E:\\survival analysis\\src\\result_9_16_0\\采用整合特征之后的数据\\2年\\logistic regression\\LogisticRegression 2019-10-14-21-02-15.xls"
    df = pd.read_excel(logistic_file,usecols=['label','pre'])
    label_real = df['label']
    label_pre = df['pre']
    results = logrank_test(time_real,time_real,event_observed_A=label_real,event_observed_B=label_pre)
    results.print_summary()
    print(results.p_value)
    print(results.test_statistic)


# 求出有特征的病人的平均值排序
def get_average_weights():
    weights = np.load("average_weight.npy")
    weight_reshape = weights.reshape(-1,92)  # 得到所有病人的weight
    features = np.load("pick_5_visit_features_merge_1.npy")[0:2100,:,1:93]
    feature_reshape = features.reshape(-1,92)  # 得到所有病人的feature
    # 找到有用的weight
    weight_useful = weight_reshape * feature_reshape

    # 计算每个feature的权重和
    weight_useful_sum = np.sum(weight_useful, axis=0)

    # 计算有用的patient数目
    patient_useful = np.sum(feature_reshape, axis=0)

    # 计算最后的结果
    feature_selection = weight_useful_sum * np.reciprocal(patient_useful)
    np.save("patient_weight.npy", feature_selection)
    np.savetxt("patient_weight.csv",feature_selection, delimiter=',')


def do_calibration():
    group = [0,1]
    df = pd.read_excel("C:\\Users\\szh\\Desktop\\2.xlsx")
    df['rank'] = df['probability'].rank(method='first')
    df['decile'] = pd.qcut(df['rank'].values, 10).codes
    print(df['decile'])
    obsevents_pos = df['label'].groupby(df['decile']).sum()
    print("obsevents_pos",obsevents_pos)
    obsevents_neg = df['label'].groupby(df['decile']).count()-obsevents_pos
    print("obseevent_neg",obsevents_neg)
    expevents_pos = df['probability'].groupby(df['decile']).sum()
    print("expevents",expevents_pos)
    expevents_neg = df['probability'].groupby(df['decile']).count() - expevents_pos
    print("expevents_neg",expevents_neg)
    h1 = (((obsevents_pos - expevents_pos) ** 2 / expevents_pos) + (
                (obsevents_neg - expevents_neg) ** 2 / expevents_neg)).sum()
    print(h1)
    p_value = 1.0-stats.chi2.cdf(h1,9)
    print(p_value)


def embedding_features():
    two_year_features = np.load("pick_5_features_half_year.npy")
    all_patient_features = np.zeros(shape=(0,5,93,2))
    for patient in range(two_year_features.shape[0]):
        one_patient_feature = np.zeros(shape=(0,93,2))
        for visit in range(two_year_features.shape[1]):
            feature = two_year_features[patient,visit,1:]
            if np.sum(feature)==0:
                one_visit_feature = np.zeros(shape=(len(feature),2))
            else:
                one_visit_feature =[]
                for m in range(len(feature)):
                    new_feature = np.zeros(shape=(0,2))
                    if feature[m]==1:
                        new_feature= np.concatenate((new_feature,np.array([1,0]).reshape(-1,2)))
                    else:
                        new_feature = np.concatenate((new_feature,np.array([0,1]).reshape(-1,2)))
                    new_feature = np.array(new_feature)
                    new_feature = new_feature.reshape(-1,2)
                    one_visit_feature.append(new_feature)
            one_visit_feature = np.array(one_visit_feature)
            one_visit_feature = one_visit_feature.reshape(-1,len(feature),2)
            print("one_patient_nums",one_patient_feature.shape[0])
            one_patient_feature = np.concatenate((one_patient_feature,one_visit_feature))
        one_patient_feature = one_patient_feature.reshape(-1,5,93,2)
        all_patient_features = np.concatenate((all_patient_features,one_patient_feature))
        print("all_patient_nums",all_patient_features.shape[0])
    np.save("embedding_features_half_year_2dims.npy",all_patient_features)
    return all_patient_features


def calculate_hazard_ratio():
    features = np.load("pick_logistic_features_two_year.npy")[:,1:]
    label = np.load("pick_logistic_labels_two_year.npy")
    HR = list()
    treatment_observeds = list()
    treatment_expecteds = list()
    control_observeds = list()
    control_expecteds = list()
    for i in range(features.shape[1]):
        treatment_observed = 0
        treatment_expected = 0
        control_observed = 0
        control_expected = 0
        feature = features[:,i]
        for j in range(features.shape[0]):
            feature_m = int(feature[j])
            label_m = int(label[j])
            if feature_m ==1 and label_m == 1:
                treatment_observed += 1
            if feature_m ==1 and label_m == 0:
                treatment_expected += 1
            if feature_m ==0 and label_m == 1:
                control_observed += 1
            if feature_m ==0 and label_m == 0:
                control_expected += 1
        treatment_observeds.append(treatment_observed)
        treatment_expecteds.append(treatment_expected)
        control_observeds.append(control_observed)
        control_expecteds.append(control_expected)
        m = treatment_observed/(treatment_expected+treatment_observed)*(control_expected+control_observed)/control_observed
        print(m)
        HR.append(m)
    treatment_observeds_ = np.array(treatment_observeds).reshape(-1,1)
    treatment_expecteds_ = np.array(treatment_expecteds).reshape(-1,1)
    control_observeds_ = np.array(control_observeds).reshape(-1,1)
    control_expecteds_ = np.array(control_expecteds).reshape(-1,1)
    result_1 = np.concatenate((treatment_observeds_,treatment_expecteds_),axis=1)
    result_2 = np.concatenate((control_observeds_,control_expecteds_),axis=1)
    result = np.concatenate((result_1,result_2),axis=1)
    np.savetxt("result_two_year.csv",result,delimiter=',')
    np.savetxt("hr_two_year.csv",HR)


def get_part_matrix():
    adjency = np.loadtxt("adjacency_ave_.csv",delimiter=",")
    result = list()
    m = [39, 42, 86, 7, 90, 84, 8, 19,  51, 60]
    for i in range(10):
        for j in range(10):
            x = m[i]
            y = m[j]
            result.append(adjency[x,y])
    result = np.array(result).reshape(-1,10)
    np.savetxt("part_adjacency_half_year.csv",result,delimiter=',')
    fina = list()
    for i in range(10):
        element = result[i,i]
        if i==0:
            result_ = result[i,i+1:]
        else:
            result_ = np.concatenate((result[i,0:i],result[i,i+1:]))
        result_ = np.exp(result_)/np.sum(np.exp(result_))
        soft_result = np.insert(result_,i,element)
        fina.append(soft_result)
    fina = np.array(fina).reshape(-1,10)
    print(fina)
    np.savetxt("soft_adjacency_half_year.csv",fina,delimiter=',')


def init_relative_day():
    # 相对入院时间初始化
    data = pd.read_excel('j:/result.xlsx')
    data = data.fillna(0)
    id_group = data.groupby('id').groups
    id_list = set(data['id'])
    relative_day_list = []
    for id in id_list:
        visit_id_index_list = id_group[id]
        id_features_group = data.iloc[visit_id_index_list]
        day = -1
        base_date = ''
        d0 = ''
        for single_visit in id_features_group.iterrows():
            if day == -1:
                day = 0
                base_date = single_visit[1]['出院时间']
                d0 = datetime.datetime.strptime(str(base_date), "%Y-%m-%d  %H:%M:%S")
                relative_day_list.append([single_visit[0], day])
            else:

                this_date = single_visit[1]['出院时间']
                d1 = datetime.datetime.strptime(str(this_date), "%Y-%m-%d  %H:%M:%S")
                day = (d1-d0).days
                relative_day_list.append([single_visit[0], day])
                
    relative_day_list = pd.DataFrame(relative_day_list)
    relative_day_list = relative_day_list.sort_values(0)
    data['day'] = list(relative_day_list[1])
    data.to_excel('j:/result2.xlsx')



"""

['Unnamed: 0', 'id', 'visit_id', '出院时间', 'day', '年龄', '住院时长', '血红蛋白测定',
       '红细胞计数', '白细胞计数', '淋巴细胞', '平均红细胞红蛋白量', '平均红细胞红蛋白浓度', '血小板计数',
       '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '总蛋白', '血清白蛋白', '总胆红素', '直接胆红素', '尿素', '血清尿酸',
       '无机磷', '钾', '镁', '碱性磷酸酶', 'γ-谷氨酰基转移酶', '肌酸激酶', '乳酸脱氢酶', '二氧化碳', '总胆固醇',
       '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '死亡']


"""
def init_survival_label():
    # 初始化标签数据用于生存分析预测

    data = pd.read_excel('j:/result2.xlsx')
    id_group = data.groupby('id').groups
    id_list = []
    [id_list.append(i) for i in list(data['id']) if not i in id_list]
    six_month_death_right_labels = []
    one_year_death_right_labels = []
    two_year_right_labels = []

    six_month_death_right_day = []
    one_year_death_right_day = []
    two_year_right_day = []
    #
    for id in id_list:
        visit_id_index_list = id_group[id]
        id_features_group = data.iloc[visit_id_index_list]

        final_vid_data = id_features_group.tail(1)
        final_day = final_vid_data['day'].values[0]
        final_event = final_vid_data['死亡'].values[0]
        # 两年
        for single_visit in id_features_group.iterrows():
            day = single_visit[1]['day']
            if final_event == 1:
                if final_day == day:
                    two_year_right_labels.append([single_visit[0], -1])
                    two_year_right_day.append([single_visit[0], -1])
                elif final_day - day > 365 * 2:
                    two_year_right_labels.append([single_visit[0], 0])
                    two_year_right_day.append([single_visit[0], 365 * 2])
                else:
                    two_year_right_labels.append([single_visit[0], 1])
                    two_year_right_day.append([single_visit[0], int(final_day - day)])
                if final_day == day:
                    one_year_death_right_labels.append([single_visit[0], -1])
                    one_year_death_right_day.append([single_visit[0], -1])
                elif final_day - day > 365 :
                    one_year_death_right_labels.append([single_visit[0], 0])
                    one_year_death_right_day.append([single_visit[0], 365 ])
                else:
                    one_year_death_right_labels.append([single_visit[0], 1])
                    one_year_death_right_day.append([single_visit[0], int(final_day - day)])
                if final_day == day:
                    six_month_death_right_labels.append([single_visit[0], -1])
                    six_month_death_right_day.append([single_visit[0], -1])
                elif final_day - day > 180 :
                    six_month_death_right_labels.append([single_visit[0], 0])
                    six_month_death_right_day.append([single_visit[0], 180 ])
                else:
                    six_month_death_right_labels.append([single_visit[0], 1])
                    six_month_death_right_day.append([single_visit[0], int(final_day - day)])

            else:
                if final_day - day < 365 * 2:
                    two_year_right_labels.append([single_visit[0], -1])
                    two_year_right_day.append([single_visit[0], -1])
                else:
                    two_year_right_labels.append([single_visit[0], 0])
                    two_year_right_day.append([single_visit[0], 365 * 2])
                if  final_day - day< 365:
                    one_year_death_right_labels.append([single_visit[0], -1])
                    one_year_death_right_day.append([single_visit[0], -1])
                else:
                    one_year_death_right_labels.append([single_visit[0], 0])
                    one_year_death_right_day.append([single_visit[0], 365])
                if final_day - day< 180:
                    six_month_death_right_labels.append([single_visit[0], -1])
                    six_month_death_right_day.append([single_visit[0], -1])
                else:
                    six_month_death_right_labels.append([single_visit[0], 0])
                    six_month_death_right_day.append([single_visit[0], 180])

    six_month_death_right_labels = pd.DataFrame(six_month_death_right_labels)
    six_month_death_right_labels = six_month_death_right_labels.sort_values(0)
    data['six_month_death_right_labels'] = list(six_month_death_right_labels[1])
    one_year_death_right_labels = pd.DataFrame(one_year_death_right_labels)
    one_year_death_right_labels = one_year_death_right_labels.sort_values(0)
    data['one_year_death_right_labels'] = list(one_year_death_right_labels[1])
    two_year_right_labels = pd.DataFrame(two_year_right_labels)
    two_year_right_labels = two_year_right_labels.sort_values(0)
    data['two_year_death_right_labels'] = list(two_year_right_labels[1])

    six_month_death_right_day = pd.DataFrame(six_month_death_right_day)
    six_month_death_right_day = six_month_death_right_day.sort_values(0)
    data['six_month_death_right_day'] = list(six_month_death_right_day[1])
    one_year_death_right_day = pd.DataFrame(one_year_death_right_day)
    one_year_death_right_day = one_year_death_right_day.sort_values(0)
    data['one_year_death_right_day'] = list(one_year_death_right_day[1])
    two_year_right_day = pd.DataFrame(two_year_right_day)
    two_year_right_day = two_year_right_day.sort_values(0)
    data['two_year_death_right_day'] = list(two_year_right_day[1])

    data.to_excel('j:/dataset_with_labels.xlsx')


# def init_survival_label2():
#     # 初始化标签数据用于生存分析预测
#     data = pd.read_excel('j:/result2.xlsx')
#     id_group = data.groupby('id').groups
#     id_list = set(data['id'])
#     six_month_death_right_labels = []
#     one_year_death_right_labels = []
#     two_year_right_labels = []
#
#     six_month_death_right_day = []
#     one_year_death_right_day = []
#     two_year_right_day = []
#
#     for id in id_list:
#         visit_id_index_list = id_group[id]
#         id_features_group = data.iloc[visit_id_index_list]
#
#         final_vid_data = id_features_group.tail(1)
#         final_day = final_vid_data['day'].values
#         final_event = final_vid_data['死亡'].values
#         # 两年
#         for single_visit in id_features_group.iterrows():
#             day = single_visit[1]['day']
#             if final_day - day < 365 * 2:
#                 two_year_right_labels.append([single_visit[0], 1])
#                 two_year_right_day.append([single_visit[0], final_day,final_event[0]])
#             elif final_day - day > 0:
#                 two_year_right_labels.append([single_visit[0], 0])
#                 two_year_right_day.append([single_visit[0], final_day,final_event[0]])
#             else:
#                 two_year_right_labels.append([single_visit[0], -1])
#                 two_year_right_day.append([single_visit[0], -1,final_event[0]])
#     data['tte'] = np.asarray(two_year_right_day)[:,1]
#     data['time'] = (data['day']/30).astype(int)
#     data['e'] = np.asarray(two_year_right_day)[:,2]
#
#     data.to_excel('j:/dataset_with_labels_dynmic1221.xlsx')
#
#     sequence_length = 5
#     sequence_count = 0
#     sequence_data_store = []
#
#
#     columns = ['id', 'visit_id', '年龄', '血红蛋白测定', '红细胞计数', '白细胞计数', '淋巴细胞', '平均红细胞红蛋白量', '平均红细胞红蛋白浓度', '血小板计数',
#                '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '总蛋白', '血清白蛋白', '总胆红素', '直接胆红素', '尿素', '血清尿酸',
#                '无机磷', '钾', '镁', '碱性磷酸酶', 'γ-谷氨酰基转移酶', '肌酸激酶', '乳酸脱氢酶', '二氧化碳', '总胆固醇',
#                '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', 'day', 'time','tte','e','死亡']
#
#     data = data[columns]
#     id_group = data.groupby('id').groups
#     id_list = set(data['id'])
#     count = 0
#
#     for id in id_list:
#
#         visit_id_index_list = id_group[id]
#         id_features_group = data.iloc[visit_id_index_list]
#         selected_id_features_group = id_features_group.loc[id_features_group['tte'] != -1]
#         group_length = selected_id_features_group.shape[0]
#         if group_length < sequence_length:
#             continue
#         else:
#             for tail in range(sequence_length, group_length + 1):
#                 selected_id_features = selected_id_features_group.iloc[tail - sequence_length:tail]
#                 day = selected_id_features['day'].values
#                 relative_day_list = []
#                 first_day = 0
#                 for i in range(sequence_length):
#                     if i == 0:
#                         relative_day_list.append(0)
#                         first_day = day[i]
#                     else:
#                         relative_day_list.append(day[i] - first_day)
#                 selected_id_features['interval'] = np.array(relative_day_list)
#                 count += 1
#                 selected_id_features['ID'] = np.tile(count, sequence_length)
#                 sequence_data_store.append(selected_id_features.values)
#                 sequence_count = sequence_count + 1
#
#     #
#     # with open('j:/sequence_dataset_with4.pkl', 'rb') as f:
#     #     sequence_data_store = pickle.load(f)
#     sequence_data_store = np.asarray(sequence_data_store)
#     sequence_data_store = sequence_data_store.reshape(-1, sequence_data_store.shape[2])
#     sequence_data_store = pd.DataFrame(sequence_data_store, columns=['id', 'visit_id', '年龄', '血红蛋白测定', '红细胞计数', '白细胞计数', '淋巴细胞', '平均红细胞红蛋白量', '平均红细胞红蛋白浓度', '血小板计数',
#                    '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '总蛋白', '血清白蛋白', '总胆红素', '直接胆红素', '尿素', '血清尿酸',
#                    '无机磷', '钾', '镁', '碱性磷酸酶', 'γ-谷氨酰基转移酶', '肌酸激酶', '乳酸脱氢酶', '二氧化碳', '总胆固醇',
#                    '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', 'day', 'time','tte','e','死亡','interval','ID'])
#     sequence_data_store = sequence_data_store.drop(columns=['id', 'visit_id','死亡','interval','day'])
#     c = ['年龄', '血红蛋白测定', '红细胞计数', '白细胞计数', '淋巴细胞', '平均红细胞红蛋白量', '平均红细胞红蛋白浓度', '血小板计数',
#                    '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '总蛋白', '血清白蛋白', '总胆红素', '直接胆红素', '尿素', '血清尿酸',
#                    '无机磷', '钾', '镁', '碱性磷酸酶', 'γ-谷氨酰基转移酶', '肌酸激酶', '乳酸脱氢酶', '二氧化碳', '总胆固醇',
#                    '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇']
#     sequence_data_store[c] = (sequence_data_store[c].max()-sequence_data_store[c])/ (sequence_data_store[c].max()-sequence_data_store[c].min())
#     f = open('j:/sequence_dataset_with5.pkl', 'wb')
#     pickle.dump(sequence_data_store, f)
def init_survival_label2():
    # 初始化标签数据用于生存分析预测
    data = pd.read_excel('j:/result2.xlsx')
    id_group = data.groupby('id').groups
    id_list = []
    [id_list.append(i) for i in list(data['id']) if not i in id_list]
    two_year_right_labels = []
    two_year_right_day = []
    #
    for id in id_list:
        visit_id_index_list = id_group[id]
        id_features_group = data.iloc[visit_id_index_list]

        final_vid_data = id_features_group.tail(1)
        final_day = final_vid_data['day'].values[0]
        final_event = final_vid_data['死亡'].values[0]
        # 两年
        for single_visit in id_features_group.iterrows():
            day = single_visit[1]['day']
            if final_event == 1:
                if final_day == day:
                    two_year_right_labels.append([single_visit[0], -1])
                    two_year_right_day.append([single_visit[0], -1])
                elif final_day - day > 365 * 2:
                    two_year_right_labels.append([single_visit[0], 0])
                    two_year_right_day.append([single_visit[0], 365 * 2])
                else:
                    two_year_right_labels.append([single_visit[0], 1])
                    two_year_right_day.append([single_visit[0], int(final_day - day)])

            else:
                if final_day - day < 365 * 2:
                    two_year_right_labels.append([single_visit[0], -1])
                    two_year_right_day.append([single_visit[0], -1])
                else:
                    two_year_right_labels.append([single_visit[0], 0])
                    two_year_right_day.append([single_visit[0], 365 * 2])


    #  event time in time windows
    data['time'] = (data['day']/1).astype(int)
    data['ett'] = np.asarray(two_year_right_day)[:, 1]
    data['label'] = np.asarray(two_year_right_labels)[:, 1]

    data.to_excel('j:/dataset_with_labels_dynmic1221.xlsx')
    # 输入：1
    sequence_length = 5
    sequence_count = 0
    sequence_data_store = []


    columns = ['id', 'visit_id', '年龄', '血红蛋白测定', '红细胞计数', '白细胞计数', '淋巴细胞', '平均红细胞红蛋白量', '平均红细胞红蛋白浓度', '血小板计数',
               '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '总蛋白', '血清白蛋白', '总胆红素', '直接胆红素', '尿素', '血清尿酸',
               '无机磷', '钾', '镁', '碱性磷酸酶', 'γ-谷氨酰基转移酶', '肌酸激酶', '乳酸脱氢酶', '二氧化碳', '总胆固醇',
               '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', 'day', 'time','ett','label','死亡']

    data = data[columns]
    id_group = data.groupby('id').groups
    count = 0

    for id in id_list:

        visit_id_index_list = id_group[id]
        id_features_group = data.iloc[visit_id_index_list]
        selected_id_features_group = id_features_group.loc[id_features_group['ett'] != -1]
        group_length = selected_id_features_group.shape[0]
        if group_length < sequence_length:
            continue
        else:
            for tail in range(sequence_length, group_length + 1):
                selected_id_features = selected_id_features_group.iloc[tail - sequence_length:tail]
                day = selected_id_features['day'].values
                e = selected_id_features['label'].values[-1]
                last_ett = selected_id_features['ett'].values[-1]
                # tte: time to event
                tte = day[-1] + last_ett

                relative_day_list = []
                first_day = 0
                for i in range(sequence_length):
                    if i == 0:
                        relative_day_list.append(0)
                        first_day = day[i]
                    else:
                        relative_day_list.append(day[i] - first_day)

                count += 1
                selected_id_features.insert(selected_id_features.shape[1], 'interval', np.array(relative_day_list))
                selected_id_features.insert(selected_id_features.shape[1], 'ID', count)
                selected_id_features.insert(selected_id_features.shape[1], 'e', e)
                selected_id_features.insert(selected_id_features.shape[1], 'tte', tte)
                #
                # selected_id_features.loc['ID'] = np.tile(count, sequence_length)
                # selected_id_features.loc['e'] = np.tile(e, sequence_length)
                # selected_id_features.loc['tte'] = np.tile(tte, sequence_length)
                sequence_data_store.append(selected_id_features.values)
                sequence_count = sequence_count + 1

    #
    # with open('j:/sequence_dataset_with4.pkl', 'rb') as f:
    #     sequence_data_store = pickle.load(f)
    sequence_data_store = np.asarray(sequence_data_store)
    sequence_data_store = sequence_data_store.reshape(-1, sequence_data_store.shape[2])
    sequence_data_store = pd.DataFrame(sequence_data_store, columns=['id', 'visit_id', '年龄', '血红蛋白测定', '红细胞计数', '白细胞计数', '淋巴细胞', '平均红细胞红蛋白量', '平均红细胞红蛋白浓度', '血小板计数',
                   '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '总蛋白', '血清白蛋白', '总胆红素', '直接胆红素', '尿素', '血清尿酸',
                   '无机磷', '钾', '镁', '碱性磷酸酶', 'γ-谷氨酰基转移酶', '肌酸激酶', '乳酸脱氢酶', '二氧化碳', '总胆固醇',
                   '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', 'day','time','ett', 'label','死亡','interval','ID','e','tte'])
    c = ['年龄', '血红蛋白测定', '红细胞计数', '白细胞计数', '淋巴细胞', '平均红细胞红蛋白量', '平均红细胞红蛋白浓度', '血小板计数',
         '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '总蛋白', '血清白蛋白', '总胆红素', '直接胆红素', '尿素', '血清尿酸',
         '无机磷', '钾', '镁', '碱性磷酸酶', 'γ-谷氨酰基转移酶', '肌酸激酶', '乳酸脱氢酶', '二氧化碳', '总胆固醇',
         '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇']
    sequence_data_store[c] = (sequence_data_store[c].max() - sequence_data_store[c]) / (
                sequence_data_store[c].max() - sequence_data_store[c].min())
    f = open('j:/sequence_dataset_whole_two_year.pkl', 'wb')
    pickle.dump(sequence_data_store, f)
    # ett:时间窗口下
    sequence_data_store = sequence_data_store.drop(columns=['id', 'visit_id','死亡','day','time',])
    f = open('j:/sequence_dataset_with5.pkl', 'wb')
    pickle.dump(sequence_data_store, f)

def  generate_sequence_dataset_for_training():
    """
    按照模型需要的每位患者入院记录序列长度把二维数据划分为三维
    """

    sequence_length = 5
    sequence_count = 0
    sequence_data_store = []
    time = 'six_month'
    target_label = '{}_death_right_labels'.format(time)
    target_label_day = '{}_death_right_day'.format(time)
    columns = ['id', 'visit_id', '年龄', '血红蛋白测定', '红细胞计数', '白细胞计数', '淋巴细胞', '平均红细胞红蛋白量', '平均红细胞红蛋白浓度', '血小板计数',
               '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '总蛋白', '血清白蛋白', '总胆红素', '直接胆红素', '尿素', '血清尿酸',
               '无机磷', '钾', '镁', '碱性磷酸酶', 'γ-谷氨酰基转移酶', '肌酸激酶', '乳酸脱氢酶', '二氧化碳', '总胆固醇',
               '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', 'day', target_label, target_label_day]
    data = pd.read_excel(SAVE_DIR + 'dataset_with_labels.xlsx')
    data = data[columns]
    con_c = [ '年龄', '血红蛋白测定', '红细胞计数', '白细胞计数', '淋巴细胞', '平均红细胞红蛋白量', '平均红细胞红蛋白浓度', '血小板计数',
               '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '总蛋白', '血清白蛋白', '总胆红素', '直接胆红素', '尿素', '血清尿酸',
               '无机磷', '钾', '镁', '碱性磷酸酶', 'γ-谷氨酰基转移酶', '肌酸激酶', '乳酸脱氢酶', '二氧化碳', '总胆固醇',
               '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇']
    data[con_c] = (data[con_c]-data[con_c].min())/(data[con_c].max()-data[con_c].min())
    id_group = data.groupby('id').groups
    id_list = set(data['id'])

    for id in id_list:
        visit_id_index_list = id_group[id]
        id_features_group = data.iloc[visit_id_index_list]
        selected_id_features_group = id_features_group.loc[id_features_group[target_label] != -1]
        group_length = selected_id_features_group.shape[0]
        if group_length < sequence_length:
            continue
        else:
            for tail in range(sequence_length, group_length+1):
                selected_id_features = selected_id_features_group.iloc[tail-sequence_length:tail]
                day = selected_id_features['day'].values
                relative_day_list = []
                first_day = 0
                for i in range(sequence_length):
                    if i == 0:
                        relative_day_list.append(0)
                        first_day = day[i]
                    else:
                        relative_day_list.append(day[i] - first_day)
                selected_id_features['interval'] = np.array(relative_day_list)
                sequence_data_store.append(selected_id_features.values)
                sequence_count = sequence_count + 1

    # sequence_data_store
    f = open(SAVE_DIR + 'sequence_dataset_with_' + target_label + '.pkl', 'wb')
    pickle.dump(sequence_data_store, f)

def data_stats():
    data = pd.read_excel('j:/result2.xlsx')
    s = data.describe()
    stats = []
    for c in s.columns:
        mean = round(s[c]['mean'], 2)
        std = round(s[c]['std'], 2)
        output = '{}±{}'.format(mean, std)
        stats.append([c, output])

    print(stats)
    stats = pd.DataFrame(stats)
    stats.to_excel('j:/stats121.xlsx', index=None)


def save_data_as_data_set(load_path, time_window, data_type):
    k_folds = 5

    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    data = np.asarray(data)
    # data = data.astype('float32')
    labels = data[:, :, 31].astype('float32')
    features = data[:, :, 2:30].astype('float32')
    times = data[:, :, 32].astype('float32')
    days = data[:, :, 33].astype('float32')
    dataset_spilt_and_store([features, labels, times, days], k_folds, data_type, time_window)


def read_data(time_window, data_type):
    with open('{}/train_set_{}{}.pkl'.format(DATA_SAVE_DIR, time_window, data_type), 'rb') as f:
        test_set_ = pickle.load(f)
    with open('{}/train_set_{}{}.pkl'.format(DATA_SAVE_DIR, time_window, data_type), 'rb') as f:
        train_set_ = pickle.load(f)
    return train_set_, test_set_


def read_plagh_data(time_windows):
    dynamic_features, labels, times = None, None, None
    if time_windows == 'half_year':
        all_features = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_5_features_half_year.npy')
        dynamic_features = all_features[:, :, 1:-1]
        days = all_features[:, :, -1]
        # dynamic_features2 = np.load('data/embedding_features_two_year_2dims.npy')[:, :, 1:]
        labels = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_5_labels_half_year.npy')
        times = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_5_features_half_year.npy')[:, :, 0].reshape(-1, 5, 1)
    elif time_windows == 'one_year':
        all_features = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_4_features_one_year.npy')
        dynamic_features = all_features[:, :, 1:-1]
        labels = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_4_labels_one_year.npy')
        times = all_features[:, :, 0]
        days = all_features[:, :, -1]
    elif time_windows == 'two_year':
        all_features = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_4_features_two_year.npy')
        dynamic_features = all_features[:, :, 1:-1]
        days = all_features[:, :,  -1]
        labels = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_4_labels_two_year.npy')
        times = all_features[:, :, 0]
    print(dynamic_features.shape)
    features_index = dynamic_features.reshape(dynamic_features.shape[0], -1)
    label_index = labels.reshape(labels.shape[0], -1)
    interference_index = ((np.squeeze(times) + np.squeeze(labels)) != 0)
    mask = interference_index * 1
    # ((times + np.squeeze(labels)) == 0).values
    k_folds = 5
    test_size = 1 / k_folds
    train_size = 1 - test_size
    split = StratifiedShuffleSplit(k_folds, test_size, train_size, 1).split(features_index, label_index)
    train_index, test_index = next(split)
    train_set = DataSetWithMask(dynamic_features[train_index], times[train_index], labels[train_index], mask[train_index])
    test_set = DataSetWithMask(dynamic_features[test_index], times[test_index], labels[test_index], mask[test_index])
    f = open('j:/train_set_{}2.pkl'.format(time_windows), 'wb')
    pickle.dump(train_set, f)
    f2 = open('j:/test_set_{}2.pkl'.format(time_windows), 'wb')
    pickle.dump(test_set, f2)
    return train_set, test_set


def read_plagh_data_with_interval(time_windows):
    dynamic_features, labels, times = None, None, None
    if time_windows == 'half_year':
        all_features = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_5_features_half_year.npy')
        dynamic_features = all_features[:, :, 1:-1]
        days = all_features[:, :, -1]
        # dynamic_features2 = np.load('data/embedding_features_two_year_2dims.npy')[:, :, 1:]
        labels = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_5_labels_half_year.npy')
        times = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_5_features_half_year.npy')[:, :, 0].reshape(-1, 5, 1)
    elif time_windows == 'one_year':
        all_features = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_4_features_one_year.npy')
        dynamic_features = all_features[:, :, 1:-1]
        labels = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_4_labels_one_year.npy')
        times = all_features[:, :, 0]
        days = all_features[:, :, -1]
    elif time_windows == 'two_year':
        all_features = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_4_features_two_year.npy')
        dynamic_features = all_features[:, :, 1:-1]
        days = all_features[:, :,  -1]
        labels = np.load('j:/实验室/survival-analysis_code20.10.29/model/pick_4_labels_two_year.npy')
        times = all_features[:, :, 0]
    print(dynamic_features.shape)
    features_index = dynamic_features.reshape(dynamic_features.shape[0], -1)
    label_index = labels.reshape(labels.shape[0], -1)
    interference_index = ((np.squeeze(times) + np.squeeze(labels)) != 0)
    mask = interference_index * 1
    # ((times + np.squeeze(labels)) == 0).values
    k_folds = 5
    test_size = 1 / k_folds
    train_size = 1 - test_size
    split = StratifiedShuffleSplit(k_folds, test_size, train_size, 1).split(features_index, label_index)
    train_index, test_index = next(split)
    train_set = DataSetWithMask2(dynamic_features[train_index], times[train_index], labels[train_index], days[train_index], mask[train_index])
    test_set = DataSetWithMask2(dynamic_features[test_index], times[test_index], labels[test_index], days[test_index], mask[test_index])
    f = open('j:/train_set_{}2.pkl'.format(time_windows), 'wb')
    pickle.dump(train_set, f)
    f2 = open('j:/test_set_{}2.pkl'.format(time_windows), 'wb')
    pickle.dump(test_set, f2)
    return train_set, test_set
# 划分三个数据集
# test_set val_set test_set
if __name__ == '__main__':
    read_plagh_data_with_interval('half_year')
    read_plagh_data_with_interval('one_year')
    read_plagh_data_with_interval('two_year')

    # init_relative_day()
    # init_survival_label()


    # generate_sequence_dataset_for_training()
    # save_data_as_data_set(path, time_window, data_type)
    # path = 'j:/sequence_dataset_with_two_year_right_labels3.pkl'
    # time_window = 'two_year'
    # data_type = 3
    # a = pd.read_csv('I:/微信/WeChat Files/wxid_15g9oyah8afg22/FileStorage/File/2021-01/合并特征值之后的特征.csv')
    # print(a)


    # train_set, test_set = read_data(time_window, data_type)
    # print(train_set)

    # data_stats()
    # f = open('j:/sequence_dataset_with4.pkl', 'wb')
    # pickle.dump(sequence_data_store, f)

    # target_label = 'two_year'
    # with open('j:/sequence_dataset_with_two_year_right_labels3.pkl', 'rb') as f:
    #     test_set = pickle.load(f)
    # with open('j:/实验室/survival-analysis_code20.10.29/model/test_set_{}3.pkl'.format(target_label), 'rb') as f:
    #     test_set2 = pickle.load(f)
    # with open('j:/sequence_dataset_whole_two_year.pkl',  'rb') as f:
    #     S1 = pickle.load(f)
    # with open('j:/sequence_dataset_with_one_year_death_right_labels.pkl', 'rb') as f:
    #     S2 = pickle.load(f)
    # with open('j:/sequence_dataset_with_six_month_death_right_labels.pkl', 'rb') as f:
    #     S3 = pickle.load(f)
    #
    # for i in range(len(S1)):
    #     data = S1[i]
    #     print(1)

    # s1 = S1['arr_0']
    # print(S1)
    # s2 = S2['arr_0']
    # s3 = S3['arr_0']

    # get_part_matrix()
    # calculate_hazard_ratio()
    # get_feature_selection()
    # embedding_features()
    # do_calibration()
    # get_insert_sql()
    # get_all_patients_features()
    # get_average_weights()
    # get_logistic_log_rank()
    # get_all_patients_features()
    # get_feature_selection()
    # read_features()
    # read_labels()
    # read_features1()
    # get_right_label()
    # get_logistic_features()
