import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 학습용 데이터셋 불러오기
dataset_train1 = pd.read_csv("AM_210329_COVID2.csv", encoding="utf-8")
# print(dataset_train1.head())

# 데이터 전처리
# dataset_train1.drop(["MID", "SEND_TIME", "SEND_LOC", "SEND_PLATFORM", "DISASTER"], axis=1, inplace=True)
dataset_train1.drop(["SUB"], axis=1, inplace=True)
# print(dataset_train1.head())

# 대표 분류명 추출 (이미 csv 파일에서 전처리했으므로 안해도 무관)
data1 = dataset_train1.loc[dataset_train1["CATEGORY"] == "확진자발생"]
data2 = dataset_train1.loc[dataset_train1["CATEGORY"] == "보건소방문"]
data3 = dataset_train1.loc[dataset_train1["CATEGORY"] == "안내"]
new_data = data1.append([data2, data3], sort=False)
new_df = pd.DataFrame(new_data)
new_df = new_df[["MESSAGE", "CATEGORY"]]
# new_data = dataset_train1
# print(new_data.head())

# 분류명 라벨링
encoder = LabelEncoder()
encoder.fit(new_df["CATEGORY"])
new_df["CATEGORY"] = encoder.transform(new_df["CATEGORY"])
# print(new_data.head())

# 라벨링된 분류명 매핑
# {0: '보건소방문', 1: '안내', 2: '확진자발생'}
mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
print(mapping)

# 학습, 테스트 셋 분리
train, test = train_test_split(new_df, test_size=0.1, random_state=42)
print("train shape is:", len(train))
print(train.head())
print("test shape is:", len(test))
print(test.head())

train.to_csv("./covid_train.txt", sep="\t", na_rep="NaN")
test.to_csv("./covid_test.txt", sep="\t", na_rep="NaN")

