import pandas as pd
import numpy as np

dataset = pd.read_csv("../AM_210329_COVID3.csv", encoding="utf-8")

message_list = dataset["MESSAGE"].values.tolist()
category_list = dataset["CATEGORY"].values.tolist()
sub_list = dataset["SUB"].values.tolist()

MAX_LEN = 10

keyword_1 = ["확진", "발생", "동선", "역학", "접촉", "코로나", "공개", "홈페이지", "방역", "예정"]
keyword_2 = ["보건소", "방문", "검사", "선별", "진료소", "확진", "이용", "상담", "코로나", "교회"]
keyword_3 = ["착용", "자제", "코로나", "방문", "마스크", "지역", "모임", "방역", "수칙", "시설"]
keyword_3_1 = ["착용", "자제", "코로나", "마스크", "방문", "수칙", "방역", "모임", "예방", "지역"]
keyword_3_2 = ["자제", "코로나", "지역", "방문", "착용", "확진", "발생", "모임", "감염", "마스크"]
keyword_3_3 = ["운영", "착용", "시설", "금지", "단계", "선별", "코로나", "사회", "방역", "두기"]


def getMainCategory(count):
    count1 = count["1"] / MAX_LEN
    count2 = count["2"] / MAX_LEN
    count3 = count["3"] / MAX_LEN
    # main_category = []
    # if count1 >= 0.6:
    #     main_category.append(1)
    # if count2 >= 0.6:
    #     main_category.append(2)
    # if count3 >= 0.6:
    #     main_category.append(3)
    # return main_category
    # 1 2 3 순서대로
    if count1 >= 0.6:
        return 1
    elif count2 >= 0.6:
        return 2
    elif count3 >= 0.6:
        return 3
    return 4
    # 확률이 같으면 미분류 하는것도 해보자


def getSubCategory(count):
    count3_1 = count["3-1"] / MAX_LEN
    count3_2 = count["3-2"] / MAX_LEN
    count3_3 = count["3-3"] / MAX_LEN
    # sub_category = []
    # if count1 >= 0.6:
    #     sub_category.append(1)
    # if count2 >= 0.6:
    #     sub_category.append(2)
    # if count3 >= 0.6:
    #     sub_category.append(3)
    # return sub_category
    # 1 2 3 순서대로
    if count3_1 >= 0.6:
        return 1
    elif count3_2 >= 0.6:
        return 2
    elif count3_3 >= 0.6:
        return 3
    return 4
    # 확률이 같으면 미분류 하는것도 해보자


main_dict = {"확진자발생": 1, "보건소방문": 2, "안내": 3}
sub_dict = {"개인방역수칙": 1, "발생현황": 2, "행정안내": 3}

con_matrix_main = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
con_matrix_sub = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
accuracy_main = 0
accuracy_sub = 0
sub_count = 0
for i in range(0, len(message_list)):
    message = message_list[i]
    count = {"1": 0, "2": 0, "3": 0, "3-1": 0, "3-2": 0, "3-3": 0}
    for j in range(0, MAX_LEN):
        if keyword_1[j] in message:
            count["1"] += 1
        if keyword_2[j] in message:
            count["2"] += 1
        if keyword_3[j] in message:
            count["3"] += 1
        if keyword_3_1[j] in message:
            count["3-1"] += 1
        if keyword_3_2[j] in message:
            count["3-2"] += 1
        if keyword_3_3[j] in message:
            count["3-3"] += 1
    main_category = getMainCategory(count)
    con_matrix_main[main_dict[category_list[i]] - 1][main_category - 1] += 1
    if category_list[i] == main_category:
        accuracy_main += 1

    if sub_list[i]:
        print(i, sub_list[i], type(sub_list[i]))
        if sub_dict[sub_list[i]] == 3:
            sub_count += 1
            sub_category = getSubCategory(count)
            con_matrix_main[sub_dict[sub_list[i]] - 1][sub_category - 1] += 1
            if sub_list[i] == sub_category:
                accuracy_sub += 1

accuracy_main /= len(message_list)
accuracy_sub /= sub_count
precision_main = 0
recall_main = 0
f1_score_main = 0
print(accuracy_main, accuracy_sub)
# for i in range(0, 3):
#     print(main_category[i])
#     TP = con_matrix_main[i][i]
#     FP = 0
#     FN = 0
#     for j in range(0, 3):
#         if j != i:
#             FP += con_matrix[j][i]

for i in range(0, 4):
    print(main_category[i])
for j in range(0, 4):
    print(sub_category[i])
