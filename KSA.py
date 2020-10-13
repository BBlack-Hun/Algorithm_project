import numpy as np
import time

from sklearn import datasets
iris = datasets.load_iris()
v = iris.data


#v = np.array([[1,1],[1,2],[2,1],[2,2],[10,1],[10,2],[11,1],[11,2],[5,5],[5,6]])

#2차원까지 지원합니다. 3D 지원 안함....
from matplotlib import pyplot as plt
#주어진 데이터를 가지고 시각화
x = v[:, 0]
y = v[:, 1]

plt.xlabel('xCuk')
plt.ylabel('yCuk')

#클러스터의 개수
k = 3
#SA의 결과중 best 3개를 초기 값으로 하여 K-means와 SA를 혼합한다.
#랜덤으로 3개의 중심 좌표를 생성합니다.
#np.random.uniform은 주어진 최소, 최대값 사이에서  k개 만큼의 centroid를 생성합니다.
centroids_x = np.random.uniform(min(x), max(x), k)
centroids_y = np.random.uniform(min(y), max(y), k)
centroids = list(zip(centroids_x, centroids_y))
#튜플 데이터를 np.array로 변환시켜줘야 plt.scatter 에서 작동한다.
centroids = np.array(centroids)
#초기의 centroid 3개를 출력해주는 것...!
print(centroids)

#거리를 측정하기 위한 함수 작성
def distance(a, b):
    result = 0
    for el_a, el_b in list(zip(a,b)):
        result += (abs(el_a-el_b))**2

    return result**0.5

def inter_distance(sepal_length_width, centroids):
    sum_UC_1 = 0
    sum_UC_2 = 0
    sum_UC_3 = 0
    for j in range(len(v)):
        if labels[j] == 0:
            UC = math.sqrt(math.pow((sepal_length_width[j][0]-centroids[0][0]),2)+pow((sepal_length_width[j][1]-centroids[0][1]),2))
            #print("0번 클러스터의 유클리드 거리는: ", UC)
            sum_UC_1 +=UC

        elif labels[j] == 1:
            UC = math.sqrt(math.pow((sepal_length_width[j][0] - centroids[1][0]),2) + pow(
                (sepal_length_width[j][1] - centroids[1][1]),2))
            #print("1번 클러스터의 유클리드 거리는: ", UC)
            sum_UC_2 += UC
        elif labels[j] ==2:
            UC = math.sqrt(math.pow((sepal_length_width[j][0] - centroids[2][0]),2) + pow(
                (sepal_length_width[j][1] - centroids[2][1]),2))
            #print("2번 클러스터의 유클리드 거리는: ", UC)
            sum_UC_3 += UC

    return sum_UC_1+ sum_UC_2 +sum_UC_3


#백터의 개수만큼 배열을 생성한다.
labels = np.zeros(len(v))
sepal_length_width = np.array(list(zip(x,y)))
#print(sepal_length_width)


#각 데이터를 순회 하면서 중심과의 길이를 측정한다.
for i in range(len(v)):
    #각 백터의 거리를 계산하기 위해 초기 거리는 0으로 초기화, zeros는 배열의 값이 0인 배열을 생성, 즉, 배열은 [0, 0, 0]이 저장
    distances = np.zeros(k)
    for j in range(k):
        #각각의 2차원 백터와 cnetroid와의 거리를 측정, 리스트를 반환
        distances[j] = distance(sepal_length_width[i], centroids[j])

    #print(distances)
    #클러스터에 가장 작은 거리의 Index를 반환
    cluster= np.argmin(distances)
    #각 데이터에 대한 할당받은 클러스터 이름을 출력
    #print(cluster)
    #클러스터의 이름 지정
    labels[i] = cluster

from copy import deepcopy
centroids_old = deepcopy(centroids)

for i in range(k):
    #각  그룹에 속한 데이터들만 골라 points에 저장합니다.
    points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
    #poins의 평균 지점으로 centroid를 이동합니다.
    centroids[i] = np.mean(points, axis=0)
#한번 이동한 centroid
print("한번 이동한 centroids")
print(centroids)
plt.scatter(x, y, c=labels, alpha=0.5)
'''
처음에서 한번 움직인 곳의 위치를 표시
'''
# 과거 점의 위치를 표시합니다. 파랑색으로 표시
plt.scatter(centroids_old[:, 0], centroids_old[:, 1], c='blue')

# 데이터들의 중심으로 이동한 중심값의 위치
plt.scatter(centroids[:, 0], centroids[:, 1], c='red')

# 화면에 출력
plt.show()

#제일 처음 old는 0으로 초기화
centroids_old = np.zeros(centroids.shape)

# 클러스터명도 초기화를 한다.
labels=  np.zeros(len(v))

#클러스터의 갯수만큼 error 초기화를 한다.
error = np.zeros(k)

for i in range(k):
    #바뀐 위치와 데이터의 위치를 측정하여 error에 저장한다.
    error[i] = distance(centroids_old[i], centroids[i])

cnt = 2
#step4, error가 0에 수렴할때 까지 2~3단계를 반복한다.
while(error.all() != 0):
    print(cnt,'번 반복합니다.')
    #step2, 가장 가까운 cnetroids에 데이터를 할당한다.
    for i in range(len(v)):
        distances = np.zeros(k)
        for j in range(k):
            distances[j] = distance(sepal_length_width[i], centroids[j])
        cluster = np.argmin(distances)
        labels[i] = cluster
    print(cnt,"번 이동한 centroids")
    print(centroids)
    #step3: centroids를 업데이트 시킨다.
    centroids_old = deepcopy(centroids)

    print("평가 값은: ", inter_distance(sepal_length_width, centroids))

    for i in range(k):
        points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
        #points의 각 feature, 즉 각 좌표 평균 지점을 centroids로 지정한다.
        centroids[i] = np.mean(points, axis=0)

    '''
    두번째에서 한번 움직인 곳의 위치를 표시(error를 수정할때마다 움직인 포인트가 계속 출력된다)
    '''

    plt.scatter(x, y, c=labels, alpha=0.5)

    # 과거 점의 위치를 표시합니다. 파랑색으로 표시
    plt.scatter(centroids_old[:, 0], centroids_old[:, 1], c='blue')

    # 데이터들의 중심으로 이동한 중심값의 위치
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    # 화면에 출력
    #plt.show()

    #새롭게 평균을 다시 계산 했으므로 error를 다시 계산한다.
    for i in range(k):
        error[i] = distance(centroids_old[i], centroids[i])
    cnt+=1

    #print(error)

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math

labels = np.zeros(len(v))
sepal_length_width = np.array(list(zip(x,y)))


print("\n==================================================================================================\n")
print("계승받은건\n", centroids)


plt.scatter(x, y, c=labels, alpha=0.5)
plt.show()

n = 0
T0 = 1000
M = 100
N = 15
alpha = 0.5  #T를 얼마나 줄일지
k_n = 3 #클러스터의 개수
s_n = 0 #솔루션이 받아드려진 횟수
computing_time = 50  # second(s)

#랜덤하게 클러스터를 할당한다.
i = 0
while i<10:
    if not 0 in labels:
        labels[i] = np.random.randint(0,3)
    elif not 1 in labels:
        labels[i] = np.random.randint(0, 3)
    elif not 2 in labels:
        labels[i] = np.random.randint(0, 3)
    else:
        labels[i] = np.random.randint(0, 3)
    i+=1

current_solution = labels
best_solution = labels

for k in range(k_n):
    # 각  그룹에 속한 데이터들만 골라 points에 저장합니다.
    points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if current_solution[j] == k]
    # poins의 평균 지점으로 centroid를 이동합니다.
    centroids[k] = np.mean(points)

centroids_old = deepcopy(centroids)

'''
처음에서 한번 움직인 곳의 위치를 표시
'''

# 과거 점의 위치를 표시합니다. 파랑색으로 표시
plt.scatter(centroids_old[:, 0], centroids_old[:, 1], c='blue')
# 데이터들의 중심으로 이동한 중심값의 위치
plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
# 화면에 출력
plt.show()

start = time.time()
# inter uclide distance 계산 ( 기존의 데이터와 중심점과의 거리 및 평가값 계산)
best_fitness = inter_distance(sepal_length_width, centroids)
cnt = 0
s_cnt = 0
for i in range(M):
    for j in range(N):
        distances = np.zeros(k_n)
        for l in range(k_n):
            distances[l] = distance(sepal_length_width[cnt], centroids[l])

        cluster = np.argmin(distances)
        current_solution[cnt] = cluster

        centroids_old = deepcopy(centroids)

        for k in range(k_n):
            # 각  그룹에 속한 데이터들만 골라 points에 저장합니다.
            points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if current_solution[j] == k]
            # poins의 평균 지점으로 centroid를 이동합니다.
            centroids[k] = np.mean(points, axis=0)



        # 한번 이동한 centroid
        print(n, "번 이동한 centroids")
        print(centroids_old)
        print(centroids)
        plt.scatter(x, y, c=labels, alpha=0.5)

        '''
        처음에서 한번 움직인 곳의 위치를 표시
        '''
        # 과거 점의 위치를 표시합니다. 파랑색으로 표시
        plt.scatter(centroids_old[:, 0], centroids_old[:, 1], c='blue')
        # 데이터들의 중심으로 이동한 중심값의 위치
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
        # 화면에 출력
        if n%50 == 0:
            plt.show()

        current_fitness = inter_distance(sepal_length_width, centroids)
        E = abs(current_fitness - best_fitness)
        if i == 0 and j == 0:
            EA = E


        if current_fitness > best_fitness:
            p = math.exp(-E/(EA *T0))
            if np.random.random() < p:
                accept = True
            else:
                accept = False
        else:
            accept = True
        if accept == True:
            best_solution = current_solution
            best_fitness = inter_distance(sepal_length_width, centroids)
            s_n +=1
            EA = (EA * (s_n-1) + E) / s_n

        #평가값 측정, 이값을 가장 작게 유지하는 것이 가장 이상적인 클러스터의 위치이다.
        print("평가 값은: ", best_fitness)

        n+=1
        cnt += 1
        if cnt == len(v):
            cnt = 0
    T0 *= alpha
    end = time.time()
    if end - start >= computing_time:
        break


colors = ['r', 'g', 'y']
for i in range(k_n):
    points = np.array([sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i])
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=150)
plt.show()

