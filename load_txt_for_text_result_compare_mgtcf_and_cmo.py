import argparse
import os
import yaml
from easydict import EasyDict
import numpy as np
import pdb
import torch
import matplotlib.pyplot as plt
import os
#WP_WP_1-CTOT
#WP_WP_1-CTOT-2
#output_WP_WP_1-4-1d-2d-alone

# output_WP_WP_1-4-1d-2d-guide-each-other
# output_WP_WP_1-4-2d-guide-1d


name = 'add-ERA5-16-z-uv2-vocen'  #0
save_folder = 'fig/'+name
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# test = [270,260,250,240,230,220,210,200,190,180,170,160,150,140,130,120,110,100,90,80,70,60,50,40,30,20,10]
test = []
i = 290
while i>0:
    test.append(i)
    i=i-5
    if i<120:
        i=i-5
epoch_num = len(test)
# uv2
# traj_sum = 143.61
# tc_diffuser_traj = [18.52 ,	19.58 ,	37.34 ,	68.16]
# pres_sum_tc_diffuser = 6.29
# tc_diffuser_pres = [1.33 ,	0.80 ,	1.70 ,	2.45]
# wind_sum_tc_diffuser= 3.19
# tc_diffuser_wind = [0.76 ,	0.38 ,	0.85 ,	1.20]



# TC-Diffuser
traj_sum = 159.13781581
tc_diffuser_traj = [20.50,22.63,40.85,75.15]
pres_sum_tc_diffuser = 5.9813
tc_diffuser_pres = [1.12,0.60,1.59,2.67]
wind_sum_tc_diffuser=3.4141
tc_diffuser_wind = [0.69,0.34,0.88,1.50]

# TC-Diffuser-platform-2023
traj_sum = 143.323
tc_diffuser_traj = [18.98 ,	18.56 ,	37.04 ,	68.75]
pres_sum_tc_diffuser = 5.927566
tc_diffuser_pres = [1.37 ,	0.77 ,	1.55 	,2.24]
wind_sum_tc_diffuser=3.4683338
tc_diffuser_wind = [0.78 ,	0.44 	,0.93 ,	1.32]

# MGTCF
# traj_sum = 226.68
# tc_diffuser_traj = [23.14,43.37,67.09,75.49]
# pres_sum_tc_diffuser = 9.36
# tc_diffuser_pres = [1.37,2.04,2.66,3.29]
# wind_sum_tc_diffuser=5.31
# tc_diffuser_wind = [0.73,1.17,1.55,1.86]



# 加载txt文件
txt_name = name+'.txt'
try:
    with open(txt_name, 'r') as file:
        content = file.read()
    print("File content read successfully:")
    # print(content)
except FileNotFoundError:
    print(f"File {txt_name} not found.")
except Exception as e:
    print(f"An error occurred: {e}")

plt.figure(figsize=(20, 20))
fig, ax = plt.subplots(figsize=(15, 10), dpi=400)
line_size=2
markeredgewidth = 2

# 使用 splitlines() 方法将字符串拆分成行的列表
lines = content.splitlines()

list_for_traj_sum = []
list_for_traj_4dot = []
for i in range(epoch_num):
    list_for_traj_sum.append(float(lines[1+i*7][24:]))
    input_string = lines[4+i*7][24:-1]
    float_list = [float(x) for x in input_string.split(',')]
    # tensor_list = [torch.tensor(float_list)]
    list_for_traj_4dot.append(torch.tensor(float_list))
traj_4dot = torch.stack(list_for_traj_4dot, dim=0)   #[epoch_num,4]
plt.plot(test, list_for_traj_sum,marker='o', markersize=5, markeredgewidth=markeredgewidth, linestyle='--', linewidth=line_size, color='dodgerblue', label='sum_traj', fillstyle='none')
y = [traj_sum] * len(test)
plt.plot(test, y, label='y=traj_sum', color='red')
plt.xticks(test)
save_path = os.path.join(save_folder, f'sum_traj.png')  # 图像保存路径
plt.savefig(save_path)
# plt.show()

'''
traj, 四个点 traj_4dot:#[epoch_num,4]
'''
plt.figure(figsize=(20, 20))
fig, ax = plt.subplots(figsize=(15, 10), dpi=400)
min_value_of_4_time = torch.tensor([0.0, 0.0, 0.0, 0.0])
min_index = torch.tensor([0, 0, 0, 0])
for t in range(traj_4dot.size(1)):
    min_value_of_4_time[t], min_index[t] = torch.min(traj_4dot[:, t], dim=0)
# min_index  0--epoch_num   1-26
min_index = epoch_num-min_index
print("==================trajectory=====================")
print("6h,12h,18,24h's min value is at epoch:", min_index)
print("6h,12h,18,24h's:", min_value_of_4_time)
for i in range(traj_4dot.size(1)):
    print(test[epoch_num-min_index[i].item()], ":", traj_4dot[epoch_num-min_index[i],:])

file_path = save_folder + '/' + name + '_result.txt'
with open(file_path, 'a') as file:  # output_WP_WP_env_10_fusion5
    file.write(f"==================trajectory=====================\n")
    file.write(f"6h,12h,18,24h's min value is at epoch: {min_index}\n")
    file.write(f"6h,12h,18,24h's min value is: {min_value_of_4_time}\n")
    for i in range(traj_4dot.size(1)):
        file.write(f"{test[epoch_num-min_index[i].item()]}: {traj_4dot[epoch_num - min_index[i], :]}\n")

traj_better = torch.zeros_like(traj_4dot)
t = [6,12,18,24]
plt.xticks(t)

for i in range(epoch_num):
    plt.plot(t, list_for_traj_4dot[i],marker='o', markersize=5,
             markeredgewidth=markeredgewidth, linestyle='--',
             linewidth=1,
             color='dodgerblue', label='sum_traj', fillstyle='none')
    traj_better[i] = np.around(list_for_traj_4dot[i], decimals=2) <= torch.tensor(tc_diffuser_traj)
plt.plot(t, tc_diffuser_traj,marker='o', markersize=5,
             markeredgewidth=markeredgewidth, linestyle='--',
             linewidth=1,
             color='red', label='tc_diffuser_traj', fillstyle='none')

save_path = os.path.join(save_folder, f'traj_4time.png')  # 图像保存路径
plt.savefig(save_path)
# plt.show()


plt.figure(figsize=(20, 20))
fig, ax = plt.subplots(figsize=(15, 10), dpi=400)
# 遍历每一行并进行处理
list_for_pres_sum = []
list_for_pres_4dot = []
for i in range(epoch_num):
    list_for_pres_sum.append(float(lines[2+i*7][24:]))
    input_string = lines[5+i*7][23:-2]
    float_list = [float(x) for x in input_string.split(',')]
    list_for_pres_4dot.append(torch.tensor(float_list))
pres_4dot = torch.stack(list_for_pres_4dot, dim=0)   #[epoch_num,4]
plt.plot(test, list_for_pres_sum, marker='o', markersize=5,
         markeredgewidth=markeredgewidth, linestyle='--',
         linewidth=line_size, color='mediumseagreen',
         label='sum_pres', fillstyle='none')
plt.xticks(test)
y = [pres_sum_tc_diffuser] * len(test)
plt.plot(test, y, color='red')
save_path = os.path.join(save_folder, f'sum_pres.png')  # 图像保存路径
plt.savefig(save_path)
# plt.show()

plt.figure(figsize=(20, 20))
fig, ax = plt.subplots(figsize=(15, 10), dpi=400)
min_value_of_4_time = torch.tensor([0.0, 0.0, 0.0, 0.0])
min_index = torch.tensor([0, 0, 0, 0])
for t in range(pres_4dot.size(1)):
    min_value_of_4_time[t], min_index[t] = torch.min(pres_4dot[:, t], dim=0)
# min_index  0--epoch_num   1-26
min_index = epoch_num-min_index
print("==================pressure=====================")
print("6h,12h,18,24h's min value is at epoch:", min_index)
print("6h,12h,18,24h's min value is:", min_value_of_4_time)
for i in range(pres_4dot.size(1)):
    print(test[epoch_num-min_index[i].item()], ":", pres_4dot[epoch_num-min_index[i],:])

#save_folder+'/'+
file_path = save_folder + '/' + name + '_result.txt'
with open(file_path, 'a') as file:  # output_WP_WP_env_10_fusion5
    file.write(f"==================pressure=====================\n")
    file.write(f"6h,12h,18,24h's min value is at epoch: {min_index}\n")
    file.write(f"6h,12h,18,24h's min value is: {min_value_of_4_time}\n")
    for i in range(pres_4dot.size(1)):
        file.write(f"{test[epoch_num-min_index[i].item()]}: {pres_4dot[epoch_num - min_index[i], :]}\n")

t = [6,12,18,24]
plt.xticks(t)

pres_better = torch.zeros_like(pres_4dot)
for i in range(epoch_num):
    plt.plot(t, list_for_pres_4dot[i],marker='o', markersize=5,
             markeredgewidth=markeredgewidth, linestyle='--',
             linewidth=1,
             color='mediumseagreen', label='sum_traj', fillstyle='none')
    pres_better[i] = np.around(list_for_pres_4dot[i], decimals=2) <= torch.tensor(tc_diffuser_pres)
plt.plot(t, tc_diffuser_pres,marker='o', markersize=5,
             markeredgewidth=markeredgewidth, linestyle='--',
             linewidth=1,
             color='red', label='tc_diffuser_pres', fillstyle='none')

save_path = os.path.join(save_folder, f'pres_4time.png')  # 图像保存路径
plt.savefig(save_path)
# plt.show()


# plt.plot(hours, y4, marker='o', markersize=10, markeredgewidth=markeredgewidth, linestyle='--', linewidth=line_size, color='gold', label=r'w/o $P_{task}$', fillstyle='none')
# plt.plot(hours, y5, marker='^', markersize=10, markeredgewidth=markeredgewidth, linestyle='--', linewidth=line_size, color='darkorchid', label='all', fillstyle='none')
plt.figure(figsize=(20, 20))
fig, ax = plt.subplots(figsize=(15, 10), dpi=400)
list_for_wind_sum = []
list_for_wind_4dot = []
for i in range(epoch_num):
    list_for_wind_sum.append(float(lines[3+i*7][25:]))
    input_string = lines[6+i*7][18:-2]
    float_list = [float(x) for x in input_string.split(',')]
    list_for_wind_4dot.append(torch.tensor(float_list))
wind_4dot = torch.stack(list_for_wind_4dot, dim=0)   #[epoch_num,4]
plt.plot(test, list_for_wind_sum, marker='o', markersize=5,
         markeredgewidth=markeredgewidth, linestyle='--',
         linewidth=line_size, color='gold',
         label='sum_wind', fillstyle='none')
plt.xticks(test)
y = [wind_sum_tc_diffuser] * len(test)
plt.plot(test, y, color='red')
save_path = os.path.join(save_folder, f'sum_wind.png')  # 图像保存路径
plt.savefig(save_path)
# plt.show()

plt.figure(figsize=(20, 20))
fig, ax = plt.subplots(figsize=(15, 10), dpi=400)
min_value_of_4_time = torch.tensor([0.0, 0.0, 0.0, 0.0])
min_index = torch.tensor([0, 0, 0, 0])
for t in range(wind_4dot.size(1)):
    min_value_of_4_time[t], min_index[t] = torch.min(wind_4dot[:, t], dim=0)
# min_index  0--epoch_num   1-26
min_index = epoch_num-min_index
print("==================wind=====================")
print("6h,12h,18,24h's min value is at epoch:", min_index)
print("6h,12h,18,24h's min value is:", min_value_of_4_time)
for i in range(wind_4dot.size(1)):
    print(test[epoch_num-min_index[i].item()], ":", wind_4dot[epoch_num-min_index[i],:])

file_path = save_folder + '/' + name + '_result.txt'
with open(file_path, 'a') as file:  # output_WP_WP_env_10_fusion5
    file.write(f"==================wind=====================\n")
    file.write(f"6h,12h,18,24h's min value is at epoch: {min_index}\n")
    file.write(f"6h,12h,18,24h's min value is: {min_value_of_4_time}\n")
    for i in range(wind_4dot.size(1)):
        file.write(f"{test[epoch_num-min_index[i].item()]}: {wind_4dot[epoch_num - min_index[i], :]}\n")


t = [6,12,18,24]
plt.xticks(t)

wind_better = torch.zeros_like(wind_4dot)
for i in range(epoch_num):
    plt.plot(t, list_for_wind_4dot[i],marker='o', markersize=5,
             markeredgewidth=markeredgewidth, linestyle='--',
             linewidth=1,
             color='gold', label='sum_traj', fillstyle='none')
    wind_better[i] = np.around(list_for_wind_4dot[i], decimals=2) <= torch.tensor(tc_diffuser_wind)
plt.plot(t, tc_diffuser_wind,marker='o', markersize=5,
             markeredgewidth=markeredgewidth, linestyle='--',
             linewidth=1,
             color='red', label='tc_diffuser_wind', fillstyle='none')

save_path = os.path.join(save_folder, f'wind_4time.png')  # 图像保存路径
plt.savefig(save_path)
# plt.show()

'''
traj_better: [epoch_num,4]
pres_better: [epoch_num,4]
wind_better: [epoch_num,4]
'''
all_better = torch.concat((traj_better,pres_better,wind_better),dim=1) #[epoch_num,12]
for i in range(epoch_num):
    print("epoch" ,test[i],": " ,round((all_better[i].sum()/12).item(), 2), round((all_better[i][0:4].sum()/4).item(),2),
          round((all_better[i][4:8].sum()/4).item(),2), round((all_better[i][8:12].sum()/4).item(),2))


best_epoch_value, best_epoch = 0,0
with open(file_path, 'a') as file:  # output_WP_WP_env_10_fusion5
    file.write(f"===========================================\n")
    for i in range(wind_4dot.size(0)):
        file.write(f"epoch{test[i]}:  {round((all_better[i].sum()/12).item(),2)} / "
                   f"{round((all_better[i][0:4].sum() / 4).item(),2)} {round((all_better[i][4:8].sum() / 4).item(),2)} {round((all_better[i][8:12].sum() / 4).item(),2)}\n")
        if best_epoch_value < all_better[i].sum()/12:
            best_epoch_value = all_better[i].sum()/12
            best_epoch = test[i]
            item_for_next = i

    best_epoch_num = 0
    for i in range(wind_4dot.size(0)):
        if best_epoch_value == all_better[i].sum() / 12:
            best_epoch_num += 1

print("best epoch:", best_epoch)
item = item_for_next
item = int(item)
print("best_epoch_value:", round((all_better[item].sum()/12).item(), 2), round((all_better[item][0:4].sum()/4).item(),2),
          round((all_better[item][4:8].sum()/4).item(),2), round((all_better[item][8:12].sum()/4).item(),2) )
print(list_for_traj_4dot[int(item)],list_for_pres_4dot[int(item)],list_for_wind_4dot[int(item)])
with open(file_path, 'a') as file:  # output_WP_WP_env_10_fusion5
    file.write(f"===========================================\n")
    file.write(f"best epoch: {best_epoch}\n")
    file.write(f"{round((all_better[item].sum() / 12).item(), 2)} "
               f"{round((all_better[item][0:4].sum() / 4).item(), 2)} {round((all_better[item][4:8].sum() / 4).item(), 2)} "
               f"{round((all_better[item][8:12].sum() / 4).item(), 2)}\n")
    file.write(f"{list_for_traj_4dot[int(item)]} {list_for_pres_4dot[int(item)]} {list_for_wind_4dot[int(item)]}\n")

print(name)
print("最佳epoch的数量为：",best_epoch_num)

