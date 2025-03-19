import sys
import os
import numpy as np
import pandas as pd
import dill
import pickle
import netCDF4
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, gaussian_filter, maximum_filter, minimum_filter

from environment import Environment, Scene, Node, derivative_of

data_folder_name = 'process_data_1D-ENV-ERA5-ALL-WE-NEED-vocen-vodis-global'
# delete_processed_data_noise_traj_inten_wind_gph_env_era5_200_500_gph_u_v：test有点问题  train也变小了
# valid_test_gph_200_500: 对齐了200和500

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.4

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }

        ,
        'intensity': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity_i': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration_i': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }

        # ,
        # 'gph':{
        #     'x': {'mean': 0, 'std': 1,
        #             'mean': 0, 'std': 1
        #
        #
        #           },
        #     'y': {'mean': 0, 'std': 1}
        # }


    }
}

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration',
                                                'intensity', 'velocity_i', 'acceleration_i'
                                                # ,'gph'
                                                ], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        #intensity
        x_i = node.data.intensity.x.copy()
        y_i = node.data.intensity.y.copy()

        x_i, y_i = rotate_pc(np.array([x_i, y_i]), alpha)

        vx_i = derivative_of(x_i, scene.dt)
        vy_i = derivative_of(y_i, scene.dt)
        ax_i = derivative_of(vx_i, scene.dt)
        ay_i = derivative_of(vy_i, scene.dt)

        # gph = node.gph_data
        env_data = node.env_data
        gph_all_hpa_data = node.gph_all_hpa_data
        # sst_data = node.sst_data
        # tcwv_data = node.tcwv_data
        # temperature_data = node.temperature_data
        wind_data = node.wind_data

        q_data = node.q_data
        t_data = node.t_data

        vo_cen = node.vo_cen
        vo_dis = node.vo_dis


        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay,

                     ('intensity', 'x'): x_i,
                     ('intensity', 'y'): y_i,
                     ('velocity_i', 'x'): vx_i,
                     ('velocity_i', 'y'): vy_i,
                     ('acceleration_i', 'x'): ax_i,
                     ('acceleration_i', 'y'): ay_i

                     # ,
                     # ('gph', 'x'): gph,
                     # ('gph', 'y'): gph1

                     }

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data,
                    # gph_data = gph,
                    env_data = env_data,
                    gph_all_hpa_data = gph_all_hpa_data
                    # , sst_data = sst_data
                    # , tcwv_data = tcwv_data
                    # , temperature_data = temperature_data
                    , wind_data = wind_data
                    , q_data = q_data
                    , t_data = t_data
                    , vo_cen = vo_cen
                    , vo_dis = vo_dis
                    , first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug

def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


nl = 0
l = 0

maybe_makedirs(data_folder_name)
# data_columns这个多级索引对象被用于表示一个包含位置、速度和加速度的数据集，其中每个变量都有x和y两个维度
# 使用这个多级索引对象data_columns可以创建一个包含位置、速度和加速度数据的DataFrame，其中每个变量都有x和y两个维度的取值
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration',
                                            'intensity','velocity_i', 'acceleration_i'
                                            # ,'gph'
                                            ], ['x', 'y']])
def read_nc2array(in_file):
    with netCDF4.Dataset(in_file) as nf:
        # for var in nf.variables:
        #     print(var)
        '''5 pres-level: 925, 850, 700, 500, 200'''
        z = nf.variables["z"][:].data
        q = nf.variables["q"][:].data
        t = nf.variables["t"][:].data
        u = nf.variables["u"][:].data
        v = nf.variables["v"][:].data
        u200 = u[:, 4]
        u500 = u[:, 3]
        u850 = u[:, 1]
        v200 = v[:, 4]
        v500 = v[:, 3]
        v850 = v[:, 1]
        vws = np.sqrt((u200 - u850) * (u200 - u850) + (v200 - v850) * (v200 - v850))
        vorticity200 = np.gradient(v200, 1, axis=1) - np.gradient(u200, 1, axis=2)
        vorticity500 = np.gradient(v500, 1, axis=1) - np.gradient(u500, 1, axis=2)
        vorticity850 = np.gradient(v850, 1, axis=1) - np.gradient(u850, 1, axis=2)
        '''5, 5, 81, 81''' '''5个变量z, q, t, u, v, 5个气压层925, 850, 700, 500, 200, 81, 81'''
        era_var5 = np.vstack((z, q, t, u, v)) #z, q, t, u, v
        '''4 vws, vorticity200, vorticity500, vorticity850, 81, 81'''
        era_aux = np.vstack((vws, vorticity200, vorticity500, vorticity850))
    return era_var5, era_aux

# Process ETH-UCY
#        五个不同的场景： ETH    ETH      UCY      UCY      UCY
#        6个不同的大洋

# for desired_source in ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']:

def compute_vorticity(u_wind, v_wind, dx, dy):
    dv_dx = np.gradient(v_wind, dx, axis=-1)

        # 计算相对于y（纬度）的u风偏导数
    du_dy = np.gradient(u_wind, dy, axis=-2)

        # 计算涡度
    vorticity = dv_dx - du_dy
    return vorticity

def find_vorticity_centers(vorticity, threshold=0.01):
    num = 6
    # 对涡度数据进行高斯平滑，以减少噪声
    smoothed_vorticity = gaussian_filter(vorticity, sigma=1)

    # 创建极小值的二值图
    minima = (smoothed_vorticity == minimum_filter(smoothed_vorticity, size=num)) & (smoothed_vorticity < -threshold)

    # 标记极小值
    labeled_minima, num_minima = label(minima)
    while num_minima<3:
        threshold -= 0.01
        minima = (smoothed_vorticity == minimum_filter(smoothed_vorticity, size=num)) & (
                    smoothed_vorticity < -threshold)
        labeled_minima, num_minima = label(minima)

    centers = []
    centers_coor = []

    # 提取极小值的中心
    for i in range(1, num_minima + 1):
        indices = np.argwhere(labeled_minima == i)
        if len(indices) > 0:
            center = np.mean(indices, axis=0)
            centers.append((center, smoothed_vorticity[int(center[0]), int(center[1])]))
            centers_coor.append(center)

    # 按照涡度值排序，返回前 1-6 个中心（极小值）
    centers = sorted(centers, key=lambda x: x[1])[:num]
    center_coorr = [center[0] for center in centers]

    # 计算距离
    center_coordinates = np.array([40, 40])
    distances_to_center = np.linalg.norm(np.array(center_coorr) - center_coordinates, axis=1)
    sorted_indices = np.argsort(distances_to_center)

    # 选取最近的三个中心
    closest_centers = [center_coorr[i] for i in sorted_indices[:3]]
    distances = np.sum(distances_to_center[sorted_indices[:3]])

    return closest_centers,distances


def process_by_year(year_list, ocean, test_or_train):
    year_left = year_list[0]
    year_right = year_list[1]
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()  # 存储节点之间的注意半径信息
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius  # 将 attention_radius 应用于环境。

    scenes = []
    for desired_source in ocean:
        print("desired_source:", desired_source)
        # for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:

        for data_class in test_or_train:  # 'test','train'  ,'train' 'train','test','val'
            # for data_class in ['test']:
            print("data_class:", data_class)

            # 数据字典路径 pkl文件的路径 'processed_data_noise_new/ep_train.pkl'
            data_dict_path = os.path.join(data_folder_name, '_'.join(
                [desired_source, data_class, str(year_left), str(year_right)]) + '.pkl')  # 根据指定的数据路径和文件名，使用 os.path.join 函数构建数据字典路径 data_dict_path。
            # 从这里不行了 唉
            print(os.path.join('bst_divi10_train_val_test_inlcude15', desired_source, data_class))

            for subdir, dirs, files in os.walk(os.path.join(
                    '/opt/data/private/global_area_tropicalcyclone_data_1950_2023/TropiCycloneNet/bst_divi10_train_val_test_inlcude15',
                    desired_source, data_class)):
                print("subdir:", subdir)
                env_path = os.path.join(
                    '/opt/data/private/global_area_tropicalcyclone_data_1950_2023/TropiCycloneNet/all_area_correct_location_includeunder15',
                    desired_source)  # /年份/大洋的名字/年yyyy月mm日dd小时hh.npy
                # gph_path_500 = os.path.join(
                #     '/opt/data/private/global_area_tropicalcyclone_data_1950_2023/TropiCycloneNet/all_ocean_gph500')  # /年份/大洋的名字/大洋的名字_年yyyy月mm日dd小时hh_geo...npy
                # gph_all_hpa_path = os.path.join(
                #     '/root/AAAI_data_zyc/opt/data/private/MGTCF/scripts/datasets/AAAI_data/geopotential_centercrop')  # /年份/大洋的名字/大洋的名字_年yyyy月mm日dd小时hh_geo...npy
                # sst_path = os.path.join(
                #     '/root/datasets/AAAI_data/opt/data/private/MGTCF/scripts/datasets/AAAI_data/sst_centercrop')  # /年份/大洋的名字/大洋的名字_年yyyy月mm日dd小时hh_geo...npy
                # tcwv_path = os.path.join(
                #     '/root/datasets/AAAI_data/opt/data/private/MGTCF/scripts/datasets/AAAI_data/tcwv_centercrop')  # /年份/大洋的名字/大洋的名字_年yyyy月mm日dd小时hh_geo...npy
                # temperature_path = os.path.join(
                #     '/root/datasets/AAAI_data/opt/data/private/MGTCF/scripts/datasets/AAAI_data/temperature_centercrop')  # /年份/大洋的名字/大洋的名字_年yyyy月mm日dd小时hh_geo...npy
                # wind_path = os.path.join(
                #     '/root/AAAI_data_zyc/opt/data/private/MGTCF/scripts/datasets/AAAI_data/wind_centercrop')  # /年份/大洋的名字/大洋的名字_年yyyy月mm日dd小时hh_geo...npy
                all_we_need_path = os.path.join(
                    '/opt/data/private/ALL-TC-correlation/era5_download/era5_download/era5_data/core_nc_925_850_700_500_200_all_we_need/')
                num = 0
                for file in files:  # 600 keyi 700可以 700不行
                    if file.endswith('.txt'):  # and num<=600:# and file[2:6]=='1950': #通过检查文件后缀为 .txt 的文件来获取数据文件
                        input_data_dict = dict()  # 创建一个空字典 读取文件内容并进行数据处理
                        full_data_path = os.path.join(subdir, file)  # subdir变量表示当前文件夹的路径 files变量是一个列表，包含当前文件夹中的文件名称
                        print('At', full_data_path)
                        num = num + 1
                        print("num:", num)
                        yy_ori = file[2:6]
                        tc_name = file[9:-4].capitalize()
                        if int(yy_ori)<year_left or int(yy_ori)>year_right:
                            print("jump")
                            continue


                        #@@@ 这个数据缺失 暂时删掉
                        # if int(yy_ori) ==2021 and tc_name=='Mindulle':
                        #     continue
                        #
                        # if file!= 'WP1976BSTOPAL.txt': #@@@
                        #     continue


                        data = pd.read_csv(full_data_path, sep='\t', index_col=False,
                                           header=None)  # 使用 pd.read_csv 函数加载文件数据，
                        # 设置分隔符为 \t， 刚好前四列是这个
                        data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y', 'inten_x', 'inten_y',
                                        'yymmddtt', 'name']  # 列名为 ['frame_id', 'track_id', 'pos_x', 'pos_y']

                        data['frame_id'] = pd.to_numeric(data['frame_id'],
                                                         downcast='integer')  # 将 frame_id 和 track_id 列转换为整数类型
                        data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')
                        # ？？？？？？？？？？？？？？？？？？？？？？？
                        # data['frame_id'] = data['frame_id'] // 10 #将 frame_id 除以 10，并将结果减去最小值，以对时间步长进行调整。可能有问题
                        data['frame_id'] -= data['frame_id'].min()  # 所以时间步长都是从0开始的
                        data['node_type'] = 'PEDESTRIAN'
                        data['node_id'] = data['track_id'].astype(str)  # same in one txt file

                        data.sort_values('frame_id', inplace=True)  # 根据 frame_id 列对数据进行排序。

                        max_timesteps = data['frame_id'].max()
                        scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=desired_source + "_" + data_class,
                                      aug_func=augment if data_class == 'train' else None)

                        for node_id in pd.unique(data['node_id']):  # 遍历每个唯一的node_id（节点标识符），
                            # 从数据中选择具有相同node_id的行，并提取出'pos_x'和'pos_y'列的数值。
                            # 如果节点的数据点数少于2，则忽略该节点

                            node_df = data[data['node_id'] == node_id]

                            node_values = node_df[['pos_x', 'pos_y']].values

                            node_values_i = node_df[['inten_x', 'inten_y']].values

                            node_ymdt = node_df[['yymmddtt']].values
                            node_name = node_df[['name']].values

                            if node_values.shape[0] < 2:
                                continue

                            new_first_idx = node_df['frame_id'].iloc[0]  # 第一行的"frame_id"值

                            x = node_values[:, 0]  # 横坐标
                            y = node_values[:, 1]  # 纵坐标
                            vx = derivative_of(x, scene.dt)  # dt=0.4
                            vy = derivative_of(y, scene.dt)
                            ax = derivative_of(vx, scene.dt)
                            ay = derivative_of(vy, scene.dt)

                            x_i = node_values_i[:, 0]  # 横坐标
                            y_i = node_values_i[:, 1]  # 纵坐标
                            vx_i = derivative_of(x_i, scene.dt)  # dt=0.4
                            vy_i = derivative_of(y_i, scene.dt)
                            ax_i = derivative_of(vx_i, scene.dt)
                            ay_i = derivative_of(vy_i, scene.dt)

                            gph_data_500 = []
                            env_data = []
                            gph_all_hpa_data = []
                            sst_data = []
                            tcwv_data = []
                            temperature_data = []
                            wind_data = []
                            q_data = []
                            t_data = []

                            vo_cen = []
                            vo_dis = []

                            if (desired_source == 'WP' and data_class == 'test'):
                                name = file[9:-4]
                            else:
                                name = str(node_name[0])[2:-2]

                            name_only_first_big = name.capitalize()
                            # 若gph数据里，直接没有这个台风名的文件夹，直接跳过这个台风
                            yy_test = str(node_ymdt[0])[1:5]
                            # test_gph_tc_path = os.path.join(gph_all_hpa_path, yy_test, name)
                            # if not os.path.exists(test_gph_tc_path):
                            #     test_gph_tc_path = os.path.join(gph_all_hpa_path, yy_test, name_only_first_big)
                            # if not os.path.exists(test_gph_tc_path):
                            #     print(test_gph_tc_path, " not exist")
                            #     continue

                            # # 若gph500数据里，直接没有这个台风名的文件夹，直接跳过这个台风
                            # test_gph500_tc_path = os.path.join(gph_path_500,desired_source,  yy_test, name)
                            # if not os.path.exists(test_gph500_tc_path):
                            #     test_gph500_tc_path = os.path.join(gph_path_500,desired_source, yy_test, name_only_first_big)
                            # if not os.path.exists(test_gph500_tc_path):
                            #     print(test_gph500_tc_path, " not exist")
                            #     continue
                            #
                            # # 若wind数据里，直接没有这个台风名的文件夹，直接跳过这个台风
                            # test_wind_tc_path = os.path.join(wind_path, yy_test, name)
                            # if not os.path.exists(test_wind_tc_path):
                            #     test_wind_tc_path = os.path.join(wind_path, yy_test, name_only_first_big)
                            # if not os.path.exists(test_wind_tc_path):
                            #     print(test_wind_tc_path, " not exist")
                            #     continue

                            # 若all_we_need数据里，直接没有这个台风名的文件夹，直接跳过这个台风
                            test_all_we_need_tc_path = os.path.join(all_we_need_path, yy_test, name)
                            if not os.path.exists(test_all_we_need_tc_path):
                                test_all_we_need_tc_path = os.path.join(all_we_need_path, yy_test, name_only_first_big)
                            if not os.path.exists(test_all_we_need_tc_path):
                                print(test_all_we_need_tc_path, " not exist")
                                continue

                            for i in range(node_ymdt.shape[0]):
                                # yy = str(node_ymdt[0])[1:-1][0:4]
                                yymmddtt = str(node_ymdt[i])[1:-1]
                                npy_name = yymmddtt + '.npy'

                                yy = yymmddtt[0:4]
                                # # 该时刻的gph文件，两种台风名都试试，若都不存在报错
                                # full_gph_all_hpa_path = os.path.join(gph_all_hpa_path, yy, name,
                                #                                      name + '_' + yymmddtt + '_geopotential.npy')
                                # if not os.path.exists(full_gph_all_hpa_path):
                                #     name_only_first_big = name.capitalize()
                                #     full_gph_all_hpa_path = os.path.join(gph_all_hpa_path, yy, name_only_first_big,
                                #                                          name_only_first_big + '_' + yymmddtt + '_geopotential.npy')
                                #
                                # if not os.path.exists(full_gph_all_hpa_path):
                                #     print(full_gph_all_hpa_path, " not exist")
                                #     # continue

                                # # # 该时刻的wind文件，两种台风名都试试，若都不存在报错
                                # full_wind_path = os.path.join(wind_path, yy, name, name + '_' + yymmddtt + '_wind.npy')
                                # if not os.path.exists(full_wind_path):
                                #     name_only_first_big = name.capitalize()
                                #     full_wind_path = os.path.join(wind_path, yy, name_only_first_big,
                                #                                   name_only_first_big + '_' + yymmddtt + '_wind.npy')
                                # if not os.path.exists(full_wind_path):
                                #     print(full_wind_path, " not exist")
                                #     # continue

                                # # 该时刻的gph500文件，两种台风名都试试，若都不存在报错
                                # full_gph_path_500 = os.path.join(gph_path_500,desired_source, yy_ori, name, npy_name)
                                # if not os.path.exists(full_gph_path_500):
                                #     full_gph_path_500 = os.path.join(gph_path_500,desired_source, yy_ori, name_only_first_big, npy_name)
                                # if not os.path.exists(full_gph_path_500):
                                #     print(full_gph_path_500, " not exist")

                                # # 该时刻的all_we_need文件，两种台风名都试试，若都不存在报错
                                full_all_we_need_path = os.path.join(all_we_need_path, yy, name,
                                                                     name + '_' + yymmddtt + '_200-925_gph_sh_temp_u_v.nc')
                                if not os.path.exists(full_all_we_need_path):
                                    name_only_first_big = name.capitalize()
                                    full_all_we_need_path = os.path.join(all_we_need_path, yy, name_only_first_big,
                                                                         name_only_first_big + '_' + yymmddtt + '_200-925_gph_sh_temp_u_v.nc')
                                if not os.path.exists(full_all_we_need_path):
                                    print(full_all_we_need_path, " not exist")

                                full_env_path = os.path.join(env_path, yy_ori, name, npy_name)
                                env_alone = np.load(full_env_path, allow_pickle=True).item()
                                env_data.append(env_alone)  # 将读取的数据添加到列表中

                                # 加载_200-925_gph_sh_temp_u_v.nc数据
                                '''era_var5：5个变量zqtuv, 5个气压层925-850-700-500-200, 81, 81'''
                                '''era_aux：4 vws, vorticity200, vorticity500, vorticity850, 81, 81'''
                                era_var5, era_aux = read_nc2array(
                                    full_all_we_need_path)  # z, q, t, u, v/vws, vorticity200, vorticity500, vorticity850

                                '''2, 81, 81'''
                                gph_200_500 = era_var5[0][3:5]

                                uv = era_var5[3:5]
                                '''2,4,81,81'''
                                wind_alone = uv[:, 1:5] #850-700-500-200 2,4,81,81

                                u_4_hPa = wind_alone[0, ]  # [4, 81, 81]
                                u_4_hPa = np.flip(u_4_hPa, axis=0)

                                  # 气压层200-500-700-850
                                v_4_hPa = wind_alone[1, ]  # [4, 81, 81]
                                v_4_hPa = np.flip(v_4_hPa, axis=0)

                                avg_latitude = 15.0  # 台风活动常见的纬度范围

                                dy = 111.32 * 0.25  # 0.25度纬度分辨率

                                # 计算在平均纬度下的经度方向距离 (dx)
                                dx = 111.32 * 0.25 * np.cos(np.radians(avg_latitude))  # 固定纬度下的 dx

                                # 提取200 hPa、500 hPa 和 850 hPa层
                                # 选择 [0, 1, 3] 索引的数据
                                u_selected = u_4_hPa[[0, 1, 3], :, :]  # 提取200、500、850 hPa
                                v_selected = v_4_hPa[[0, 1, 3], :, :]

                                # 计算200、500、850 hPa的涡度 (3,81,81)
                                vorticity_selected = compute_vorticity(u_selected, v_selected, dx, dy)

                                closest_centers = []
                                distances = []

                                # if len(vo_cen)==11:
                                #     print(vo_cen)
                                for p in range(vorticity_selected.shape[0]):
                                    closest_center, distance = find_vorticity_centers(vorticity_selected[p])
                                    closest_centers.append(closest_center)
                                    distances.append(distance)

                                print(closest_centers)
                                centemp=np.array(closest_centers).astype(np.float32)
                                vo_cen.append(centemp)
                                vo_dis.append(np.array(distances).astype(np.float32))

                                q = era_var5[1]  # 5,81,81
                                q500 = q[3]  # 81,81
                                q850 = q[1]  # 81,81
                                q_500_850 = np.concatenate((q500[np.newaxis, :], q850[np.newaxis, :]), axis=0)

                                t = era_var5[2]
                                t700 = t[2]
                                t850 = t[1]
                                t925 = t[0]
                                t_700_850_925 = np.concatenate(
                                    (t700[np.newaxis, :], t850[np.newaxis, :], t925[np.newaxis, :]), axis=0)

                                gph_all_hpa_data.append(gph_200_500)
                                wind_data.append(wind_alone)
                                q_data.append(q_500_850)
                                t_data.append(t_700_850_925)

                                # npy_name_sst = name + '_' + yymmddtt + '_sst.npy'
                                # full_sst_path = os.path.join(sst_path, yy, name, npy_name_sst)
                                # sst_alone = np.load(full_sst_path)
                                # sst_data.append(sst_alone)
                                #
                                # npy_name_tcwv = name + '_' + yymmddtt + '_tcwv.npy'
                                # full_tcwv_path = os.path.join(tcwv_path, yy, name, npy_name_tcwv)
                                # tcwv_alone = np.load(full_tcwv_path)
                                # tcwv_data.append(tcwv_alone)
                                #
                                # npy_name_temperature = name + '_' + yymmddtt + '_temperature.npy'
                                # full_temperature_path = os.path.join(temperature_path, yy, name, npy_name_temperature)
                                # temperature_alone = np.load(full_temperature_path)
                                # temperature_data.append(temperature_alone)
                                #

                                # wind_alone = np.load(full_wind_path)

                            # gph_data = np.array(gph_data)  # 将列表转换为tuple [22,100,100]
                            env_data = np.array(env_data)  # env[0]--env[n]
                            gph_all_hpa_data = np.array(gph_all_hpa_data)  # [22,4,101,101]
                            # sst_data = np.array(sst_data) #[22,101,101]
                            # tcwv_data = np.array(tcwv_data) #[22,101,101]
                            # temperature_data = np.array(temperature_data) #[22,4,101,101]
                            wind_data = np.array(wind_data)  # [22,4,101,101,2]

                            q_data = np.array(q_data)
                            t_data = np.array(t_data)

                            vo_cen = np.array(vo_cen)
                            vo_dis = np.array(vo_dis)

                            # gph1 = 0

                            data_dict = {('position', 'x'): x,
                                         ('position', 'y'): y,
                                         ('velocity', 'x'): vx,
                                         ('velocity', 'y'): vy,
                                         ('acceleration', 'x'): ax,
                                         ('acceleration', 'y'): ay,

                                         ('intensity', 'x'): x_i,
                                         ('intensity', 'y'): y_i,
                                         ('velocity_i', 'x'): vx_i,
                                         ('velocity_i', 'y'): vy_i,
                                         ('acceleration_i', 'x'): ax_i,
                                         ('acceleration_i', 'y'): ay_i

                                         # ,
                                         # ('gph', 'x'): gph,
                                         # ('gph', 'y'): gph1

                                         }
                            # 要不 把gph单独加到node里面 不在data_dict里了？
                            node_data = pd.DataFrame(data_dict, columns=data_columns)
                            node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data
                                        # , gph_data = gph_data
                                        , env_data=env_data
                                        , gph_all_hpa_data=gph_all_hpa_data
                                        # , sst_data = sst_data
                                        # , tcwv_data = tcwv_data
                                        # , temperature_data = temperature_data
                                        , wind_data=wind_data
                                        , q_data=q_data
                                        , t_data=t_data
                                        , vo_cen = vo_cen
                                        , vo_dis = vo_dis
                                        )  # 创建一个Node对象来表示该节点，包括节点类型、节点标识符和数据
                            node.first_timestep = new_first_idx  # 将节点的第一个时间步索引设置为new_first_idx

                            scene.nodes.append(node)  # 创建的节点添加到场景的节点列表 多个节点 分开了
                        if data_class == 'train':  # 如果data_class为'train'，则创建一个空列表augmented，然后对场景进行增强操作。
                            # 在这里，使用不同的角度对场景进行增强，每个角度增强一个新的场景，
                            # 并将其添加到augmented列表中。
                            scene.augmented = list()
                            angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                            for angle in angles:
                                scene.augmented.append(augment_scene(scene, angle))

                        print(scene)
                        scenes.append(scene)
            # 这句输出了
            print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

    env.scenes = scenes

            # name_tt='0618'+'.pkl'
            #
            # with open(name_tt, 'wb') as f:
            #     dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

    if len(scenes) > 0:  # data_dict_path：数据字典路径 pkl文件的路径
        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f,
                        protocol=dill.HIGHEST_PROTOCOL)  # 使用dill.dump函数将env对象（包含所有场景）序列化并写入文件。protocol=dill.HIGHEST_PROTOCOL参数指定了序列化协议，使用dill库的最高协议级别。
                    # 所有的场景数据就被保存到了文件中，可以在需要时进行加载和使用
# all
# process_by_year([1951,1971], ['WP'], ['train'])
# process_by_year([1972,1994], ['WP'], ['train'])
# process_by_year([1995,2016], ['WP'], ['train'])
# process_by_year([2017,2023], ['WP'], ['test'])
#
# # small train
# process_by_year([1951,1952], ['WP'], ['train'])

# 处理其他五个大洋的train数据成pkl
# process_by_year([1970,2016], ['SP','NI'], ['train']) #278+28
# process_by_year([1988,2016], ['EP'], ['train'])
# process_by_year([1959,2016], ['NA'], ['train'])
# process_by_year([1973,2016], ['SI'], ['train'])

process_by_year([2017,2023], ['SP'], ['test'])
process_by_year([2017,2023], ['SI'], ['test'])
process_by_year([2017,2023], ['NI'], ['test'])
process_by_year([2017,2023], ['NA'], ['test'])
process_by_year([2017,2023], ['EP'], ['test'])



exit() #到这就不跑了 所以之前没有处理stanford的数据
