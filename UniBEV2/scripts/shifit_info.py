import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
import mmcv



det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/left_stereo/det_3d.json'
infos = mmcv.load(det_3d_path)
print(infos.keys())
# dict_keys(['frames', 'config'])
print(len(infos['frames']))
# 150000
print(infos['config']['imageSize'])
#{'width': 1280, 'height': 800}
print(infos['config']['categories'])
# [{'name': 'pedestrian'}, {'name': 'car'}, {'name': 'truck'}, {'name': 'bus'}, {'name': 'motorcycle'}, {'name': 'bicycle'}]
print(infos['frames'][0].keys())
# dict_keys(['name', 'videoName', 'intrinsics', 'extrinsics', 'attributes', 'frameIndex', 'labels'])
# {'name': '00000000_img_left_stereo.jpg',
#  'videoName': '0003-17fb',
#  'intrinsics': {'focal': [640.0, 640.0],
#                 'center': [640.0, 400.0], 'skew': 0},
#  'extrinsics': {'location': [32.4959831237793, -67.69702911376953, 1.5402708053588867],
#                 'rotation': [0.11117553182098182, -0.14815363057230854, 179.9752524025178]},
#  'attributes': {'weather_coarse': 'rainy', 'timeofday_coarse': 'night',
#                 'weather_fine': 'heavy rain', 'timeofday_fine': 'night', 'view': 'left_stereo',
#                 'town': '10HD', 'sun_altitude_angle': '-2', 'cloudiness': '100.0', 'precipitation': '100.0',
#                 'precipitation_deposits': '100.0', 'wind_intensity': '100.0', 'sun_azimuth_angle': '0.0',
#                 'fog_density': '7.0', 'fog_distance': '0.75', 'wetness': '0.0', 'fog_falloff': '0.1'},
#  'frameIndex': 0,
#  'labels': [{'id': '11367', 'attributes': {'type': 'vehicle.volkswagen.t2_2021'},
#                               'category': 'bus', 'box2d': {'x1': 278.0, 'y1': 391.0, 'x2': 383.0, 'y2': 448.0},
#                               'box3d': {'alpha': 0.0, 'orientation': [-0.0004453301835687373, 1.5708124626212618, 0.002251448747240946],
#                                         'location': [-10.290006768576262, 0.7253934110501639, 21.583345587779966],
#                                         'dimension': [1.9872066974639893, 1.7213283789157867, 4.308918452262878]}},
#          {'id': '11343', 'attributes': {'type': 'vehicle.carlamotors.carlacola'},
#           'category': 'truck', 'box2d': {'x1': 560.0, 'y1': 393.0, 'x2': 587.0, 'y2': 417.0},
#           'box3d': {'alpha': 0.0, 'orientation': [0.0029943990159684297, 1.5816452979359816, 0.000253542432400721],
#                     'location': [-6.390415849677126, 0.6025290553648639, 60.29239925599211],
#                     'dimension': [2.467444658279419, 2.5481799149513242, 5.047723197937011]}},
#          {'id': '11342', 'attributes': {'type': 'vehicle.tesla.model3'},
#           'category': 'car', 'box2d': {'x1': 426.0, 'y1': 404.0, 'x2': 500.0, 'y2': 442.0},
#           'box3d': {'alpha': 0.0, 'orientation': [0.001940320000007878, 1.5708191652730294, 0.0025858129527228124],
#                     'location': [-6.789403236291797, 0.9023235623423336, 25.083595187634337],
#                     'dimension': [1.4876600503921509, 2.0985465025901795, 4.648026132583618]}},
#          {'id': '11334', 'attributes': {'type': 'vehicle.bmw.grandtourer'},
#           'category': 'car', 'box2d': {'x1': 606.0, 'y1': 399.0, 'x2': 689.0, 'y2': 459.0},
#           'box3d': {'alpha': 0.0, 'orientation': [-0.0012612191407663786, -1.5707763097129446, -0.003758152236470214],
#                     'location': [0.20768482321682313, 0.7389984515613159, 18.217383091792062],
#                     'dimension': [1.6672759056091309, 2.174461886882782, 4.472675609588623]}},
#          {'id': '11299', 'attributes': {'type': 'vehicle.tesla.model3'},
#           'category': 'car', 'box2d':
#              {'x1': 513.0, 'y1': 403.0, 'x2': 532.0, 'y2': 418.0},
#           'box3d': {'alpha': 0.0, 'orientation': [0.0019122179305548492, 1.5816432053926759, 0.002606663155939502], 'location': [-9.890216049872002, 0.978403007700287, 56.83272573434367], 'dimension': [1.4876600503921509, 2.0985465025901795, 4.648026132583618]}}, {'id': '11297', 'attributes': {'type': 'vehicle.toyota.prius'}, 'category': 'car', 'box2d': {'x1': 393.0, 'y1': 402.0, 'x2': 427.0, 'y2': 413.0}, 'box3d': {'alpha': 0.0, 'orientation': [0.0025914948468375165, 0.008048368805860682, -0.0020279856100484358], 'location': [-26.29246275177723, 0.9347772249343982, 74.25286835322527], 'dimension': [1.5248334407806396, 1.9466100454330444, 4.3781169462203975]}}, {'id': '11292', 'attributes': {'type': 'vehicle.nissan.patrol_2021'}, 'category': 'car', 'box2d': {'x1': 526.0, 'y1': 397.0, 'x2': 565.0, 'y2': 423.0}, 'box3d': {'alpha': 0.0, 'orientation': [0.000501891845774427, 1.5816384097315268, 0.0017939820474380198], 'location': [-6.6307727191716594, 0.7602559215758626, 45.83343021068602], 'dimension': [2.045147180557251, 2.085467946529388, 5.398853936195374]}}, {'id': '11290', 'attributes': {'type': 'vehicle.dodge.charger_2020'}, 'category': 'car', 'box2d': {'x1': 436.0, 'y1': 402.0, 'x2': 459.0, 'y2': 409.0}, 'box3d': {'alpha': 0.0, 'orientation': [-0.00260115990053289, -3.133620596985647, 0.001919697291896354], 'location': [-24.3115509416928, 0.9913163082398668, 81.14479366263103], 'dimension': [1.5347249507904053, 1.8251732981204987, 4.857590613365173]}}, {'id': '11690', 'attributes': {'type': 'walker.pedestrian.0037'}, 'category': 'pedestrian', 'box2d': {'x1': 427.0, 'y1': 395.0, 'x2': 432.0, 'y2': 418.0}, 'box3d': {'alpha': 0.0, 'orientation': [0.0016189843116336533, 1.6900433204985612, 0.0027982420567478966], 'location': [-16.593576376020216, 0.5357561721351514, 50.886317649190474], 'dimension': [1.8600000143051147, 0.3753577768802643, 0.3753577768802643]}}, {'id': '11680', 'attributes': {'type': 'walker.pedestrian.0024'}, 'category': 'pedestrian', 'box2d': {'x1': 421.0, 'y1': 396.0, 'x2': 427.0, 'y2': 416.0}, 'box3d': {'alpha': 0.0, 'orientation': [0.0016505251025089596, 1.6787344519094618, 0.002779754641730673], 'location': [-18.673896059877507, 0.5437050364307796, 55.52146902663553], 'dimension': [1.8600000143051147, 0.3753577768802643, 0.3753577768802643]}}, {'id': '11677', 'attributes': {'type': 'walker.pedestrian.0037'}, 'category': 'pedestrian', 'box2d': {'x1': 58.0, 'y1': 386.0, 'x2': 74.0, 'y2': 442.0}, 'box3d': {'alpha': 0.0, 'orientation': [0.0015279697909533763, 1.7222739733185692, 0.002848960619315545], 'location': [-18.831402024980015, 0.4539047889485268, 20.911157141661256], 'dimension': [1.8600000143051147, 0.3753577768802643, 0.3753577768802643]}}, {'id': '11645', 'attributes': {'type': 'walker.pedestrian.0005'}, 'category': 'pedestrian', 'box2d': {'x1': 768.0, 'y1': 395.0, 'x2': 775.0, 'y2': 413.0}, 'box3d': {'alpha': 0.0, 'orientation': [0.00035245009107698344, -0.817797577162662, -0.0032135714480728056], 'location': [8.652359511579434, 0.5617555240401371, 41.99638788277343], 'dimension': [1.8600000143051147, 0.3753577768802643, 0.3753577768802643]}}, {'id': '11634', 'attributes': {'type': 'walker.pedestrian.0010'}, 'category': 'pedestrian', 'box2d': {'x1': 207.0, 'y1': 408.0, 'x2': 215.0, 'y2': 433.0}, 'box3d': {'alpha': 0.0, 'orientation': [0.0017326261537693455, 1.6489310275894198, 0.002729336187356797], 'location': [-17.75154929419425, 0.8505488327903881, 26.536868778258853], 'dimension': [1.100000023841858, 0.5, 0.5]}}, {'id': '11609', 'attributes': {'type': 'walker.pedestrian.0010'}, 'category': 'pedestrian', 'box2d': {'x1': 48.0, 'y1': 412.0, 'x2': 59.0, 'y2': 452.0}, 'box3d': {'alpha': 0.0, 'orientation': [5.2044414473151335e-05, 2.1984572438735595, 0.0032324222277730487], 'location': [-15.58612432838897, 0.8301698205580031, 17.030735191825705], 'dimension': [1.100000023841858, 0.5, 0.5]}}, {'id': '11581', 'attributes': {'type': 'walker.pedestrian.0026'}, 'category': 'pedestrian', 'box2d': {'x1': 776.0, 'y1': 390.0, 'x2': 785.0, 'y2': 427.0}, 'box3d': {'alpha': 0.0, 'orientation': [0.0032305158362264663, 0.681691466274039, 0.0001225960364179297], 'location': [5.501658961769394, 0.5113562655585425, 24.869752818494742], 'dimension': [1.8600000143051147, 0.3753577768802643, 0.3753577768802643]}}, {'id': '11518', 'attributes': {'type': 'walker.pedestrian.0035'}, 'category': 'pedestrian', 'box2d': {'x1': 269.0, 'y1': 392.0, 'x2': 283.0, 'y2': 426.0}, 'box3d': {'alpha': 0.0, 'orientation': [-0.0031043748023142914, -2.2149748478276745, -0.0009022871190283045], 'location': [-19.52950279570159, 0.4872719829338541, 34.33913094010768], 'dimension': [1.8600000143051147, 0.3753577768802643, 0.3753577768802643]}}, {'id': '11497', 'attributes': {'type': 'walker.pedestrian.0038'}, 'category': 'pedestrian', 'box2d': {'x1': 40.0, 'y1': 386.0, 'x2': 64.0, 'y2': 445.0}, 'box3d': {'alpha': 0.0, 'orientation': [0.0012278301873165365, 1.8249786422174321, 0.002990601866607095], 'location': [-17.59498916120704, 0.451793179317175, 19.1667227161834], 'dimension': [1.8600000143051147, 0.3753577768802643, 0.3753577768802643]}}, {'id': '11443', 'attributes': {'type': 'walker.pedestrian.0041'}, 'category': 'pedestrian', 'box2d': {'x1': 432.0, 'y1': 395.0, 'x2': 437.0, 'y2': 418.0}, 'box3d': {'alpha': 0.0, 'orientation': [-0.003120664894052716, -2.7620360024075756, 0.0008442246103080051], 'location': [-15.376253065568953, 0.5305062975789439, 47.94254498739313], 'dimension': [1.8600000143051147, 0.3753577768802643, 0.3753577768802643]}}, {'id': '11430', 'attributes': {'type': 'walker.pedestrian.0013'}, 'category': 'pedestrian', 'box2d': {'x1': 1178.0, 'y1': 411.0, 'x2': 1205.0, 'y2': 473.0}, 'box3d': {'alpha': 0.0, 'orientation': [0.0016790909446662283, 1.6684263457686106, 0.0027625934798380953], 'location': [9.092251572366237, 0.7612702170208611, 10.539647427547543], 'dimension': [1.2999999523162842, 0.3753577768802643, 0.3753577768802643]}}]}




det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/left_stereo/det_3d.json'
infos_ls = mmcv.load(det_3d_path)
det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/front/det_3d.json'
infos_f = mmcv.load(det_3d_path)
det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/left_45/det_3d.json'
infos_l45 = mmcv.load(det_3d_path)
det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/left_90/det_3d.json'
infos_l90 = mmcv.load(det_3d_path)
det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/right_45/det_3d.json'
infos_r45 = mmcv.load(det_3d_path)
det_3d_path = '/mnt/cfs/algorithm/hao.lu/Data/SHIFT/shift-dev/data/discrete/images/train/right_90/det_3d.json'
infos_r90 = mmcv.load(det_3d_path)


i=0
infos_ls['frames'][i]['extrinsics']['location']
infos_f['frames'][i]['extrinsics']['location']
infos_l45['frames'][i]['extrinsics']['location']
infos_l90['frames'][i]['extrinsics']['location']
infos_r45['frames'][i]['extrinsics']['location']
infos_r90['frames'][i]['extrinsics']['location']

i=2
list(map(lambda x: x[0]-x[1], zip(infos_ls['frames'][i]['extrinsics']['location'], infos_f['frames'][i]['extrinsics']['location'])))
infos_ls['frames'][i]['name']
infos_f['frames'][i]['name']
infos_ls['frames'][i]['extrinsics']['location']-infos_f['frames'][i]['extrinsics']['location']
infos_l45['frames'][i]['extrinsics']['location']-infos_f['frames'][i]['extrinsics']['location']
infos_l90['frames'][i]['extrinsics']['location']-infos_f['frames'][i]['extrinsics']['location']
infos_r45['frames'][i]['extrinsics']['location']-infos_f['frames'][i]['extrinsics']['location']
infos_r90['frames'][i]['extrinsics']['location']-infos_f['frames'][i]['extrinsics']['location']



center = list(map(lambda x: (x[0] + x[1])/2, zip(infos_l90['frames'][i]['extrinsics']['location'],
                                        infos_r90['frames'][i]['extrinsics']['location'])))
rotation = list(map(lambda x: (x[0] + x[1])/2, zip(infos_f['frames'][i]['extrinsics']['location'],
                                        center)))



