import os
import shutil


rootpath = './framenumbers'
for filename in sorted(os.listdir(rootpath)):
    fullpath = os.path.join(rootpath, filename)

    new_filename = filename.replace('_15e_kinetics400_rgb', '')
    new_fullpath = os.path.join(rootpath, new_filename)

    shutil.copy(
        fullpath,
        new_fullpath
    )

# configfileset=[
#     '220117_timesformer_jointST_8x32x1_15e_kinetics400_rgb.py',
#     '220117_timesformer_divST_8x32x1_15e_kinetics400_rgb.py',
#     '220117_timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb.py',
# ]

# framenumber_set = [
#     2, 4, 6, 10, 12, 14, 16 , 18, 20
# ]

# for configfile in configfileset:
#     prevname = configfile.split('8')[0]
#     postname = configfile.split('8')[1]
#     for framenumber in framenumber_set:
#         newname = '{0}{1}{2}'.format(
#             prevname, framenumber, postname
#         )
#         shutil.copy(
#             configfile,
#             os.path.join('./framenumbers/', newname)
#         )
