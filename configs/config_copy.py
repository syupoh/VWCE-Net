
import os
import shutil

root_path = '/data/syupoh/python/_mine/mmaction2/configs'
# target_name = 'recognition/timesformer'
# target_name = 'recognition/tpn'
# target_name = 'recognition/tsn/'
target_name = 'recognition/x3d'



target_config = os.path.join(
    root_path,
    target_name    
)

for filename in os.listdir(target_config):
    
    if filename.endswith('.py') and filename.startswith(
        # os.path.basename(target_config)
        '220107'
    ):
        shutil.copy(
            os.path.join(
                target_config, 
                filename
                ), 
            os.path.join(
                target_config,
                filename.replace('220107', '220117')
                )
        )
        # shutil.copy(
        #     os.path.join(
        #         target_config, 
        #         filename
        #         ), 
        #     os.path.join(
        #         target_config,
        #         '220107_{0}'.format(filename)
        #         )
        # )
        
        # shutil.copy(
        #     os.path.join(
        #         target_config, 
        #         filename
        #         ), 
        #     os.path.join(
        #         target_config,
        #         '220107_4cls_{0}'.format(filename)
        #         )
        # )

