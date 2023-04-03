#!/bin/sh
#SOURCE=$1
#TARGET=$2
#ARCH1=$3
#ARCH2=$4
#ARCH3=$5

# MYVAR="pre/users/joebloggs/domain.com" 

# # Remove the path leaving file name (all characters up to a slash):
# echo ${MYVAR##*/}
# # domain.com
# echo ${MYVAR#*/}
# # users/joebloggs/domain.com

# # Remove the file name, leaving the path (delete shortest match after last /):
# echo ${MYVAR%/*}
# # pre/users/joebloggs
# echo ${MYVAR%%/*}
# # pre

# # Get just the file extension (remove all before last period):
# echo ${MYVAR##*.}
# # com

# # NOTE: To do two operations, you can't combine them, but have to assign to an intermediate variable. So to get the file name without path or extension:

# NAME=${MYVAR##*/}      # remove part before last slash
# echo ${NAME%.*}        # from the new var remove the part after the last period
# domain

# ${MYVAR:3}   # Remove the first three chars (leaving 4..end)
# ${MYVAR::3}  # Return the first three characters
# ${MYVAR:3:5} # The next five characters after removing the first 3 (chars 4-9)
#

if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

#  conda update -n base -c defaults conda
if [ ${gpu} -eq "0" ]
  then
  
    nframes_set='
    4 6 2 8 3 5 
    '
    for nframes in ${nframes_set}
    do
      
      annfile_set="
      test_2_offset_${nframes}
      "
      # test_2_offset_${nframes}_half
      pthfile_set="
      ./work_dirs/220209_train_2_offset_${nframes}_*/epoch_100.pth
      "
      # echo ${nframes}
      for pthfile in ${pthfile_set}
      do
        for annfile in ${annfile_set}
        do
          # echo ${pthfile}
          CUDA_VISIBLE_DEVICES=${gpu} python tools/inference.py \
            --evaluate_pth=${pthfile} \
            --dataname=220209 --annfile=${annfile}
        done
      done
    done
    
    nframes_set='
    2 4 6 8
    '
    for nframes in ${nframes_set}
    do
      annfile_set="
      test_2_offset_${nframes}
      test_2_offset_${nframes}_half
      "
      pthfile_set="
      ./work_dirs/_220204/220204_*_${nframes}x32x1*/epoch_100.pth
      "
      # echo ${nframes}
      for pthfile in ${pthfile_set}
      do
        for annfile in ${annfile_set}
        do
          # echo ${pthfile}
          CUDA_VISIBLE_DEVICES=${gpu} python tools/inference.py \
            --evaluate_pth=${pthfile} \
            --dataname=220209 --annfile=${annfile}
        done
      done
    done
    
    nframes_set='
    2 4 6 8
    '
    for nframes in ${nframes_set}
    do
      
      annfile_set="
      test_2_offset_${nframes}
      test_2_offset_${nframes}_half
      "
      pthfile_set="
      ./work_dirs/220209_train_2_offset_${nframes}_*/epoch_100.pth
      "
      # echo ${nframes}
      for pthfile in ${pthfile_set}
      do
        for annfile in ${annfile_set}
        do
          # echo ${pthfile}
          CUDA_VISIBLE_DEVICES=${gpu} python tools/inference.py \
            --evaluate_pth=${pthfile} \
            --dataname=220209 --annfile=${annfile}
        done
      done
    done
  



#########################
    echo
  elif [ ${gpu} -eq "1" ]
  then

  # 220204_normal_timesformer_spaceOnly_2x32x1_220208T1528
    nframes_set='
    4 6 2 8 
    '
    for nframes in ${nframes_set}
    do
      annfile="offset_${nframes}"
      pthfile_set="
      ./work_dirs/220209_train_2_offset_${nframes}*/epoch_100.pth
      ./work_dirs/_220204/220204_*${nframes}x32*/epoch_100.pth
      "
      # echo ${nframes}
      for pthfile in ${pthfile_set}
      do
        # echo ${pthfile}
        CUDA_VISIBLE_DEVICES=${gpu} python tools/inference.py \
          --evaluate_pth=${pthfile} \
          --dataname=220209 --annfile=${annfile}
      done
    done
    
    
    
#########################
    echo
  elif [ ${gpu} -eq "2" ]
  then
  # 220204_normal_timesformer_spaceOnly_2x32x1_220208T1528
    nframes_set='
    8 2 6 4   
    '
    for nframes in ${nframes_set}
    do
      annfile="offset_${nframes}"
      pthfile_set="
      ./work_dirs/220209_train_2_offset_${nframes}*/epoch_100.pth
      ./work_dirs/_220204/220204_*${nframes}x32*/epoch_100.pth
      "
      # echo ${nframes}
      for pthfile in ${pthfile_set}
      do
        # echo ${pthfile}
        CUDA_VISIBLE_DEVICES=${gpu} python tools/inference.py \
          --evaluate_pth=${pthfile} \
          --dataname=220209 --annfile=${annfile}
      done
    done



#########################
    echo
  elif [ ${gpu} -eq "3" ]
  then
  
    configname_set=' 
    configs/recognition/timesformer/220209_timesformer_divST_3x1x1.py 
    '
    numframes_set='4'
    splitnumber_set='2'
    setname='trainable'
    for configname in ${configname_set}
    do
      for splitnumber in ${splitnumber_set}
      do
        for nframes in ${numframes_set}
        do
          CUDA_VISIBLE_DEVICES=${gpu} python tools/train.py ${configname} \
            --gpu-ids=${gpu} --test-last --dataname=220209 --offset \
            --splitnumber=${splitnumber} --numframes=${nframes} --setname=${setname} # --attntype=${attn} 
        done
      done
    done
    
    configname_set=' 
    configs/recognition/timesformer/220209_timesformer_jointST_3x1x1.py 
    '
    numframes_set='8'
    splitnumber_set='2'
    setname='trainable'
    for configname in ${configname_set}
    do
      for splitnumber in ${splitnumber_set}
      do
        for nframes in ${numframes_set}
        do
          CUDA_VISIBLE_DEVICES=${gpu} python tools/train.py ${configname} \
            --gpu-ids=${gpu} --test-last --dataname=220209 --offset \
            --splitnumber=${splitnumber} --numframes=${nframes} --setname=${setname} # --attntype=${attn} 
        done
      done
    done
  
    # configname_set='
    # configs/recognition/timesformer/220209_timesformer_spaceOnly_3x1x1.py 
    # '
    # numframes_set='3 5 4'
    # splitnumber_set='1'
    
    # for configname in ${configname_set}
    # do
    #   for splitnumber in ${splitnumber_set}
    #   do
    #     for nframes in ${numframes_set}
    #     do
    #       CUDA_VISIBLE_DEVICES=${gpu} python tools/train.py ${configname} \
    #         --gpu-ids=${gpu} --test-last --dataname=220209 --offset \
    #         --splitnumber=${splitnumber} --numframes=${nframes} # setname=${setname} --attntype=${attn} 
    #     done
    #   done
    # done
    
    # configname_set=' 
    # configs/recognition/timesformer/220209_timesformer_jointST_3x1x1.py 
    # configs/recognition/timesformer/220209_timesformer_divST_3x1x1.py 
    # '
    # numframes_set='4 6 2 8 3 5'
    # splitnumber_set='1'
    
    # for configname in ${configname_set}
    # do
    #   for splitnumber in ${splitnumber_set}
    #   do
    #     for nframes in ${numframes_set}
    #     do
    #       CUDA_VISIBLE_DEVICES=${gpu} python tools/train.py ${configname} \
    #         --gpu-ids=${gpu} --test-last --dataname=220209 --offset \
    #         --splitnumber=${splitnumber} --numframes=${nframes} # setname=${setname} --attntype=${attn} 
    #     done
    #   done
    # done


    # configname='configs/recognition/timesformer/220117_timesformer_spaceOnly_3x1x1_withoffset.py'
    # CUDA_VISIBLE_DEVICES=3 python tools/train.py ${configname} \
    #   --gpu-ids=3 --test-last --dataname=220209 --offset \
    #   --splitnumber=0 --numframes=2 #--attntype=space_only --setname=${setname}
    
    

#########################
    echo
  elif [ ${gpu} -eq "4" ]
  then
    gpu=1
    



#########################
    echo
  elif [ ${gpu} -eq "5" ]
  then
    
    prefix_set='
    220209
    '
    root_set="
    ./work_dirs/
    "
    dataname_set="
    220209_offset
    220209_test_2
    "
    for prefix in ${prefix_set}
    do
      # echo ${nframes}
      for root in ${root_set}
      do
        for dataname in ${dataname_set}
        do
        # echo ${pthfile}
          CUDA_VISIBLE_DEVICES=${gpu} python metric_evaluation.py \
            --prefix=${prefix} --root=${root} --dataname=${dataname}
        done 
      done
    done
    
    
    prefix_set='
    220204
    '
    root_set="
    ./work_dirs/_220204
    "
    dataname_set="
    220209_offset
    220209_test_2
    "
    for prefix in ${prefix_set}
    do
      # echo ${nframes}
      for root in ${root_set}
      do
        for dataname in ${dataname_set}
        do
        # echo ${pthfile}
          CUDA_VISIBLE_DEVICES=${gpu} python metric_evaluation.py \
            --prefix=${prefix} --root=${root} --dataname=${dataname}
        done 
      done
    done


#########################
    echo
  elif [ ${gpu} -eq "6" ]
  then
    
    for nframes in ${nframes_set}
    do
      annfile_set="
      test_2_offset_${nframes}_half
      test_2_offset_${nframes}_trainable
      test_2_offset_${nframes}
      "
      pthfile_set="
      ./work_dirs/_220204/220204_*spaceOnly_${nframes}x32*/epoch_100.pth
      "
      # echo ${nframes}
      for pthfile in ${pthfile_set}
      do
        for annfile in ${annfile_set}
        do
          # echo ${pthfile}
          echo ${nframes} ${annfile}
        done
      done
    done
#########################
    echo
fi


######## metric_evaluation
    # python metric_evaluation.py --dataname=220209_offset --root=./work_dirs/ --prefix=220209
    # python metric_evaluation.py --dataname=220209_offset --root=./work_dirs/_220204/ --prefix=220204


    # prefix_set='
    # 220204_normal_timesformer
    # '
    # root_set="
    # ./work_dirs/
    # "
    # dataname="220209"
    # for prefix in ${prefix_set}
    # do
    #   # echo ${nframes}
    #   for root in ${root_set}
    #   do
    #     # echo ${pthfile}
    #     CUDA_VISIBLE_DEVICES=${gpu} python metric_evaluation.py \
    #        --prefix=${prefix} --root=${root} --dataname=${dataname}
    #   done
    # done

######## inference
      # nframes_set='
    # 4 6 2 8 10 12 14
    # '
    # gpu=1
    # for nframes in ${nframes_set}
    # do
    #   annfile_set="
    #   test_2_offset_${nframes}_half
    #   test_2_offset_${nframes}
    #   test_2_offset_${nframes}_trainable
    #   "
    #   pthfile_set="
    #   ./work_dirs/_220204/220204_*spaceOnly_${nframes}x32*/epoch_100.pth
    #   ./work_dirs/_220204/220204_*divST_${nframes}x32*/epoch_100.pth
    #   ./work_dirs/_220204/220204_*jointST_${nframes}x32*/epoch_100.pth
    #   ./work_dirs/220209_offset_${nframes}_train_1_timesformer*spaceOnly*/epoch_100.pth
    #   ./work_dirs/220209_offset_${nframes}_train_1_timesformer*divST*/epoch_100.pth
    #   ./work_dirs/220209_offset_${nframes}_train_1_timesformer*jointST*/epoch_100.pth
    #   ./work_dirs/220209_offset_${nframes}_train_1_half*spaceOnly*/epoch_100.pth
    #   ./work_dirs/220209_offset_${nframes}_train_1_half*divST*/epoch_100.pth
    #   ./work_dirs/220209_offset_${nframes}_train_1_half*jointST*/epoch_100.pth
    #   ./work_dirs/220209_train_2_offset_${nframes}_trainable_*spaceOnly*/epoch_100.pth
    #   ./work_dirs/220209_train_2_offset_${nframes}_trainable_*divST*/epoch_100.pth
    #   ./work_dirs/220209_train_2_offset_${nframes}_trainable_*jointST*/epoch_100.pth
    #   "
    #   # echo ${nframes}
    #   for pthfile in ${pthfile_set}
    #   do
    #     for annfile in ${annfile_set}
    #     do
    #       # echo ${pthfile}
    #       CUDA_VISIBLE_DEVICES=${gpu} python tools/inference.py \
    #         --evaluate_pth=${pthfile} \
    #         --dataname=220209 --annfile=${annfile}
    #     done
    #   done
    # done


######## Train 
    
    # configs/recognition/timesformer/framenumbers/*.py
    
    # setname_set='
    # bleeding
    # inflammed
    # vascular
    # random
    # '
    # attn_set='
    # divided_space_time
    # joint_space_time
    # space_only
    # '
    # configname_set ='
    # configs/recognition/timesformer/220209_timesformer_divST_3x1x1.py
    # configs/recognition/timesformer/220209_timesformer_jointST_3x1x1.py
    # configs/recognition/timesformer/220209_timesformer_spaceOnly_3x1x1.py
    # '
    # numframes_set='2 3 4 5'
    # splitnumber_set='0'
    
    # for setname in ${setname_set}
    # do
    #   for configname in ${configname_set}
    #   do
    #     for splitnumber in ${splitnumber_set}
    #     do
    #       for attn in ${attn_set}
    #       do
    #         for nframes in ${numframes_set}
    #         do
              # CUDA_VISIBLE_DEVICES=${gpu} python tools/train.py ${configname} \
              #   --gpu-ids=${gpu} --test-last --dataname=220209 --offset \
              #   --splitnumber=${splitnumber} --numframes=${nframes} # setname=${setname} --attntype=${attn} 
    #         done
    #       done
    #     done
    #   done
    # done