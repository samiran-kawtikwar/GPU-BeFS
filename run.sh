#!/usr/bin/env bash

exec_home=~/GPU-BeFS
# baseline_home=~/RCAP-Project/baseline/cpu-bnb

size=5
range=(10)
seeds=(1000 2000 3000 4000 5000)
# seeds=(45345)
TIMEOUT=10000
echo "*** Starting the RUN ****"
# for n=size upto 35

for ((n=size; n<=35; n+=1))
do
    for r in ${range[@]}
    do
        for s in ${seeds[@]}
        do
            echo "n=$n, k=$n running ..."
            /usr/bin/timeout $TIMEOUT ${exec_home}/build/main.exe -n $n -k $n -s $s -f $r -d 0 >> rtx-5090.log            
            # ${baseline_home}/build/main.exe -n $n -k $n -s $s -f $r -d 0 >> cpu_run.log
            out=($?)
            if [ $out -eq 0 ]
            then
                echo "n=$n, k=$n, s=$s success 🎉🎉"
            else
                #print Out of memory to the log file
                echo "OOM $out" >> gpu_run_fixed.log
                echo "n=$n, k=$n, s=$s failed !🚨🚨"
                # print exit code
                echo "exit code: $out"
            fi
        done
    done
done

echo "*** Script Finished ****"