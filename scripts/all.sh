ma_type=ema
alpha=0.3
beta=0.3

current_time=$(date +"%m%d_%H%M")

seq_len_meta=96
testname=Newfusion_cnn_mamba
pred_lens_meta=(96 192 336 720) #96 192 336 720
epochs=(40 40 40 40) #20 20 20 20
patch_lens=(16)
strides_meta=(8)
label_len_mata=48
lradj='type3' #cosine

# === 定义每个数据集的专属参数（使用关联数组）===
declare -A batch_sizes_per_dataset
batch_sizes_per_dataset["ETTh1"]=128
batch_sizes_per_dataset["ETTh2"]=32
batch_sizes_per_dataset["ETTm1"]=256
batch_sizes_per_dataset["ETTm2"]=256
batch_sizes_per_dataset["weather"]=128
batch_sizes_per_dataset["traffic"]=16
batch_sizes_per_dataset["electricity"]=32
batch_sizes_per_dataset["exchange_rate"]=32
batch_sizes_per_dataset["national_illness"]=32
batch_sizes_per_dataset["solar"]=96
batch_sizes_per_dataset["PEMS03"]=32
batch_sizes_per_dataset["PEMS04"]=32
batch_sizes_per_dataset["PEMS07"]=32
batch_sizes_per_dataset["PEMS08"]=32

declare -A learning_rates_per_dataset
learning_rates_per_dataset["ETTh1"]=0.0005
learning_rates_per_dataset["ETTh2"]=0.0001
learning_rates_per_dataset["ETTm1"]=0.0001
learning_rates_per_dataset["ETTm2"]=0.0005
learning_rates_per_dataset["weather"]=0.0001
learning_rates_per_dataset["traffic"]=0.0003
learning_rates_per_dataset["electricity"]=0.0005
learning_rates_per_dataset["exchange_rate"]=0.0001
learning_rates_per_dataset["national_illness"]=0.0005
learning_rates_per_dataset["solar"]=0.0001
learning_rates_per_dataset["PEMS03"]=0.0003
learning_rates_per_dataset["PEMS04"]=0.0003
learning_rates_per_dataset["PEMS07"]=0.0003
learning_rates_per_dataset["PEMS08"]=0.0003

declare -A enc_in_per_dataset
enc_in_per_dataset["ETTh1"]=7
enc_in_per_dataset["ETTh2"]=7  
enc_in_per_dataset["ETTm1"]=7
enc_in_per_dataset["ETTm2"]=7 
enc_in_per_dataset["weather"]=21
enc_in_per_dataset["traffic"]=862
enc_in_per_dataset["electricity"]=321
enc_in_per_dataset["exchange_rate"]=8
enc_in_per_dataset["national_illness"]=7
enc_in_per_dataset["solar"]=137
enc_in_per_dataset["PEMS03"]=358
enc_in_per_dataset["PEMS04"]=307
enc_in_per_dataset["PEMS07"]=883
enc_in_per_dataset["PEMS08"]=170

declare -A data_per_dataset 
data_per_dataset["ETTh1"]=ETTh1
data_per_dataset["ETTh2"]=ETTh2
data_per_dataset["ETTm1"]=ETTm1
data_per_dataset["ETTm2"]=ETTm2
data_per_dataset["weather"]=custom
data_per_dataset["traffic"]=custom
data_per_dataset["electricity"]=custom
data_per_dataset["exchange_rate"]=custom
data_per_dataset["national_illness"]=custom
data_per_dataset["solar"]=Solar
data_per_dataset["PEMS03"]=PEMS
data_per_dataset["PEMS04"]=PEMS
data_per_dataset["PEMS07"]=PEMS
data_per_dataset["PEMS08"]=PEMS

declare -A data_path_per_dataset
data_path_per_dataset["ETTh1"]=ETTh1.csv
data_path_per_dataset["ETTh2"]=ETTh2.csv
data_path_per_dataset["ETTm1"]=ETTm1.csv
data_path_per_dataset["ETTm2"]=ETTm2.csv
data_path_per_dataset["weather"]=weather.csv
data_path_per_dataset["traffic"]=traffic.csv
data_path_per_dataset["electricity"]=electricity.csv
data_path_per_dataset["exchange_rate"]=exchange_rate.csv
data_path_per_dataset["national_illness"]=national_illness.csv
data_path_per_dataset["solar"]=solar.txt
data_path_per_dataset["PEMS03"]=PEMS03.npz
data_path_per_dataset["PEMS04"]=PEMS04.npz
data_path_per_dataset["PEMS07"]=PEMS07.npz
data_path_per_dataset["PEMS08"]=PEMS08.npz

declare -A d_state_per_dataset
d_state_per_dataset["ETTh1"]=2
d_state_per_dataset["ETTh2"]=2
d_state_per_dataset["ETTm1"]=2
d_state_per_dataset["ETTm2"]=2
d_state_per_dataset["weather"]=2
d_state_per_dataset["traffic"]=32
d_state_per_dataset["electricity"]=32
d_state_per_dataset["exchange_rate"]=32
d_state_per_dataset["national_illness"]=32
d_state_per_dataset["solar"]=32
d_state_per_dataset["PEMS03"]=32
d_state_per_dataset["PEMS04"]=32
d_state_per_dataset["PEMS07"]=32
d_state_per_dataset["PEMS08"]=32

declare -A d_model_per_dataset
d_model_per_dataset["ETTh1"]=256
d_model_per_dataset["ETTh2"]=256
d_model_per_dataset["ETTm1"]=128
d_model_per_dataset["ETTm2"]=128
d_model_per_dataset["weather"]=512
d_model_per_dataset["traffic"]=256
d_model_per_dataset["electricity"]=256
d_model_per_dataset["exchange_rate"]=128
d_model_per_dataset["national_illness"]=256
d_model_per_dataset["solar"]=256
d_model_per_dataset["PEMS03"]=512
d_model_per_dataset["PEMS04"]=512
d_model_per_dataset["PEMS07"]=512
d_model_per_dataset["PEMS08"]=512

declare -A d_core_per_dataset
d_core_per_dataset["ETTh1"]=128
d_core_per_dataset["ETTh2"]=128
d_core_per_dataset["ETTm1"]=128
d_core_per_dataset["ETTm2"]=128
d_core_per_dataset["weather"]=128
d_core_per_dataset["traffic"]=128
d_core_per_dataset["electricity"]=128
d_core_per_dataset["exchange_rate"]=128
d_core_per_dataset["national_illness"]=128
d_core_per_dataset["solar"]=128
d_core_per_dataset["PEMS03"]=512
d_core_per_dataset["PEMS04"]=512
d_core_per_dataset["PEMS07"]=512
d_core_per_dataset["PEMS08"]=512

datasets=("solar" "traffic" "PEMS03" "PEMS04" "PEMS07")
#("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather"  "exchange_rate" "national_illness" "electricity" "solar" "traffic" "PEMS03" "PEMS04" "PEMS07" "PEMS08")
models=("SGF") #"PaDuM" "SGF"

# === 为每个模型 + 测试名 + ma_type + 数据集 创建独立目录 ===
for model_name in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        log_dir="./logs/$model_name/$testname/$dataset"
        if [ ! -d "$log_dir" ]; then
            mkdir -p "$log_dir"
            echo "Created directory: $log_dir"
        fi
    done
done

# === 主循环 ===
for dataset in "${datasets[@]}"; do
    # 获取该数据集的专属参数
    batch_size=${batch_sizes_per_dataset[$dataset]}
    learning_rate=${learning_rates_per_dataset[$dataset]}
    enc_in=${enc_in_per_dataset[$dataset]}
    data=${data_per_dataset[$dataset]}
    data_path=${data_path_per_dataset[$dataset]}
    d_state=${d_state_per_dataset[$dataset]}
    d_model=${d_model_per_dataset[$dataset]}
    d_core=${d_core_per_dataset[$dataset]}
    # 数据集特殊处理：national_illness 与 PEMS 系列
    if [ "$dataset" = "national_illness" ]; then
        seq_len=36
        pred_lens=(24 36 48 60)
        strides=(3)
        label_len=18
    elif [ "$dataset" = "PEMS03" ] || [ "$dataset" = "PEMS04" ] || [ "$dataset" = "PEMS07" ] || [ "$dataset" = "PEMS08" ]; then
        # PEMS 数据集使用更短的预测步长集合
        seq_len=$seq_len_meta
        pred_lens=(12 24 48 96)
        strides=("${strides_meta[@]}")
        label_len=$label_len_mata
    else
        seq_len=$seq_len_meta
        pred_lens=("${pred_lens_meta[@]}")
        strides=("${strides_meta[@]}")
        label_len=$label_len_mata
    fi
    for model_name in "${models[@]}"; do
        for i in "${!pred_lens[@]}"; do
            for j in "${!patch_lens[@]}"; do
                for n in "${!strides[@]}"; do
                    pred_len=${pred_lens[$i]}
                    train_epoch=${epochs[$i]}
                    patch_len=${patch_lens[$j]}
                    stride=${strides[$n]}


                    # 日志文件名
                    log_file="logs/$model_name/$testname/$dataset/${model_name}_${dataset}_${seq_len}_${pred_len}_${current_time}.log"

                    echo "Running: $model_name on $dataset with lr=$learning_rate, bs=$batch_size, pred_len=$pred_len"
                    
                    python -u run.py \
                        --Exp exp_main \
                        --revin 1 \
                        --is_training 1 \
                        --donot_save 0 \
                        --root_path ./dataset/ \
                        --data_path "$data_path" \
                        --dataset_id "${dataset}" \
                        --model "$model_name" \
                        --data "$data" \
                        --features M \
                        --seq_len "$seq_len" \
                        --pred_len "$pred_len" \
                        --patch_len "$patch_len" \
                        --d_state "$d_state" \
                        --d_model "$d_model" \
                        --d_core "$d_core" \
                        --stride "$stride" \
                        --label_len "$label_len" \
                        --enc_in "$enc_in" \
                        --des "$testname" \
                        --itr 1 \
                        --batch_size "$batch_size" \
                        --learning_rate "$learning_rate" \
                        --lradj "$lradj" \
                        --train_epochs "$train_epoch" \
                        --ma_type "$ma_type" \
                        --alpha "$alpha" \
                        --beta "$beta" > "$log_file" 2>&1
                done
            done
        done
    done
done

for model_name in "${models[@]}"; do
# === 所有任务执行完毕后，生成并保存表格 ===
echo "All training tasks are completed. Generating the results table..."

log_base_dir="./logs/$model_name/$testname"

table_log_dir=$(find "$log_base_dir" -maxdepth 1 -type d -print | head -n 1)

# 生成CSV文件的路径
# 例如：./logs/mymodel/fulltest/ema/results_table.csv
output_csv_path="$log_base_dir/${model_name}_${testname}.csv"

python generate_table.py --log_dir "$table_log_dir" --output_csv_path "$output_csv_path"

echo "Results table has been generated and saved."
done
