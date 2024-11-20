import subprocess
import sys
import argparse
import re


DATA =[
    # "9_7_1_chamfer_128_400_with_0.00_noise",
    # "9_7_1_chamfer_128_400_with_0.02_noise",
    # "9_7_1_chamfer_128_400_with_0.04_noise",
    # "9_7_1_cos_SW_128_400_experiment_0.00_noise",
    # "9_7_1_cos_SW_128_400_experiment_0.02_noise",
    # "9_7_1_cos_SW_128_400_experiment_0.04_noise",
    # "9_5_1_cos_SW_128_128_experiment",
    # "9_5_1_cos_SW_128_128_experiment_0.02_noise",
    # "9_5_1_cos_SW_128_128_experiment_0.04_noise",
    # # "9_6_1_cos_SW_128_128_experiment_0.02_noise",
    # "9_6_1_cos_SW_128_128_experiment_0.04_noise",
    # "9_8_1_cos_SW_128_400_experiment_0.00_noise",
    # "9_8_1_cos_SW_128_400_experiment_0.02_noise",
    # "9_7_1_cos_SW_128_1024_experiment_0.00_noise_planar_early_stop",
    # "9_7_1_cos_SW_128_1024_experiment_0.02_noise_planar_early_stop",
    # "9_19_1_chamfer_128_128_0.00_noise",
    # "9_19_1_chamfer_128_128_0.2_noise",
    # "9_19_1_chamfer_128_128_0.02_noise",
    # "9_19_3_CDW_with_128_128_0.00_noise_restart",
    # "9_19_3_CDW_with_128_128_0.1_noise",
    # "9_19_3_CDW_with_128_128_0.2_noise",
    # "9_19_1_CDW_with_128_128_0.00_noise",
    # "9_19_2_CDW_with_128_128_0.00_noise_restart",
    # "9_19_1_chamfer_128_128_0.04_noise",

    "1_21_1_CD_with_128_128_0.00_noise",
    "1_21_1_CD_with_128_128_0.02_noise",
    "1_21_1_CD_with_128_128_0.04_noise",
    "1_21_1_CD_with_128_128_0.1_noise",
    # "1_20_4_WD_with_128_128_0.00_noise",
    # "1_20_4_WD_with_128_128_0.02_noise",
    # "1_20_4_WD_with_128_128_0.04_noise",
    # "1_20_4_WD_with_128_128_0.1_noise",

    # "1_20_2_CD_with_128_128_0.00_noise"
    # "1_20_2_CD_with_128_128_0.02_noise"
    # "1_20_2_CD_with_128_128_0.04_noise"
    # "1_20_2_CD_with_128_128_0.1_noise"
    
    
    
    
    
    # "10_30_2_CDW_with_128_128_0.00_noise_W2",
    # "10_30_2_CDW_with_128_128_0.1_noise_W2",
    # "10_30_2_CDW_with_128_128_0.02_noise_W2",
    # "10_30_2_CDW_with_128_128_0.04_noise_W2",
    
    
    
    
    
    # "10_30_3_CDW_with_128_128_0.00_noise_W2",
    # "10_30_3_CDW_with_128_128_0.1_noise_W2",
    # "10_30_3_CDW_with_128_128_0.02_noise_W2",
    # "10_30_3_CDW_with_128_128_0.04_noise_W2",
    
    
    
    
    # "10_30_4_CDW_with_128_128_0.00_noise_W1",
    # "10_30_4_CDW_with_128_128_0.1_noise_W1",
    # "10_30_4_CDW_with_128_128_0.02_noise_W1",
    # "10_30_4_CDW_with_128_128_0.04_noise_W1",




    # "10_30_5_CDW_with_128_128_0.00_noise_W1",
    # "10_30_5_CDW_with_128_128_0.1_noise_W1",
    # "10_30_5_CDW_with_128_128_0.02_noise_W1",
    # "10_30_5_CDW_with_128_128_0.04_noise_W1",




    # "10_30_6_CDW_with_128_128_0.00_noise_W2",
    # "10_30_6_CDW_with_128_128_0.1_noise_W2",
    # "10_30_6_CDW_with_128_128_0.02_noise_W2",
    # "10_30_6_CDW_with_128_128_0.04_noise_W2",
    
    # "10_30_7_CDW_with_128_128_0.00_noise_W1",
    # "10_30_7_CDW_with_128_128_0.1_noise_W1",
    # "10_30_7_CDW_with_128_128_0.02_noise_W1",
    # "10_30_7_CDW_with_128_128_0.04_noise_W1",





    # "10_31_1_Chamfer_128_128_0.00_noise",
    # "10_31_1_Chamfer_128_128_0.1_noise",
    # "10_31_1_Chamfer_128_128_0.02_noise",
    # "10_31_1_Chamfer_128_128_0.04_noise",
    
    
    
    
    


    # "10_31_2_Chamfer_128_128_0.00_noise",
    # "10_31_2_Chamfer_128_128_0.1_noise",
    # "10_31_2_Chamfer_128_128_0.02_noise",
    # "10_31_2_Chamfer_128_128_0.04_noise",


    # "10_31_3_Chamfer_128_128_0.00_noise",
    # "10_31_3_Chamfer_128_128_0.1_noise",
    # "10_31_3_Chamfer_128_128_0.02_noise",
    # "10_31_3_Chamfer_128_128_0.04_noise",


    # "10_31_4_Chamfer_128_128_0.00_noise",
    # "10_31_4_Chamfer_128_128_0.1_noise",
    # "10_31_4_Chamfer_128_128_0.02_noise",
    # "10_31_4_Chamfer_128_128_0.04_noise",



    # "10_31_5_Chamfer_128_128_0.00_noise",
    # "10_31_5_Chamfer_128_128_0.1_noise",
    # "10_31_5_Chamfer_128_128_0.02_noise",
    # "10_31_5_Chamfer_128_128_0.04_noise",




    # "10_31_6_Chamfer_128_128_0.00_noise",
    # "10_31_6_Chamfer_128_128_0.1_noise",
    # "10_31_6_Chamfer_128_128_0.02_noise",
    # "10_31_6_Chamfer_128_128_0.04_noise",
    




    # "10_31_7_Chamfer_128_128_0.00_noise",
    # "10_31_7_Chamfer_128_128_0.1_noise",
    # "10_31_7_Chamfer_128_128_0.02_noise",
    # "10_31_7_Chamfer_128_128_0.04_noise",
    
    


    # "10_31_8_Chamfer_128_128_0.00_noise",
    # "10_31_8_Chamfer_128_128_0.1_noise",
    # "10_31_8_Chamfer_128_128_0.02_noise",
    # "10_31_8_Chamfer_128_128_0.04_noise",



    # "10_31_9_Chamfer_128_128_0.00_noise",
    # "10_31_9_Chamfer_128_128_0.1_noise",
    # "10_31_9_Chamfer_128_128_0.02_noise",
    # "10_31_9_Chamfer_128_128_0.04_noise",


    # "10_31_10_Chamfer_128_128_0.00_noise",
    # "10_31_10_Chamfer_128_128_0.1_noise",
    # "10_31_10_Chamfer_128_128_0.02_noise",
    # "10_31_10_Chamfer_128_128_0.04_noise",


    # "10_31_11_Chamfer_128_128_0.00_noise",
    # "10_31_11_Chamfer_128_128_0.1_noise",
    # "10_31_11_Chamfer_128_128_0.02_noise",
    # "10_31_11_Chamfer_128_128_0.04_noise",


    # "10_31_12_Chamfer_128_128_0.00_noise",
    # "10_31_12_Chamfer_128_128_0.1_noise",
    # "10_31_12_Chamfer_128_128_0.02_noise",
    # "10_31_12_Chamfer_128_128_0.04_noise",
    
    
    
    
    # "10_31_13_Chamfer_128_128_0.00_noise",
    # "10_31_13_Chamfer_128_128_0.1_noise",
    # "10_31_13_Chamfer_128_128_0.02_noise",
    # "10_31_13_Chamfer_128_128_0.04_noise",



    # "10_31_14_Chamfer_128_128_0.00_noise",
    # "10_31_14_Chamfer_128_128_0.1_noise",
    # "10_31_14_Chamfer_128_128_0.02_noise",
    # "10_31_14_Chamfer_128_128_0.04_noise",

    # "10_31_15_Chamfer_128_128_0.00_noise",
    # "10_31_15_Chamfer_128_128_0.1_noise",
    # "10_31_15_Chamfer_128_128_0.02_noise",
    # "10_31_15_Chamfer_128_128_0.04_noise",

    # "10_31_16_Chamfer_128_128_0.00_noise",
    # "10_31_16_Chamfer_128_128_0.1_noise",
    # "10_31_16_Chamfer_128_128_0.02_noise",
    # "10_31_16_Chamfer_128_128_0.04_noise",


    # "10_31_17_Chamfer_128_128_0.00_noise",
    # "10_31_17_Chamfer_128_128_0.1_noise",
    # "10_31_17_Chamfer_128_128_0.02_noise",
    # "10_31_17_Chamfer_128_128_0.04_noise",

    # "10_31_18_Chamfer_128_128_0.00_noise",
    # "10_31_18_Chamfer_128_128_0.1_noise",
    # "10_31_18_Chamfer_128_128_0.02_noise",
    # "10_31_18_Chamfer_128_128_0.04_noise",


    # "10_31_19_Chamfer_128_128_0.00_noise",
    # "10_31_19_Chamfer_128_128_0.1_noise",
    # "10_31_19_Chamfer_128_128_0.02_noise",
    # "10_31_19_Chamfer_128_128_0.04_noise",

    # "10_31_20_Chamfer_128_128_0.00_noise",
    # "10_31_20_Chamfer_128_128_0.1_noise",
    # "10_31_20_Chamfer_128_128_0.02_noise",
    # "10_31_20_Chamfer_128_128_0.04_noise",
    ]











test_epoch_num = "100"
# ===================================================================================================
# 関数群
def find_lines_with_keyword_and_extract_numbers(file_path, keyword):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    matching_lines = [line.strip() for line in lines if keyword in line]
    return extract_numbers(matching_lines[0])

def extract_numbers(input_string):
    pattern = r"[-+]?\d*\.\d+|\d+"
    numbers = re.findall(pattern, input_string)
    return ''.join(numbers)

# ===================================================================================================
source_point_num = {"source_point_num":[]}
target_point_num = {"target_point_num":[]}
experiment_version = {"experiment_version":[]}
sigma = {"sigma":[]}
mean = {"mean":[]}
cuda_number = {"cuda_number":[]}
code_name = {"code_name":[]}
test_epoch = {"test_epoch":[]}
pretrained = {"pretrained":[]}
angle_range = {"angle_range":[]}
translation_range = {"translation_range":[]}
batch_size = {"batch_size":[]}
workers = {"workers":[]}
iteration_num = {"iteration_num":[]}
SEED = {"SEED":[]}


for i in range(len(DATA)):
    pretrained["pretrained"].append("log/" + DATA[i] + "/models/best_model.t7")
    # pretrained["pretrained"].append("log/" + DATA[i] + "/models/model.t7")
    log_dir = "log/" + DATA[i] + "/run.log"
    cuda_number["cuda_number"].append(find_lines_with_keyword_and_extract_numbers(log_dir, "cuda_num"))
    # cuda_number["cuda_number"].append(i + 1)
    source_point_num["source_point_num"].append(find_lines_with_keyword_and_extract_numbers(log_dir, "source_point_num"))
    target_point_num["target_point_num"].append(find_lines_with_keyword_and_extract_numbers(log_dir, "target_point_num"))
    sigma["sigma"].append(find_lines_with_keyword_and_extract_numbers(log_dir, "sigma"))
    experiment_version["experiment_version"].append(DATA[i])
    code_name["code_name"].append("test_ERROR.py")
    test_epoch["test_epoch"].append(test_epoch_num)
    mean["mean"].append(find_lines_with_keyword_and_extract_numbers(log_dir, "mean"))
    angle_range["angle_range"].append(find_lines_with_keyword_and_extract_numbers(log_dir, "angle_range"))
    translation_range["translation_range"].append(find_lines_with_keyword_and_extract_numbers(log_dir, "translation_range"))
    batch_size["batch_size"].append(find_lines_with_keyword_and_extract_numbers(log_dir, "batch_size"))
    workers["workers"].append(2)
    iteration_num["iteration_num"].append(find_lines_with_keyword_and_extract_numbers(log_dir, "iteration_num"))
    SEED["SEED"].append(find_lines_with_keyword_and_extract_numbers(log_dir, "SEED"))



#===================================================================================================
#タスク実行
i = 0
for _ in range(len(pretrained["pretrained"])):
    i += 1
    command = ["python3", code_name["code_name"][_] , "--ex_ver", experiment_version["experiment_version"][_], "--cuda_num", str(cuda_number["cuda_number"][_]), "--noise_m", str(mean["mean"][_]), "--noise_s", str(sigma["sigma"][_]), "--source_p_n", str(source_point_num["source_point_num"][_]), "--target_p_n", str(target_point_num["target_point_num"][_]), "--angle_r", str(angle_range["angle_range"][_]), "--translation_r", str(translation_range["translation_range"][_]), "--batch_size", str(batch_size["batch_size"][_]), "--workers", str(workers["workers"][_]), "--pcr_iteration_num", str(iteration_num["iteration_num"][_]), "--seed", str(SEED["SEED"][_]), "--pretrained", str(pretrained["pretrained"][_]), "--test_epoch", str(test_epoch["test_epoch"][_])]
    print(command)
    if i < len(pretrained["pretrained"]):
        res = subprocess.Popen(command)
    else:
        res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
sys.stdout.buffer.write(res.stdout)


