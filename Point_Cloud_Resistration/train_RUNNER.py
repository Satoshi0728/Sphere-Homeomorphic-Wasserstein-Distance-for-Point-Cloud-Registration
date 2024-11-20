import subprocess
import sys
import argparse


date = "1_20"

#128_128
# best_params = {'phi_weight_decay' : 1.1025643227748046e-10, 'phi_op_lr' : 1.733751361724018e-05,'lamda_of_regularization' : 1.6259939449760638e-05, 'adam_lr' : 0.001043225875916206, 'weight_decay' : 1.4096013153858628e-08}
#128 128 W_2のバージョン
# best_params =  {'phi_weight_decay': 0.011791088504562659, 'phi_op_lr': 0.003502567667793813, 'lamda_of_regularization': 2.0424131988890235e-10, 'weight_decay': 3.2442434155367296e-08, 'adam_lr': 0.00040255529104310896}
# best_params = {'phi_weight_decay': 6.465249942503775e-10, 'phi_op_lr':2.4191084494525337e-07, 'lamda_of_regularization': 0.0001161281837394294, 'weight_decay': 2.6151152737953995e-10, 'adam_lr': 0.0024388988676612594}
# best_params = {'phi_weight_decay':9.160841619203245e-07, 'phi_op_lr': 2.9238431638589966e-08, 'lamda_of_regularization': 4.087235711414848e-06, 'weight_decay':  3.256887136839647e-15, 'adam_lr': 0.00023826630702010018}
# best_params = {'phi_weight_decay': 8.63741943045349e-14, 'phi_op_lr': 5.760179086663032e-05, 'lamda_of_regularization': 0.0004970068867943073, 'weight_decay':  3.054025227375986e-14, 'adam_lr': 0.0006873004901397991}
# best_params = {'phi_weight_decay': 3.7713842621035476e-05, 'phi_op_lr': 0.00038320053153412877, 'lamda_of_regularization': 4.279718671240612e-05, 'weight_decay':  4.504436764459625e-13, 'adam_lr': 0.000581708896266405}
# best_params = {'phi_weight_decay': 6.223888985166097e-06, 'phi_op_lr': 0.00013553767317879214, 'lamda_of_regularization': 3.260716176038329e-05, 'weight_decay':  1.0068559271286764e-15, 'adam_lr': 0.0006045420371214191}


#128_128WD
# best_pWrams = {'phi_weight_decay': 1, 'phi_op_lr':1, 'lamda_of_regularization': 1, 'weight_decay': 4.08486475244071e-06, 'adam_lr':0.0004493243680531813}
# best_pWrams = {'phi_weight_decay': 1, 'phi_op_lr':1, 'lamda_of_regularization': 1, 'weight_decay': 1.0476499757811175e-06, 'adam_lr': 0.00029030651451812217}W




# best_params = {'phi_weight_decay': 1, 'phi_op_lr':1, 'lamda_of_regularization': 1, 'weight_decay':0.00010328807285526202, 'adam_lr':  0.0013906898092881567 }
# best_params = {'phi_weight_decay': 1, 'phi_op_lr':1, 'lamda_of_regularization': 1, 'weight_decay':0.0009040119818979512, 'adam_lr':  0.00013047065535606772 }
# best_params = {'phi_weight_decay': 1, 'phi_op_lr':1, 'lamda_of_regularization': 1, 'weight_decay':7.285567245478216e-05, 'adam_lr':  0.0011400857829070026}
# best_params = {'phi_weight_decay': 1, 'phi_op_lr':1, 'lamda_of_regularization': 1, 'weight_decay':7.285567245478216e-05, 'adam_lr': 0.0011400857829070026}
best_params = {'phi_weight_decay': 1e-5, 'phi_op_lr':1e-6, 'lamda_of_regularization': 1e-5, 'weight_decay':1e-10, 'adam_lr': 1e-3}







# adam_lr: 0.0013906898092881567, weight_decay: 0.00010328807285526202

# adam_lr: 0.00013047065535606772, weight_decay: 0.0009040119818979512

# adam_lr: 0.0011400857829070026, weight_decay: 7.285567245478216e-05

# adam_lr: 0.0011400857829070026, weight_decay: 7.285567245478216e-05
# adam_lr: 0.0004493243680531813, weight_decay: 4.08486475244071e-06
# adam_lr: 0.0006531458172036332, weight_decay: 3.315831379850749e-07
# adam_lr: 0.00029030651451812217, weight_decay: 1.0476499757811175e-06
# adam_lr: 0.00025312026074939616, weight_decay: 1.442780023727746e-08

# adam_lr: 0.0003091298745142204, weight_decay: 3.1336653177723305e-08
# adam_lr: 0.00047100093476440744, weight_decay: 1.694248184082232e-08

# adam_lr: 0.0010362037367595274, weight_decay: 1.306851030191792e-07
# adam_lr: 0.00032605002665823544, weight_decay: 6.664765004772386e-07

# adam_lr: 0.00015567872073374194, weight_decay: 2.525485882248828e-06
# adam_lr: 0.0005042622839336507, weight_decay: 1.5927421838742948e-07

# adam_lr: 0.000360830057744495, weight_decay: 2.840446085259368e-06
# adam_lr: 0.00034604908171352177, weight_decay: 3.3336249677428362e-06

# adam_lr: 0.0003629161981660605, weight_decay: 8.652513133936281e-06
# adam_lr: 0.0014903337741762925, weight_decay: 1.6944271054759886e-08

# adam_lr: 0.0026927752058110636, weight_decay: 7.58253703558647e-07
# adam_lr: 0.002446687003505258, weight_decay: 7.227070489444873e-07

# adam_lr: 7.788011722817536e-05, weight_decay: 0.00013232467519577376
# adam_lr: 2.246225400666461e-05, weight_decay: 0.00025762256429836145


#256 256 CD
# best_params = {'phi_weight_decay': 1, 'phi_op_lr': 1, 'lamda_of_regularization': 1, 'adam_lr': 0.0007955052737606003, 'weight_decay' : 3.077680230834752e-05 }
#256 256 W_2
# best_params = {'phi_weight_decay': 1.0634480939504657e-10, 'phi_op_lr': 7.331385207948084e-08, 'lamda_of_regularization': 0.0004175394122140929, 'weight_decay':  7.2116782447156984e-09, 'adam_lr': 0.0010030081852267323}
# best_params = {'phi_weight_decay': 0.00032198753813338766, 'phi_op_lr': 8.828823471507909e-07, 'lamda_of_regularization': 4.812653340651418e-07, 'weight_decay':  1.188181927401176e-11, 'adam_lr': 0.0007530603825163931}



#===================================================================================================
#実行したい複数のタスク
code_name = {"code_name":
            [
            # "train__with_sinkhorn.py",
            # "train__with_chamfer_vs_cos_SW.py",
            # "train__with_chamfer_vs_cos_SW.py",
            # "train_CD.py",
            # "train_CD.py",
            # "train_CD.py",
            # "train_CD.py",
            # "train_CD.py",
            # "train_CD.py",
            # "train_CD.py",
            # "train_CD.py",
            "train_W_COS.py",
            "train_W_COS.py",
            "train_W_COS.py",
            "train_W_COS.py",
            # "train_W1_COS.py",
            # "train_W1_COS.py",
            # "train_W1_COS.py",
            # "train_W1_COS.py",
            # "train_Pseudo_W_COS.py",
            # "train_W_COS.py",
             ]}


load_model = {"load_model":
                [
            # "/home/ishikawa/Research/Point_Cloud_Registration/pcrnet_ishikawa_ver/log/9_19_5_CDW_with_128_128_0.1_noise_W2/models/best_model_snap.t7",
            # "/home/ishikawa/Research/Point_Cloud_Registration/pcrnet_ishikawa_ver/log/9_19_5_CDW_with_128_128_0.2_noise_W2/models/best_model_snap.t7",
            None,
            None,
            None,
            None,
            None,
            None,
            # "/home/ishikawa/Research/Point_Cloud_Registration/pcrnet_ishikawa_ver/log/10_9_1_CDW_with_256_256_0.00_noise_W2/models/best_model_snap.t7",
            ]}


experiment_version = {"experiment_version":
            [
            "4_WD_with_128_128_0.00_noise",
            "4_WD_with_128_128_0.02_noise",
            "4_WD_with_128_128_0.04_noise",
            "4_WD_with_128_128_0.1_noise",
            # "4_CD_with_128_128_0.00_noise",
            # "4_CD_with_128_128_0.02_noise",
            # "4_CD_with_128_128_0.04_noise",
            # "4_CD_with_128_128_0.1_noise",
             ]}
# "21_Chamfer_128_128_0.00_noise",
# "21_Chamfer_128_128_0.02_noise",
# "21_Chamfer_128_128_0.04_noise",
# "21_Chamfer_128_128_0.1_noise",
# "1_chamfer_256_256_0.00_noise",
# "1_chamfer_256_256_0.02_noise",
# "1_chamfer_256_256_0.04_noise",
# "1_chamfer_256_256_0.1_noise",


SEED = {"SEED":
            [
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            ]}

cuda_number ={"cuda_number":
            [
            1,
            2,
            3,
            0,
            # 4,
            # 5,
            # 6,
            # 7,
            # 7,
            # 2,
            # 3,
            # 4,
            # 5,
            # 6,
            # 7
            ]}

sigma = {"sigma":
            [
            0.00,
            0.02,
            0.04,
            0.1,
            # 0.2,
            # 0.04,
            # 0.04,
            # 0.04,
            # 0.06,
            # 0.00,
            # 0.02,
            # 0.04,
            # 0.00,
            # 0.00,
            ]}


source_point_num = {"source_point_num":
            [
            128,
            128,
            128,
            128,
            128,
            # 128,
            # 128,
            # 128,
            # 128,
            ]}

target_point_num = {"target_point_num":
            [
            128,
            128,
            128,
            128,
            128,
            # 256,
            # 256,
            # 256,
            # 512,
            # 512,
            ]}



phi_num_flow_layer = {"phi_num_flow_layer":
            [
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            # 20,
            # 20,
            # 20,
            # 20,
            # 20,
            # 20,
            # 20,
            # 20,
            ]}


phi_max_iter = {"phi_max_iter":
            [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            ]}

lr = {"lr":
            [
            best_params["adam_lr"],
            best_params["adam_lr"],
            best_params["adam_lr"],
            best_params["adam_lr"],
            best_params["adam_lr"],
            best_params["adam_lr"],
            # 0.0007868927008969214,
            # 0.0007868927008969214,
            # 2.7836719226615914e-07,
            # 2.7836719226615914e-07,
            # 0.0015358385939485738,
            # 0.00037714788769350284,
            # 1e-3,
            ]}

weight_decay = {"weight_decay":
            [
            best_params["weight_decay"],
            best_params["weight_decay"],
            best_params["weight_decay"],
            best_params["weight_decay"],
            best_params["weight_decay"],
            best_params["weight_decay"],
            best_params["weight_decay"],
            # 2.4121590344040404e-05,
            # 2.4121590344040404e-05,
            # 0.07505944053859534,
            # 0.07505944053859534,
            # 1.1465159883456407e-10,
            ]}

num_epoch = {"num_epoch":
            [
            2000,
            2000,
            2000,
            2000,
            2000,
            2000,
            2000,
            2000,
            ]}

batch_size = {"batch_size":
            [
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            ]}


phi_op_lr = {"phi_op_lr":
            [
            best_params["phi_op_lr"],
            best_params["phi_op_lr"],
            best_params["phi_op_lr"],
            best_params["phi_op_lr"],
            best_params["phi_op_lr"],
            best_params["phi_op_lr"],
            best_params["phi_op_lr"],
            best_params["phi_op_lr"],
            # 0.002179090658161783,
            # 0.002179090658161783,
            # 8.338961263026378e-07,
            # 8.338961263026378e-07,
            # 2.8626265291412935e-07,
            # 2.8626265291412935e-07,
            # 5.62197672416649e-05,
            ]}

phi_op_weight_decay = {"phi_op_weight_decay":
            [
            best_params["phi_weight_decay"],
            best_params["phi_weight_decay"],
            best_params["phi_weight_decay"],
            best_params["phi_weight_decay"],
            best_params["phi_weight_decay"],
            best_params["phi_weight_decay"],
            best_params["phi_weight_decay"],
            best_params["phi_weight_decay"],
            # 6.41140679107157e-05,
            # 6.41140679107157e-05,
            # 0.047105796398105054,
            # 1e-4,
            # 1e-4,
            # 1e-3,
            ]}


phi_lamda_of_regularization = {"phi_lamda_of_regularization":
            [
            best_params["lamda_of_regularization"],
            best_params["lamda_of_regularization"],
            best_params["lamda_of_regularization"],
            best_params["lamda_of_regularization"],
            best_params["lamda_of_regularization"],
            best_params["lamda_of_regularization"],
            best_params["lamda_of_regularization"],
            best_params["lamda_of_regularization"],
            # 1.6441027191571424e-05,
            # 0.000887563593057382,
            # 0.000887563593057382,
            # 0.0001347809471632583,
            # 1.5136573589053853e-05,
            ]}


flow_name = {"flow_name":
            [
            "Residual",
            "Residual",
            "Residual",
            "Residual",
            "Residual",
            "Residual",
            "Residual",
            ]}


iteration_num = {"iteration_num":
            [
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            ]}



experiment_date = {"experiment_date":
            [
             date,
             date,
             date,
             date,
             date,
             date,
             date,
             date,
             ]}

workers = {"workers":
            [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            ]}


sinkhorn_eps = {"sinkhorn_eps":
            [
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            ]}


sinkhorn_iter = {"sinkhorn_iter":
            [
            200,
            200,
            200,
            200,
            200,
            200,
            200,
            200,
            ]}


angle_range = {"angle_range":
            [
            45,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
            ]}

mean = {"mean":
            [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            ]}


translation_range = {"translation_range":
            [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            ]}



#===================================================================================================
#タスク実行
i = 0
for _ in range(len(code_name["code_name"])):
    i += 1
    # コマンドライン引数を指定する場合
    command = ["python3", code_name["code_name"][_] , "--ex_date", experiment_date["experiment_date"][_] , "--ex_ver", experiment_version["experiment_version"][_], "--cuda_num", str(cuda_number["cuda_number"][_]), "--noise_m", str(mean["mean"][_]), "--noise_s", str(sigma["sigma"][_]), "--source_p_n", str(source_point_num["source_point_num"][_]), "--target_p_n", str(target_point_num["target_point_num"][_]), "--angle_r", str(angle_range["angle_range"][_]), "--translation_r", str(translation_range["translation_range"][_]), "--num_epoch", str(num_epoch["num_epoch"][_]), "--batch_size", str(batch_size["batch_size"][_]), "--workers", str(workers["workers"][_]), "--lr", str(lr["lr"][_]), "--pcr_iteration_num", str(iteration_num["iteration_num"][_]), "--seed", str(SEED["SEED"][_]), "--load_model", str(load_model["load_model"][_]), "--sinkhorn_eps", str(sinkhorn_eps["sinkhorn_eps"][_]), "--sinkhorn_iter", str(sinkhorn_iter["sinkhorn_iter"][_]), "--phi_op_lr", str(phi_op_lr["phi_op_lr"][_]), "--phi_op_weight_decay", str(phi_op_weight_decay["phi_op_weight_decay"][_]), "--phi_num_flow_layer", str(phi_num_flow_layer["phi_num_flow_layer"][_]), "--phi_lamda_of_regularization", str(phi_lamda_of_regularization["phi_lamda_of_regularization"][_]), "--phi_max_iter", str(phi_max_iter["phi_max_iter"][_]), "--flow_name", str(flow_name["flow_name"][_]), "--weight_decay", str(weight_decay["weight_decay"][_])]
    print(command)

    if i < len(code_name["code_name"]):
        res = subprocess.Popen(command)
    else:
        res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
sys.stdout.buffer.write(res.stdout)

