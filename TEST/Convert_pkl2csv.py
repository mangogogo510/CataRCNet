import pickle
import shutil
import os
import argparse
# yilin add csv
import csv

with open('../gen_datasets/cataracts_test.pkl', 'rb') as f:
    test_paths_labels = pickle.load(f)

parser = argparse.ArgumentParser(description='lstm testing')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-n', '--name', default = './test_results_pkl/NLB-resnest_test_6937_crop_0.pkl',type=str, help='name of pred')


args = parser.parse_args()
sequence_length = args.seq



## 
############################# 1 ####################
# create the dir firstly
pred_filepath = './test_results_csv/LFB/pred-step-resnet-10'
gt_filepath = './test_results_csv/LFB/gt-step-resnet-10'
# the prediction pkl results name
pred_name = './test_results_pkl/NLB-resnet-ec_test_6942_crop_0.pkl'  # NLB-resnet_test_7110_crop_0.pkl
########################################################

# pred_filepath = './test_results/NLB-resnet/add_train24_newtest/pred-phase'
# gt_filepath = './test_results/NLB-resnet/add_train24_newtest/gt-phase'
# pred_name = './test_results_pkl/NLB-resnet_test_7080_crop_0.pkl'


#  ./latest_model_2_test_6714_crop_1.pkl
# non-local_test_4998_crop_0.pkl 0.69 f1
# non-local_test_6953_crop_0.pkl  0.7651
# NLB-resnet_test_7078_crop_0.pkl 0.7654 using +train24



# finalpkl

# './test_results_pkl/NLB-resnet_test_7598_crop_0.pkl'



with open(pred_name, 'rb') as f:
    ori_preds = pickle.load(f)

   ###########################   2    ########################
num_video = 25 # 40   #  gai
num_labels = 0
for i in range(25): #(40,80)
    num_labels += len(test_paths_labels[i])

num_preds = len(ori_preds)

print('num of labels  : {:6d}'.format(num_labels))
print("num ori preds  : {:6d}".format(num_preds))
print("labels example : ", test_paths_labels[0][0][1])
print("preds example  : ", ori_preds[0])

if num_labels == (num_preds + (sequence_length - 1) * num_video):


    preds_all = []
    label_all = []
    count = 0
    ###########################   3   ########################
    for i in range(25): #(40,80)

        
        filename = pred_filepath +'/video' + str(1 + i) + '-phase.csv'  #txt
        gt_filename = gt_filepath+'/video' + str(1 + i) + '-phase.csv'  #txt
        
        f = open(filename, 'w') # encoding='utf-8'
        csv_writer_pred = csv.writer(f)
        

        f2 = open(gt_filename, 'w')
        csv_writer_gt = csv.writer(f2)
        #f2.write('Frame Phase')
        #f2.write('\n')

        preds_each = []
        for j in range(count, count + len(test_paths_labels[i]) - (sequence_length - 1)):
            if j == count:
                # 
                for k in range(sequence_length - 1):
                    preds_each.append(0)
                    preds_all.append(0)
            preds_each.append(ori_preds[j])
            preds_all.append(ori_preds[j])
            
        ###########################   4   ########################    
        samplerate = 30 # sample rate 30-> 1fps
        
        for k in range(len(preds_each)):

            
            csv_writer_pred.writerow([str(samplerate * k),str(int(preds_each[k]))])
            csv_writer_gt.writerow([str(samplerate * k),str(int(test_paths_labels[i][k][1]))])
           
            label_all.append(test_paths_labels[i][k][1])

        f.close()
        f2.close()
        count += len(test_paths_labels[i]) - (sequence_length - 1)
    test_corrects = 0

    print('num of labels       : {:6d}'.format(len(label_all)))
    print('result of all preds  : {:6d}'.format(len(preds_all))) # result of all preds

    for i in range(num_labels):
# TODO
        if int(label_all[i]) == int(preds_all[i]):
            test_corrects += 1

    print('right number preds  : {:6d}'.format(test_corrects))
    print('test accuracy       : {:.4f}'.format(test_corrects / num_labels))
else:
    print('number error, please check')
