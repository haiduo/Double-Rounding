import numpy as np
import matplotlib.pyplot as plt

log_path = "/root/code/Multi_Precision/results/log_resnet18q.txt" 

bit32_loss = []
bit8_loss = []
bit7_loss = []
bit6_loss = []
bit5_loss = []
bit4_loss = []
bit3_loss = []
bit2_loss = []
bit1_loss = []

bit32_acc = []
bit8_acc = []
bit7_acc = []
bit6_acc = []
bit5_acc = []
bit4_acc = []
bit3_acc = []
bit2_acc = []
bit1_acc = []

with open(log_path) as f:
    for lines_f in f:
        if "topk_bit" in lines_f:
            if "val prec" in lines_f:
                lines = lines_f.split("val prec:")[1].split(",")
                for line in lines:
                    if '1_1' in line:
                        acc1 = float(line.split(':')[1])
                        bit1_acc.append(acc1)
                    elif '1_2' in line:
                        acc1 = float(line.split(':')[1])
                        bit2_acc.append(acc1)
                    elif '1_3:' in line:
                        acc1 = float(line.split(':')[1])
                        bit3_acc.append(acc1)
                    elif '1_4' in line:
                        acc1 = float(line.split(':')[1])
                        bit4_acc.append(acc1)
                    elif '1_5' in line:
                        acc1 = float(line.split(':')[1])
                        bit5_acc.append(acc1)
                    elif '1_6' in line:
                        acc1 = float(line.split(':')[1])
                        bit6_acc.append(acc1)
                    elif '1_7' in line:
                        acc1 = float(line.split(':')[1])
                        bit7_acc.append(acc1)
                    elif '1_8' in line:
                        acc1 = float(line.split(':')[1])
                        bit8_acc.append(acc1)
                    elif '1_32' in line:
                        acc1 = float(line.split(':')[1])
                        bit32_acc.append(acc1)
            else:
                lines = lines_f.split("train loss:")[1].split(",")
                for line in lines:
                    if '1_1' in line:
                        acc1 = float(line.split(':')[1])
                        bit1_loss.append(acc1)
                    elif '1_2' in line:
                        acc1 = float(line.split(':')[1])
                        bit2_loss.append(acc1)
                    elif '1_3' in line:
                        acc1 = float(line.split(':')[1])
                        bit3_loss.append(acc1)
                    elif '1_4' in line:
                        acc1 = float(line.split(':')[1])
                        bit4_loss.append(acc1)
                    elif '1_5' in line:
                        acc1 = float(line.split(':')[1])
                        bit5_loss.append(acc1)
                    elif '1_6' in line:
                        acc1 = float(line.split(':')[1])
                        bit6_loss.append(acc1)
                    elif '1_7' in line:
                        acc1 = float(line.split(':')[1])
                        bit7_loss.append(acc1)
                    elif '1_8' in line:
                        acc1 = float(line.split(':')[1])
                        bit8_loss.append(acc1)
                    elif '1_32' in line:
                        acc1 = float(line.split(':')[1])
                        bit32_loss.append(acc1)

mean_value_index = np.argmax(bit32_acc)

print(f'float_avgacc:{bit32_acc[mean_value_index]:3.2f} max_acc:{max(bit32_acc):3.2f}')

epoch = len(bit32_acc)

plt.figure(num= 'Acc1 of cifar10', figsize=(8,6), dpi=300)
ax = plt.subplot(1,1,1)

tilte_str = 'mobileV2_float'
plt.title(f"{tilte_str}")

epoch_list = [i+1 for i in range(epoch)]

plt.plot(epoch_list, bit32_acc, 'r', label='32_acc', linewidth=1)
# plt.plot(epoch_list, bit8_acc, 'y', label='8_acc', linewidth=1)
# plt.plot(epoch_list, bit8_acc, 'g', label='8_acc', linewidth=1)
# plt.plot(epoch_list, bit8_acc, 'c', label='8_acc', linewidth=1)
# plt.plot(epoch_list, bit8_acc, 'b', label='8_acc', linewidth=1)
# plt.plot(epoch_list, bit8_acc, 'gold', label='8_acc', linewidth=1)
# plt.plot(epoch_list, bit8_acc, 'm', label='8_acc', linewidth=1)
# plt.plot(epoch_list, bit8_acc, 'pink', label='8_acc', linewidth=1)

min_value_acc = min(bit32_acc)
max_value_acc = max(bit32_acc)

min_value = min(min_value_acc, max_value_acc)
max_value = max(min_value_acc, max_value_acc)

import matplotlib
matplotlib.rc('font', size=14)

plt.xlim(min(epoch_list)*0.1, max(epoch_list)*1.01)
plt.ylim(min_value*0.95, max_value*1.1)

plt.xlabel(u'Epoch')
plt.ylabel(u'Acc1(%)')
plt.legend()

figPath = f'/root/code/Multi_Precision/results/plots/{tilte_str}.png'
plt.savefig(figPath, dpi=300)

# plt.show()
plt.close()
