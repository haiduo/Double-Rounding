import numpy as np
import matplotlib.pyplot as plt

log_path = "/root/code/Multi_bits/results/log_resnet18q.txt"

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
        if "****val" in lines_f:
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
        elif "5_average:" in lines_f:
            lines = lines_f.split("5_average:")[0].split(",")
            for line in lines:
                if '5_1' in line:
                    acc1 = float(line.split(':')[1])
                    bit1_loss.append(acc1)
                elif '5_2' in line:
                    acc1 = float(line.split(':')[1])
                    bit2_loss.append(acc1)
                elif '5_3' in line:
                    acc1 = float(line.split(':')[1])
                    bit3_loss.append(acc1)
                elif '5_4' in line:
                    acc1 = float(line.split(':')[1])
                    bit4_loss.append(acc1)
                elif '5_5' in line:
                    acc1 = float(line.split(':')[1])
                    bit5_loss.append(acc1)
                elif '5_6' in line:
                    acc1 = float(line.split(':')[1])
                    bit6_loss.append(acc1)
                elif '5_7' in line:
                    acc1 = float(line.split(':')[1])
                    bit7_loss.append(acc1)
                elif '5_8' in line:
                    acc1 = float(line.split(':')[1])
                    bit8_loss.append(acc1)
                elif '5_32' in line:
                    acc1 = float(line.split(':')[1])
                    bit32_loss.append(acc1)

mean_value_index = np.argmax(np.mean([bit8_acc, bit4_acc], axis=0))

print(f'8_avgacc:{bit8_acc[mean_value_index]:3.2f} max_acc:{max(bit8_acc):3.2f}')
print(f'4_avgacc:{bit4_acc[mean_value_index]:3.2f} max_acc:{max(bit4_acc):3.2f}')

epoch = len(bit4_acc)

plt.figure(num= 'Acc1 of cifar10', figsize=(8,6), dpi=300)
ax = plt.subplot(1,1,1)
tilte_str = "resnet20_2bit_basas"
plt.title(f"{tilte_str}")

epoch_list = [i+1 for i in range(epoch)]

plt.plot(epoch_list, bit8_acc, 'r', label='8_acc', linewidth=1)
plt.plot(epoch_list, bit4_acc, 'g', label='4_acc', linewidth=1)


min_value_acc = min([min(bit8_acc), min(bit8_acc)])
max_value_acc = max([max(bit4_acc), max(bit4_acc)])

min_value = min(min_value_acc, max_value_acc)
max_value = max(min_value_acc, max_value_acc)

import matplotlib
matplotlib.rc('font', size=14)

plt.xlim(min(epoch_list)*0.1, max(epoch_list)*1.01)
plt.ylim(min_value*0.95, max_value*1.1)

plt.xlabel(u'Epoch')
plt.ylabel(u'Acc1(%)')
plt.legend()

figPath = f''
plt.savefig(figPath, dpi=300)

# plt.show()
plt.close()
