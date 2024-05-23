import matplotlib.pyplot as plt
from pathlib import Path

log_file_path = Path("/root/code/Super_net_solve/results/log_resnet20_cal_HMT.txt")
name_file = log_file_path.name

# List used to store data
average_bits = []
val_prec1_values = []

with open(log_file_path, 'r') as file:
    for line in file:
        if "Current candidate_bits" in line:
            bits_string = line.split('[')[-1].split(']')[0]
            bits = list(map(int, bits_string.split(', ')))
            average = sum(bits) / len(bits)
            average_bits.append(average)
        
        if "val prec1" in line:
            val_prec1 = float(line.split('val prec1: ')[-1].split(',')[0])
            val_prec1_values.append(val_prec1)


plt.figure(figsize=(10, 5))
plt.scatter(average_bits, val_prec1_values, color='red', alpha=0.5, label='Weights')
plt.title('Average Bit Values vs. Val Precision 1', fontdict={'fontsize':25, 'color':'black', 'family':'Times New Roman', 'weight':'normal'})
plt.xlabel('Average Bits', fontdict={'fontsize':20, 'color':'black', 'family':'Times New Roman', 'weight':'normal'})
plt.ylabel('Val Precision 1 (%)', fontdict={'fontsize':20, 'color':'black', 'family':'Times New Roman', 'weight':'normal'})
plt.grid(True)
plt.legend(loc='lower right', fontsize='x-large')
plt.tight_layout()

plt.savefig(Path("code/Super_net_solve/results/plots")/ name_file.replace("txt", "png"), dpi=300)
# plt.show()
