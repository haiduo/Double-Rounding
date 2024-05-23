import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path


file_path = Path("/root/code/Multi_Precision/results/log_resnet18q_8642bits_gradient_statistic-imagenet.txt")
name_file = file_path.name

pattern_act = re.compile(r'layers\.(\d+)\.(act|conv)\d+\.quantize_fn\.scale(?:_dict)?\.(\d+)-bit:\[([-\d.,e]+)\]')
pattern_conv = re.compile(r'layers\.(\d+)\.(act|conv)\d+\.quantize_fn\.scale(?:_dict)?\.(\d+)-bit:\[(-?[\d.e]+(?:,\s?-?[\d.e]+)*)\]')

data = {'layer': [], 'type': [], 'bit': [], 'scale': []}

with open(file_path, 'r') as file:
    for line in file:
        match_act = pattern_act.search(line)
        match_conv = pattern_conv.search(line)
        if match_act:
            layer = int(match_act.group(1))
            scale_type = match_act.group(2)
            scale_bit = match_act.group(3)
            scale_values = match_act.group(4).split(', ')
            for scale_value in scale_values:
                data['layer'].append(layer)
                data['type'].append(scale_type)
                data['bit'].append(int(scale_bit))
                data['scale'].append(float(scale_value))
        elif match_conv:
            layer = int(match_conv.group(1))
            scale_type = match_conv.group(2)
            scale_bit = match_conv.group(3)
            scale_values = match_conv.group(4).split(', ')
            for scale_value in scale_values:
                data['layer'].append(layer)
                data['type'].append(scale_type)
                data['bit'].append(int(scale_bit))
                data['scale'].append(float(scale_value))

print('Extracted data:', data['layer'][:5])
print('Extracted data:', data['type'][:5])
print('Extracted data:', data['bit'][:5])
print('Extracted data:', data['scale'][:5])

df = pd.DataFrame(data)

print(df.head())

# Draw box plots of act and conv respectively
for scale_type in ['act', 'conv']:
    plt.figure(figsize=(12, 8))
    df_filtered = df[df['type'] == scale_type]
    
    # Calculate mean and standard deviation
    mean = df_filtered['scale'].mean()
    std_dev = df_filtered['scale'].std()
    # define threshold
    threshold = 3
    # Determine upper and lower bounds for outliers
    upper_limit = mean + threshold * std_dev
    lower_limit = mean - threshold * std_dev
    # Remove outliers based on threshold
    df_filtered = df_filtered[(df_filtered['scale'] >= lower_limit) & (df_filtered['scale'] <= upper_limit)]

    # y_min = df_filtered['bit'].min()
    # y_max = df_filtered['bit'].max()

    y_min = lower_limit
    y_max = upper_limit

    for bit in [8,6,4,2]:
        df_filtered_bit = df_filtered[df_filtered['bit'] == bit]
        df_filtered_bit.boxplot(column='scale', by='layer', grid=False, showfliers=True)
        if not df_filtered_bit.empty:
            # plt.title(f'{bit}-bit Scale Values for {scale_type.upper()}', fontdict={'fontsize':20, 'color':'black', 'family':'Times New Roman', 'weight':'normal'})
            plt.title(f'{bit}-bit', fontdict={'fontsize':20, 'color':'black', 'family':'Times New Roman', 'weight':'normal'})
            plt.suptitle('')
            plt.ylim(y_min, y_max)  # Set the y-axis range
            plt.xlabel('Layer', fontdict={'fontsize':15, 'color':'black', 'family':'Times New Roman', 'weight':'normal'})
            if scale_type == 'act':
                plt.ylabel('The gradients of activation scale', fontdict={'fontsize':15, 'color':'black', 'family':'Times New Roman', 'weight':'normal'})
            else:
                plt.ylabel('The gradients of weight scale', fontdict={'fontsize':15, 'color':'black', 'family':'Times New Roman', 'weight':'normal'})
            plt.xticks(rotation=90)
            plt.tight_layout()
            # plt.legend(loc='lower right', fontsize='x-large')
            plt.savefig(Path("/root/code/Multi_Precision/results/plots")/ str(f'{bit}-bit-'+scale_type+name_file.replace("txt", "png")), dpi=300)
            # plt.show()
        else:
            print(f'No data found for type: {scale_type}')
        plt.close()
