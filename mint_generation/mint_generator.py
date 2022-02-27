import os
import random

def assign_device_num(device_list):
    device_dict = {}
    new_list = []

    for device in device_list:
        device_dict[device] = 1

    for device in device_list:
        new_list.append(f"{device}_{device_dict[device]}")
        device_dict[device] += 1

    return new_list

current_dir = os.getcwd()
mint_dir = current_dir + "/mint_examples"

# create a mint dir if not exist
if not os.path.exists(mint_dir):
    os.mkdir(mint_dir)
    print("makine a new directory")

NUM_FILE = 10
DEVICE_NUM = 5

device_list = ["NOZZLE DROPLET GENERATOR nozzle_droplet_generator", "DROPLET SORTER droplet_sorter", "MIXER mixer", "DROPLET SPLITTER droplet_splitter", "PICOINJECTOR picoinjector", "DIAMOND CHAMBER diamond_chamber"]

# print(randomized_list)
# print(assign_device_num(randomized_list))

for i in range(NUM_FILE):
    file_name = "mint" + str(i) + ".mint"

    f = open(os.path.join(mint_dir, file_name), "w")

    device_name = "ml" + str(i)
    context = f"DEVICE {device_name}\n\nLAYER FLOW\n\n"

    randomized_list = assign_device_num(random.choices(device_list, k=5))

    for device in randomized_list:
        context += f"{device}\n"

    context += "\n\n"
    channelWidth = 400;

    count = 1
    for i in range(DEVICE_NUM - 1):
        context += f"CHANNEL channel_{count} from {randomized_list[i]} 1 to {randomized_list[i+1]} 2 channelWidth={channelWidth};\n"
        count += 1

    context += "\nEND LAYER"

    f.write(context)

    f.close()
