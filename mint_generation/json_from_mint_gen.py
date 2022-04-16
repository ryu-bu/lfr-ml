from parchmint import Device
import os
import json
from pymint.mintdevice import MINTDevice

current_dir = os.getcwd()
mint_dir = current_dir + "/mint_examples"

files = os.listdir(mint_dir)

# with open(mint_dir + '/' + files[0], 'r') as f:
# file_path = mint_dir + '/dropx_test1.mint'
file_path = mint_dir + '/' + files[0]

print(file_path)

device = MINTDevice.from_mint_file(file_path)

json_dir = current_dir + "/mint_json"

print(device.get_components)

tt = "{}.json".format(device.name)
dump = device.to_parchmint_v1()
with open(os.path.join(json_dir, tt), "w") as f:
    json.dump(dump, f)
