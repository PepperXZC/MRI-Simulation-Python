import main_again
import yaml
import numpy as np
import bssfp
import copy
test_info = main_again.info(thickness=0.5)

def sign(num):
    if num > 0:
        return 1
    else:
        return -1
    
def TR_node(fa_sequence, TR_sequence, i, info):
    fa = fa_sequence.tolist()[i]
    t = TR_sequence.tolist()[i]
    init_pulse = {
        "type": "PULSE",
        "FA": fa,
        "t": 0,
        "slice_thickness": info.thickness # 选层
    }
    res = [init_pulse]
    if i == 0: # 只给了一个 prep
        free = {
            "type": "NONE",
            "t": t,
        }
        res.append(free)
        return res
    else:
        center = info.N_pe // 2
        num = center + sign((i) % 2) * ((i) // 2)
        G_diff = info.Gyp - num * info.Gyi
        Gx = info.Gx
        phase_encode = {
            "type": "ENCODING",
            "Gy": G_diff,
            "Gx": - Gx,
            "t": info.tau_y
        }
        
        rewind = {
            "type": "ENCODING",
            "Gy": - G_diff,
            "Gx": - Gx,
            "t": info.tau_y
        }
        res.append(copy.deepcopy(phase_encode))
        for i in range(info.N_read):
            readout = {
                "type": "ADC",
                "Gx": Gx,
                "t": info.delta_t,
                "line_index": num,
                "sample_index": i
            }
            res.append(copy.deepcopy(readout))
        res.append(copy.deepcopy(rewind))
        return res
    
def yaml_generate(info:main_again.info):
    # bssfp
    prep_num = 1
    fa_sequence, TR_sequence = bssfp.get_sequence_info(info, prep_num)

    seq_list = []
    for i in range(info.N_pe + 1):
        seq_list += TR_node(fa_sequence, TR_sequence, i, info)
    print(seq_list[-5:-1])
    with open("test.yaml", "w") as f:
        yaml.safe_dump(seq_list, f)

yaml_generate(test_info)
# print(sign((0 - 1) % 2) * ((0 + 1) // 2))