import json
import os
quantization_encodings = '../../example1/output_dir_Qwen2-VL-2B-Instruct/onnx/llama31.encodings.bak'
quantization_encodings_update = 'llama31.encodings'
w41fp16 = True


def write_all_downproj_into_fp16():
    with open(quantization_encodings, 'r') as file:

        data = json.load(file)
        
        for key,item in data['activation_encodings'].items():
            if 'mlp_down_proj_conv_Conv/Conv_output_0' in key:
                data['activation_encodings'][key][0]["dtype"] = "float"
                print(f'write {key} as fp16 or w4fp16')

        if data['quantizer_args']['param_bitwidth'] != 4:
            w41fp16 = False
            print('converter to fp16 directly instead of w4fp16, which will increase memory usage')
            
        with open(quantization_encodings_update, 'w') as file:
            json.dump(data, file, indent=4)

def write_partial_downproj_into_fp16():
    with open(quantization_encodings, 'r') as file:

        data = json.load(file)
        #the best logic here is using (max - min) to detimine whether using fp16
        for key,item in data['activation_encodings'].items():
            if 'mlp_down_proj_conv_Conv/Conv_output_0' in key:
                max_value = data['activation_encodings'][key][0]["max"]
                min_value = data['activation_encodings'][key][0]["min"]
                #no solid reason why using 50 as threhold, just balance accuracy and perf
                if (max_value - min_value) > 40.0:
                    data['activation_encodings'][key][0]["dtype"] = "float"
                    print(f'write {key} as fp16 or w4fp16')

        if data['quantizer_args']['param_bitwidth'] != 4:
            w41fp16 = False
            print('converter to fp16 directly instead of w4fp16, which will increase memory usage')
            
        with open(quantization_encodings_update, 'w') as file:
            json.dump(data, file, indent=4)        


write_all_downproj_into_fp16()
#write_partial_downproj_into_fp16()

os.system('cp llama31.encodings ../../example1/output_dir_Qwen2-VL-2B-Instruct/onnx/llama31.encodings')
