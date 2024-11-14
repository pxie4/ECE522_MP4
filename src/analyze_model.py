import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import re
import ast

workloads = ['BERT','Inceptionv3','ResNet152','SENet154','VIT']
# Main Goals:
# (1) Can you determine the mininum memory demand to execute each model for each batch size, when no data is allowed to
# be evicted from the GPU before a tensor is destroyed?
# (2) What is the distribution of the tensor size and lifetime? Plot the total size of the tensors against lifetime in milliseconds.
# (3) Plot a figure to show the distribution of the active/inactive time of tensors in each model. Explain the potential for data
# offloading with the figure.
# (4) If we can swap tensors off GPU memory when not used, what is the minimal memory requirement for executing each
# model (assuming data transfer takes no time)?

for load in workloads:
    # print(os.path.join('..','results',load,'sim_input'))
    path = os.path.join('..','results',load,'sim_input')
    model_dir = os.path.join('..','img_plot', load)
    os.makedirs(model_dir)
    batch_sizes = set()                         # per workload
    for dataset in os.listdir(path):
        pattern = r"(\d+)(Kernel|Tensor|AuxTime)\.info"
        match = re.match(pattern, dataset)
        if match:
            batch_len = int(match.group(1))      # Extracts the number (e.g., 1024)
            info_type = match.group(2)            # Extracts "Kernel" or "Tensor"
            batch_sizes.add(batch_len)
        else:
            print("Filename format is incorrect")

    for batch in batch_sizes:
        print('Parsing Through: {}, {}'.format(load, batch))
        tensor_path = os.path.join('..','results',load,'sim_input', str(batch)+'Tensor.info' )
        kernel_path = os.path.join('..','results',load,'sim_input', str(batch)+'Kernel.info')
        # create Data Structures
        tensor_info = {}
        kernel_info = {}
        size_per_tensor = []
        time_series = [0]
        if os.path.isfile(tensor_path):
            with open(tensor_path, "r") as tensor_file:
                for line in tensor_file:  
                    parts = line.strip().split()
                    tensor_id = int(parts[0])                 # Tensor ID
                    size = int(parts[1])                      # Tensor size in bytes
                    is_global = parts[2].lower() == "true"    # Global status as a boolean

                    # Initialize tensor information in dictionary
                    tensor_info[tensor_id] = {
                        "size": size,
                        "is_global": is_global,
                        "created": float('inf'),              # Set to high value for min comparison
                        "last_used": float('-inf'),           # Set to low value for max comparison
                        "used_in": set()
                    }
                    size_per_tensor.append(size)
        else:
            print('Path to {} is invalid'.format(tensor_path))

        if os.path.isfile(kernel_path):
            with open(kernel_path, "r") as kernel_file:
                for line in kernel_file: 
                    parts = line.strip().split()
                    kernel_id = int(parts[0])                       # Kernel ID
                    kernel_type = parts[1]                          # Kernel Type
                    execute_time = float(parts[2])                  # Kernel Execute Time (ms)
                    input_tensors = ast.literal_eval(parts[3]) if parts[3] else []  # Input Tensor IDs
                    output_tensors = ast.literal_eval(parts[4]) if parts[4] else [] # Output Tensor IDs

                    time_series.append(time_series[-1] + execute_time)
                    workspace_tensor_id = int(parts[5]) if len(parts) > 5 else None

                    # Update workspace tensor if it exists
                    if workspace_tensor_id is not None:
                        tensor_info[workspace_tensor_id]['created'] = min(kernel_id, tensor_info[workspace_tensor_id]['created'])
                        tensor_info[workspace_tensor_id]['last_used'] = max(kernel_id, tensor_info[workspace_tensor_id]['last_used'])
                        tensor_info[workspace_tensor_id]['used_in'].add(kernel_id)

                    # Initialize kernel information
                    kernel_info[kernel_id] = {
                        "kernel_type": kernel_type,
                        "execute_time": execute_time,
                        "workspace_tensor": workspace_tensor_id
                    }

                    # For input tensors
                    for tensor in input_tensors:
                        if not tensor_info[tensor]["is_global"]:
                            tensor_info[tensor]['created'] = min(kernel_id, tensor_info[tensor]['created'])
                            tensor_info[tensor]['last_used'] = max(kernel_id, tensor_info[tensor]['last_used'])
                        tensor_info[tensor]['used_in'].add(kernel_id)

                    # For output tensors
                    for tensor in output_tensors:
                        if not tensor_info[tensor]["is_global"]:
                            tensor_info[tensor]['created'] = min(kernel_id, tensor_info[tensor]['created'])
                            tensor_info[tensor]['last_used'] = max(kernel_id, tensor_info[tensor]['last_used'])
                        tensor_info[tensor]['used_in'].add(kernel_id)
        else:
            print('Path to {} is invalid'.format(kernel_path))
 
        # For Q1: 
        active_mem_bytes = np.zeros(len(kernel_info)) # index refers to kernel id, value refers to amt of bytes active based on tensors
        # For Q2:
        lifetimes_per_tensor = np.zeros(len(tensor_info)) # index refers to tensor id, value refers to lifetime in ms
        # For Q3:
        active_tensor = np.zeros(len(kernel_info)) # index refers to kernel id, value refers to number of active tensors
        # For Q4:
        active_offload = np.zeros(len(kernel_info)) # index refers to kernel id, value refers to amt of bytes active in each kernel

        for tensor_id, tensor in tensor_info.items():
            if not tensor["is_global"]:
                # print("TensorID: {}, Created: {}, LastUsed: {}".format(tensor_id,tensor["created"],tensor['last_used']))
                for i in range(tensor['created'], tensor['last_used'] + 1):
                    active_mem_bytes[i] += tensor['size']
                    lifetimes_per_tensor[tensor_id] += kernel_info[i]["execute_time"]
                    active_tensor[i] += 1
            else:
                for i in range(len(kernel_info)):
                    active_mem_bytes[i] += tensor['size']
                    lifetimes_per_tensor[tensor_id] = -1 # always alive
                    active_tensor[i] += 1

            for j in tensor["used_in"]:
                active_offload[j] += tensor['size']

        plt.figure(figsize=(10, 6))
        plt.plot(active_mem_bytes, label='Active Memory (Bytes)', marker='o', color='b')
        plt.xlabel('Kernel ID')
        plt.ylabel('Active Memory (Bytes)')
        plt.title('Active Memory Usage across Kernels, Model {}, Batch Size: {}'.format(load,batch))
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, 'active_memory_across_kernel_{}_{}.png'.format(load, batch)))  # Save the figure
        plt.close()

        plt.hist2d(size_per_tensor, lifetimes_per_tensor, bins=[20, 20], cmap='viridis'.format(load,batch))
        plt.colorbar(label='Count')
        plt.xlabel('Memory Size (MB)')
        plt.ylabel('Lifetime (s)')
        plt.title('Lifetime vs Memory Size Distribution, Model {}, Batch Size: {}')
        plt.savefig(os.path.join(model_dir, 'lifetime_memory_size_{}_{}.png'.format(load, batch)))  # Save the figure
        plt.close()
    
        plt.figure(figsize=(10, 6))
        plt.plot(time_series[:-1], active_mem_bytes, label='Active Memory (Bytes)', marker='o', color='b')
        plt.xlabel('Time (ms)')
        plt.ylabel('Active Memory (Bytes)')
        plt.title('Active Memory across Time, Model {}, Batch Size: {}'.format(load,batch))
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, 'active_memory_across_time{}_{}.png'.format(load, batch)))  # Save the figure
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(active_offload, label='Active Memory (Bytes)', marker='o', color='b')
        plt.xlabel('Kernel ID')
        plt.ylabel('Active Memory (Bytes)')
        plt.title('Memory Usage per Kernels, Model {}, Batch Size: {}'.format(load,batch))
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, 'mem_usage_per_kernel_{}_{}.png'.format(load, batch)))  # Save the figure
        plt.close()

