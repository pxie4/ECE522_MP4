
#include "analysis.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>

#include "ast.h"
#include "simulationUtils.h"
#include "simulator.h"
#include "printUtils.h"

using Simulator::TensorMovementHint;
using Simulator::TensorLocation;

extern std::string migration_policy_str;
extern std::string eviction_policy_str;

extern double GPU_frequency_GHz;
extern double GPU_memory_size_GB;
extern double CPU_PCIe_bandwidth_GBps;
extern double SSD_PCIe_bandwidth_GBps;
extern double GPU_malloc_uspB;
// extern double GPU_free_uspB;
extern int prefetch_degree;
extern int borden;
extern int is_transformer;
double CPU_memory_line_GB = -1;
double SSD_latency_us = -1;
double system_latency_us = -1;
double delta_parameter = -1;
double loosen_parameter = 1;

long long memory_offset_intermediate = 0;
long long memory_offset_weights = 0;
int kernel_index = 0;
int prefetch_optimize = 1;
std::vector<Tensor *> tensor_list;
std::vector<CUDAKernel> kernel_list;

// TODO: Important: The global list for all inactive tensor periods
std::vector<InactivePeriod *> inactive_periods_list;

std::vector<double> kernel_time_table;
std::vector<EvictionGuideEntry> EvictionGuideTable;
std::vector<long> GPU_resident_memory_estimation;
std::vector<long> CPU_resident_memory_estimation;

std::vector<TensorMovementHint> movement_hints;
std::vector<InactivePeriod *> offloaded_local_intervals;

string Tensor::name() const { return "tensor" + std::to_string(tensor_id); }

bool Tensor::is_alive(int current_kernel) const {
  return is_global_weight || (live_interval.second == -1 ? current_kernel == live_interval.first
                                                         : current_kernel >= live_interval.first &&
                                                               current_kernel < live_interval.second);
}

void Tensor::print() const {
  std::cout << "tensor" << tensor_id << " Is weight (global)?: " << this->is_global_weight << ", "
            << "Size in byte: " << size_in_byte << std::endl;
}

Tensor::Tensor(long long size, bool glob) {
  static int tensor_count = 0;
  tensor_id = tensor_count++;
  size_in_byte = size;
  raw_size_byte = size;
  is_global_weight = glob;
  if (glob) {
    address_offset = memory_offset_weights;
    // page-level alignment
    long N_pages = (size % 4096 == 0) ? (size / 4096) : ((size / 4096) + 1);
    memory_offset_weights += N_pages * 4096;
    size_in_byte = N_pages * 4096;
  } else {
    address_offset = memory_offset_intermediate;
    // page-level alignment
    long N_pages = (size % 4096 == 0) ? (size / 4096) : ((size / 4096) + 1);
    memory_offset_intermediate += N_pages * 4096;
    size_in_byte = N_pages * 4096;
  }
}

unsigned long Tensor::getGlobalOffset() {
  return address_offset + (is_global_weight ? 0 : memory_offset_weights);
}

CUDAKernel::CUDAKernel(int kernel_id, CUDAKernelType t, std::vector<Tensor *> input_tensor_list,
                       std::vector<Tensor *> output_tensor_list, Tensor *workspace_tensor) {
  this->kernel_id = kernel_id;
  this->type = t;
  this->inputs.insert(input_tensor_list.begin(), input_tensor_list.end());
  this->outputs.insert(output_tensor_list.begin(), output_tensor_list.end());
  this->workspace = workspace;
}

void CUDAKernel::print() {
  std::cout << "---------------------------------------------------------------"
               "---------------"
            << std::endl;
  std::cout << "Kernel ID: " << kernel_id << ", "
            << "Name: " << print_kerneltype_array[type] << std::endl;
  std::cout << "Execution Time:            " << execution_cycles << std::endl;
  std::cout << "Input Tensors:" << std::endl;
  for (auto it = inputs.begin(); it != inputs.end(); it++) {
    (*it)->print();
  }
  std::cout << "Output Tensors:" << std::endl;
  for (auto it = outputs.begin(); it != outputs.end(); it++) {
    (*it)->print();
  }
}

void CUDAKernel::getRequiredTensors(std::vector<Tensor *> &required_tensors) const {
  std::unordered_set<Tensor *> set;
  getRequiredTensors(set);
  for (Tensor *tensor : set) required_tensors.push_back(tensor);
}

void CUDAKernel::getRequiredTensors(std::unordered_set<Tensor *> &required_tensors) const {
  for (Tensor *tensor : inputs) required_tensors.insert(tensor);
  for (Tensor *tensor : outputs) required_tensors.insert(tensor);
}

void CUDAKernel::getRequiredTensors(std::vector<Tensor *> &required_tensors,
                                    std::vector<Tensor *> &required_input_tensors,
                                    std::vector<Tensor *> &required_output_tensors) const {
  std::unordered_set<Tensor *> set;
  for (Tensor *tensor : inputs) {
    set.insert(tensor);
    required_tensors.push_back(tensor);
    required_input_tensors.push_back(tensor);
  }
  for (Tensor *tensor : outputs) {
    if (set.find(tensor) == set.end()) {
      required_tensors.push_back(tensor);
      required_output_tensors.push_back(tensor);
    }
  }
}

/**
 * @brief this function is used to fill the liveness information of every tensor
 * @todo you should fill the field live_interval for each tensor in the tensor_list
 *       see descriptions of live_interval in Tensor::live_interval
 */
void tensor_first_pass_liveness_analysis() {
  const int tensor_num = tensor_list.size();
  const int kernel_num = kernel_list.size();

  for (int i = 0; i < tensor_num; i++) {
    Tensor *current_tensor = tensor_list[i];

    // TODO: complete liveness analysis
    if (!current_tensor->is_global_weight) {
      // This tensor is intermediate

      // loop over kernels
      for(int j = 0; j < kernel_list.size(); ++j){
        

        // grab input/output tenors
        CUDAKernel cur = kernel_list[j];
        std::unordered_set<Tensor*> in = cur.inputs;
        std::unordered_set<Tensor*> out = cur.outputs;

        if(std::find(in.begin(), in.end(), current_tensor) != in.end() || std::find(out.begin(), out.end(), current_tensor) != out.end()){
          // populate in_kernel
          current_tensor->in_kernels.push_back(cur.kernel_id);
          std::make_heap(current_tensor->in_kernels.begin(),current_tensor->in_kernels.end(),std::greater<>{});
          // replace if find newer or older for first and second respectively
          if(cur.kernel_id < current_tensor->live_interval.first || current_tensor->live_interval.first == -1){
            current_tensor->live_interval.first = cur.kernel_id; 
          }

          if(cur.kernel_id > current_tensor->live_interval.second){
            current_tensor->live_interval.second = cur.kernel_id+1;
          }
        }
      }
    }
    // global tensors do not need this info
  }
}

void Tensor::print_liveness() {
  this->print();
  if (!this->is_global_weight) {
    std::cout << "Liveness: Birth: " << this->live_interval.first << ", Death: " << this->live_interval.second
              << "." << std::endl;
  } else {
    std::cout << "Liveness: Global" << std::endl;
  }
}

/**
 * @brief this function is used to fill the inactive period information of every tensor
 * @todo you should fill the field inactive_periods for each tensor in the tensor_list
 *       see descriptions of inactive_periods in Tensor::inactive_periods
 */
void tensor_second_pass_interval_formation() {
  const int tensor_num = tensor_list.size();
  const int kernel_num = kernel_list.size();
  for (int i = 0; i < tensor_num; i++) {
    // std::cout<<i<<std::endl;
    Tensor *current_tensor = tensor_list[i];
    // TODO: complete inactive period analysis
    if (!current_tensor->is_global_weight) {
      // This tensor is intermediate
      
      std::pair<int,int> period;
      bool fill_status = true; // false - first populated, true - second populated
      for(int j = 0; j < kernel_list.size(); ++j){
        // if(i == 1052){
        //   std::cout<<i<<" "<<j<<std::endl;
        //   std::cout<<kernel_list[i].kernel_id<<std::endl;
        // }
        
        CUDAKernel cur = kernel_list[j];
        std::unordered_set<Tensor*> in = cur.inputs;
        std::unordered_set<Tensor*> out = cur.outputs;  
        // if(i == 1052){
        //   std::cout<<"grabbed in and out"<<std::endl;
        // }
        // if not in either in or output, label the first part of period
        if(fill_status && std::find(in.begin(), in.end(), current_tensor) == in.end() && std::find(out.begin(), out.end(), current_tensor) == out.end()){
          period.first = cur.kernel_id;
          fill_status = false;
          // if(i == 1052){
          //   std::cout<<"START OF PERIOD: "<<period.first<<std::endl;
          // }
          
        // if present, label the second part of period
        }else if((fill_status == false) && (std::find(in.begin(), in.end(), current_tensor) != in.end() || std::find(out.begin(), out.end(), current_tensor) != out.end())){
          period.second = cur.kernel_id;
          fill_status = true;
          InactivePeriod * p  = new InactivePeriod(current_tensor);
          p->kernelLevel_interval = period;
          current_tensor->inactive_periods.push_back(p);
          // std::cout<<p->kernelLevel_interval.first<<" "<<p->kernelLevel_interval.second<<std::endl;
        }
      }
      
      if(fill_status == false){
          period.second = kernel_list[kernel_list.size()-1].kernel_id+1;
          fill_status = true;
          InactivePeriod * p  = new InactivePeriod(current_tensor);
          p->kernelLevel_interval = period;
          current_tensor->inactive_periods.push_back(p);
          // std::cout<<i<<" "<<p->kernelLevel_interval.first<<" "<<p->kernelLevel_interval.second<<std::endl;
      }

    } else {
      // This tensor is global
      InactivePeriod * p  = new InactivePeriod(current_tensor);
      p->is_looped = true;
      p->kernelLevel_interval = std::pair<int,int>(0,-1); // first must be > second
      current_tensor->inactive_periods.push_back(p);
    }
  }
}

void Tensor::print_inactive_periods() {
  // print();
  std::cout << "Inactive Periods:" << std::endl;
  for (int i = 0; i < inactive_periods.size(); i++) {
    std::cout << "interval " << i << ": " << inactive_periods[i]->kernelLevel_interval.first << "--------"
              << inactive_periods[i]->kernelLevel_interval.second << std::endl;
    std::cout << "Estimated Time:" << inactive_periods[i]->time_estimated << std::endl;
  }
  std::cout << "_______________________________________________________________" << std::endl;
}

// A provided compiler pass to calculate the estimated execution time for every
// tensors' inactive period length(time)
void get_inactive_periods_time() {
  int kernel_num = kernel_list.size();

  // Setup a cumulative time list;
  double time = 0;
  kernel_time_table.push_back(0);
  for (int i = 0; i < kernel_num; i++) {
    time += (double)kernel_list[i].execution_cycles / (double)(GPU_frequency_GHz * 1000);
    kernel_time_table.push_back(time);
  }

  // Fill the looped extend kernel time table      0 - 2 * kernel_num
  std::vector<double> kernel_time_table_extended;
  kernel_time_table_extended.resize(kernel_num);
  for (int j = 0; j < kernel_num; j++) {
    kernel_time_table_extended[j] = kernel_time_table[j];
  }
  double last_time = kernel_time_table[kernel_num];
  kernel_time_table_extended.push_back(last_time);
  for (int j = 0; j < kernel_num; j++) {
    last_time += (double)kernel_list[j].execution_cycles / (double)(GPU_frequency_GHz * 1000);
    kernel_time_table_extended.push_back(last_time);
  }

  for (int i = 0; i < inactive_periods_list.size(); i++) {
    if (!inactive_periods_list[i]->is_looped) {
      assert(inactive_periods_list[i]->kernelLevel_interval.second >
             inactive_periods_list[i]->kernelLevel_interval.first);
      inactive_periods_list[i]->time_estimated =
          kernel_time_table[inactive_periods_list[i]->kernelLevel_interval.second] -
          kernel_time_table[inactive_periods_list[i]->kernelLevel_interval.first];
    } else {
      assert(inactive_periods_list[i]->kernelLevel_interval.second <
             inactive_periods_list[i]->kernelLevel_interval.first);
      int end = inactive_periods_list[i]->kernelLevel_interval.second;
      int start = inactive_periods_list[i]->kernelLevel_interval.first;
      end += kernel_num;
      inactive_periods_list[i]->time_estimated =
          kernel_time_table_extended[end] - kernel_time_table_extended[start];
    }
  }
}

void InactivePeriod::print() {
  std::cout << "interval " << ": " << kernelLevel_interval.first << "--------" << kernelLevel_interval.second
            << std::endl;
  std::cout << "Estimated Time:" << time_estimated << std::endl;
  std::cout << "Tensor: ";
  this->tensor_back_ptr->print();
  std::cout << "_______________________________________________________________" << std::endl;
}

void print_GPU_mem_really_in_use() {
  for (int i = 0; i < kernel_list.size(); i++) {
    std::vector<Tensor *> r;
    kernel_list[i].getRequiredTensors(r);
    long size_bite = 0;
    for (int j = 0; j < r.size(); j++) {
      size_bite += r[j]->size_in_byte;
    }
    std::cout << "Kernel " << i << ": " << size_bite << std::endl;
  }
}

/**
 * @brief fill this function to schedule your movement hints
 */

void scheduling_movement_hints() {
  // TODO: fill the data structure "std::vector<TensorMovementHint> movement_hints" with your own hints!
  for (int i = 0; i < kernel_list.size(); i++) {
    CUDAKernel cur_kernel = kernel_list[i]; 
    
    std::unordered_set<Tensor*> cur_inputs = cur_kernel.inputs;
    std::unordered_set<Tensor*> cur_outputs = cur_kernel.outputs;
    std::cout<<"KERNEL ID: "<<cur_kernel.kernel_id<<" INPUT SIZE: "<<cur_inputs.size()<<std::endl;
    // prefetch hints
    for(auto it = cur_inputs.begin(); it != cur_inputs.end(); it++){
      int temp;
      if(cur_kernel.kernel_id-1 < 0){
        temp = 0;
      }else{
        temp = cur_kernel.kernel_id-1;
      }
      std::cout<<(*it)->tensor_id<<std::endl;
      TensorMovementHint * hint = new TensorMovementHint(Simulator::NOT_KNOWN, Simulator::IN_GPU,temp,*it);
      movement_hints.push_back(*hint);
    }
    // preallocate hints
    // for(auto it = cur_outputs.begin(); it != cur_outputs.end(); it++){
    //   int temp;
    //   if(cur_kernel.kernel_id-1 < 0){
    //     temp = 0;
    //   }else{
    //     temp = cur_kernel.kernel_id-1;
    //   }
    //   TensorMovementHint * hint = new TensorMovementHint(Simulator::NOT_PRESENT, Simulator::IN_GPU,temp,*it);
    //   movement_hints.push_back(*hint);
    // }    
  }

  // //pre eviction hints
  // for(int i = 0; i < tensor_list.size(); i++){
  //   int last_used = tensor_list[i]->live_interval.second;
  //   TensorMovementHint * hint = new TensorMovementHint(Simulator::IN_GPU, Simulator::IN_SSD,last_used,tensor_list[i]);
  //   movement_hints.push_back(*hint);
  // }

  // make sure the movement hints are sorted, the simulator depends on this
  std::sort(movement_hints.begin(), movement_hints.end());
  // for(int i = 0; i < movement_hints.size(); i++){
  //   // std::cout<<*it.from<<" "<<*it.to<<std::endl;
  //   std::cout<<movement_hints[i].issued_kernel_id<<std::endl;
  // }
}
