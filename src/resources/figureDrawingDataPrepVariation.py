import os, json, codecs
import numpy as np
from statsFiguresUtil import *

script_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.abspath(os.path.join(script_path, os.path.pardir, "configs"))
result_path = os.path.abspath(os.path.join(script_path, os.path.pardir, os.path.pardir, "results"))
output_path = os.path.abspath(os.path.join(script_path, "raw_output"))

def retrieve_data_from_json_file(filename: str):
  overall_json = json.load(codecs.open(filename, 'r', encoding='utf-8'))
  dimension_num = overall_json["dimension_num"]
  dimension_names = overall_json["dimension_names"]
  dimension_details = []
  print(f"Parsing json file <{filename}>")
  for dimension in dimension_names:
    exec(f"{dimension}s = overall_json[\"{dimension}\"]")
    exec(f"print(f\"  {dimension:12s}: {{{dimension}s}}\")")
    exec(f"dimension_details.append({dimension}s)")
  data = np.array(overall_json["data"])
  print(f"  Dimension Length: {[len(dimension_detail) for dimension_detail in dimension_details]}")
  return dimension_num, dimension_names, dimension_details, data

def fuse_data_matrices(items: list):
  assert all(item[0] == len(all_dimensions) for item in items)
  assert all(item[1] == list(all_dimensions.keys()) for item in items)
  for dimension in all_dimensions.keys():
    exec(f"{dimension}s_set = set()")
  # gather all the properties in all dimensions into a set
  for item in items:
    for dimension_idx, dimension in enumerate(all_dimensions.keys()):
      exec(f"for property in item[2][dimension_idx].keys(): {dimension}s_set.add(property)")
  # sorting keys for all dimension according to lambda expression
  print("Unified Dimensions:")
  dimension_details = []
  for dimension in all_dimensions.keys():
    exec(f"global {dimension}s; {dimension}s = {{ key : i for i, key in enumerate(sorted(list({dimension}s_set), key=all_dimensions[\"{dimension}\"])) }}")
    exec(f"print(f\"  {dimension:12s}: {{{dimension}s}}\")")
    exec(f"dimension_details.append(list({dimension}s.keys()))")
  # generate unified data matrix
  data, data_dimension = -np.ones(0, dtype=float), []
  for dimension in all_dimensions.keys():
    exec(f"data_dimension.append(str(len({dimension}s)))")
  # python3 does not support exec to directly reassign local variable
  exec(f"data.resize(({', '.join(data_dimension)}), refcheck=False)")
  exec(f"data.fill(-1)")
  data_dimension = tuple(int(dimension_len) for dimension_len in data_dimension)
  print(f"  Dimension Length: {list(data_dimension)}")
  for data_idx in np.ndindex(data_dimension):
    properties = [dimension_details[i][data_idx[i]] for i in range(len(all_dimensions))]
    for item_idx, item in enumerate(items):
      _, _, item_dimension_details, item_data = item
      if all([properties[i] in item_dimension_details[i] for i in range(len(all_dimensions))]):
        item_data_index = [item_dimension_details[i][properties[i]] for i in range(len(all_dimensions))]
        data[data_idx] = max(data[data_idx], item_data[tuple(item_data_index)])
  return data

if __name__ == "__main__":
  data_used = ["data"]
  # data_used = ["data100copy", "data113copy"]
  data = fuse_data_matrices([
      retrieve_data_from_json_file(f"{output_path}/{data_json}.json")
      for data_json in data_used])

  header_printed = False
  # for model in [INCEPTION, RESNET, SENET, VIT, BERT]:
  for model in [BERT, VIT, INCEPTION, RESNET, SENET]:
    # settings BEGIN ========================================
    # model = INCEPTION
    # model = RESNET
    # model = SENET
    # model = RESNEXT
    # model = VIT
    # model = BERT

    # stat_candidate = "stall_percentage"
    # stat_candidate = "overlap_percentage"
    # stat_candidate = "compute_percentage"

    # transpose = True
    transpose = False 

    plot_format = True
    latex_format = False

    # plot_format = True
    # latex_format = True

    include_ideal = True
    normal_have_axis = True

    # unit = ""
    unit = "k"
    # unit = "M"
    
    if model == INCEPTION:
      # Inception ###################################
      x_axis_idxs, y_axis_idxs = [
        tuple([
            batch_sizes["1536"]
        ]),
        # tuple(
        #     batch_sizes.values()
        # ),
        tuple([
            ktime_vars["0"],
            ktime_vars["0.05"],
            ktime_vars["0.10"],
            ktime_vars["0.15"],
            ktime_vars["0.20"],
            ktime_vars["0.25"]
        ])
      ]
    elif model == RESNET:
      # ResNet ######################################
      x_axis_idxs, y_axis_idxs = [
        tuple([
            batch_sizes["1280"]
        ]),
        # tuple(
        #     batch_sizes.values()
        # ),
        tuple([
            ktime_vars["0"],
            ktime_vars["0.05"],
            ktime_vars["0.10"],
            ktime_vars["0.15"],
            ktime_vars["0.20"],
            ktime_vars["0.25"]
        ])
      ]
    elif model == SENET:
      # SENet #######################################
      x_axis_idxs, y_axis_idxs = [
        tuple([
            batch_sizes["1024"]
        ]),
        tuple([
            ktime_vars["0"],
            ktime_vars["0.05"],
            ktime_vars["0.10"],
            ktime_vars["0.15"],
            ktime_vars["0.20"],
            ktime_vars["0.25"]
        ])
      ]
    elif model == BERT:
      # BERT ########################################
      x_axis_idxs, y_axis_idxs = [
        tuple([
            batch_sizes["256"]
        ]),
        # tuple(
        #     batch_sizes.values()
        # ),
        tuple([
            ktime_vars["0"],
            ktime_vars["0.05"],
            ktime_vars["0.10"],
            ktime_vars["0.15"],
            ktime_vars["0.20"],
            ktime_vars["0.25"]
        ])
      ]
    elif model == VIT:
      # VIT #########################################
      x_axis_idxs, y_axis_idxs = [
        tuple([
            batch_sizes["1280"]
        ]),
        # tuple(
        #     batch_sizes.values()
        # ),
        tuple([
            ktime_vars["0"],
            ktime_vars["0.05"],
            ktime_vars["0.10"],
            ktime_vars["0.15"],
            ktime_vars["0.20"],
            ktime_vars["0.25"]
        ])
      ]
    ###############################################
    # settings END ==========================================
    model_desc = list(net_name_detail_translation.values())[model]
    model = list(net_name_translation.values())[model]
    # sanity check
    assert plot_format + latex_format <= 1

    # data transformation & auto generation
    print(f"Model being processed: {model}")
    if not plot_format:
      x_ticks, y_ticks = [
        [list(batch_sizes.keys())[i] for i in x_axis_idxs],
        [setting_translation[list(settings.keys())[i]] for i in y_axis_idxs]
      ]
    else:
      x_ticks, y_ticks = [
        [list(batch_sizes.keys())[i] for i in x_axis_idxs],
        [list(settings.keys())[i] for i in y_axis_idxs]
      ]
    print(f"Index selection:\n  X:{x_ticks}\n  Y:{y_ticks}")

    data_slice = data[
        models[model], 
        x_axis_idxs, 
        settings["prefetch_lru"], 
        cpu_mems["128"], 
        ssd_bws["3.2"],
        pcie_bws["15.754"],
        :,
        tuple([stats["exe_time"], stats["ideal_exe_time"]])
    ].astype(float)
    data_slice = data_slice[:, y_axis_idxs]
    speedup_ratio = data_slice[1, :] / data_slice[0, :]
    
    if not header_printed:
      with open(f"figure_drawing/sensitivity_variation/all.txt", "w") as f:
        int_percentage = [f"{int(float(d) * 100)}%" for d in [list(ktime_vars.keys())[v] for v in y_axis_idxs]]
        f.write(" | ".join(int_percentage) + "\n\n")
      header_printed = True

    with open(f"figure_drawing/sensitivity_variation/all.txt", "a") as f:
      f.write(model_desc + "\n")
      f.write(" ".join(str(d) for d in speedup_ratio) + "\n\n")
