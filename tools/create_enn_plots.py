from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_grid, facet_wrap,
    scale_y_continuous, geom_hline, position_dodge,
    geom_errorbar
)
from plotnine.data import economics
from pandas import DataFrame
from plotnine.scales.limits import ylim
from plotnine.scales.scale_xy import scale_x_discrete
from glob import glob
import re

float_fields = [
    "noise_scale",
    "prior_scale",
    "dropout_rate",
    "regularization_scale",
    "sigma_0",
    "learning_rate",
]
int_fields = [
    "num_ensemble",
    "num_layers",
    "hidden_size",
    "index_dim",
    # "num_batches",
]

def read_data(file):
    with open(file, "r") as f:
        lines = f.readlines()

        agent_frames = {}

        for line in lines:
            id, kl, agent, *params = line.replace('\n', '').split(' ')

            id = int(id)
            kl = float(kl)
            agent = agent.split('=')[1]

            if agent not in agent_frames:
                agent_frames[agent] = {"kl": []}
            
            agent_frames[agent]["kl"].append(min(2, kl))

            for p in params:
                k, v = p.split('=')

                if k in float_fields:
                    v = float(v)
                elif k in int_fields:
                    v = int(v)
                
                # if k == "num_batches":
                #     v //= 1000

                if k not in agent_frames[agent]:
                    agent_frames[agent][k] = []
                
                agent_frames[agent][k].append(v)
        
        for agent in agent_frames.keys():
            agent_frames[agent] = DataFrame(agent_frames[agent])

        return agent_frames

def plot_single_frame(frame, agent, output_file_name):

    if agent in ["vnn"]:
        
        plot = (
            ggplot(frame) + aes(x="num_batches", y="kl") +
            facet_wrap(["activation_mode", "global_std_mode", "num_index_samples"], nrow=2) +
            geom_hline(yintercept = 1) + 
            ylim(0, 2) + 
            geom_point(aes(colour="factor(num_layers)", shape="factor(hidden_size)", fill="activation"), size=3, position=position_dodge(width=0.8), stroke=0.2)
        )
        plot.save(output_file_name, dpi=600)
    else:
        raise ValueError("Unknown agent")

def plot_multiple_frames(frames, agent, output_file_name):
    
    result = frames[0].copy()
    result["kl"]=sum(f["kl"] for f in frames)/ len(frames)
    std = (sum((f["kl"] - result["kl"])**2 for f in frames) / len(frames))**0.5
    result["kl_std"] = std


    if agent in ["vnn"]:
        
        dodge = position_dodge(width=0.8)

        plot = (
            ggplot(result) + aes(x="num_batches", y="kl") +
            facet_wrap(["activation_mode", "global_std_mode", "num_index_samples"], nrow=2) +
            geom_hline(yintercept = 1) + 
            ylim(0, 2) + 
            geom_point(aes(colour="factor(num_layers)", shape="factor(hidden_size)", fill="activation"), size=3, position=dodge, stroke=0.2) +
            geom_errorbar(aes(colour="factor(num_layers)", shape="factor(hidden_size)", fill="activation", ymin="kl-kl_std", ymax="kl+kl_std"), position=dodge, width=0.8)
        )
        plot.save(output_file_name, dpi=600)
    else:
        raise ValueError("Unknown agent")


files = glob("results_vnn_selected*")

def plot_all_single_frames(files):

    for file in files:
        agent_frames = read_data(file)
        for agent in ["vnn"]:
            frame = agent_frames[agent]
                
            plot_single_frame(frame, agent, "enn_plot_" + agent + "_" + file.replace('.txt', '') + ".png")


def plot_all_total_frames(files):

    all_agent_frames = {}
    
    for file in files:
        agent_frames = read_data(file)
        for agent in agent_frames.keys():
            frame = agent_frames[agent]
            
            if agent not in all_agent_frames:
                all_agent_frames[agent] = []
            
            all_agent_frames[agent].append(frame)

    for agent in ["vnn"]:

        frames = all_agent_frames[agent]

        if len(frames) > 0:

            plot_multiple_frames(frames, agent, "total_enn_plot_" + agent + ".png")


def parse_enn_experiment_parameters(file):
    
    param_string = file.split("_")[-1]
    input_dim, data_ratio, noise_std = re.findall(r'\d+(?:\.\d+|\d*)', param_string)

    input_dim = int(input_dim)
    data_ratio = float(data_ratio)
    noise_std = float(noise_std)

    return {
        "input_dim": input_dim,
        "data_ratio": data_ratio,
        "noise_std": noise_std,
    }


def plot_all_hyperexperiment_frames(files, parse_experiment_parameters=parse_enn_experiment_parameters):

    all_experiment_agent_frames = {}
    
    for file in files:
        agent_frames = read_data(file)
        for agent in agent_frames.keys():
            frame = agent_frames[agent]

            experiment_params = parse_experiment_parameters(file)

            for name, value in experiment_params.items():
                key = str(name) + ":" + str(value)
                if key not in all_experiment_agent_frames:
                    all_experiment_agent_frames[key] = {}
            
                if agent not in all_experiment_agent_frames[key]:
                    all_experiment_agent_frames[key][agent] = []
            
                all_experiment_agent_frames[key][agent].append(frame)

    for experiment_param, all_agent_frames in all_experiment_agent_frames.items(): 
        for agent in ["vnn"]:
            frames = all_agent_frames[agent]
            if len(frames) > 0:
                plot_multiple_frames(frames, agent, "hyperexperiment_enn_plot_" + experiment_param + "_" + agent + ".png")


plot_all_hyperexperiment_frames(files)