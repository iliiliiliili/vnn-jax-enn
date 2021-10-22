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

# files = glob("results_vnn_selected*")
files = glob("results_id*")
# files =  glob("results_vnn_selected*") +  glob("results_f*")

float_fields = [
    "noise_scale",
    "prior_scale",
    "dropout_rate",
    "regularization_scale",
    "sigma_0",
    "learning_rate",
    "mean_error",
    "std_error",
]
int_fields = [
    "num_ensemble",
    "num_layers",
    "hidden_size",
    "index_dim",
    # "num_batches",
]
int_list_fields = [
    "num_ensembles",
]

agent_plot_params = {
    "ensemble": {
        "x": "num_ensemble",
        "y": "kl",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "dropout": {
        "x": "dropout_rate",
        "y": "kl",
        "facet": ["regularization_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "hypermodel": {
        "x": "index_dim",
        "y": "kl",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "bbb": {
        "x": "sigma_0",
        "y": "kl",
        "facet": ["learning_rate"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "vnn": {
        "x": "num_batches",
        "y": "kl",
        "facet": ["activation_mode", "global_std_mode"],
        # "facet": ["activation_mode", "global_std_mode", "num_index_samples"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
        "fill": "activation",
    },
    "layer_ensemble": {
        "x": "num_ensembles",
        "y": "kl",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
}


def read_data(file):
    with open(file, "r") as f:
        lines = f.readlines()

        agent_frames = {}

        for line in lines:
            id, kl, agent, *params = line.replace("\n", "").split(" ")

            f = []
            for p in params:
                if "=" in p:
                    f.append(p)
                else:
                    f[-1] += " " + p
            params = f

            id = int(id)
            kl = float(kl)
            agent = agent.split('=')[1]

            if agent not in agent_frames:
                agent_frames[agent] = {"kl": []}
            
            agent_frames[agent]["kl"].append(kl)
            # agent_frames[agent]["kl"].append(min(2, kl))

            for p in params:
                k, v = p.split('=')

                if k in float_fields:
                    v = float(v)
                elif k in int_fields:
                    v = int(v)
                elif k in int_list_fields:
                    v = int(v.split("]")[0].split(" ")[-1])
                
                # if k == "num_batches":
                #     v //= 1000

                if k not in agent_frames[agent]:
                    agent_frames[agent][k] = []
                
                agent_frames[agent][k].append(v)
        
        for agent in agent_frames.keys():
            agent_frames[agent] = DataFrame(agent_frames[agent])

        return agent_frames

def plot_single_frame(frame, agent, output_file_name):

    params = agent_plot_params[agent]

    point_aes_params = {}

    for key in ["colour", "shape", "fill"]:
        if key in params:
            point_aes_params[key] = params[key]

    plot = (
        ggplot(frame) + aes(x=params["x"], y=params["y"]) +
        facet_wrap(params["facet"], nrow=2, labeller="label_both") +
        geom_hline(yintercept = 1) + 
        ylim(0, 2) + 
        geom_point(aes(**point_aes_params), size=3, position=position_dodge(width=0.8), stroke=0.2)
    )
    plot.save(output_file_name, dpi=600)

def plot_multiple_frames(frames, agent, output_file_name):

    params = agent_plot_params[agent]
    
    result = frames[0].copy()
    result[params["y"]]=sum(f[params["y"]] for f in frames)/ len(frames)
    std = (sum((f[params["y"]] - result[params["y"]])**2 for f in frames) / len(frames))**0.5
    result[params["y"] + "_std"] = std

    point_aes_params = {}

    for key in ["colour", "shape", "fill"]:
        if key in params:
            point_aes_params[key] = params[key]

    dodge = position_dodge(width=0.8)

    plot = (
        ggplot(result) + aes(x=params["x"], y=params["y"]) +
        facet_wrap(params["facet"], nrow=2, labeller="label_both") +
        geom_hline(yintercept = 1) + 
        ylim(0, 2) + 
        geom_point(aes(**point_aes_params), size=3, position=position_dodge(width=0.8), stroke=0.2) +
        geom_errorbar(aes(
            **point_aes_params,
            ymin=params["y"] + "-" + params["y"] + "_std",
            ymax=params["y"] + "+" + params["y"] + "_std"
        ), position=dodge, width=0.8)
    )
    plot.save(output_file_name, dpi=600)


def plot_all_single_frames(files):

    for file in files:
        agent_frames = read_data(file)
        for agent, frame in agent_frames.items():
                
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

    for agent, frames in all_agent_frames.items():
        if len(frames) > 0:
            plot_multiple_frames(frames, agent, "total_enn_plot_" + agent + ".png")


def plot_summary(files):

    all_agent_frames = {}
    
    for file in files:
        agent_frames = read_data(file)
        for agent in agent_frames.keys():
            frame = agent_frames[agent]
            
            if agent not in all_agent_frames:
                all_agent_frames[agent] = []
            
            all_agent_frames[agent].append(frame)

    data = {
        "agent": [],
        "mean": [],
        "std": [],
    }

    for agent, frames in all_agent_frames.items():
        
        params = agent_plot_params[agent]
        mean = sum(sum(f[params["y"]]) for f in frames)/sum(len(f) for f in frames)
        std = sum(sum((f[params["y"]]-mean)**2) for f in frames)/sum(len(f) for f in frames)

        data["agent"].append(agent)
        data["mean"].append(mean)
        data["std"].append(std)

    frame = DataFrame(data)

    plot = (
        ggplot(frame) + aes(x="agent", y="mean") +
        geom_hline(yintercept = 1) + 
        # ylim(0, 2) + 
        scale_y_continuous(trans = "log10") +
        geom_point(aes(colour="agent"), size=3, stroke=0.2) +
        geom_errorbar(aes(
            colour="agent",
            ymin="mean-std",
            ymax="mean+std"
        ), width=0.8)
    )
    plot.save("summary_enn_plot.png", dpi=600)
    frame.to_csv("summary_enn.csv")


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
        for agent, frames in all_agent_frames.items():
            if len(frames) > 0:
                plot_multiple_frames(frames, agent, "hyperexperiment_enn_plot_" + experiment_param + "_" + agent + ".png")


plot_summary(files)
# plot_all_hyperexperiment_frames(files)
# plot_all_single_frames(files)