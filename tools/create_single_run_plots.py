from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    facet_grid,
    facet_wrap,
    scale_y_continuous,
    geom_hline,
    position_dodge,
    geom_errorbar,
)
from plotnine.data import economics
from pandas import DataFrame
from plotnine.scales.limits import ylim
from plotnine.scales.scale_xy import scale_x_discrete
from glob import glob
import cv2
import numpy as np
import os

# files = glob("single_run_*")
files = glob("single_run_init_std_1_*")

float_fields = [
    "loss",
    "kl_estimate",
    "mean_error",
    "std_error",
]
int_fields = [
    "step",
]
int_list_fields = []


def read_data(file):
    with open(file, "r") as f:
        lines = f.readlines()

        trainig_data = {}
        testing_data = {}

        last_step = 0

        for line in lines:

            current_dict = {}

            params = line.replace("\n", "").split(" ")

            f = []
            for p in params:
                if "=" in p:
                    f.append(p)
                else:
                    f[-1] += " " + p
            raw_params = f

            params = []

            for p in raw_params:
                k, v = p.split("=")
                if k == "agent":
                    agent = v
                else:
                    params.append(p)

            for p in params:
                k, v = p.split("=")

                if k in float_fields:
                    v = float(v)
                elif k in int_fields:
                    v = int(v)
                elif k in int_list_fields:
                    v = int(v.split("]")[0].split(" ")[-1])

                current_dict[k] = v

            target = None

            if "kl_estimate" in current_dict:
                target = testing_data
                current_dict["step"] = last_step
            else:
                target = trainig_data
                last_step = current_dict["step"]

            for k, v in current_dict.items():
                if k not in target:
                    target[k] = []

                target[k].append(v)

        return DataFrame(trainig_data), DataFrame(testing_data)


def vstack(img_list):
    max_width = 0
    total_height = 200  # padding
    for img in img_list:
        if img.shape[1] > max_width:
            max_width = img.shape[1]
        total_height += img.shape[0]

    # create a new array with a size large enough to contain all the images
    final_image = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in img_list:
        # add an image to the final array and increment the y coordinate
        image = np.hstack(
            (image, 255 * np.ones((image.shape[0], max_width - image.shape[1], 3)))
        )
        final_image[current_y : current_y + image.shape[0], :, :] = image
        current_y += image.shape[0]
    return final_image

def hstack(img_list):
    max_height = 0
    total_width = 200  # padding
    for img in img_list:
        if img.shape[0] > max_height:
            max_height = img.shape[0]
        total_width += img.shape[1]

    # create a new array with a size large enough to contain all the images
    final_image = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255

    current_x = 0  # keep track of where your current image was last placed in the y coordinate
    for image in img_list:
        # add an image to the final array and increment the y coordinate
        image = np.vstack(
            (image, 255 * np.ones((max_height - image.shape[0], image.shape[1], 3)))
        )
        final_image[:, current_x : current_x + image.shape[1], :] = image
        current_x += image.shape[1]
    return final_image


def plot_single_frame(training_frame, testing_frame, output_file_name):

    plot_train = ggplot(training_frame) + aes(x="step", y="loss") + geom_line()
    plot_train.save(
        "plots/single/sub/" + output_file_name + "_train.png", dpi=600
    )
    plot_test_kl = (
        ggplot(testing_frame) + aes(x="step", y="kl_estimate") + geom_line()
    )
    plot_test_kl.save(
        "plots/single/sub/" + output_file_name + "_test_kl.png", dpi=600
    )
    plot_test_mean = (
        ggplot(testing_frame) + aes(x="step", y="mean_error") + geom_line()
    )
    plot_test_mean.save(
        "plots/single/sub/" + output_file_name + "_test_mean.png", dpi=600
    )
    plot_test_std = (
        ggplot(testing_frame) + aes(x="step", y="std_error") + geom_line()
    )
    plot_test_std.save(
        "plots/single/sub/" + output_file_name + "_test_std.png", dpi=600
    )

    img_train = cv2.imread(
        "plots/single/sub/" + output_file_name + "_train.png"
    )
    img_kl = cv2.imread("plots/single/sub/" + output_file_name + "_test_kl.png")
    img_mean = cv2.imread(
        "plots/single/sub/" + output_file_name + "_test_mean.png"
    )
    img_std = cv2.imread(
        "plots/single/sub/" + output_file_name + "_test_std.png"
    )

    total_img = hstack(
        [
            vstack([img_train, img_kl]),
            vstack([img_mean, img_std]),
        ]
    )

    cv2.imwrite("plots/single/" + output_file_name + ".png", total_img)

    print()


def plot_all_single_frames(files):

    os.makedirs("plots/single/sub/", exist_ok=True)

    for file in files:
        training_frame, testing_frame = read_data(file)
        plot_single_frame(
            training_frame,
            testing_frame,
            "single_plot_" + file.replace(".txt", ""),
        )


# plot_all_hyperexperiment_frames(files)
plot_all_single_frames(files)
