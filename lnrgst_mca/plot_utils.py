# from load_utils import load_file
import shutil
from pathlib import Path
from PIL import Image
import nibabel as nib
from nilearn import plotting
from nilearn.datasets import load_mni152_template
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from constants import *


# add this to the load module
def create_directory(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def plotter(data_1, data_2, title, axis_labels=None, labels=None, ylim=None, path=None, ylable=None):

    if labels is None:
        labels = ["PD", "HC"]

    if data_1.ndim == 1:
        data_1 = data_1.reshape(-1, 1)
        data_2 = data_2.reshape(-1, 1)

    dims = data_1.shape[-1]

    if axis_labels is None:
        axis_labels = ["Group"] * dims

    for i in range(dims):
        plt.subplot(1, dims, i + 1)

        # Create a DataFrame for each dataset
        df_1 = pd.DataFrame({axis_labels[i]: labels[0], "Value": data_1[:, i]})

        df_2 = pd.DataFrame({axis_labels[i]: labels[1], "Value": data_2[:, i]})

        # Concatenate the DataFrames
        df = pd.concat([df_1, df_2])

        # Create the swarmplot
        sns.swarmplot(x=axis_labels[i], y="Value", data=df, palette=["orange", "blue"])
        sns.boxplot(x=axis_labels[i], y="Value", data=df, color="white")
        if i == 1 or (i == 0 and dims == 1):
            plt.title(title)
        if ylable:
            plt.ylabel(ylable)

        # Set the y-axis limits
        if ylim is not None:
            plt.ylim(ylim[i])
        # else:
        #     plt.ylim([df["Value"].min() * (0.9), df["Value"].max() * (1.01)])

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        plt.tight_layout()
        if path:
            create_directory(path)
            plt.savefig(path / f"{title}.png")

    plt.show()


def hist_plotter(data_1, data_2, title, labels=None, path=None, bins=[10, 10], s=10, ylable=None):

    if labels is None:
        labels = ["Software 1", "Software 2"]

    if data_1.ndim == 1:
        data_1 = data_1.reshape(-1, 1)
        data_2 = data_2.reshape(-1, 1)

    dims = data_1.shape[-1]

    for i in range(dims):
        plt.subplot(1, dims, i + 1)
        plt.hist(data_1[:, i], color="r", alpha=0.2, bins=bins[0], label=labels[0])
        plt.hist(data_2[:, i], color="b", alpha=0.2, bins=bins[1], label=labels[1])
        median1 = np.median(data_1[:, i])
        median2 = np.median(data_2[:, i])
        plt.axvline(median1, color="r", linestyle="dashed", linewidth=1)
        plt.axvline(median2, color="b", linestyle="dashed", linewidth=1)
        plt.text(median1, plt.ylim()[1] / 2, f"Median: {median1:.2f}", color="r", ha="right", rotation=90)
        plt.text(median2 * s, plt.ylim()[1] / 2, f"Median: {median2:.2f}", color="b", ha="left", rotation=90)
        if i == 1 or (i == 0 and dims == 1):
            plt.title(title)
        plt.legend()

        if ylable:
            plt.ylabel(ylable)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.75, hspace=0.75)
        plt.tight_layout()

        if path:
            create_directory(path)
            plt.savefig(path / f"{title}.png")
        plt.show()


def generate_gif(images, filename, duration=200):
    with Image.open(images[0]) as frame_one:
        frame_one.save(filename, format="GIF", append_images=[Image.open(image) for image in images[1:]], save_all=True, duration=duration, loop=0)


def slice_plotter(
    images_paths,
    title_prefix,
    output_dir,
    template="mni152",
    cut_coords=None,
    levels=[0.6],
    display_mode="ortho",
    colorbar=True,
    threshold=None,
    annotate=True,
    draw_cross=True,
    dim=-0.5,
    **kwargs,
):

    create_directory(output_dir)

    if display_mode == "ortho" and cut_coords is None:
        cut_coords = (3, 3, 3)

    if template == "mni152":
        template = load_mni152_template()
    elif isinstance(template, str):
        template = nib.load(template)
    else:
        raise ValueError("Template must be either 'mni152' or a path to a Nifti file")

    for image_path in images_paths:

        image_path = Path(image_path)
        try:
            img = nib.load(image_path)
            output_file = output_dir / f"{'_'.join([image_path.parent.name, image_path.name])}.png"
            display = plotting.plot_anat(
                img,
                cut_coords=cut_coords,
                display_mode=display_mode,
                colorbar=colorbar,
                threshold=threshold,
                annotate=annotate,
                draw_cross=draw_cross,
                title=f"{title_prefix} {image_path.parent.name}",
                dim=dim,
                **kwargs,
            )

            display.add_contours(template, levels=levels, cut_coords=cut_coords, colors="r")
            display.savefig(output_file)
            display.close()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue


def make_gif(image_paths, subject_name, output_dir, duration=200, **kwargs):

    create_directory(output_dir)
    try:
        output_dir_preprocess = output_dir / f"preprocess_{subject_name}"
        create_directory(output_dir_preprocess)
        slice_plotter(image_paths, subject_name, output_dir_preprocess, **kwargs)

        png_files = list(output_dir_preprocess.glob("*.png"))
        generate_gif(png_files, output_dir / f"{subject_name}.gif", duration=duration)

        shutil.rmtree(output_dir_preprocess)
    except Exception as e:
        print(f"No images found for {subject_name}: {e}")


# add docstring for all of them and handel kwargs here
def QC_plotter(paths_map, output_dir, template=None):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ieee_dir = output_dir / ORIGINAL
    ieee_dir.mkdir(parents=True, exist_ok=True)

    mca_dir = output_dir / MCA
    mca_dir.mkdir(parents=True, exist_ok=True)

    for subject in paths_map.keys():

        subject_dir = mca_dir / subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        slice_plotter(paths_map[subject][MCA], template=template, title_prefix="", output_dir=subject_dir, display_mode="mosaic", levels=[0.4])

        slice_plotter(paths_map[subject][ORIGINAL], template=template, title_prefix="", output_dir=ieee_dir, display_mode="mosaic", levels=[0.4])
