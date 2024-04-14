# from load_utils import load_file
import shutil
from pathlib import Path
from PIL import Image
import nibabel as nib
from nilearn import plotting
from nilearn.datasets import load_mni152_template
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# add this to the load module
def create_directory(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def plotter(data_1, data_2, title, axis_labels=None, labels=None, ylim=None, path=None):

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

        # Set the y-axis limits
        if ylim is not None:
            plt.ylim(ylim[i])
        else:
            plt.ylim([df["Value"].min() * (0.99), df["Value"].max() * (1.03)])

        if path:
            create_directory(path)
            plt.savefig(path / f"{title}.png")

    plt.show()


def generate_gif(images, filename, duration=200):
    with Image.open(images[0]) as frame_one:
        frame_one.save(filename, format="GIF", append_images=[Image.open(image) for image in images[1:]], save_all=True, duration=duration, loop=0)


def process_images(
    images_paths,
    title_prefix,
    output_dir,
    template=load_mni152_template,
    cut_coords=(0, 0, 0),
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
    template = template()

    for image_path in images_paths:

        image_path = Path(image_path)
        try:
            img = nib.load(image_path)
            output_file = output_dir / f"{image_path.parent.name}.png"
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
            # template=template not sure about this
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
        process_images(image_paths, subject_name, output_dir_preprocess, **kwargs)

        png_files = list(output_dir_preprocess.glob("*.png"))
        generate_gif(png_files, output_dir / f"{subject_name}.gif", duration=duration)

        shutil.rmtree(output_dir_preprocess)
    except Exception as e:
        print(f"No images found for {subject_name}: {e}")
