# from load_utils import load_file
import shutil
from pathlib import Path
from PIL import Image
import nibabel as nib
from nilearn import plotting
from nilearn.datasets import load_mni152_template


# add this to the load module
def create_directory(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


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
                dim=dim**kwargs,
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
        output_dir_preprocess = output_dir / "preprocess"
        create_directory(output_dir_preprocess)
        process_images(image_paths, subject_name, output_dir_preprocess, **kwargs)

        png_files = list(output_dir_preprocess.glob("*.png"))
        generate_gif(png_files, output_dir / f"{subject_name}.gif", duration=duration)

        shutil.rmtree(output_dir_preprocess)
    except Exception as e:
        print(f"No images found for {subject_name}: {e}")
