import src.utils.images as _image_utils
import imagecorruptions
import os
import pathlib
import shutil
import typing

_SUPPORTED_EXTENSIONS = ('png', 'jpg', 'jpeg', 'gif', 'ppm', 'tga')


def corrupt_dataset(original_ds_path: str, corrupted_ds_path: str, corruption_name: str,
                    corruption_severity: int) -> None:
    """Creates a copy of the dataset and corrupts it with the given corruption and severity

    :param original_ds_path: Path to the dataset to corrupt
    :param corrupted_ds_path: Path where the newly generated, corrupted dataset will be stored. Path mustn't exist yet
    :param corruption_name: Name of the corruption to apply
    :param corruption_severity: Severity (1-5) of the corruption
    :return: None
    """
    # Check that parameters are valid
    if corruption_name not in imagecorruptions.get_corruption_names():
        raise Exception(f'"{corruption_name}" is not a valid corruption.\n'
                        f'Available options are: {", ".join(imagecorruptions.get_corruption_names())}')
    elif corruption_severity not in range(1, 6):
        raise Exception(
            f'Corruption severity must be between 1 and 5 (was {corruption_severity}).')

    # Get paths to all images in the dataset
    image_paths = []
    for extension in _SUPPORTED_EXTENSIONS:
        image_paths.extend(
            list(pathlib.Path(original_ds_path).rglob(f'*.{extension}')))
        image_paths.extend(
            list(pathlib.Path(original_ds_path).rglob(f'*.{extension.upper()}')))

    for image_path in image_paths:
        # Define the path for the corrupted image
        corrupted_image_path = pathlib.Path(
            corrupted_ds_path, pathlib.Path(image_path).relative_to(original_ds_path))

        # Create the directory structure for the corrupted image in the corrupted dataset
        os.makedirs(corrupted_image_path.parent, exist_ok=True)

        # Corrupt and save the image
        _generate_corrupted_copy_of_image(
            image_path, corrupted_image_path, corruption_name, corruption_severity)


def _generate_corrupted_copy_of_image(orig_image_path: typing.Union[str, pathlib.Path],
                                      corrupted_image_path: typing.Union[str, pathlib.Path],
                                      corruption_name: str, corruption_severity: int) -> None:
    """Create a copy of the image at the specified location and corrupts it with the given corruption and severity

    :param orig_image_path: Path to the image to corrupt
    :param corrupted_image_path: Path where the corrupted image will be stored
    :param corruption_name: Name of the corruption to apply
    :param corruption_severity: Severity (1-5) of the corruption
    :return: None
    """
    image_array, orig_size = _image_utils.load_image_as_numpy(orig_image_path)
    corrupted_image_array = imagecorruptions.corrupt(
        image_array, corruption_severity, corruption_name)
    _image_utils.save_numpy_as_image(
        corrupted_image_array, corrupted_image_path, orig_size)
