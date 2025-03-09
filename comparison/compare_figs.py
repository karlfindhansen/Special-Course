from PIL import Image
import os

folder = os.path.join("comparison", "figures")

image_files = [os.path.join(folder, f"output_{i}.png") for i in range(121)]

def create_mosaic(image_files, output_path="mosaic.png"):
    """
    Creates a mosaic image from a list of image files.

    Args:
        image_files: A list of image file paths.
        output_path: The path to save the resulting mosaic image.
    """

    num_images = len(image_files)
    if num_images != 121:
        raise ValueError("Expected 121 images, but got {}".format(num_images))

    image_size = (36, 36)
    mosaic_size = (image_size[0] * 11, image_size[1] * 11)

    mosaic = Image.new("RGB", mosaic_size)

    for i, image_file in enumerate(image_files):
        try:
            img = Image.open(image_file).convert("RGB") # Ensure RGB mode
            print(f"Image size: {img.size}")
            img.show()
            img = img.resize((image_size[0], image_size[1]))
            img.show()
            exit()
            if img.size != image_size:
                raise ValueError(f"Image {image_file} has incorrect size: {img.size}, expected {image_size}")
        except FileNotFoundError:
            print(f"Warning: Image file not found: {image_file}")
            continue # Skip if file not found
        except Exception as e:
            print(f"Warning: Error processing image {image_file}: {e}")
            continue

        row = i // 11
        col = i % 11
        x = col * image_size[0]
        y = row * image_size[1]
        mosaic.paste(img, (x, y))

    mosaic.save(output_path)
    print(f"Mosaic image saved to {output_path}")

# Example usage (assuming image_files is defined and 'folder' is set):
# folder = "your_image_folder" # replace with your folder path
# image_files = [os.path.join(folder, f"output_{i}.png") for i in range(121)]
create_mosaic(image_files)