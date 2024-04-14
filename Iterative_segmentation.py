from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage import color as skimage_color
from skimage import filters
from skimage import morphology
import os
import shutil
from PIL import Image, ImageDraw
import matplotlib.colors as mcolors
from scipy.ndimage import label
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.5)))


def process_image_with_contour_method(img, area_threshold=200):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_impurities = [c for c in contours if cv2.contourArea(c) < area_threshold]
    cv2.drawContours(img, small_impurities, -1, (255, 255, 255), thickness=cv2.FILLED)
    return img

def process_image_skimage(image_opencv, min_size=1000, color_threshold=[200, 200, 200]):
    image_rgb = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)
    gray_sk = skimage_color.rgb2gray(image_rgb)
    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value
    cleaned = morphology.remove_small_objects(binary, min_size=min_size)
    mask = np.where(cleaned == 0, 1, 0).astype(bool)
    image_rgb[mask] = [255, 255, 255]
    whitening_mask_sk = np.all(image_rgb > color_threshold, axis=-1)
    image_rgb[whitening_mask_sk] = [255, 255, 255]
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def resize_image(i):
    image_path = f'M reg/{i}.jpg'
    with Image.open(image_path) as img:
        if img.size != (1000, 1000):
            img = img.resize((1000, 1000), Image.Resampling.LANCZOS)
            img.save(image_path)


def remove_files(run_folder_path):
    if os.path.exists(run_folder_path) and os.path.isdir(run_folder_path):
        files_and_folders = os.listdir(run_folder_path)

        if files_and_folders:
            for item in files_and_folders:
                item_path = os.path.join(run_folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print("All files have been deleted.")
        else:
            print("The Run folder is empty and there are no files to delete.")
    else:
        print("The Run folder does not exist.")


def check_colored_area_proportion(file_path):
    image = Image.open(file_path)
    image_np = np.array(image)
    if image_np.shape[2] == 4:
        alpha_channel = image_np[:, :, 3]
    else:
        alpha_channel = np.full(image_np.shape[:2], 255)

    colored_pixels = (np.any(image_np[:, :, :3] != 255, axis=2) | (alpha_channel < 255))

    colored_proportion = np.sum(colored_pixels) / colored_pixels.size
    return colored_proportion >= 0.05


def round_1(i):
    image_path = f'M reg/{i}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    masks = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks]

    masks = np.stack(masks)
    np.save(f'Run/masks_{i}_1.npy', masks)

    npy_path = f'Run/masks_{i}_1.npy'
    npy_array = np.load(npy_path)

    dilated_array = np.zeros_like(npy_array)

    for j in range(npy_array.shape[0]):
        dilated_array[j] = binary_dilation(npy_array[j], iterations=3)
    two_d_array = np.max(dilated_array, axis=0)

    np.save(f'Run/masks_{i}_1_2d.npy', two_d_array)

    image_path = f'M reg/{i}.jpg'
    image = cv2.imread(image_path)

    masks_path = f'Run/masks_{i}_1_2d.npy'
    masks = np.load(masks_path)

    image[masks != 0] = [255, 255, 255]

    output_path = f'Run/{i}_upgrade_1.jpg'
    cv2.imwrite(output_path, image)

    img_path = f'Run/{i}_upgrade_1.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    img_no_residual = process_image_with_contour_method(img)

    img_whitened = process_image_skimage(img_no_residual)

    output_path = f'Run/{i}_upgrade_1.jpg'
    cv2.imwrite(output_path, img_whitened)

def round_2(i):
    image_path = f'Run/{i}_upgrade_1.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    masks = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks]
    masks = np.stack(masks)
    np.save(f'Run/masks_{i}_2.npy', masks)
    image_path = f'Run/{i}_upgrade_1.jpg'
    mask_path = f'Run/masks_{i}_2.npy'

    image = cv2.imread(image_path)

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"No mask file found at {mask_path}")
    masks = np.load(mask_path)
    remaining_masks = []

    for current_mask in masks:
        mask_indices = np.where(current_mask == 1)
        masked_image = image[mask_indices]

        white_pixels = np.sum(np.all(masked_image == [255, 255, 255], axis=-1))

        white_percentage = (white_pixels / masked_image.shape[0]) * 100 if masked_image.shape[0] > 0 else 0

        if white_percentage < 50:
            remaining_masks.append(current_mask)

    if remaining_masks:
        remaining_masks = np.array(remaining_masks)
        new_mask_path = mask_path.replace('.npy', '.npy')
        np.save(new_mask_path, remaining_masks)
        print(f"Remaining masks saved to {new_mask_path}")
    else:
        print("No masks remaining after filtering.")

    npy_path = f'Run/masks_{i}_2.npy'
    npy_array = np.load(npy_path)

    dilated_array = np.zeros_like(npy_array)

    for j in range(npy_array.shape[0]):
        dilated_array[j] = binary_dilation(npy_array[j], iterations=3)

    two_d_array = np.max(dilated_array, axis=0)
    np.save(f'Run/masks_{i}_2_2d.npy', two_d_array)
    image_path = f'Run/{i}_upgrade_1.jpg'
    image = cv2.imread(image_path)

    masks_path = f'Run/masks_{i}_2_2d.npy'
    masks = np.load(masks_path)

    if image.shape[:2] != masks.shape:
        raise ValueError("The shape of the masks does not match the shape of the image")

    image[masks != 0] = [255, 255, 255]

    output_path = f'Run/{i}_upgrade_2.jpg'
    cv2.imwrite(output_path, image)

    img_path = f'Run/{i}_upgrade_2.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_no_residual = process_image_with_contour_method(img)
    img_whitened = process_image_skimage(img_no_residual)
    output_path = f'Run/{i}_upgrade_2.jpg'
    cv2.imwrite(output_path, img_whitened)

def round_3(i):
    image_path = f'Run/{i}_upgrade_2.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    masks = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks]
    masks = np.stack(masks)
    np.save(f'Run/masks_{i}_3.npy', masks)

    image_path = f'Run/{i}_upgrade_2.jpg'
    mask_path = f'Run/masks_{i}_3.npy'

    image = cv2.imread(image_path)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"No mask file found at {mask_path}")
    masks = np.load(mask_path)
    remaining_masks = []

    for current_mask in masks:
        mask_indices = np.where(current_mask == 1)
        masked_image = image[mask_indices]
        white_pixels = np.sum(np.all(masked_image == [255, 255, 255], axis=-1))
        white_percentage = (white_pixels / masked_image.shape[0]) * 100 if masked_image.shape[0] > 0 else 0
        if white_percentage < 90:
            remaining_masks.append(current_mask)

    if remaining_masks:
        remaining_masks = np.array(remaining_masks)
        new_mask_path = mask_path.replace('.npy', '.npy')
        np.save(new_mask_path, remaining_masks)
        print(f"Remaining masks saved to {new_mask_path}")
    else:
        print("No masks remaining after filtering.")

    npy_path = f'Run/masks_{i}_3.npy'
    npy_array = np.load(npy_path)
    dilated_array = np.zeros_like(npy_array)
    for j in range(npy_array.shape[0]):
        dilated_array[j] = binary_dilation(npy_array[j], iterations=3)
    two_d_array = np.max(dilated_array, axis=0)

    np.save(f'Run/masks_{i}_3_2d.npy', two_d_array)

    image_path = f'Run/{i}_upgrade_2.jpg'
    image = cv2.imread(image_path)
    masks_path = f'Run/masks_{i}_3_2d.npy'
    masks = np.load(masks_path)
    if image.shape[:2] != masks.shape:
        raise ValueError("The shape of the masks does not match the shape of the image")
    image[masks != 0] = [255, 255, 255]
    output_path = f'Run/{i}_upgrade_3.jpg'
    cv2.imwrite(output_path, image)

    img_path = f'Run/{i}_upgrade_3.jpg'
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=3)
    mask = np.zeros_like(img)

    mask[edges_dilated == 255] = (255, 255, 255)

    blurred_edges = cv2.GaussianBlur(mask, (21, 21), sigmaX=0, sigmaY=0)

    img_with_blurred_edges = np.where(blurred_edges == (255, 255, 255), blurred_edges, img)

    output_path = f'Run/{i}_upgrade_3.jpg'
    cv2.imwrite(output_path, img_with_blurred_edges)

    img_path = f'Run/{i}_upgrade_3.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_no_residual = process_image_with_contour_method(img)

    img_whitened = process_image_skimage(img_no_residual)
    output_path = f'Run/{i}_upgrade_3.jpg'
    cv2.imwrite(output_path, img_whitened)

def round_4(i):
    image_path = f'Run/{i}_upgrade_3.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    masks = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks]
    masks = np.stack(masks)
    np.save(f'Run/masks_{i}_4.npy', masks)

    image_path = f'Run/{i}_upgrade_3.jpg'
    mask_path = f'Run/masks_{i}_4.npy'

    image = cv2.imread(image_path)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"No mask file found at {mask_path}")
    masks = np.load(mask_path)
    remaining_masks = []
    for current_mask in masks:
        mask_indices = np.where(current_mask == 1)
        masked_image = image[mask_indices]
        white_pixels = np.sum(np.all(masked_image == [255, 255, 255], axis=-1))
        white_percentage = (white_pixels / masked_image.shape[0]) * 100 if masked_image.shape[0] > 0 else 0
        if white_percentage < 90:
            remaining_masks.append(current_mask)
    if remaining_masks:
        remaining_masks = np.array(remaining_masks)
        new_mask_path = mask_path.replace('.npy', '.npy')
        np.save(new_mask_path, remaining_masks)
        print(f"Remaining masks saved to {new_mask_path}")
    else:
        print("No masks remaining after filtering.")

    npy_path = f'Run/masks_{i}_4.npy'
    npy_array = np.load(npy_path)

    dilated_array = np.zeros_like(npy_array)
    for j in range(npy_array.shape[0]):
        dilated_array[j] = binary_dilation(npy_array[j], iterations=3)

    two_d_array = np.max(dilated_array, axis=0)

    np.save(f'Run/masks_{i}_4_2d.npy', two_d_array)

    image_path = f'Run/{i}_upgrade_3.jpg'
    image = cv2.imread(image_path)

    masks_path = f'Run/masks_{i}_4_2d.npy'
    masks = np.load(masks_path)

    if image.shape[:2] != masks.shape:
        raise ValueError("The shape of the masks does not match the shape of the image")
    image[masks != 0] = [255, 255, 255]

    output_path = f'Run/{i}_upgrade_4.jpg'
    cv2.imwrite(output_path, image)

    img_path = f'Run/{i}_upgrade_4.jpg'
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=3)
    mask = np.zeros_like(img)
    mask[edges_dilated == 255] = (255, 255, 255)
    blurred_edges = cv2.GaussianBlur(mask, (21, 21), sigmaX=0, sigmaY=0)
    img_with_blurred_edges = np.where(blurred_edges == (255, 255, 255), blurred_edges, img)
    output_path = f'Run/{i}_upgrade_4.jpg'
    cv2.imwrite(output_path, img_with_blurred_edges)

    img_path = f'Run/{i}_upgrade_4.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_no_residual = process_image_with_contour_method(img)

    img_whitened = process_image_skimage(img_no_residual)
    output_path = f'Run/{i}_upgrade_4.jpg'
    cv2.imwrite(output_path, img_whitened)


def combine(i):
    base_path = 'Run'

    mask_files = [
        f'{base_path}/masks_{i}_1.npy',
        f'{base_path}/masks_{i}_2.npy',
        f'{base_path}/masks_{i}_3.npy',
        f'{base_path}/masks_{i}_4.npy'
    ]
    shapes = []
    existing_mask_files = []
    for mask_file in mask_files:
        if os.path.exists(mask_file):
            masks = np.load(mask_file)
            shapes.append(masks.shape[0])
            existing_mask_files.append(mask_file)
        else:
            print(f"Warning: Mask file {mask_file} not found and will be skipped.")
    if existing_mask_files:
        max_layers = max(shapes)

        combined_mask = np.zeros((max_layers, 1000, 1000), dtype=np.uint8)

        for mask_file in existing_mask_files:
            masks = np.load(mask_file)
            for j in range(masks.shape[0]):
                combined_mask[j] = np.maximum(combined_mask[j], masks[j])

        combined_mask_file = f'{base_path}/masks_{i}_5.npy'
        np.save(combined_mask_file, combined_mask)


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0


def delete_iou():
    masks = np.load(f'Run/masks_{i}_5.npy')
    while True:
        to_delete = []
        for layer_idx in range(masks.shape[0]):
            if layer_idx in to_delete:
                continue
            current_mask = masks[layer_idx]
            for other_idx in range(masks.shape[0]):
                if layer_idx == other_idx or other_idx in to_delete:
                    continue
                iou = calculate_iou(current_mask, masks[other_idx])
                if iou > 0.8:
                    if np.sum(current_mask) <= np.sum(masks[other_idx]):
                        to_delete.append(layer_idx)
                        print(f"Layer {layer_idx} deleted due to IoU > 80% with layer {other_idx}")
                        break
                    else:
                        to_delete.append(other_idx)
                        print(f"Layer {other_idx} deleted due to IoU > 80% with layer {layer_idx}")

        if not to_delete:
            break
        to_delete = np.unique(to_delete)
        masks = np.delete(masks, to_delete, axis=0)

    new_mask_file = f'Run/masks_{i}_6.npy'
    np.save(new_mask_file, masks)


def split_connected_regions_and_save(mask_path, output_path):
    mask = np.load(mask_path)
    new_layers = []

    for layer in mask:
        labeled_array, num_features = label(layer)

        for i in range(1, num_features + 1):
            new_layer = (labeled_array == i)
            new_layers.append(new_layer)
    new_layers_array = np.array(new_layers, dtype=mask.dtype)

    np.save(output_path, new_layers_array)


def remove_layers_with_few_pixels(mask_path, output_path, threshold=50):
    mask = np.load(mask_path)
    retained_layers = []
    for layer in mask:
        if np.count_nonzero(layer) >= threshold:
            retained_layers.append(layer)
    if retained_layers:
        retained_layers_array = np.array(retained_layers)
        np.save(output_path, retained_layers_array)


def apply_colored_mask(image, mask, color, alpha=0.35):
    foreground = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == 1:
                foreground.putpixel((x, y), color + (int(255 * alpha),))

    return Image.alpha_composite(image.convert("RGBA"), foreground)


def draw(i):
    extended_colors = ['orange', 'red', 'gold', 'skyblue', 'green', 'purple', 'pink', 'brown',
                       'blue', 'lime', 'teal', 'maroon', 'navy', 'olive', 'gray', 'cyan', 'magenta', 'yellowgreen']
    rgba_colors = [tuple(int(255 * x) for x in mcolors.to_rgb(color)) for color in extended_colors]

    image_path = f'M reg/{i}.jpg'
    mask_file = f'Run/masks_{i}_6.npy'
    image = Image.open(image_path)
    masks = np.load(mask_file)

    for idx, mask in enumerate(masks):
        color = rgba_colors[idx % len(rgba_colors)]
        image = apply_colored_mask(image, mask, color)
    output_path = f'Run/masked_{i}.png'
    image.save(output_path)


def visualize_colored_masks_with_custom_borders(image_path, npy_path, colors, border_color, border_width, border_alpha, save_path):
    original_image = plt.imread(image_path)
    processed_layers = np.load(npy_path)

    cmap = ListedColormap([(plt.cm.colors.to_rgba(c, alpha=0.45)) for c in colors])

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.imshow(original_image)

    for j, layer in enumerate(processed_layers):
        color = colors[j % len(colors)]
        rgba_color = plt.cm.colors.to_rgba(color, alpha=0.45)

        overlay = np.zeros((*layer.shape, 4))
        overlay[layer > 0] = rgba_color

        ax.imshow(overlay, extent=(0, original_image.shape[1], original_image.shape[0], 0))

        boundaries = find_boundaries(layer, mode='thick').astype(np.uint8)
        boundary_overlay = np.zeros((*layer.shape, 4))
        boundary_overlay[boundaries > 0, :3] = np.array(border_color) / 255.0
        boundary_overlay[boundaries > 0, 3] = border_alpha
        for _ in range(border_width - 1):
            boundaries = binary_dilation(boundaries)
        boundary_overlay[boundaries > 0, 3] = border_alpha

        ax.imshow(boundary_overlay, extent=(0, original_image.shape[1], original_image.shape[0], 0))

    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()


def main():

    run_folder_path = 'Run'

    remove_files(run_folder_path)

    resize_image(i)
    round_1(i)

    if check_colored_area_proportion(f'{run_folder_path}/{i}_upgrade_1.jpg'):
        round_2(i)

        if check_colored_area_proportion(f'{run_folder_path}/{i}_upgrade_2.jpg'):
            round_3(i)

            if check_colored_area_proportion(f'{run_folder_path}/{i}_upgrade_3.jpg'):
                round_4(i)
    combine(i)
    delete_iou()
    npy_path = f'Run/masks_{i}_6.npy'
    split_connected_regions_and_save(npy_path, npy_path)
    remove_layers_with_few_pixels(npy_path, npy_path)

    # draw(i)

    image_path = f'M reg/{i}.jpg'
    npy_path = f'Run/masks_{i}_6.npy'
    save_path = f'Run/{i}.png'
    colors = ['orange', 'red', 'gold', 'red', 'skyblue', 'green', 'purple', 'pink', 'brown']
    border_color = (100, 115, 232)
    border_width = 1
    border_alpha = 0.8
    visualize_colored_masks_with_custom_borders(image_path, npy_path, colors, border_color, border_width, border_alpha, save_path)

if __name__ == '__main__':
    i = '0017'

    # sam_checkpoint = "sam_vit_h_4b8939.pth"
    # sam_checkpoint = "sam_vit_b_01ec64.pth"
    sam_checkpoint = "sam_vit_l_0b3195.pth"
    # model_type = "vit_b"
    model_type = "vit_l"
    # model_type = "default"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    main()