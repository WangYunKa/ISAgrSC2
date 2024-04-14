from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from PIL import Image
from scipy.ndimage import label
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries
import shutil

def filter_and_save_layers(input_mask_path, output_mask_path, min_pixels=10000):
    mask = np.load(input_mask_path)
    filtered_layers = []

    for layer in mask:
        if np.count_nonzero(layer) > min_pixels:
            filtered_layers.append(layer)

    if filtered_layers:
        filtered_layers_array = np.array(filtered_layers)
        np.save(output_mask_path, filtered_layers_array)
        print(f"Filtered mask saved to {output_mask_path}. Total retained layers: {len(filtered_layers)}")
    else:
        print("No layers retained based on the specified pixel count threshold.")


def apply_mask_and_detect_edges_for_all_layers(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask = np.load(mask_path)
    mask = mask.astype(bool)

    layers_with_high_ratio = []

    for layer_index in range(mask.shape[0]):
        layer_mask = mask[layer_index]

        masked_image = np.array(image)
        masked_image[~layer_mask] = 255

        gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 100)

        edge_pixels_total = np.count_nonzero(edges)

        mask_edges = np.zeros_like(gray)
        contours, _ = cv2.findContours(layer_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_edges, contours, -1, (255), 3)

        overlap = cv2.bitwise_and(edges, mask_edges)
        overlap_edge_pixels = np.count_nonzero(overlap)

        if edge_pixels_total > 0:  # 计算比值并检查是否大于0.4
            ratio = (edge_pixels_total - overlap_edge_pixels) / edge_pixels_total
            if ratio > 0.4:
                layers_with_high_ratio.append(layer_index)
                print(f"Layer {layer_index}: Ratio of difference to total edge pixels = {ratio:.4f}")
    print("Layers with ratio of difference to total edge pixels greater than 0.4:", layers_with_high_ratio)

    return layers_with_high_ratio


def evaluate_color_distribution(image_array):
    color = ('r', 'g', 'b')
    metrics = {}

    mask = np.all(image_array != [255, 255, 255], axis=-1)
    filtered_image = image_array[mask]

    if filtered_image.size == 0:
        print("No non-white pixels found.")
        return metrics

    filtered_image = filtered_image.reshape(-1, 3)
    filtered_image = np.expand_dims(filtered_image, 1)

    total_nonzero_bins = 0
    total_entropy = 0
    for i, col in enumerate(color):
        histogram = cv2.calcHist([filtered_image], [i], None, [256], [0, 256])
        histogram = histogram / histogram.sum()
        nonzero_bins = np.count_nonzero(histogram)
        total_nonzero_bins += nonzero_bins

        entropy = -np.sum(histogram[histogram > 0] * np.log2(histogram[histogram > 0]))
        total_entropy += entropy

        metrics[col] = {'nonzero_bins': nonzero_bins, 'entropy': entropy}

    metrics['average_nonzero_bins'] = total_nonzero_bins / 3
    metrics['average_entropy'] = total_entropy / 3
    return metrics


def process_layer(image, mask, layer_index):
    first_layer_mask = mask[layer_index]

    masked_image = np.array(image)
    masked_image[~first_layer_mask] = 255

    masked_image_cv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

    metrics = evaluate_color_distribution(masked_image_cv)
    return metrics

def calculate_average_entropy():
    image_path = f'M reg/{i}.jpg'
    mask_path = f'Run/masks_{i}_7.npy'

    image = Image.open(image_path).convert("RGB")
    mask = np.load(mask_path)

    mask = mask.astype(bool)
    layers_with_high_entropy = []

    for layer_index in range(mask.shape[0]):
        metrics = process_layer(image, mask, layer_index)
        if metrics and metrics['average_entropy'] > 6.0:
            layers_with_high_entropy.append(layer_index)
    print("Layers with average entropy greater than 6.0:", layers_with_high_entropy)
    return layers_with_high_entropy

def select_layers(layers_with_high_ratio, layers_with_high_entropy):
    set_ratio = set(layers_with_high_ratio)
    set_entropy = set(layers_with_high_entropy)

    intersection = set_ratio.intersection(set_entropy)

    mask_path = f'Run/masks_{i}_7.npy'
    mask = np.load(mask_path)

    selected_layers = mask[list(intersection)]

    new_mask_path = f'Run/masks_{i}_8.npy'
    np.save(new_mask_path, selected_layers)

    print(f"Selected layers saved to {new_mask_path}")


def apply_mask_and_get_min_rect(image_path, mask_path, layer_index):
    image = Image.open(image_path).convert("RGB")
    mask = np.load(mask_path)
    mask = mask.astype(bool)
    layer_mask = mask[layer_index]

    masked_image = np.array(image)
    for c in range(3):
        channel = masked_image[:, :, c]
        channel[~layer_mask] = 255
        masked_image[:, :, c] = channel

    contours, _ = cv2.findContours(layer_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = masked_image[y:y+h, x:x+w]
        return Image.fromarray(cropped_image), (x, y, w, h)
    else:
        return None, None

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


def filter_masks_by_color(cropped_image, masks):
    cropped_image_np = np.array(cropped_image)

    remaining_masks = []

    for current_mask in masks:
        mask_indices = np.where(current_mask == 1)
        masked_image = cropped_image_np[mask_indices]

        white_pixels = np.sum(np.all(masked_image == [255, 255, 255], axis=-1))
        white_percentage = (white_pixels / masked_image.shape[0]) * 100 if masked_image.shape[0] > 0 else 0

        if white_percentage < 90:
            remaining_masks.append(current_mask)

    masks = np.array(remaining_masks) if remaining_masks else np.empty((0, masks.shape[1], masks.shape[2]))

    return masks

def visualize_with_sam(image):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    image_np = np.array(image)
    masks = mask_generator.generate(image_np)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image_np)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()
    masks = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks]
    masks = np.stack(masks)
    return masks

def visualize_mask_layers(npy_file_path):
    mask_layers = np.load(npy_file_path)

    if mask_layers.ndim == 3 and mask_layers.shape[1:] == (1000, 1000):
        merged_mask = np.sum(mask_layers, axis=0)

        plt.imshow(merged_mask, cmap='gray')
        plt.axis('off')  # Hide the axes
        plt.title("Visualization of Merged Mask Layers")
        plt.show()
    else:
        print("The mask layers are not in the expected format for visualization.")


def refined_segmentation():
    mask_path = f'Run/masks_{i}_8.npy'
    image_path = f'M reg/{i}.jpg'
    masks = np.load(mask_path)
    num_layers = masks.shape[0]

    all_processed_masks = []

    for layer_index in range(num_layers):
        cropped_image, rect_coords = apply_mask_and_get_min_rect(image_path, mask_path, layer_index)
        if cropped_image:
            print("Processing layer", layer_index)
            print("Rectangle Coordinates:", rect_coords)

            detected_masks = visualize_with_sam(cropped_image)
            filtered_masks = filter_masks_by_color(cropped_image, detected_masks)

            for mask in filtered_masks:
                resized_mask = cv2.resize(mask, (rect_coords[2], rect_coords[3]), interpolation=cv2.INTER_NEAREST)
                full_mask = np.zeros((1000, 1000), dtype=np.uint8)
                full_mask[rect_coords[1]:rect_coords[1] + rect_coords[3],
                rect_coords[0]:rect_coords[0] + rect_coords[2]] = resized_mask
                all_processed_masks.append(full_mask)

    final_masks = np.stack(all_processed_masks) if all_processed_masks else np.empty((0, 1000, 1000), dtype=np.uint8)

    final_mask_path = f'Run/masks_{i}_10.npy'
    np.save(final_mask_path, final_masks)


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
    print(f"Processed mask saved to {output_path}. Total new layers: {len(new_layers)}")


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0


def delete_iou():
    masks = np.load(f'Run/masks_{i}_10.npy')
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
                if iou > 0.9:
                    if np.sum(current_mask) <= np.sum(masks[other_idx]):
                        to_delete.append(layer_idx)
                        print(f"Layer {layer_idx} deleted due to IoU > 90% with layer {other_idx}")
                        break
                    else:
                        to_delete.append(other_idx)
                        print(f"Layer {other_idx} deleted due to IoU > 90% with layer {layer_idx}")

        if not to_delete:
            break
        to_delete = np.unique(to_delete)
        masks = np.delete(masks, to_delete, axis=0)

    new_mask_file = f'Run/masks_{i}_10.npy'
    np.save(new_mask_file, masks)


def delete_iou_60():
    masks = np.load(f'Run/masks_{i}_11.npy')
    keep_going = True

    while keep_going:
        to_delete = set()
        for layer_idx in range(masks.shape[0]):
            if layer_idx in to_delete:
                continue
            current_mask = masks[layer_idx]
            for other_idx in range(layer_idx + 1, masks.shape[0]):
                if other_idx in to_delete:
                    continue
                iou = calculate_iou(current_mask, masks[other_idx])
                if iou > 0.6:
                    if np.sum(current_mask) <= np.sum(masks[other_idx]):
                        to_delete.add(layer_idx)
                        print(f"Layer {layer_idx} deleted due to IoU > 60% with layer {other_idx}")
                        break
                    else:
                        to_delete.add(other_idx)
                        print(f"Layer {other_idx} deleted due to IoU > 60% with layer {layer_idx}")

        if to_delete:
            masks = np.delete(masks, list(to_delete), axis=0)
        else:
            keep_going = False

    new_mask_file = f'Run/masks_{i}_11.npy'
    np.save(new_mask_file, masks)


def remove_layers_with_few_pixels(mask_path, output_path, threshold=50):
    mask = np.load(mask_path)
    retained_layers = []

    for layer in mask:
        if np.count_nonzero(layer) >= threshold:
            retained_layers.append(layer)

    if retained_layers:
        retained_layers_array = np.array(retained_layers)
        np.save(output_path, retained_layers_array)
        print(f"Processed mask saved to {output_path}. Total retained layers: {len(retained_layers)}")
    else:
        print("No layers retained based on the specified threshold.")


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
    input_mask_path = f'Run/masks_{i}_6.npy'
    output_mask_path = f'Run/masks_{i}_7.npy'
    filter_and_save_layers(input_mask_path, output_mask_path)

    image_path = f'M reg/{i}.jpg'
    mask_path = f'Run/masks_{i}_7.npy'
    layers_with_high_ratio = apply_mask_and_detect_edges_for_all_layers(image_path, mask_path)
    layers_with_high_entropy = calculate_average_entropy()

    select_layers(layers_with_high_ratio, layers_with_high_entropy)

    refined_segmentation()

    mask_path = f'Run/masks_{i}_10.npy'
    output_path = f'Run/masks_{i}_10.npy'

    split_connected_regions_and_save(mask_path, output_path)

    delete_iou()

    source_file = f'Run/masks_{i}_6.npy'
    data = np.load(output_path)
    if data.size == 0:
        shutil.copyfile(source_file, output_path)

    array1 = np.load(f'Run/masks_{i}_6.npy')
    array2 = np.load(f'Run/masks_{i}_10.npy')

    merged_array = np.concatenate((array1, array2), axis=0)

    np.save(f'Run/masks_{i}_11.npy', merged_array)

    print(f"Successfully merged arrays. The shape of the merged array is: {merged_array.shape}")

    delete_iou_60()

    mask_path = f'Run/masks_{i}_11.npy'
    output_path = f'Run/masks_{i}_11.npy'

    remove_layers_with_few_pixels(mask_path, output_path)

    image_path = f'M reg/{i}.jpg'
    npy_path = f'Run/masks_{i}_11.npy'
    save_path = f'Run/{i}_refined.png'
    colors = ['orange', 'red', 'gold', 'red', 'skyblue', 'green', 'purple', 'pink', 'brown']
    border_color = (100, 115, 232)
    border_width = 1
    border_alpha = 0.8
    visualize_colored_masks_with_custom_borders(image_path, npy_path, colors, border_color, border_width, border_alpha, save_path)



if __name__ == '__main__':
    i = '0017'

    # sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    # sam_checkpoint = "weights/sam_vit_b_01ec64.pth"
    sam_checkpoint = "weights/sam_vit_l_0b3195.pth"
    # model_type = "vit_b"
    model_type = "vit_l"
    # model_type = "default"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    main()
