import json
import cv2
import numpy as np
import os
import glob

def create_mask_for_polygon(shape, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(shape, dtype=np.int32)], 255)
    return mask

def get_bounding_square(mask):
    where = np.where(mask)
    y_min, x_min = np.min(where, axis=1)
    y_max, x_max = np.max(where, axis=1)
    side_length = max(x_max - x_min, y_max - y_min)
    return x_min, y_min, side_length


def mask_to_polygon_points(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        polygon = contour.reshape(-1, 2).tolist()
        polygons.append(polygon)

    return polygons

def json_file_generation(i):
    final_npy_path = f'Run/masks_{i}_11.npy'

    masks_final = np.load(final_npy_path, allow_pickle=True)
    masks_dict = {}
    for j, mask in enumerate(masks_final):
        key_name = f'Head_{j}_Tail_{j + 1}'
        masks_dict[key_name] = mask
    converted_npy_path = f'Run/masks_{i}_12.npy'
    np.save(converted_npy_path, masks_dict)

    npy_file_path = f'Run/masks_{i}_12.npy'
    masks = np.load(npy_file_path, allow_pickle=True)

    mask_content = masks.item()

    json_output = {
        "version": "2.3.0",
        "flags": {},
        "shapes": [],
        "imagePath": f"{i}.jpg",
        "imageData": None,
        "imageHeight": 1000,
        "imageWidth": 1000
    }

    for label, mask in mask_content.items():
        polygons = mask_to_polygon_points(mask)
        for polygon in polygons:
            shape_data = {
                "label": "1",
                "points": polygon,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            json_output['shapes'].append(shape_data)

    json_output_path = f'Run/{i}.json'
    with open(json_output_path, 'w') as json_file:
        json.dump(json_output, json_file, indent=4)

def Acquisition_of_cultivated_land(base_dir):
    json_pattern = os.path.join(base_dir, '*.json')
    json_files = glob.glob(json_pattern)

    output_dir = os.path.join(base_dir, '1')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_index = 1

    for json_path in json_files:
        with open(json_path, 'r') as file:
            data = json.load(file)

        i = os.path.splitext(os.path.basename(json_path))[0]
        img_path = f'{base_dir}/{i}.jpg'
        image = cv2.imread(img_path)

        if image is None:
            print(f"Could not open or find the image {img_path}. Skipping...")
            continue

        for shape in data['shapes']:
            if shape['label'] == "1":
                mask = create_mask_for_polygon(shape['points'], image.shape)
                x_min, y_min, side_length = get_bounding_square(mask)
                square_mask = np.zeros((side_length, side_length), dtype=np.uint8)
                square_image = np.zeros((side_length, side_length, 3), dtype=np.uint8)

                x_max = x_min + side_length
                y_max = y_min + side_length

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, image.shape[1])
                y_max = min(y_max, image.shape[0])

                mask_cropped = mask[y_min:y_max, x_min:x_max]
                image_cropped = image[y_min:y_max, x_min:x_max]

                x_offset = (square_mask.shape[1] - mask_cropped.shape[1]) // 2
                y_offset = (square_mask.shape[0] - mask_cropped.shape[0]) // 2

                square_mask[y_offset:y_offset + mask_cropped.shape[0],
                x_offset:x_offset + mask_cropped.shape[1]] = mask_cropped
                square_image[y_offset:y_offset + image_cropped.shape[0],
                x_offset:x_offset + image_cropped.shape[1]] = image_cropped

                final_image = cv2.bitwise_and(square_image, square_image, mask=square_mask)

                output_filename = os.path.join(output_dir, f'{image_index:04d}.jpg')
                try:
                    cv2.imwrite(output_filename, final_image)
                    image_index += 1
                except Exception as e:
                    print(f"Failed to save image {output_filename}: {e}")

    print(f"Processed {image_index - 1} images with label '1'.")


def Acquisition_of_non_cultivated_land(base_dir):
    json_pattern = os.path.join(base_dir, '*.json')
    json_files = glob.glob(json_pattern)

    output_dir = os.path.join(base_dir, '0')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_index = 1

    for json_path in json_files:
        with open(json_path, 'r') as file:
            data = json.load(file)

        i = os.path.splitext(os.path.basename(json_path))[0]
        img_path = f'{base_dir}/{i}.jpg'
        image = cv2.imread(img_path)

        if image is None:
            print(f"Could not open or find the image {img_path}. Skipping...")
            continue

        for shape in data['shapes']:
            if shape['label'] == "0":
                mask = create_mask_for_polygon(shape['points'], image.shape)
                x_min, y_min, side_length = get_bounding_square(mask)
                square_mask = np.zeros((side_length, side_length), dtype=np.uint8)
                square_image = np.zeros((side_length, side_length, 3), dtype=np.uint8)

                x_max = x_min + side_length
                y_max = y_min + side_length

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, image.shape[1])
                y_max = min(y_max, image.shape[0])

                mask_cropped = mask[y_min:y_max, x_min:x_max]
                image_cropped = image[y_min:y_max, x_min:x_max]

                x_offset = (square_mask.shape[1] - mask_cropped.shape[1]) // 2
                y_offset = (square_mask.shape[0] - mask_cropped.shape[0]) // 2

                square_mask[y_offset:y_offset + mask_cropped.shape[0],
                x_offset:x_offset + mask_cropped.shape[1]] = mask_cropped
                square_image[y_offset:y_offset + image_cropped.shape[0],
                x_offset:x_offset + image_cropped.shape[1]] = image_cropped

                final_image = cv2.bitwise_and(square_image, square_image, mask=square_mask)

                output_filename = os.path.join(output_dir, f'{image_index:04d}.jpg')
                try:
                    cv2.imwrite(output_filename, final_image)
                    image_index += 1
                except Exception as e:
                    print(f"Failed to save image {output_filename}: {e}")

    print(f"Processed {image_index - 1} images with label '0'.")


def main():
    i = '0017'
    json_file_generation(i)
    # base_dir = 'Label'
    # Acquisition_of_cultivated_land(base_dir)
    # Acquisition_of_non_cultivated_land(base_dir)
if __name__ == '__main__':
    main()
