import argparse
import dataclasses
import json

import cv2
import numpy as np

from collections import abc, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

from megapose.utils.logging import get_logger


JsonValue = Union[None, str, bool, int, float, List[Any], Dict[str, Any]]


def rle_to_binary_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Converts a COCOs run-length encoding (RLE) to a binary mask.

    Args:
        rle: Mask in RLE format
    Returns:
        A 2D binary numpy array where '1's represent the object.
    """
    binary_array = np.zeros(np.prod(rle["size"]), dtype=bool)
    counts = rle["counts"]
    start = 0
    for i in range(len(counts) - 1):
        start += counts[i]
        end = start + counts[i + 1]
        binary_array[start:end] = (i + 1) % 2
    binary_mask = binary_array.reshape(*rle["size"], order="F")
    return binary_mask


def calc_bbox_from_mask(mask: np.ndarray) -> List[int]:
    """Calculates the bounding box from a binary mask.

    Args:
        mask: A 2D binary numpy array where '1's represent the object.
    Returns:
        A list of four integers representing the bounding box
        in the format [x, y, w, h].
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy array")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if mask.dtype != bool:
        raise ValueError("mask must be a binary mask (dtype=bool)")
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return [0, 0, 0, 0]  # No object found
    x_min, y_min, w, h = cv2.boundingRect(contours[0])

    return [x_min, y_min, w, h]


def str_from_any(x: Any) -> str:
    k = from_any(x)
    if not isinstance(k, str):
        raise TypeError(f"use of {x!r} as a json key is not supported")
    return k


def from_any(x: Any) -> JsonValue:
    """Converts a Python object to json. This does not require type hints,
       since we have the actual object available.
    """

    # primitives
    if type(x) in (str, bool, int, float, type(None)):
        return x

    if isinstance(x, Enum):
        return from_any(x.value)

    if isinstance(x, abc.Mapping):
        return {str_from_any(k): from_any(v) for (k, v) in x.items()}

    if dataclasses.is_dataclass(x):
        return from_any(dataclasses.asdict())

    # NamedTuple
    if hasattr(x, "_asdict"):
        return from_any(x._asdict())

    if isinstance(x, (abc.Sequence, abc.Set)):
        return [from_any(x) for x in x]

    # NumPy floating-point types
    if isinstance(x, np.floating):
        return float(x)

    # NumPy integer types
    if isinstance(x, np.integer):
        return int(x)

    # NumPy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    raise ValueError(f"conversion of {x!r} to json not supported")


def save_json(path: str, content: Any):
    """Saves a content to a JSON file.

    Args:
        path: The path to the output JSON file.
        content: The content to save (typically a dictionary).
    """

    with open(path, "w", encoding="utf-8") as f:
        content_json = from_any(content)
        json.dump(content_json, f)


def load_json(path: str) -> Any:
    """Loads the content of a JSON file.

    Args:
        path: The path to the input JSON file.
    Returns:
        The loaded content (typically a dictionary).
    """
    with open(path, "r") as f:
        return json.load(f)


def get_best_detection_per_object(
    pred_mask_list: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Get the best detection for each object based on the highest score.

    Args:
        pred_mask_list: A list of predictions, each containing 'category_id',
                        'image_id', 'bbox', 'segmentation', and 'score'.
    Returns:
        A dictionary mapping keys of the form "image_id_category_id" to the
        best detection for that object.
    """
    best_detections_per_object = dict()

    for pred in pred_mask_list:
        category_id = pred["category_id"]
        image_id = pred["image_id"]
        key = f"{image_id:06d}_{category_id:06d}"
        detection = best_detections_per_object.get(key, None)
        if not detection or (detection["score"] < pred["score"]):
            best_detections_per_object[key] = {
                "category_id": category_id,
                "image_id": image_id,
                "bbox_obj": pred["bbox"],
                "segmentation": pred["segmentation"],
                "score": pred["score"],
            }

    return best_detections_per_object


def collect_detections_per_image(
    best_detections_per_object: Dict[str, Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Collects detections per image from the best detections.

    Args:
        best_detections_per_object: A dictionary mapping keys of the form
                                    "image_id_category_id" to the best detection
                                    for that object.
    Returns:
        A dictionary mapping image IDs to a list of detections for that image.
    """
    detections = defaultdict(list)

    for detection in best_detections_per_object.values():
        image_id = detection["image_id"]
        key = str(image_id)
        detections[key].append(
            {
                "category_id": detection["category_id"],
                "image_id": image_id,
                "bbox_obj": detection["bbox_obj"],
                "score": detection["score"],
                "segmentation": detection["segmentation"],
            }
        )

    return detections


def save_cnos_as_bop(
    detections: Dict[str, List[Dict[str, Any]]],
    output_dir: str
) -> None:
    """Saves detections in BOP format.

    Args:
        detections: A dictionary mapping image IDs to a list of detections
                    for that image.
        output_dir: The directory where the BOP files will be saved.
    """
    sorted_detections = defaultdict(list)
    sorted_instances = defaultdict(list)
    masks_dir = Path(output_dir) / "mask_visib"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for detection in detections.values():
        # get_logger("convert_cnos_to_bop").info(f"detection = {detection}")
        for value in sorted(detection, key=lambda x: x["category_id"]):
            category_id = value["category_id"]
            image_id = value["image_id"]
            key = str(image_id)
            binary_mask = rle_to_binary_mask(value["segmentation"])
            bbox_visib = calc_bbox_from_mask(binary_mask)
            px_count_visib = np.sum(binary_mask)
            px_count_all = bbox_visib[2] * bbox_visib[3]
            visib_fract = px_count_visib / px_count_all if px_count_all > 0 else 0
            sorted_instances[key].append(
                {
                    "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "cam_t_m2c": [0.0, 0.0, 0.0],
                    "obj_id": category_id,
                }
            )
            sorted_detections[key].append(
                {
                    "bbox_obj": value["bbox_obj"],
                    "bbox_visib": bbox_visib,
                    "visib_fract": visib_fract,
                }
            )
            segmentation_filepath = masks_dir / f"{image_id:06d}_{category_id:06d}.png"
            cv2.imwrite(str(segmentation_filepath), binary_mask * 255)

    instances_filepath = Path(output_dir) / "scene_gt.json"
    save_json(instances_filepath, sorted_instances)
    detections_filepath = Path(output_dir) / "scene_gt_info.json"
    save_json(detections_filepath, sorted_detections)


def main():
    """Main function to convert CNOS masks to BOP format."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    pred_mask_list = load_json(args.input)
    best_detections_per_object = get_best_detection_per_object(pred_mask_list)
    detections_per_image = collect_detections_per_image(best_detections_per_object)
    save_cnos_as_bop(detections_per_image, args.output)


if __name__ == "__main__":
    main()
