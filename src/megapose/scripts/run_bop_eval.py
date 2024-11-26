# Standard Library
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import ObservationTensor, PoseEstimatesType
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.symmetries import ContinuousSymmetry, DiscreteSymmetry
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

BOP_DS_DIR = Path("/home/gfloros/data/bop/datasets")
LMO_TEST_DIR = BOP_DS_DIR / "lmo/test/000002"
LMO_ID_TO_LABEL_MAP = {
    1: "ape",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
}
LMO_EXCLUDED_IMAGES = ["001117", "000930", "000097", "001159", "001160", "000203"]

logger = get_logger(__name__)


def load_observation(
    rgb_image: Path,
    camera_data: CameraData,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    rgb = np.array(Image.open(rgb_image), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution
    depth = None
    if load_depth:
        depth_image = rgb_image.parent / "depth" / f"{rgb_image.stem}.png"
        depth = np.array(Image.open(depth_image), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution
    return rgb, depth, camera_data


def load_observation_tensor(
    rgb_image: Path,
    camera_data: CameraData,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(rgb_image, camera_data, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(test_dir: Path, id_label_map: dict[int, str]) -> dict[str, List[ObjectData]]:
    bboxes_filepath = test_dir / "scene_gt_info.json"
    bbox_data = json.loads(bboxes_filepath.read_text())
    gt_filepath = test_dir / "scene_gt.json"
    gt_data = json.loads(gt_filepath.read_text())
    object_data = {}
    for bbox_key, gt_key in zip(bbox_data.keys(), gt_data.keys()):
        assert bbox_key == gt_key
        assert len(bbox_data[bbox_key]) == len(gt_data[gt_key])
        object_data[bbox_key] = ObjectData.from_bop(bbox_data[bbox_key], gt_data[gt_key], id_label_map)
    return object_data


def make_object_dataset(dataset_dir: Path, id_label_map: dict[int, str]) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    models_dir = dataset_dir / "models"
    infos_file = models_dir / "models_info.json"
    infos = json.loads(infos_file.read_text())
    for obj_id, bop_info in infos.items():
        obj_id = int(obj_id)
        obj_label = id_label_map[obj_id]
        mesh_path = (models_dir / f"obj_{obj_id:06d}").with_suffix(".ply").as_posix()
        symmetries_discrete = [
            DiscreteSymmetry(pose=np.array(x).reshape((4, 4)))
            for x in bop_info.get("symmetries_discrete", [])
        ]
        symmetries_continuous = [
            ContinuousSymmetry(offset=d["offset"], axis=d["axis"])
            for d in bop_info.get("symmetries_continuous", [])
        ]
        rigid_object = RigidObject(
            label=obj_label,
            mesh_path=Path(mesh_path),
            mesh_units=mesh_units,
            symmetries_discrete=symmetries_discrete,
            symmetries_continuous=symmetries_continuous,
            mesh_diameter=bop_info["diameter"],
        )
        rigid_objects.append(rigid_object)
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_detections_visualization(
    dataset_dir: Path,
    rgb_image: Path,
    camera_data: CameraData,
    input_object_data: List[ObjectData],
) -> None:
    rgb, _, _ = load_observation(rgb_image, camera_data, load_depth=False)
    detections = make_detections_from_object_data(input_object_data).cuda()
    plotter = BokehPlotter()
    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    output_fn = dataset_dir / "visualizations" / f"detections_{rgb_image.stem}.png"
    output_fn.parent.mkdir(exist_ok=True)
    export_png(fig_det, filename=output_fn)


def save_predictions(
    dataset_dir: Path,
    pose_estimates: PoseEstimatesType,
    rgb_image_stem: str,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = dataset_dir / "outputs" / f"object_data_{rgb_image_stem}.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)


def run_inference(
    dataset_dir: Path,
    rgb_image: Path,
    camera_data: CameraData,
    pose_estimator: PoseEstimatesType,
    model_name: str,
    input_object_data: List[ObjectData],
) -> None:
    model_info = NAMED_MODELS[model_name]
    observation = load_observation_tensor(
        rgb_image, camera_data, load_depth=model_info["requires_depth"]
    ).cuda()
    detections = make_detections_from_object_data(input_object_data).cuda()
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )
    save_predictions(dataset_dir, output, rgb_image.stem)


def make_output_visualization(
    dataset_dir: Path,
    rgb_image: Path,
    camera_data: CameraData,
    object_dataset: RigidObjectDataset,
) -> None:
    rgb, _, camera_data = load_observation(rgb_image, camera_data, load_depth=False)
    camera_data.TWC = Transform(np.eye(4))
    object_datas = dataset_dir / "outputs" / f"object_data_{rgb_image.stem}.json"
    renderer = Panda3dSceneRenderer(object_dataset)
    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]
    plotter = BokehPlotter()
    fig_rgb = plotter.plot_image(rgb)
    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
    vis_dir = dataset_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    export_png(fig_mesh_overlay, filename=vis_dir / f"mesh_overlay_{rgb_image.stem}.png")
    export_png(fig_contour_overlay, filename=vis_dir / f"contour_overlay{rgb_image.stem}.png")
    export_png(fig_all, filename=vis_dir / f"all_results_{rgb_image.stem}.png")

def main():
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument('--images-list', nargs='+', default=[])
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    args = parser.parse_args()

    dataset_dir = BOP_DS_DIR / args.dataset_name
    if (args.dataset_name == "lmo"):
        test_dir = LMO_TEST_DIR
        id_label_map = LMO_ID_TO_LABEL_MAP
        excluded_images = LMO_EXCLUDED_IMAGES

    logger.info(f"Loading camera data...")
    camera_data = CameraData.from_bop(dataset_dir / "camera.json")
    logger.info(f"Loading object detections...")
    object_data = load_object_data(test_dir, id_label_map)
    if args.run_inference or args.vis_outputs:
        logger.info(f"Loading 3D models...")
        object_dataset = make_object_dataset(dataset_dir, id_label_map)
    if args.run_inference:
        logger.info(f"Loading pose estimation model {args.model}...")
        pose_estimator = load_named_model(args.model, object_dataset).cuda()

    logger.info(f"Running inference...")
    if args.images_list:
        rgb_images = [Path(test_dir / "rgb" / image) for image in args.images_list]
        excluded_images = []
    else:
        rgb_images = Path(test_dir / "rgb").glob("*.png")
    for rgb_image in rgb_images:
        if rgb_image.stem in excluded_images:
            continue
        logger.info(f"Processing {rgb_image.stem}...")
        image_id = str(int(rgb_image.stem))
        input_object_data = object_data[image_id]
        if args.vis_detections:
            make_detections_visualization(dataset_dir, rgb_image, camera_data, input_object_data)
        if args.run_inference:
            if Path(dataset_dir / "outputs" / f"object_data_{rgb_image.stem}.json").exists():
                logger.info(f"Skipping {rgb_image.stem} as it already exists...")
                continue
            run_inference(dataset_dir, rgb_image, camera_data, pose_estimator, args.model, input_object_data)
        if args.vis_outputs:
            make_output_visualization(dataset_dir, rgb_image, camera_data, object_dataset)


if __name__ == "__main__":
    main()
