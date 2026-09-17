import os
import cv2
import mmcv
import numpy as np
import torch

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = r"D:\mvxnet-mmdetection3d"

RESULT_FILE = os.path.join(
    ROOT,
    "work_dirs",
    "mvxnet_epoch80_results.pkl"
)

INFO_FILE = os.path.join(
    ROOT,
    "data",
    "kitti",
    "kitti_infos_val.pkl"
)

IMAGE_DIR = os.path.join(
    ROOT,
    "data",
    "kitti",
    "training",
    "image_2"
)

CALIB_DIR = os.path.join(
    ROOT,
    "data",
    "kitti",
    "training",
    "calib"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "work_dirs",
    "mvxnet_visualization"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

SCORE_THRESHOLD = 0.30

# First test only 100 images.
# Later change this to None if everything looks correct.
MAX_IMAGES = None

CLASS_NAMES = [
    "Pedestrian",
    "Cyclist",
    "Car"
]


# ---------------------------------------------------------
# READ KITTI CALIBRATION
# ---------------------------------------------------------

def read_calib(calib_file):

    calib = {}

    with open(calib_file, "r") as f:

        for line in f:

            if ":" not in line:
                continue

            key, value = line.strip().split(":", 1)

            values = np.array(
                [float(x) for x in value.strip().split()],
                dtype=np.float32
            )

            calib[key] = values

    P2 = calib["P2"].reshape(3, 4)

    R0 = np.eye(4, dtype=np.float32)
    R0[:3, :3] = calib["R0_rect"].reshape(3, 3)

    Tr = np.eye(4, dtype=np.float32)
    Tr[:3, :4] = calib["Tr_velo_to_cam"].reshape(3, 4)

    lidar2img = P2 @ R0 @ Tr

    return lidar2img


# ---------------------------------------------------------
# PROJECT A LIDAR POINT INTO CAMERA IMAGE
# ---------------------------------------------------------

def project_point(point, lidar2img):

    point_h = np.array(
        [
            point[0],
            point[1],
            point[2],
            1.0
        ],
        dtype=np.float32
    )

    projected = lidar2img @ point_h

    # Point is behind camera
    if projected[2] <= 0:
        return None

    x = projected[0] / projected[2]
    y = projected[1] / projected[2]

    return int(x), int(y)


# ---------------------------------------------------------
# DRAW 3D BOX
# ---------------------------------------------------------

def draw_3d_box(image, corners, lidar2img, text):

    points = []

    for corner in corners:

        p = project_point(
            corner,
            lidar2img
        )

        if p is None:
            return image

        points.append(p)

    # Connections between the 8 corners
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),

        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),

        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7)
    ]

    for start, end in edges:

        cv2.line(
            image,
            points[start],
            points[end],
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    x = min(p[0] for p in points)
    y = min(p[1] for p in points)

    cv2.putText(
        image,
        text,
        (
            max(0, x),
            max(20, y - 5)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    return image


# ---------------------------------------------------------
# LOAD PREDICTIONS
# ---------------------------------------------------------

print("Loading predictions...")

results = mmcv.load(
    RESULT_FILE
)

print(
    "Number of predictions:",
    len(results)
)

print(
    "Loading KITTI validation information..."
)

infos = mmcv.load(
    INFO_FILE
)

print(
    "Number of validation samples:",
    len(infos)
)

if len(results) != len(infos):

    print("WARNING:")
    print(
        "Predictions =",
        len(results)
    )
    print(
        "Validation infos =",
        len(infos)
    )


# ---------------------------------------------------------
# NUMBER OF IMAGES TO PROCESS
# ---------------------------------------------------------

num_images = len(results)

if MAX_IMAGES is not None:

    num_images = min(
        MAX_IMAGES,
        num_images
    )


# ---------------------------------------------------------
# VISUALIZATION LOOP
# ---------------------------------------------------------

for i in range(num_images):

    info = infos[i]

    # -----------------------------------------------------
    # Get correct KITTI frame ID
    # -----------------------------------------------------

    image_idx = info["image"]["image_idx"]

    try:

        image_idx = int(image_idx)

        frame_id = f"{image_idx:06d}"

    except:

        frame_id = str(image_idx)


    # -----------------------------------------------------
    # Build image/calibration paths
    # -----------------------------------------------------

    image_path = os.path.join(
        IMAGE_DIR,
        frame_id + ".png"
    )

    calib_path = os.path.join(
        CALIB_DIR,
        frame_id + ".txt"
    )


    # -----------------------------------------------------
    # Make sure files exist
    # -----------------------------------------------------

    if not os.path.exists(image_path):

        print(
            "Image not found:",
            image_path
        )

        continue


    if not os.path.exists(calib_path):

        print(
            "Calibration not found:",
            calib_path
        )

        continue


    # -----------------------------------------------------
    # Load camera image
    # -----------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "Could not read image:",
            image_path
        )

        continue


    # -----------------------------------------------------
    # Load calibration
    # -----------------------------------------------------

    lidar2img = read_calib(
        calib_path
    )


    # -----------------------------------------------------
    # Get MVX-Net prediction
    # -----------------------------------------------------

    result = results[i]

    if "pts_bbox" in result:

        pred = result["pts_bbox"]

    else:

        pred = result


    # -----------------------------------------------------
    # Extract prediction components
    # -----------------------------------------------------

    boxes = pred["boxes_3d"]
    scores = pred["scores_3d"]
    labels = pred["labels_3d"]


    # -----------------------------------------------------
    # Convert tensors to numpy
    # -----------------------------------------------------

    if torch.is_tensor(scores):

        scores = (
            scores
            .detach()
            .cpu()
            .numpy()
        )


    if torch.is_tensor(labels):

        labels = (
            labels
            .detach()
            .cpu()
            .numpy()
        )


    # -----------------------------------------------------
    # Handle frames with zero boxes
    # -----------------------------------------------------

    if len(boxes) == 0:

        cv2.putText(
            image,
            f"MVX-Net | KITTI {frame_id} | detections: 0",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            frame_id + "_mvxnet.png"
        )

        cv2.imwrite(
            output_path,
            image
        )

        print(
            f"[{i + 1}/{num_images}] "
            f"{frame_id} -> 0 detections"
        )

        continue


    # -----------------------------------------------------
    # Calculate 3D box corners
    # -----------------------------------------------------

    corners = boxes.corners

    if torch.is_tensor(corners):

        corners = (
            corners
            .detach()
            .cpu()
            .numpy()
        )


    # -----------------------------------------------------
    # Draw boxes above threshold
    # -----------------------------------------------------

    detected = 0

    for box_idx in range(len(scores)):

        score = float(
            scores[box_idx]
        )

        if score < SCORE_THRESHOLD:

            continue


        label = int(
            labels[box_idx]
        )


        if 0 <= label < len(CLASS_NAMES):

            class_name = CLASS_NAMES[label]

        else:

            class_name = str(label)


        label_text = (
            f"{class_name} {score:.2f}"
        )


        image = draw_3d_box(
            image,
            corners[box_idx],
            lidar2img,
            label_text
        )

        detected += 1


    # -----------------------------------------------------
    # Add frame information
    # -----------------------------------------------------

    cv2.putText(
        image,
        f"MVX-Net | KITTI {frame_id} | detections: {detected}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # Save visualization
    # -----------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        frame_id + "_mvxnet.png"
    )

    cv2.imwrite(
        output_path,
        image
    )


    print(
        f"[{i + 1}/{num_images}] "
        f"{frame_id} -> {detected} detections"
    )


# ---------------------------------------------------------
# FINISHED
# ---------------------------------------------------------

print("")
print("Finished.")
print("Images saved to:")
print(OUTPUT_DIR)