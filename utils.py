import cv2
import json
import torch
import numpy as np

input_size = (512,512)


def data_preprocess(data_path):
    # read images
    rgb, occluder_mask, occludee_mask, contact_mask, obj_name = read_data(data_path)

    # convex_hull
    M_p = make_convex_mask(occluder_mask, occludee_mask, contact_mask)

    # crop
    bmax, bmin, crop_center, crop_size = get_crop_params(occluder_mask, occludee_mask)
    rgb = resize(crop(rgb, crop_center, crop_size), input_size) / 255.
    occluder_mask = resize(crop(occluder_mask, crop_center, crop_size), input_size) / 255.
    occludee_mask = resize(crop(occludee_mask, crop_center, crop_size), input_size) / 255.
    contact_mask = resize(crop(contact_mask, crop_center, crop_size), input_size) / 255.
    M_p = resize(crop(M_p, crop_center, crop_size), input_size) / 255.

    # make it compatible to stable diffusion pipeline
    data_dict = {}
    data_dict["obj_name"] = obj_name
    data_dict["images"] = torch.from_numpy(rgb).float().permute(2, 0, 1)
    data_dict["occluder_mask"] = torch.from_numpy(occluder_mask)
    data_dict["occludee_mask"] = torch.from_numpy(occludee_mask)
    data_dict["contact_mask"] = torch.from_numpy(contact_mask)
    data_dict["M_p"] = torch.from_numpy(M_p)
    return data_dict

def read_data(data_path):
    with open(data_path.parent / "info.json") as fp:
        obj_name = json.load(fp)["cat"]
    rgb= cv2.imread(data_path / "k1.color.jpg")[:, :, ::-1]
    occluder_mask = cv2.imread(data_path / "k1.person_mask.jpg", cv2.IMREAD_GRAYSCALE)
    occludee_mask = cv2.imread(data_path / "k1.obj_rend_mask.jpg", cv2.IMREAD_GRAYSCALE)
    contact_mask = cv2.imread(data_path / "k1.contact.jpg", cv2.IMREAD_GRAYSCALE)

    # behave dataset obj mask is not accurate, additional processing
    # can remove this line if use sam_mask
    kernel = np.ones(5, np.uint8)
    occluder_mask = cv2.dilate(np.array(occluder_mask), kernel, iterations=3)
    occludee_mask = (occluder_mask < 255//2) * occludee_mask

    return rgb, occluder_mask, occludee_mask, contact_mask, obj_name


def make_convex_mask(occluder_mask, occludee_mask, contact_mask):
    kernel = np.ones(3, np.uint8)
    occludee_mask = cv2.dilate(np.array(occludee_mask), kernel, iterations=2)
    # occluder_mask = cv2.dilate(np.array(occluder_mask), kernel, iterations=2)
    occlusion_boundary = cv2.bitwise_and(occludee_mask, occluder_mask)
    boundary_contact = cv2.bitwise_or(occlusion_boundary, contact_mask)
    contours, _ = cv2.findContours(boundary_contact, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = np.vstack(contours)
    hull = cv2.convexHull(all_contours)
    hull_image = np.zeros_like(occludee_mask)
    convex_hull = cv2.drawContours(hull_image, [hull], -1, 255, thickness=cv2.FILLED)
    convex_mask = cv2.bitwise_and(convex_hull, occluder_mask)
    return convex_mask

def crop(img, center, crop_size):
    """
    crop image around the given center, pad zeros for borders
    :param img:
    :param center: np array
    :param crop_size: np array or a float size of the resulting crop
    :return: a square crop around the center
    """
    assert isinstance(img, np.ndarray)
    h, w = img.shape[:2]
    topleft = np.round(center - crop_size / 2).astype(int)
    bottom_right = np.round(center + crop_size / 2).astype(int)

    x1 = max(0, topleft[0])
    y1 = max(0, topleft[1])
    x2 = min(w - 1, bottom_right[0])
    y2 = min(h - 1, bottom_right[1])
    cropped = img[y1:y2, x1:x2]

    p1 = max(0, -topleft[0])  # padding in x, top
    p2 = max(0, -topleft[1])  # padding in y, top
    p3 = max(0, bottom_right[0] - w + 1)  # padding in x, bottom
    p4 = max(0, bottom_right[1] - h + 1)  # padding in y, bottom

    dim = len(img.shape)
    if dim == 3:
        padded = np.pad(cropped, [[p2, p4], [p1, p3], [0, 0]])
    elif dim == 2:
        padded = np.pad(cropped, [[p2, p4], [p1, p3]])
    else:
        raise NotImplemented
    return padded


def resize(img, img_size, mode=cv2.INTER_LINEAR):
    """
    resize image to the input
    :param img:
    :param img_size: (width, height) of the target image size
    :param mode:
    :return:
    """
    h, w = img.shape[:2]
    load_ratio = 1.0 * w / h
    netin_ratio = 1.0 * img_size[0] / img_size[1]
    assert load_ratio == netin_ratio, "image aspect ration not matching, given image: {}, net input: {}".format(
        img.shape, img_size)
    resized = cv2.resize(img, img_size, interpolation=mode)
    return resized

def get_crop_params(mask_hum, mask_obj, bbox_exp=1.0):
    "compute bounding box based on masks"
    bmin, bmax = masks2bbox([mask_hum, mask_obj])
    crop_center = (bmin + bmax) // 2
    crop_size = int(np.max(bmax - bmin) * bbox_exp)
    if crop_size % 2 == 1:
        crop_size += 1  # make sure it is an even number
    return bmax, bmin, crop_center, crop_size

def masks2bbox(masks, threshold=127):
    """

    :param masks:
    :param threshold:
    :return: bounding box corner coordinate
    """
    mask_comb = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        mask_comb = mask_comb | (m > threshold)

    yid, xid = np.where(mask_comb)
    bmin = np.array([xid.min(), yid.min()])
    bmax = np.array([xid.max(), yid.max()])
    return bmin, bmax

