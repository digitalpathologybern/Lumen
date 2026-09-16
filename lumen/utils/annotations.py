# Annotation reader for MetAssist-style GeoJSON exports.

import openslide
import json
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

colormap = {
    0: (50, 50, 50),        # Background
    1: (229, 100, 84),         # Lymph node
    2: (117, 173, 81),        # Metastasis
    3: (234, 40, 241),       # Germinal center
    4: (106, 29, 125),       # Vessels
    5: (54, 90, 113),       # Primary tumor
    6: (212, 185, 60),        # Tumor deposits
    7: (234, 40, 241)       # Primary tissue
}


#: The range a level-0 resolution has to fall in to be believed. Light microscopy
#: of tissue does not resolve finer than 0.05 microns per pixel, and a scan
#: coarser than 50 is not a slide scan, so a value outside this is a metadata
#: fault rather than an unusual scanner. The check is on the number alone, so it
#: catches any vendor's bad tag rather than one known cohort's.
PLAUSIBLE_MPP = (0.05, 50.0)

#: Used only when no source of resolution is believable. 0.5 is a 20x scan, the
#: commonest in these cohorts, and is deliberately conservative: guessing too
#: coarse reads less tissue detail, guessing too fine multiplies the work.
DEFAULT_BASE_MPP = 0.5

#: Microns per pixel implied by an objective power, for slides that declare the
#: magnification but not the resolution.
_MPP_PER_OBJECTIVE = {40.0: 0.25, 20.0: 0.5, 10.0: 1.0, 5.0: 2.0}


def _plausible(value) -> Optional[float]:
    """``value`` as a float if it is a believable level-0 MPP, else ``None``."""
    try:
        mpp = float(value)
    except (TypeError, ValueError):
        return None
    lo, hi = PLAUSIBLE_MPP
    return mpp if lo <= mpp <= hi else None


def resolve_base_mpp(slide: openslide.OpenSlide) -> float:
    """Level-0 microns per pixel, from the first source that can be believed.

    Tried in order: the declared MPP, then the objective power if the slide
    states a magnification but not a resolution, then :data:`DEFAULT_BASE_MPP`.
    Every candidate passes the same :data:`PLAUSIBLE_MPP` range check, so a
    corrupt tag is rejected wherever it comes from.

    A scanner writing a resolution tag in the wrong unit is the usual cause. One
    cohort here declares 1000 MPP because its TIFF records 10 pixels per
    centimetre, and believing that turns "resample to 1 micron per pixel" into an
    instruction to upsample a thousandfold, which exhausts memory before a single
    tile is read.

    This is the single definition of the question. It previously lived inside
    :func:`prepare_read_from_slide` alone, so slide tiling rejected a bad value
    while the dense-map and figure paths accepted it, and the same slide was
    readable by one part of the pipeline and fatal to another.
    """
    import logging

    declared = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp = _plausible(declared)
    if mpp is not None:
        return mpp

    objective = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    try:
        implied = _MPP_PER_OBJECTIVE.get(float(objective))
    except (TypeError, ValueError):
        implied = None
    if implied is not None:
        logging.getLogger(__name__).warning(
            "Unusable base MPP (%s); using %.2f implied by the %sx objective.",
            declared, implied, objective,
        )
        return implied

    logging.getLogger(__name__).warning(
        "Unusable base MPP (%s) and no usable objective power (%s); "
        "falling back to %.2f MPP.", declared, objective, DEFAULT_BASE_MPP,
    )
    return DEFAULT_BASE_MPP


def slide_origin(slide: openslide.OpenSlide) -> Tuple[int, int]:
    """Level-0 pixel offset of the scanned region within the slide canvas.

    MIRAX slides record the scan inside a larger canvas and declare where it
    starts in ``openslide.bounds-x`` and ``bounds-y``. Formats that write no
    such properties return ``(0, 0)``, so callers can add this unconditionally.
    """
    bx = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0) or 0)
    by = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0) or 0)
    return bx, by


def load_geojson_polygons(path: Union[str, Path],
                          slide: Optional[openslide.OpenSlide] = None,
                          keep: Optional[str] = None) -> List[np.ndarray]:
    """Polygons from a QuPath-style GeoJSON, in level-0 canvas pixels.

    ``keep`` filters on the feature's classification name, matched
    case-insensitively as a substring, so ``"tumor"`` selects tumour objects and
    ignores the lymph-node outlines written beside them.

    Coordinates are returned in the same frame OpenSlide's ``read_region`` takes,
    which is the canvas frame including any MIRAX bounds offset. This matters
    because a writer that reads a MIRAX slide from its bounds origin, as
    MetAssist-2 does, emits coordinates relative to that origin. Overlaying those
    on a canvas-frame map puts them tens of millimetres away: on the internal
    cohorts here it moved every polygon off the tissue entirely, and the
    resulting apparent disagreement between two models was an artefact of the
    frame, not of either model. Passing ``slide`` applies the correction, which
    is a no-op for every format that declares no bounds.
    """
    import json

    data = json.loads(Path(path).read_text())
    ox, oy = slide_origin(slide) if slide is not None else (0, 0)

    polys: List[np.ndarray] = []
    for feat in data.get("features", []):
        if keep is not None:
            name = ((feat.get("properties") or {}).get("classification") or {}).get("name", "")
            if name and keep.lower() not in str(name).lower():
                continue
        geom = feat.get("geometry") or {}
        rings = ([geom.get("coordinates", [])] if geom.get("type") == "Polygon"
                 else geom.get("coordinates", []))
        for ring in rings:
            outer = ring[0] if ring and isinstance(ring[0][0], (list, tuple)) else ring
            if outer and len(outer) >= 3:
                polys.append(np.asarray(outer, dtype=float) + (ox, oy))
    return polys


def load_metassist_polygons(outdir: Union[str, Path],
                            slide: Optional[openslide.OpenSlide] = None
                            ) -> List[np.ndarray]:
    """Tumour polygons from a MetAssist output directory, canvas-frame.

    Returns an empty list when the run produced no metastasis GeoJSON, which is
    a valid result and distinct from a missing run.
    """
    hits = sorted(Path(outdir).glob("*_metastasis.geojson"))
    if not hits:
        return []
    return load_geojson_polygons(hits[0], slide, keep="tumor")


def prepare_read_from_slide(slide: Union[openslide.OpenSlide, str], resolution: float, file_type:str) -> Tuple[int, int,float, Tuple[int, int], Tuple[int, int]]:
    """This function receives a slide object and the resolution from which to read the slide.
    It returns the level of the slide to read from, the original dimensions of the slide, and the origin coordinates of the slide read.

    Args:
        slide (Union[openslide.OpenSlide, str]): The slide object to read from.
        resolution (float): The resolution to read the slide from in microns per pixel (mpp).
        file_type (str): The file type of the slide. Supported types are '.svs' and '.mrxs'.

    Raises:
        ValueError: If the file type is not supported.
        ValueError: If the resolution is too high for the slide.

    Returns:
        level (int): The level of the slide to read from.
        level_downsampling (int): The number of downsamples to reach the chosen resolution.
        reading_resolution (float): The precise resolution of the slide at the chosen level in mpp.
        original_dim (Tuple[int, int]): The original dimensions of the slide.
        read_origin (Tuple[int, int]): The origin coordinates of the slide read.
    """

    if isinstance(slide, str):
        if os.path.isfile(slide):
            slide = openslide.open_slide(slide)
        else:
            raise ValueError(f"File {slide} not found.")

    base_mpp = resolve_base_mpp(slide)

    available_mpp = [base_mpp * i for i in slide.level_downsamples]
    closest_level = np.argmin([np.abs(resolution - available_mpp[i]) for i in range(len(available_mpp))])
    level_downsampling = slide.level_downsamples[closest_level]
    reading_resolution = available_mpp[closest_level]
    downsample = round(np.log2(resolution / reading_resolution))

    if downsample != 0:
        _logging.getLogger(__name__).warning(
            "Requested %.3f MPP not available; using closest level %d at %.3f MPP.",
            resolution, closest_level, reading_resolution,
        )

    if file_type in ['.svs', '.tif', '.tiff', '.tif', '.ndpi']:
        original_dim = slide.level_dimensions[closest_level][1], slide.level_dimensions[closest_level][0]
        read_origin = (0, 0) 

    elif file_type == '.mrxs':
        # get dimensions of H&E
        x, y = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_X]), int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_Y])
        w, h = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH]), int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT])
        original_dim = int(h//level_downsampling), int(w//level_downsampling)
        read_origin = (x, y)
    else:
        raise ValueError(f"Unsupported file type {file_type}.")

    return int(closest_level), int(level_downsampling), reading_resolution, original_dim, read_origin

def remove_duplicate_points(contour):
    contour = contour.reshape(-1, 2)  # (N, 1, 2) → (N, 2)
    seen = set()
    mask = []

    for pt in map(tuple, contour):
        if pt not in seen:
            seen.add(pt)
            mask.append(True)
        else:
            mask.append(False)

    unique_contour = contour[mask]
    # add original first point to the end
    if len(unique_contour) > 0 and not np.array_equal(unique_contour[0], unique_contour[-1]):
        unique_contour = np.vstack([unique_contour, unique_contour[0]])
    unique_contour =  unique_contour.reshape(-1, 1, 2).astype(np.int32)
    return unique_contour

def extract_contours_from_geojson(geojson:dict, 
                              category_dict:Dict[str, int], 
                              level_downsampling:int) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]:
    
    dict_of_outer_contour_lists = {k:[] for k in category_dict.values()}
    dict_of_hole_contour_lists = {k:[] for k in category_dict.values()}
        
    for feature in geojson['features']:
        if 'classification' in feature['properties']:
            category = feature['properties']['classification']['name']
        elif 'type' in feature['properties']:
            category = feature['properties']['type']
        else:
            continue
        if category not in category_dict.keys():
            continue
        category = category_dict[category]
        coordinates = feature['geometry']['coordinates']

        if len(coordinates) == 0:
            continue
        elif len(coordinates) >= 1 and isinstance(coordinates[0][0], (int, float)):
            coordinates = [coordinates]

        outer_contour = np.array(coordinates[0], dtype=np.int32) // level_downsampling
        outer_contour = remove_duplicate_points(outer_contour)
        if np.unique(outer_contour, axis=0).shape[0] > 3:
            # remove duplicates keeping the same order            
            dict_of_outer_contour_lists[category].append(outer_contour)      
        else:
            continue

        for hole in coordinates[1:]:
            hole = np.array(hole, dtype=np.int32) // level_downsampling
            hole = remove_duplicate_points(hole)
            if np.unique(hole, axis=0).shape[0] > 3:
                dict_of_hole_contour_lists[category].append(hole)

    return dict_of_outer_contour_lists, dict_of_hole_contour_lists

def create_mask_from_contours(geojson:Union[str, dict], 
                              category_dict:dict, 
                              mask_shape:Tuple[int,int], 
                              level_downsampling:int, 
                              order:List[int]=None) -> np.ndarray:
    """
    Creates a segmentation mask from GeoJSON contours.

    This function generates a 2D segmentation mask based on the contours provided in a GeoJSON file. 
    Each contour is assigned a category value based on the `category_dict`. The function also supports 
    handling overlapping regions by using a specified hierarchy (`order`).

    Args:
        geojson (Union[str, dict]): A file path or dictionary containing GeoJSON data with features and their corresponding geometry.
        category_dict (dict): A dictionary mapping category names (from GeoJSON) to integer values for the mask.
        mask_shape (Tuple[int, int]): The shape of the output mask (height, width).
        level_downsampling (int): The downsampling factor to scale the coordinates from the GeoJSON file.
        order (List[int], optional): A list specifying the hierarchy of categories. If provided, overlapping 
                                    regions are resolved based on this order, where lower indices have higher priority.

    Returns:
        np.ndarray: A 2D segmentation mask of the specified shape, where each pixel is assigned a category value.

    Raises:
        AssertionError: If the `order` contains values not present in `category_dict`.
        Exception: If there is an error while drawing the outer contour.

    Notes:
        - The GeoJSON file must contain features with `classification` and `geometry` properties.
        - The `geometry` property should have `coordinates` in the format `List[List[List[float]]]`.
        - Holes within contours are supported and will be filled with a value of 0.
        - If `order` is not provided, overlapping regions are resolved by taking the maximum value.
    """

    if isinstance(geojson, str):
        with open(geojson, 'r') as f:
            geojson = json.load(f)
    elif not isinstance(geojson, dict):
        raise ValueError('geojson should be a string or a dictionary')
        
    mask = np.zeros(mask_shape, dtype=np.uint8)
    dict_of_outer_contour_lists, dict_of_hole_contour_lists = extract_contours_from_geojson(geojson, category_dict, level_downsampling)

    if order is None:
        for cat in dict_of_outer_contour_lists.keys():
            if len(dict_of_outer_contour_lists[cat]) == 0:
                continue
            cv2.fillPoly(img=mask, pts=dict_of_outer_contour_lists[cat], color=cat)
            
            if len(dict_of_hole_contour_lists[cat]) > 0:
                cv2.fillPoly(mask, dict_of_hole_contour_lists[cat], 0)

    else:
    # remove empty categories from order
        order = [cat for cat in order if len(dict_of_outer_contour_lists[cat]) > 0]
        for cat in order:
            aux_mask = np.zeros(mask_shape, dtype=np.uint8)
            cv2.drawContours(aux_mask, dict_of_outer_contour_lists[cat], -1, cat, thickness=cv2.FILLED)
            if len(dict_of_hole_contour_lists[cat]) > 0:
                cv2.drawContours(aux_mask, dict_of_hole_contour_lists[cat], -1, 0, thickness=cv2.FILLED)
            mask[(aux_mask > 0)] = cat
            
    return mask


