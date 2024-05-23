import numpy as np

cfg = {
    "lines": {
        "house.jpg": {
            "horizontal_lines": {
                "canny_blur": 5,
                "canny_thresholds": (50.0, 180.0),
                "max_edges": 300,
                "min_angle": np.pi * (89 / 180),
                "max_angle": np.pi * (91 / 180),
                "angle_spacing": np.pi / 180,
                "offset_spacing": 5.0,
                "accumulator_threshold": 0.6,
                "nms_angle_range": np.pi / 180,
                "nms_offset_range": 2,
            },
        },
    },
    "circles": {
        "coins.jpg": {
            "pennies": {
                "canny_blur": 3,
                "canny_thresholds": (50, 150),
                "max_edges": 200,
                "accumulator_threshold": 0.6,
                "nms_overlap": 25,
                "radius": 55,
                "spacing": 10
            },
            "nickels": {
                "canny_blur": 3,
                "canny_thresholds": (60, 120),
                "max_edges": 200,
                "accumulator_threshold": 0.6,
                "nms_overlap": 21,
                "radius": 63,
                "spacing": 9
            },
            "dimes": {
                "canny_blur": 3,
                "canny_thresholds": (50, 100),
                "max_edges": 200,
                "accumulator_threshold": 0.6,
                "nms_overlap": 17,
                "radius": 47,
                "spacing": 10
            },
            "quarters": {
                "canny_blur": 3,
                "canny_thresholds": (50, 100),
                "max_edges": 50,
                "accumulator_threshold": 0.6,
                "nms_overlap": 31,
                "radius": 70,
                "spacing": 10
            },
        },
    },
}
