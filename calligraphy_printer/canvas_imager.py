from pupil_apriltags import Detector, Detection
import cv2
import numpy as np
import logging
from dataclasses import dataclass

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


@dataclass
class CanvasImagerOutput:
    rectified_canvas: np.ndarray | None
    debug_image: np.ndarray | None


class CanvasImager:
    """
    Connects to a camera with CV2, detects it corner Apriltags, and returns a rectified image.

    Expects tags of family 36h11 laid out:

            ─────►X
        │
        │  ┌───┬─┬───────────────────┬─┬───┐
        ▼  │ 0 │ │                   │ │ 1 │
        Y  └───┤ │                   │ ├───┘
               │ │                   │ │
               │ │                   │ │
               │ │                   │ │
           ┌───┤ │                   │ ├───┐
           │ 3 │ │                   │ │ 2 │
           └───┴─┴───────────────────┴─┴───┘

    We detect the Apriltag corners, crop the canvas between the inner corners, and then
    further horizontally crop the result by an extra padding to prevent the apriltags
    from showing up in the final image.
    """

    def __init__(
        self,
        output_width: int,
        output_height: int,
        horizontal_padding: int = 25,
        vertical_padding: int = 25,
    ):
        self.output_height = output_height
        self.output_width = output_width
        self.horizontal_padding = horizontal_padding
        self.vertical_padding = vertical_padding
        self.detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Disable autofocus.
        # self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    def update(self, include_debug_drawing: bool = False) -> CanvasImagerOutput:
        have_image, frame = self.capture.read()
        if not have_image:
            LOG.error("No image from webcam.")
            return CanvasImagerOutput(None, None)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        tags = {tag.tag_id: tag for tag in tags}
        tag_ids = [0, 1, 2, 3]
        corner_indices = [2, 3, 0, 1]
        detection_pts = []
        for tag_id, corner_index in zip(tag_ids, corner_indices, strict=True):
            if tag_id not in tags:
                LOG.info("Tag id %d missing: have %s", tag_id, str(tags.keys()))
                break
            tag: Detection = tags[tag_id]
            detection_pts.append(tag.corners[corner_index, :])
        if len(detection_pts) == 4:
            detection_pts = np.stack(detection_pts, axis=0)
            result_pts = np.array(
                [
                    [-self.horizontal_padding, -self.vertical_padding],
                    [
                        self.output_width - 1 + self.horizontal_padding,
                        -self.vertical_padding,
                    ],
                    [
                        self.output_width - 1 + self.horizontal_padding,
                        self.output_height - 1 + self.vertical_padding,
                    ],
                    [
                        -self.horizontal_padding,
                        self.output_height - 1 + self.vertical_padding,
                    ],
                ],
            )
            # CV2 is very picky that these are float32s.
            M = cv2.getPerspectiveTransform(
                detection_pts.astype(np.float32), result_pts.astype(np.float32)
            )
            rectified_canvas = cv2.warpPerspective(
                frame, M, (self.output_width, self.output_height)
            )
        else:
            rectified_canvas = None

        if include_debug_drawing:
            # Draw apriltag detection boxes
            for tag in tags.values():
                for idx in range(len(tag.corners) + 1):
                    cv2.line(
                        frame,
                        tuple(tag.corners[idx - 1, :].astype(int)),
                        tuple(tag.corners[idx % len(tag.corners), :].astype(int)),
                        (0, 0, 255),
                        2,
                    )

            # Drop crop region box
            if rectified_canvas is not None:
                for idx in range(len(detection_pts) + 1):
                    cv2.line(
                        frame,
                        tuple(detection_pts[idx - 1, :].astype(int)),
                        tuple(detection_pts[idx % len(detection_pts), :].astype(int)),
                        (0, 255, 0),
                        2,
                    )

        return CanvasImagerOutput(rectified_canvas=rectified_canvas, debug_image=frame)


if __name__ == "__main__":
    logging.basicConfig()
    imager = CanvasImager(output_width=640, output_height=414)
    show_debug = True

    while 1:
        output = imager.update(include_debug_drawing=True)
        if show_debug and output.debug_image is not None:
            cv2.imshow("canvas", output.debug_image)
        elif not show_debug and output.rectified_canvas is not None:
            cv2.imshow("canvas", output.rectified_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            show_debug = not show_debug
