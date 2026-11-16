"""Plain detection data, independent of YOLO tensors."""

from typing import NamedTuple, Optional


class Detection(NamedTuple):
    track_id: Optional[int]
    cls: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def detections_from_arrays(xyxy, conf, cls, ids=None) -> list:
    """Build plain detections from arrays."""
    result = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (int(v) for v in xyxy[i])
        track_id = None if ids is None else int(ids[i])
        result.append(Detection(track_id, int(cls[i]), float(conf[i]), x1, y1, x2, y2))
    return result


def detections_from_boxes(boxes) -> list:
    """Copy box arrays to CPU once before iterating."""
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    ids = boxes.id.cpu().numpy() if boxes.id is not None else None

    return detections_from_arrays(xyxy, conf, cls, ids)
