"""Preserved PID road-following controller for local comparison."""

import numpy as np
import cv2


class Agent:
    """Trace the visible road centerline and follow it conservatively."""

    ROWS = tuple(range(65, 17, -3))
    KP = 0.82
    KI = 0.0
    KD = 0.20
    OBSTACLE_GAIN = 0.46
    OBSTACLE_ROAD_FRACTION = 0.40
    CURVE_OBSTACLE_ROAD_FRACTION = 0.55
    CURVE_OBSTACLE_THRESHOLD = 0.25
    AVOID_HOLD_FRAMES = 18
    AVOID_RISK_FLOOR = 0.35
    LATERAL_GAIN = 0.75
    HEADING_GAIN = 0.90
    CURVE_GAIN = 0.45
    STRAIGHT_LATERAL_GAIN = 0.90
    STRAIGHT_HEADING_GAIN = 0.75
    STRAIGHT_CURVE_GAIN = 0.30

    def __init__(self):
        """Initialize the controller without external models or weights."""
        self.reset(None)

    def reset(self, observation):
        """Reset all state at the beginning of every track."""
        self.previous_error = 0.0
        self.derivative = 0.0
        self.integral = 0.0
        self.steer = 0.0
        self.last_near_center = 42.0
        self.last_middle_center = 42.0
        self.last_far_center = 42.0
        self.last_heading = 0.0
        self.recovery_error = 0.0
        self.obstacle_risk = 0.0
        self.obstacle_bias = 0.0
        self.avoid_frames = 0
        self.avoid_direction = 0.0
        self.recenter_frames = 0
        self.lost_frames = 0
        self.still_frames = 0
        self.escape_frames = 0
        self.escape_direction = 0.0
        self.step_count = 0

    @staticmethod
    def _road_components(frame):
        """Return road-like components and their connected-component data."""
        mask = ((frame >= 0.30) & (frame <= 0.54)).astype(np.uint8)
        mask[69:, :] = 0
        close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
        return cv2.connectedComponentsWithStats(mask, connectivity=8)

    def _select_road(self, frame):
        """Select the most plausible road, including after a small excursion."""
        count, labels, stats, centroids = self._road_components(frame)
        if count <= 1:
            return np.zeros((84, 84), dtype=bool), False

        front = np.zeros((84, 84), dtype=bool)
        front[51:68, 28:57] = True
        best_label = 0
        best_score = -1e9
        connected = False

        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < 65:
                continue
            component = labels == label
            overlap = float(np.count_nonzero(component & front))
            bottom = float(
                stats[label, cv2.CC_STAT_TOP]
                + stats[label, cv2.CC_STAT_HEIGHT]
            )
            center_distance = abs(float(centroids[label, 0]) - 42.0)
            score = 20.0 * overlap + 0.06 * area + 0.65 * bottom - center_distance
            if score > best_score:
                best_label = label
                best_score = score
                connected = overlap > 0

        if best_label == 0:
            return np.zeros((84, 84), dtype=bool), False
        return labels == best_label, connected

    @staticmethod
    def _runs(xs):
        """Yield horizontal road runs from sorted pixel coordinates."""
        if xs.size == 0:
            return []
        gaps = np.flatnonzero(np.diff(xs) > 1)
        starts = np.r_[0, gaps + 1]
        ends = np.r_[gaps, xs.size - 1]
        return [(int(xs[a]), int(xs[b])) for a, b in zip(starts, ends)]

    def _centerline(self, frame, road):
        """Trace one continuous road branch from the car toward the horizon."""
        points = []
        # The car is fixed near x=42 in the rendered camera.  Starting every
        # trace there prevents a previous avoidance offset from becoming the
        # next frame's preferred (and sometimes wrong) road branch.
        expected = 42.0
        previous_center = expected
        slope = 0.0

        for row in self.ROWS:
            candidates = []
            for left, right in self._runs(np.flatnonzero(road[row])):
                width = right - left + 1
                if width < 5:
                    continue
                center = 0.5 * (left + right)
                values = frame[row, left:right + 1]
                fresh = float(np.mean((values >= 0.402) & (values <= 0.430)))
                score = abs(center - expected) - 0.045 * width - 2.0 * fresh
                candidates.append((score, center, float(width)))

            if not candidates:
                continue
            _score, center, width = min(candidates, key=lambda item: item[0])
            if points and abs(center - expected) > 22.0:
                continue

            points.append((float(row), center, width))
            measured_slope = (center - previous_center) / 3.0
            slope = 0.70 * slope + 0.30 * measured_slope
            previous_center = center
            expected = center + 3.0 * slope

        return points

    @staticmethod
    def _weighted_center(points, minimum_row, maximum_row, fallback):
        selected = [p for p in points if minimum_row <= p[0] <= maximum_row]
        if not selected:
            return fallback
        weights = np.asarray([p[2] for p in selected], dtype=np.float32)
        centers = np.asarray([p[1] for p in selected], dtype=np.float32)
        return float(np.average(centers, weights=weights))

    def _detect_obstacle(self, frame, points):
        """Detect bright compact objects inside the traced road corridor."""
        inner = np.zeros((84, 84), dtype=np.uint8)
        for row, center, width in points:
            if row < 6 or row > 57:
                continue
            half = int(np.clip(width * 0.25, 3.0, 7.0))
            left = max(0, int(round(center)) - half)
            right = min(83, int(round(center)) + half)
            y = int(row)
            inner[max(0, y - 1):min(69, y + 2), left:right + 1] = 1

        bright = ((frame > 0.56) & (inner > 0)).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bright, connectivity=8
        )
        best = None
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            bottom = int(
                stats[label, cv2.CC_STAT_TOP]
                + stats[label, cv2.CC_STAT_HEIGHT] - 1
            )
            center_x = int(round(float(centroids[label, 0])))
            center_y = int(round(float(centroids[label, 1])))
            y0, y1 = max(0, center_y - 4), min(69, center_y + 5)
            x0, x1 = max(0, center_x - 4), min(84, center_x + 5)
            neighborhood = frame[y0:y1, x0:x1]
            road_fraction = float(np.mean(
                (neighborhood >= 0.30) & (neighborhood <= 0.54)
            ))
            road_fraction_threshold = self.OBSTACLE_ROAD_FRACTION
            if abs(self.last_heading) > self.CURVE_OBSTACLE_THRESHOLD:
                road_fraction_threshold = self.CURVE_OBSTACLE_ROAD_FRACTION
            if (
                2 <= area <= 80
                and bottom >= 6
                and road_fraction >= road_fraction_threshold
            ):
                score = bottom + 0.3 * area
                if best is None or score > best[0]:
                    best = (score, bottom, float(centroids[label, 0]))

        if best is None:
            if self.avoid_frames > 0:
                self.avoid_frames -= 1
                self.obstacle_risk = max(
                    self.AVOID_RISK_FLOOR, self.obstacle_risk * 0.80
                )
                self.obstacle_bias = self.avoid_direction
            else:
                self.obstacle_risk *= 0.55
                self.obstacle_bias *= 0.55
            return

        _score, bottom, obstacle_x = best
        nearest = min(points, key=lambda point: abs(point[0] - bottom))
        road_center = nearest[1]
        risk = float(np.clip((bottom - 6.0) / 47.0, 0.0, 1.0))
        offset = obstacle_x - road_center
        if abs(offset) >= 1.5:
            # Pass through the side with more usable road width.
            direction = -np.sign(offset)
        else:
            direction = -1.0
        if risk > 0.22 and self.avoid_frames <= 0:
            self.avoid_direction = float(direction)
            self.avoid_frames = self.AVOID_HOLD_FRAMES
        if self.avoid_frames > 0:
            direction = self.avoid_direction
            self.avoid_frames -= 1
            risk = max(risk, self.AVOID_RISK_FLOOR)
        self.obstacle_risk = max(self.obstacle_risk * 0.60, risk)
        self.obstacle_bias = float(direction)

    def _observe_road(self, frame):
        """Return lateral error, heading, curve, confidence, and connectivity."""
        road, connected = self._select_road(frame)
        points = self._centerline(frame, road)
        self._detect_obstacle(frame, points)
        if len(points) < 3:
            self.lost_frames += 1
            ys, xs = np.nonzero(road)
            if xs.size >= 20:
                lower = xs[ys >= 30]
                visible_center = float(np.mean(lower if lower.size else xs))
                measured_recovery = float(np.clip(
                    (visible_center - 42.0) / 20.0, -1.5, 1.5
                ))
                self.recovery_error = (
                    0.85 * self.recovery_error + 0.15 * measured_recovery
                )
            return self.recovery_error, 0.0, 0.0, 0.0, connected

        near = self._weighted_center(points, 53, 65, self.last_near_center)
        middle = self._weighted_center(points, 38, 52, near)
        far = self._weighted_center(points, 23, 37, middle)
        if connected:
            near = float(np.clip(
                near, self.last_near_center - 7.0, self.last_near_center + 7.0
            ))
            middle = float(np.clip(
                middle,
                self.last_middle_center - 10.0,
                self.last_middle_center + 10.0,
            ))
            far = float(np.clip(
                far, self.last_far_center - 14.0, self.last_far_center + 14.0
            ))
        visible_rows = [p[0] for p in points]
        row_span = max(visible_rows) - min(visible_rows)
        confidence = min(1.0, len(points) / 11.0, row_span / 27.0)
        if not connected:
            confidence *= 0.55

        lateral = float(np.clip((near - 42.0) / 20.0, -1.5, 1.5))
        heading = float(np.clip((middle - near) / 16.0, -1.5, 1.5))
        curve = float(np.clip(
            (far - 2.0 * middle + near) / 20.0, -1.2, 1.2
        ))
        self.last_near_center = 0.75 * self.last_near_center + 0.25 * near
        self.last_middle_center = 0.65 * self.last_middle_center + 0.35 * middle
        self.last_far_center = 0.60 * self.last_far_center + 0.40 * far
        self.last_heading = heading
        if connected:
            self.recovery_error = lateral
        else:
            self.recovery_error = 0.85 * self.recovery_error + 0.15 * lateral
            lateral = self.recovery_error
        self.lost_frames = 0 if connected else self.lost_frames + 1
        return lateral, heading, curve, confidence, connected

    def _steering(self, lateral, heading, curve, confidence):
        """Use one uniform PID controller on the visible road geometry."""
        if confidence < 0.30:
            # Once outside the connected road, distant geometry is misleading;
            # point directly toward the closest visible road instead.
            error = 1.25 * lateral
        elif self.recenter_frames > 0 and self.obstacle_risk < 0.20:
            error = 0.95 * lateral + 0.45 * heading + 0.10 * curve
            self.recenter_frames -= 1
        else:
            bend = max(abs(heading), abs(curve))
            turn_mix = float(np.clip((bend - 0.12) / 0.28, 0.0, 1.0))
            lateral_gain = (
                self.STRAIGHT_LATERAL_GAIN
                + turn_mix * (self.LATERAL_GAIN - self.STRAIGHT_LATERAL_GAIN)
            )
            heading_gain = (
                self.STRAIGHT_HEADING_GAIN
                + turn_mix * (self.HEADING_GAIN - self.STRAIGHT_HEADING_GAIN)
            )
            curve_gain = (
                self.STRAIGHT_CURVE_GAIN
                + turn_mix * (self.CURVE_GAIN - self.STRAIGHT_CURVE_GAIN)
            )
            error = (
                lateral_gain * lateral
                + heading_gain * heading
                + curve_gain * curve
            )
        if abs(error) < 0.035 and abs(heading) < 0.06:
            error = 0.0

        raw_derivative = error - self.previous_error
        self.derivative = 0.55 * self.derivative + 0.45 * raw_derivative
        if confidence > 0.75 and abs(error) < 0.35:
            self.integral = float(np.clip(self.integral + error, -1.0, 1.0))
        else:
            self.integral *= 0.5

        desired = float(np.clip(
            self.KP * error + self.KI * self.integral + self.KD * self.derivative,
            -1.0,
            1.0,
        ))
        # Once the car has already moved toward the chosen passing side,
        # progressively return authority to lane centering. This prevents a
        # valid avoidance decision from carrying the car off the road.
        avoidance_displacement = max(
            0.0, -self.obstacle_bias * lateral
        )
        avoidance_room = float(np.clip(
            1.0 - avoidance_displacement / 0.55, 0.15, 1.0
        ))
        obstacle_gain = self.OBSTACLE_GAIN * avoidance_room
        desired = float(np.clip(
            desired + obstacle_gain * self.obstacle_bias * self.obstacle_risk,
            -1.0,
            1.0,
        ))
        bend = max(abs(heading), abs(curve))
        urgent = self.obstacle_risk > 0.22
        max_change = 0.48 if urgent or bend > 0.32 or confidence < 0.30 else 0.18
        desired = float(np.clip(desired, self.steer - max_change, self.steer + max_change))
        smoothing = 0.12 if urgent else (0.20 if bend > 0.32 or confidence < 0.30 else 0.52)
        self.steer = smoothing * self.steer + (1.0 - smoothing) * desired
        self.previous_error = error
        return float(np.clip(self.steer, -1.0, 1.0))

    def _speed(self, steer, heading, curve, confidence, connected, motion):
        """Favor controllability over lap time and never hold both pedals."""
        danger = max(
            abs(steer),
            0.95 * abs(heading),
            1.35 * abs(curve),
            1.0 - confidence,
            self.obstacle_risk,
        )
        if not connected or confidence < 0.30:
            return 0.18, 0.0
        if self.recenter_frames > 0 and self.obstacle_risk < 0.20:
            return 0.20, 0.0
        if motion > 0.055:
            return 0.0, 0.12
        if motion > 0.040 and danger > 0.30:
            return 0.0, 0.08
        if self.obstacle_risk > 0.15:
            if motion > 0.026:
                return 0.0, 0.15
            return 0.12, 0.0
        if danger > 0.76:
            if self.step_count % 5 == 0 and motion > 0.004:
                return 0.0, 0.14
            return 0.16, 0.0
        if danger > 0.55:
            return 0.19, 0.0
        if danger > 0.32:
            return 0.27, 0.0
        return 0.34, 0.0

    def act(self, observation) -> np.ndarray:
        """Return [steer, gas, brake] for a float32 (4, 84, 84) observation."""
        frames = np.asarray(observation, dtype=np.float32)
        if frames.shape != (4, 84, 84) or not np.all(np.isfinite(frames)):
            return np.zeros(3, dtype=np.float32)

        self.step_count += 1
        motion = float(np.mean(np.abs(frames[-1, :69] - frames[-2, :69])))
        was_avoiding = self.avoid_frames > 0
        lateral, heading, curve, confidence, connected = self._observe_road(frames[-1])
        if not connected:
            self.recenter_frames = max(self.recenter_frames, 24)
        elif was_avoiding and self.avoid_frames <= 0:
            self.recenter_frames = max(self.recenter_frames, 24)
        steer = self._steering(lateral, heading, curve, confidence)
        gas, brake = self._speed(
            steer, heading, curve, confidence, connected, motion
        )
        self.debug_metrics = (
            lateral, heading, curve, confidence, connected, motion
        )

        if motion < 0.0012:
            self.still_frames += 1
        else:
            self.still_frames = 0
            self.escape_frames = 0

        if self.still_frames >= 8 and self.escape_frames <= 0:
            # Ignore obstacle bias once stationary and use road geometry.
            # A small lateral offset alone can point opposite a sharp bend.
            roadward = (
                self.LATERAL_GAIN * lateral
                + self.HEADING_GAIN * heading
                + self.CURVE_GAIN * curve
            )
            if abs(roadward) <= 0.08:
                roadward = steer if abs(steer) > 0.08 else 1.0
            self.escape_direction = float(np.sign(roadward))
            self.escape_frames = 20

        if self.escape_frames > 0:
            steer = 0.72 * self.escape_direction
            gas = 0.65
            brake = 0.0
            self.escape_frames -= 1

        return np.array([
            np.clip(steer, -1.0, 1.0),
            np.clip(gas, 0.0, 1.0),
            np.clip(brake, 0.0, 1.0),
        ], dtype=np.float32)
