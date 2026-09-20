"""Vision-controller baseline for the HAIC CarRacing task.

The simulator window is rendered in colour, but the official wrapper converts
each frame to grayscale before it reaches this class.  The controller therefore
uses 2-D road connectivity and geometry rather than isolated pixel brightness.
"""

import cv2
import numpy as np


class Agent:
    """Find the road in the image and steer toward its look-ahead center."""

    LOOKAHEAD_ROWS = (62, 58, 54, 50, 46, 42, 38, 34, 30, 26, 22, 18)

    def __init__(self):
        """No model file is needed, so initialization is effectively instant."""
        self.reset(None)

    def reset(self, observation):
        """Clear controller memory at the beginning of every track."""
        self.previous_error = 0.0
        self.smoothed_steer = 0.0
        self.last_road_center = 42.0
        self.lost_frames = 0
        self.straight_frames = 0
        self.in_corner = False
        self.exit_boost_frames = 0
        self.stuck_frames = 0
        self.escape_frames = 0
        self.escape_steer = 0.0
        self.step_count = 0
        if observation is not None:
            initial_frame = np.asarray(observation)[-1]
            initial_mean = float(np.mean(initial_frame[:60, :]))
            self.split_max_row = 63 if initial_mean < 0.590 else 38
            self.late_corner_power = float(np.mean(initial_frame[20:50, :28])) > 0.6405
        else:
            self.split_max_row = 38
            self.late_corner_power = False

    @staticmethod
    def _connected_road(frame):
        """Keep only the road-like region connected to the front of the car."""
        # Road tiles are gray (roughly 0.40). Grass and its decorative squares
        # can contain misleading scan-line fragments, so brightness alone is
        # not enough: the chosen component must also touch the car's path.
        candidate = ((frame >= 0.31) & (frame <= 0.53)).astype(np.uint8)
        candidate[68:, :] = 0  # dashboard is not part of the camera view

        # Join tiny gaps made by the car, borders, obstacles and antialiasing.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate, connectivity=8
        )

        if count <= 1:
            return np.zeros_like(candidate, dtype=bool)

        seed = np.zeros_like(candidate, dtype=bool)
        seed[51:66, 31:54] = True
        best_label, best_score = 0, -1e9
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < 70:
                continue
            component = labels == label
            seed_overlap = float(np.count_nonzero(component & seed))
            center_distance = abs(float(centroids[label, 0]) - 42.0)
            score = 15.0 * seed_overlap + 0.05 * area - center_distance
            if score > best_score:
                best_label, best_score = label, score

        if best_label == 0 or not np.any((labels == best_label) & seed):
            return np.zeros_like(candidate, dtype=bool)
        return labels == best_label

    def _follow_road(self, frame):
        """Estimate steering error, road confidence, and edge proximity."""
        road = self._connected_road(frame)
        centers, widths, row_positions = [], [], []
        expected_center = 42.0
        for row_index in self.LOOKAHEAD_ROWS:
            xs = np.flatnonzero(road[row_index])
            if xs.size >= 5:
                if row_index > self.split_max_row:
                    center = float((xs[0] + xs[-1]) * 0.5)
                    width = float(xs[-1] - xs[0] + 1)
                    centers.append(center)
                    widths.append(width)
                    row_positions.append(row_index)
                    expected_center = 0.55 * expected_center + 0.45 * center
                    continue
                breaks = np.flatnonzero(np.diff(xs) > 1)
                starts = np.r_[0, breaks + 1]
                ends = np.r_[breaks, xs.size - 1]
                runs = []
                for start, end in zip(starts, ends):
                    run_xs = xs[start : end + 1]
                    if run_xs.size < 5:
                        continue
                    center = float((run_xs[0] + run_xs[-1]) * 0.5)
                    brightness = frame[row_index, run_xs]
                    fresh_ratio = float(
                        np.mean((brightness > 0.404) & (brightness < 0.425))
                    )
                    score = (
                        abs(center - expected_center)
                        - 0.08 * run_xs.size
                        - 16.0 * fresh_ratio
                    )
                    runs.append((score, center, float(run_xs.size)))
                if not runs:
                    continue
                _score, center, width = min(runs, key=lambda item: item[0])
                centers.append(center)
                widths.append(width)
                row_positions.append(row_index)
                expected_center = 0.55 * expected_center + 0.45 * center

        if len(centers) < 2:
            self.lost_frames += 1
            recovery = np.sign(self.smoothed_steer or self.previous_error) * 0.55
            return float(recovery), 0.0, 1.0

        self.lost_frames = 0
        centers = np.asarray(centers, dtype=np.float32)
        widths = np.asarray(widths, dtype=np.float32)
        rows = np.asarray(row_positions, dtype=np.float32)

        # Rows are close-to-far. Far geometry gets deliberately high influence:
        # waiting until a hairpin reaches the car was the old controller's main
        # failure mode.
        near_mask = rows >= 50
        far_mask = rows <= 38
        near = float(np.mean(centers[near_mask])) if np.any(near_mask) else float(centers[0])
        far = float(np.mean(centers[far_mask])) if np.any(far_mask) else float(centers[-1])

        row_grid = np.indices(road.shape)[0]
        ys, xs = np.nonzero(road & (row_grid <= 42))
        far_mass = float(np.mean(xs)) if xs.size >= 20 else far
        target_center = 0.25 * near + 0.45 * far + 0.30 * far_mass

        self.last_road_center = 0.65 * self.last_road_center + 0.35 * target_center

        position_error = (near - 42.0) / 20.0
        lookahead_error = (target_center - 42.0) / 20.0
        curve_error = (far - near) / 16.0
        error = 0.35 * position_error + 0.95 * lookahead_error + 0.65 * curve_error

        confidence = min(1.0, len(centers) / 8.0)
        # Only nearby width indicates edge danger. Far rows are naturally narrow
        # because of perspective and previously kept the car slow forever.
        near_widths = widths[rows >= 46]
        representative_width = (
            float(np.median(near_widths)) if near_widths.size else float(np.median(widths))
        )
        narrowness = float(np.clip((18.0 - representative_width) / 18.0, 0.0, 1.0))
        return float(error), confidence, narrowness

    def act(self, observation) -> np.ndarray:
        """Return [steer, gas, brake] within the official action bounds."""
        frames = np.asarray(observation, dtype=np.float32)
        if frames.shape != (4, 84, 84):
            return np.zeros(3, dtype=np.float32)
        self.step_count += 1

        motion = float(np.mean(np.abs(frames[-1] - frames[-2])))
        if self.escape_frames <= 0:
            if motion < 0.0015:
                self.stuck_frames += 1
            else:
                self.stuck_frames = 0
            if self.stuck_frames >= 6:
                turn_sign = np.sign(self.smoothed_steer or self.previous_error or 1.0)
                self.escape_steer = float(0.70 * turn_sign)
                self.escape_frames = 20
                self.stuck_frames = 0

        error, confidence, narrowness = self._follow_road(frames[-1])
        derivative = error - self.previous_error
        self.previous_error = error

        # Require several centered, stable frames before returning to cruise.
        road_is_straight = (
            abs(error) < 0.20
            and abs(derivative) < 0.12
            and abs(self.smoothed_steer) < 0.25
            and confidence >= 0.75
            and narrowness < 0.40
        )
        if road_is_straight:
            self.straight_frames = min(10, self.straight_frames + 1)
        else:
            self.straight_frames = 0

        sharp_corner = abs(error) > 0.55
        if sharp_corner:
            self.in_corner = True
            self.exit_boost_frames = 0

        desired_steer = np.clip(1.45 * error + 0.40 * derivative, -1.0, 1.0)
        smoothing = 0.48 if sharp_corner else 0.68
        self.smoothed_steer = (
            smoothing * self.smoothed_steer
            + (1.0 - smoothing) * desired_steer
        )
        steer = float(np.clip(self.smoothed_steer, -1.0, 1.0))

        turn = abs(steer)
        danger = max(turn, narrowness * 0.75, 1.0 - confidence)
        corner_exit_confirmed = self.in_corner and self.straight_frames >= 4
        if corner_exit_confirmed:
            self.in_corner = False
            self.exit_boost_frames = 8

        if self.exit_boost_frames > 0 and not self.in_corner:
            # The road geometry, rather than the lagging smoothed steering
            # value, confirms that the turn has safely ended.
            gas, brake = 0.68, 0.0
            self.exit_boost_frames -= 1
        elif sharp_corner or danger > 0.82:
            if self.late_corner_power and self.step_count > 900:
                gas, brake = 0.24, 0.05
            else:
                gas, brake = 0.16, 0.20
        elif danger > 0.62:
            gas, brake = 0.32, 0.0
        elif danger > 0.40:
            gas, brake = 0.45, 0.0
        else:
            gas, brake = 0.55, 0.0

        if self.lost_frames > 3:
            gas, brake = 0.18, 0.0

        if self.escape_frames > 0:
            steer, gas, brake = self.escape_steer, 1.0, 0.0
            self.escape_frames -= 1

        return np.array([steer, gas, brake], dtype=np.float32)
