# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
def buildpt(Point_top,point_bot):
    Vector_x=(-(point_bot[0]-Point_top[0])/(point_bot[1]-Point_top[1]),1)
    return Vector_x[1], Vector_x[0],-Vector_x[1]*Point_top[0]-Vector_x[0]*Point_top[1]

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.5, max_age=10, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, H):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:

            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])


        for track_idx in unmatched_tracks:
            
            self.tracks[track_idx].mark_missed()



        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], H)

            
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, H ):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        t = Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature)
        box_t = t.to_tlbr()
        startY, endY = int(box_t[1]), int(box_t[3])
        startX, endX = int(box_t[0]), int(box_t[2])
        XMidT = startX 
        yMidT = (endY + startY)/2

        XMidT1 = endX
        yMidT1 = (endY + startY)/2




        # line1_x=(665,58)
        # 122 67
        # 398 467 
        # line1_y=(840,285)
        line1_x=(499,86)
        line1_y=(766,326)
        line2_x=(122,67)
        line2_y=(398,467)

        a,b,c=buildpt(line1_x,line1_y)
        
        stateOutMetro1=None

        stateOutMetro = {}

        if (a*XMidT1+b*yMidT1+c)>=0:
            stateOutMetro['top'] = 1
        elif (a*XMidT1+b*yMidT1+c)<=0:
            stateOutMetro['top'] = 0

        a,b,c=buildpt(line2_x,line2_y)

        if (a*XMidT+b*yMidT+c)>=0:
            stateOutMetro['bot'] = 1
        elif (a*XMidT+b*yMidT+c)<=0:
            stateOutMetro['bot'] = 0


        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, stateOutMetro, noConsider = False))
        self._next_id += 1
