import numpy as np
import itertools
from numba import jit
from collections import OrderedDict, deque
from functions.perform_nms import nms_wrapper
from functions.log import logger
from components import assignment
from functions.kalman_filter import KalmanFilter
from models.detection.detector import CandidateClassifier
from models.appearance import appearance_model, appearance_features
from models.appearance import appearance_features
from .basetrack import BaseTrack, TrackState

class TrackLet(BaseTrack):

    def __init__(self, tlwh, score, max_n_features=100, from_det=True):

        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.max_n_features = max_n_features
        self.curr_feature = None
        self.last_feature = None
        self.features = deque([], maxlen=self.max_n_features)

        # classification
        self.from_det = from_det
        self.tracklet_len = 0
        self.time_by_tracking = 0

        # self-tracking
        self.tracker = None

    def assign_feature(self, feature):
        if feature is None:
            return False
        self.features.append(feature)
        self.curr_feature = feature
        self.last_feature = feature
        return True

    def predict(self):
        if self.time_since_update > 0:
            self.tracklet_len = 0

        self.time_since_update += 1

        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

        if self.tracker:
            self.tracker.update_roi(self.tlwh)

    def self_tracking(self, image):
        tlwh = self.tracker.predict(image) if self.tracker else self.tlwh
        return tlwh

    def activate(self, kalman_filter, frame_id, image):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter  
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        del self._tlwh

        self.time_since_update = 0
        self.time_by_tracking = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, image, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.time_since_update = 0
        self.time_by_tracking = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

        self.assign_feature(new_track.curr_feature)

    def update(self, new_track, frame_id, image, update_feature=True):

        #Update a matched track
        self.frame_id = frame_id
        self.time_since_update = 0
        if new_track.from_det:
            self.time_by_tracking = 0
        else:
            self.time_by_tracking += 1
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        if update_feature:
            self.assign_feature(new_track.curr_feature)
            if self.tracker:
                self.tracker.update(image, self.tlwh)

    @property
    @jit
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    @jit
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    @jit
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, with the aspect ratio
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    def tracklet_score(self):
        score = max(0, 1 - np.log(1 + 0.05 * self.time_by_tracking)) * (self.tracklet_len - self.time_by_tracking > 2)
        return score

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class TargetTracker(object):

    def __init__(self, min_det_score=0.4, min_ap_dist=0.6, max_time_lost=30, use_tracking=True, use_refind=True):

        self.min_det_score = min_det_score
        self.min_ap_dist = min_ap_dist
        self.max_time_lost = max_time_lost

        self.kalman_filter = KalmanFilter()

        self.tracked_tracklets = []
        self.lost_tracklets = []
        self.removed_tracklets = []

        self.use_refind = use_refind
        self.use_tracking = use_tracking
        self.classifier = CandidateClassifier()
        self.appearance_model = appearance_model()

        self.frame_id = 0

    def update(self, image, tlwhs, det_scores=None):
        self.frame_id += 1

        activated_tracklets = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []

        """step 1: estimation of new candidates"""
        for tracklet in itertools.chain(self.tracked_tracklets, self.lost_tracklets):
            tracklet.predict()

        """step 2: candidate selection"""
        if det_scores is None:
            det_scores = np.ones(len(tlwhs), dtype=float)
        detections = [TrackLet(tlwh, score, from_det=True) for tlwh, score in zip(tlwhs, det_scores)]

        if self.classifier is None:
            pred_dets = []
        else:
            self.classifier.update(image)

            n_dets = len(tlwhs)
            if self.use_tracking:
                tracks = [TrackLet(t.self_tracking(image), t.tracklet_score(), from_det=False)
                          for t in itertools.chain(self.tracked_tracklets, self.lost_tracklets) if t.is_activated]
                detections.extend(tracks)
            rois = np.asarray([d.tlbr for d in detections], dtype=np.float32)

            cls_scores = self.classifier.predict(rois)
            scores = np.asarray([d.score for d in detections], dtype=np.float)
            scores[0:n_dets] = 1.
            scores = scores * cls_scores
            # non-maximum supression
            if len(detections) > 0:
                keep = nms_wrapper(rois, scores.reshape(-1), nms_thresh=0.3)
                mask = np.zeros(len(rois), dtype=np.bool)
                mask[keep] = True
                keep = np.where(mask & (scores >= self.min_det_score))[0]
                detections = [detections[i] for i in keep]
                scores = scores[keep]
                for d, score in zip(detections, scores):
                    d.score = score
            pred_dets = [d for d in detections if not d.from_det]
            detections = [d for d in detections if d.from_det]

        # set features
        tlbrs = [det.tlbr for det in detections]
        features = appearance_features(self.appearance_model,image, tlbrs)
        features = features.cpu().numpy()
        for i, det in enumerate(detections):
            det.assign_feature(features[i])

        """step 3: data association """
        # assignment of the tracked targets
        unconfirmed = []
        tracked_tracklets = []
        for track in self.tracked_tracklets:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracklets.append(track)

        dists = assignment.nearest_feature_distance(tracked_tracklets, detections, metric='euclidean')
        dists = assignment.gate_cost_matrix(self.kalman_filter, dists, tracked_tracklets, detections)
        matches, u_track, u_detection = assignment.linear_assignment(dists, thresh=self.min_ap_dist)
        for itracked, idet in matches:
            tracked_tracklets[itracked].update(detections[idet], self.frame_id, image)

        # assignment of the missing targets
        detections = [detections[i] for i in u_detection]
        dists = assignment.nearest_feature_distance(self.lost_tracklets, detections, metric='euclidean')
        dists = assignment.gate_cost_matrix(self.kalman_filter, dists, self.lost_tracklets, detections)
        matches, u_lost, u_detection = assignment.linear_assignment(dists, thresh=self.min_ap_dist)
        for ilost, idet in matches:
            track = self.lost_tracklets[ilost]
            det = detections[idet]
            track.re_activate(det, self.frame_id, image, new_id=not self.use_refind)
            refind_tracklets.append(track)

        # remaining tracked
        len_det = len(u_detection)
        detections = [detections[i] for i in u_detection] + pred_dets
        r_tracked_tracklets = [tracked_tracklets[i] for i in u_track]
        dists = assignment.iou_distance(r_tracked_tracklets, detections)
        matches, u_track, u_detection = assignment.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            r_tracked_tracklets[itracked].update(detections[idet], self.frame_id, image, update_feature=True)
        for it in u_track:
            track = r_tracked_tracklets[it]
            track.mark_lost()
            lost_tracklets.append(track)

        detections = [detections[i] for i in u_detection if i < len_det]
        dists = assignment.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = assignment.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, image, update_feature=True)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)

        """step 4: init new tracklets"""
        for inew in u_detection:
            track = detections[inew]
            if not track.from_det or track.score < 0.6:
                continue
            track.activate(self.kalman_filter, self.frame_id, image)
            activated_tracklets.append(track)

        """step 6: update state"""
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.lost_tracklets = [t for t in self.lost_tracklets if t.state == TrackState.Lost]
        self.tracked_tracklets.extend(activated_tracklets)
        self.tracked_tracklets.extend(refind_tracklets)
        self.lost_tracklets.extend(lost_tracklets)
        self.removed_tracklets.extend(removed_tracklets)


        # get scores of the lost tracks
        rois = np.asarray([t.tlbr for t in self.lost_tracklets], dtype=np.float32)
        lost_cls_scores = self.classifier.predict(rois)
        out_lost_tracklets = [t for i, t in enumerate(self.lost_tracklets)
                            if lost_cls_scores[i] > 0.3 and self.frame_id - t.end_frame <= 4]
        output_tracked_tracklets = [track for track in self.tracked_tracklets if track.is_activated]

        output_tracklets = output_tracked_tracklets + out_lost_tracklets

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_tracklets]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_tracklets]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_tracklets]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_tracklets]))

        return output_tracklets
