# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import misc.params as params
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''

    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def associate(self, track_list, meas_list, KF):

        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []

        if len(meas_list) > 0:
            self.unassigned_meas = list(range(len(meas_list)))
        if len(track_list) > 0:
            self.unassigned_tracks = list(range(len(track_list)))
        if len(meas_list) > 0 and len(track_list) > 0:
            self.association_matrix = np.inf * \
                np.ones((len(track_list), len(meas_list)))
            for track_num, track_obj in enumerate(track_list):
                for meas_num, meas_obj in enumerate(meas_list):
                    MHD_value = self.MHD(track_obj, meas_obj, KF)
                    if self.gating(MHD_value, meas_obj.sensor):
                        self.association_matrix[track_num,
                                                meas_num] = MHD_value

        ############
        # END student code
        ############

    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # the following only works for at most one track and one measurement
        association_mat = self.association_matrix
        if np.min(association_mat) == np.inf:
            return np.nan, np.nan
        i_min, j_min = np.unravel_index(
            np.argmin(association_mat, axis=None), association_mat.shape)
        association_mat = np.delete(association_mat, i_min, 0)
        association_mat = np.delete(association_mat, j_min, 1)
        self.association_matrix = association_mat
        update_track = self.unassigned_tracks[i_min]
        update_meas = self.unassigned_meas[j_min]

        # remove from list
        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)
        ############
        # END student code
        ############
        return update_track, update_meas

    def gating(self, MHD, sensor):
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############

        return MHD < chi2.ppf(params.gating_threshold, df=2)

        ############
        # END student code
        ############

    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        X = track.x
        H = meas.sensor.get_H(X)
        res = meas.z - meas.sensor.get_hx(X)
        S = np.matmul(np.matmul(H, track.P), np.transpose(H)) + meas.R
        MHD_value = np.matmul(
            np.matmul(np.transpose(res), np.linalg.inv(S)), res)
        return MHD_value

        ############
        # END student code
        ############

    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)

        # update associated tracks with measurements
        while self.association_matrix.shape[0] > 0 and self.association_matrix.shape[1] > 0:

            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]

            # check visibility, only update tracks in fov
            if not meas_list[0].sensor.in_fov(track.x):
                continue

            # Kalman update
            print('update track', track.id, 'with',
                  meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])

            # update score and track state
            manager.handle_updated_track(track)

            # save updated track
            manager.track_list[ind_track] = track

        # run track management
        manager.manage_tracks(self.unassigned_tracks,
                              self.unassigned_meas, meas_list)

        for track in manager.track_list:
            print('track', track.id, 'score =', track.score)
