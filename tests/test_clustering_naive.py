import numpy as np
import unittest

import covid19sim.frozen.clustering_naive as clu
import covid19sim.frozen.message_utils as mu
from tests.utils import FakeHuman, generate_received_messages, Visit

never = 9999  # dirty macro to indicate a human will never get infected
ui = np.uint64  # dirty macro to cast integers to np.uint64


# TODO: THERE'S ONLY ONE FUNCTIONAL TEST WE CARE ABOUT: RUN WITH LOTS OF PEOPLE, AND...
#         - VERIFY THAT THIS IMPL RETURNS FEWER CLUSTERS THAN THE SIMPLISTIC ONE
#         - VERIFY THAT THIS IMPL HAS LESS 'HETEROGENEOUS' CLUSTERS THAN THE SIMPLISTIC ONE


class NaiveClusteringTests(unittest.TestCase):
    # note: we only ever build & test clusters for a single human, assuming it would also work for others

    def test_same_day_visit_clusters(self):
        n_trials = 100
        for _ in range(n_trials):
            # scenario: single day visits, 1 cluster per visit, not enough visits to overlap the clusters
            visits = [
                Visit(visitor_real_uid=ui(1), visited_real_uid=ui(0), exposition=False, timestamp=ui(2)),
                Visit(visitor_real_uid=ui(2), visited_real_uid=ui(0), exposition=False, timestamp=ui(2)),
            ]
            # we will cheat and hard-code some initial 4-bit uids to make sure there is no overlap
            humans = [
                FakeHuman(
                    real_uid=idx,
                    exposition_timestamp=never,
                    visits_to_adopt=visits,
                    # with only two visitors, there should never be cluster uid overlap at day 2
                    force_init_uid=np.uint8(idx),
                ) for idx in range(3)
            ]
            day2_uids = [h.rolling_uids[2] for h in humans]
            self.assertTrue(len(np.unique(day2_uids)) == 3)
            messages = generate_received_messages(humans)
            # now we need to get all messages sent to human 0 to do our actual clustering analysis
            h0_messages = messages[0]["received_messages"]
            self.assertEqual(len(h0_messages), 3)  # three timesteps in book
            self.assertEqual(sum([len(msgs) for msgs in h0_messages.values()]), 2)
            self.assertEqual(len(h0_messages[2]), 2)
            h0_messages = [msg for msgs in h0_messages.values() for msg in msgs]
            cluster_manager = clu.SimplisticClusterManager(max_history_offset=never)
            cluster_manager.add_messages(h0_messages)
            self.assertEqual(len(cluster_manager.clusters), 2)
            self.assertEqual(cluster_manager.latest_refresh_timestamp, 2)
            expositions = cluster_manager._get_expositions_array()
            self.assertEqual(len(expositions), 2)
            self.assertEqual(sum(expositions), 0)
            embeddings = cluster_manager.get_embeddings_array()
            for uid in [h.rolling_uids[2] for h in [humans[1], humans[2]]]:
                self.assertIn(uid, embeddings[:, 0])
            self.assertTrue((embeddings[:, 1] == 0).all())  # risk level
            self.assertTrue((embeddings[:, 2] == 1).all())  # message count
            self.assertTrue((embeddings[:, 3] == 2).all())  # timestamp

    def test_same_day_visit_clusters_overlap(self):
        # scenario: single day visits, and some visits will share the same cluster
        visits = [
            Visit(visitor_real_uid=ui(1), visited_real_uid=ui(0), exposition=False, timestamp=ui(0)),
            Visit(visitor_real_uid=ui(2), visited_real_uid=ui(0), exposition=False, timestamp=ui(0)),
            Visit(visitor_real_uid=ui(3), visited_real_uid=ui(0), exposition=False, timestamp=ui(0)),
            Visit(visitor_real_uid=ui(4), visited_real_uid=ui(0), exposition=False, timestamp=ui(0)),
            Visit(visitor_real_uid=ui(5), visited_real_uid=ui(0), exposition=False, timestamp=ui(0)),
        ]
        # we will cheat and hard-code some initial 4-bit uids to make sure there is overlap
        humans = [
            FakeHuman(
                real_uid=idx,
                exposition_timestamp=never,
                visits_to_adopt=visits,
                force_init_uid=np.uint8(max(idx - 1, 0) % 3),
                force_init_risk=np.uint8(7),
            ) for idx in range(6)
        ]
        day0_visitor_uids = [h.rolling_uids[0] for h in humans[1:]]
        self.assertTrue(len(np.unique(day0_visitor_uids)) == 3)  # two visits will be overlapped
        messages = generate_received_messages(humans)
        h0_messages = messages[0]["received_messages"]
        self.assertEqual(len(h0_messages), 1)  # single timestep in book
        self.assertEqual(len(h0_messages[0]), 5)  # all 5 encounter messages in day 0
        h0_messages = [msg for msgs in h0_messages.values() for msg in msgs]
        cluster_manager = clu.SimplisticClusterManager(max_history_offset=never)
        cluster_manager.add_messages(h0_messages)
        self.assertEqual(len(cluster_manager.clusters), 3)
        self.assertEqual(cluster_manager.latest_refresh_timestamp, 0)
        expositions = cluster_manager._get_expositions_array()
        self.assertTrue(len(expositions) == 3 and sum(expositions) == 0)
        embeddings = cluster_manager.get_embeddings_array()
        for uid in [h.rolling_uids[0] for h in humans[1:]]:
            self.assertIn(uid, embeddings[:, 0])
        self.assertTrue((embeddings[:, 1] == 7).all())  # risk level
        self.assertTrue(np.logical_and(embeddings[:, 2] > 0, embeddings[:, 2] < 3).all())
        self.assertTrue((embeddings[:, 3] == 0).all())  # timestamp

    def test_cluster_risk_update(self):
        # scenario: single day visits, and some visits will share the same cluster
        visits = [
            Visit(visitor_real_uid=ui(1), visited_real_uid=ui(0), exposition=False, timestamp=ui(0)),
        ]
        humans = [
            FakeHuman(real_uid=0, exposition_timestamp=never, visits_to_adopt=visits,
                      force_init_risk=np.uint8(0)),
            FakeHuman(real_uid=1, exposition_timestamp=never, visits_to_adopt=visits,
                      force_init_risk=np.uint8(7)),
        ]
        messages = generate_received_messages(humans)
        h0_messages = messages[0]["received_messages"]
        self.assertEqual(len(h0_messages), 1)  # single timestep in book
        self.assertEqual(len(h0_messages[0]), 1)  # single encounter message in day 0
        h0_messages = [msg for msgs in h0_messages.values() for msg in msgs]
        cluster_manager = clu.SimplisticClusterManager(max_history_offset=never)
        cluster_manager.add_messages(h0_messages)
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(7))
        # we will add a manual update for one of the visits
        cluster_manager.add_messages([
            mu.create_update_message(h0_messages[0], np.uint8(9), np.uint64(1))
        ])
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(9))
        # add a new encounter: it should not match the existing cluster due to diff risk
        cluster_manager.add_messages([
            mu.EncounterMessage(humans[1].rolling_uids[0], risk_level=np.uint8(1), encounter_time=ui(0))
        ])
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[1].risk_level, np.uint8(1))
        # add a new encounter: it should match the existing cluster due to same risk
        new_encounter = \
            mu.EncounterMessage(humans[1].rolling_uids[0], risk_level=np.uint8(9), encounter_time=ui(0))
        cluster_manager.add_messages([new_encounter])
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(9))
        self.assertEqual(len(cluster_manager.clusters[0].messages), 2)
        # update one of the two encounters in the first cluster; average risk should change
        cluster_manager.add_messages([
            mu.create_update_message(new_encounter, np.uint8(13), np.uint64(1))
        ])
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(11))
        self.assertEqual(len(cluster_manager.clusters[0].messages), 2)

    def test_cleanup_outdated_cluster(self):
        # scenario: a new encounter is added that is waaay outdated; it should not create a cluster
        visits = [
            Visit(visitor_real_uid=ui(1), visited_real_uid=ui(0), exposition=False, timestamp=ui(2)),
            Visit(visitor_real_uid=ui(1), visited_real_uid=ui(0), exposition=False, timestamp=ui(5)),
            Visit(visitor_real_uid=ui(1), visited_real_uid=ui(0), exposition=False, timestamp=ui(8)),
        ]
        humans = [
            FakeHuman(real_uid=0, exposition_timestamp=never, visits_to_adopt=visits),
            FakeHuman(real_uid=1, exposition_timestamp=never, visits_to_adopt=visits),
        ]
        messages = generate_received_messages(humans)
        h0_messages = messages[0]["received_messages"]
        self.assertEqual(len(h0_messages), 9)
        h0_messages = [msg for msgs in h0_messages.values() for msg in msgs]
        cluster_manager = clu.SimplisticClusterManager(max_history_offset=5)
        cluster_manager.add_messages(h0_messages)
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[0].first_update_time, np.uint8(5))
        self.assertEqual(cluster_manager.clusters[1].first_update_time, np.uint8(8))
        # new manually added encounters that are outdated should also be ignored
        cluster_manager.add_messages([
            mu.EncounterMessage(humans[1].rolling_uids[0], risk_level=np.uint8(1), encounter_time=ui(0))
        ])
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[0].first_update_time, np.uint8(5))
        self.assertEqual(cluster_manager.clusters[1].first_update_time, np.uint8(8))

    def test_random_large_scale(self):
        n_trials = 10
        n_humans = 100
        n_visits = 1000
        n_expositions = 50
        max_timestamp = 20
        for _ in range(n_trials):
            # start by sampling a bunch of non-exposition visits...
            visits = []
            for _ in range(n_visits):
                visitor_real_uid = np.random.randint(n_humans)
                visited_real_uid = visitor_real_uid
                while visitor_real_uid == visited_real_uid:
                    visited_real_uid = np.random.randint(n_humans)
                visits.append(Visit(
                    visitor_real_uid=ui(visitor_real_uid),
                    visited_real_uid=ui(visited_real_uid),
                    exposition=False,
                    timestamp=ui(np.random.randint(max_timestamp + 1)),
                ))
            # ...then, had a handful of exposition visits to increase risk levels
            for _ in range(n_expositions):
                exposer_real_uid = np.random.randint(n_humans)
                exposed_real_uid = exposer_real_uid
                while exposer_real_uid == exposed_real_uid:
                    exposed_real_uid = np.random.randint(n_humans)
                visits.append(Visit(
                    visitor_real_uid=ui(exposer_real_uid),
                    visited_real_uid=ui(exposed_real_uid),
                    exposition=True,
                    timestamp=ui(np.random.randint(max_timestamp + 1)),
                ))
            # now, generate all humans with the spurious thingy tag so we dont have to set expo flags
            humans = [
                FakeHuman(
                    real_uid=idx,
                    exposition_timestamp=never,
                    visits_to_adopt=visits,
                    allow_spurious_exposition=True,
                ) for idx in range(n_humans)
            ]
            messages = generate_received_messages(humans)  # hopefully this is not too slow
            h0_messages = [msg for msgs in messages[0]["received_messages"].values() for msg in msgs]
            cluster_manager = clu.SimplisticClusterManager(max_history_offset=never)
            cluster_manager.add_messages(h0_messages)
            self.assertLessEqual(
                len(cluster_manager.clusters),
                (mu.message_uid_mask + 1) * (mu.risk_level_mask + 1) * (max_timestamp + 1)
            )


if __name__ == "__main__":
    unittest.main()
