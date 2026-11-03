import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import unittest
from app import app
from preprocess import encode_user_profile


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_training_encoding(self):
        self.assertEqual(encode_user_profile({'gender': 'F'})['gender'], 0)
        self.assertEqual(encode_user_profile({'gender': 'M'})['gender'], 1)

    def test_model_and_popular(self):
        for profile, source in [({}, 'popular'),
                                 ({'gender': 'F', 'age': 25, 'occupation': 12}, 'model')]:
            response = self.client.post('/recommend', json=profile)
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(response.json['source'], source)
            self.assertEqual(len(response.json['recommendations']), 10)

    def test_invalid_profile(self):
        for profile in [[], {'gender': 'invalid'}, {'occupation': 100}, {'age': 20}]:
            self.assertEqual(self.client.post('/recommend', json=profile).status_code, 400)

    def test_lantern_lifecycle(self):
        user = 'integration-test'
        self.client.post('/lantern/reset', json={'user_id': user})
        cold = self.client.post('/lantern/recommend', json={'user_id': user})
        self.assertEqual(cold.json['note'], 'popular_fallback')
        response = self.client.get('/similar_items?item_id=1&k=5')
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(len(response.json['similar']), 5)
        self.assertNotIn(1, [r['item_id'] for r in response.json['similar']])
        liked = self.client.post('/event', json={'user_id': user, 'item_id': 1, 'event': 'like'})
        self.assertEqual(liked.status_code, 200, liked.json)
        profile = self.client.get('/lantern/profile', query_string={'user_id': user}).json
        self.assertGreater(profile['u_norm'], 0)
        self.assertEqual(profile['recent'][0]['item_id'], 1)
        self.client.post('/lantern/reset', json={'user_id': user})
        self.assertEqual(self.client.get('/lantern/profile', query_string={'user_id': user}).json['u_norm'], 0)

    def test_exclusions(self):
        import similar
        similar.init_similar()
        excluded = list(map(int, similar.ITEM_ID_TO_INDEX))
        u = similar.ITEM_EMB[0]
        self.assertEqual(similar.recommend_by_u(u, exclude=excluded), [])
        picks = similar.recommend_by_u(u, exclude=excluded[1:])
        self.assertEqual([r['item_id'] for r in picks], excluded[:1])


if __name__ == '__main__':
    unittest.main()
