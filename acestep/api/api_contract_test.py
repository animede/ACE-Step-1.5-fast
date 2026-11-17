"""Guard the public API contract: the routes external apps rely on must stay registered."""

import os
import unittest

REQUIRED_ROUTES = {
    ("POST", "/release_task"), ("POST", "/query_result"), ("POST", "/format_input"),
    ("POST", "/create_random_sample"), ("GET", "/v1/models"), ("GET", "/v1/audio"),
    ("GET", "/health"), ("GET", "/v1/stats"), ("POST", "/v1/init"),
    ("GET", "/v1/residency"), ("POST", "/v1/residency/activate"),
}


class ApiContractTests(unittest.TestCase):
    def test_required_routes_are_registered(self):
        os.environ["ACESTEP_NO_INIT"] = "true"
        from acestep.api_server import app

        registered = {(m, r.path) for r in app.routes for m in (getattr(r, "methods", None) or [])}
        missing = sorted(REQUIRED_ROUTES - registered)
        self.assertEqual(missing, [], f"missing API routes: {missing}")
        self.assertIn("/ui", {getattr(r, "path", "") for r in app.routes})


if __name__ == "__main__":
    unittest.main()
