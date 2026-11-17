"""HTTP tests for the residency status/activate routes."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from acestep.api.dit_residency import DitResidency
from acestep.api.http.residency_routes import register_residency_routes


def _handler(name):
    return SimpleNamespace(model=torch.nn.Linear(2, 2), silence_latent=torch.zeros(1, 2, 2), device="cpu",
                           last_init_params={"config_path": name})


def _app(residency):
    app = FastAPI()
    h1, h2 = _handler("acestep-v15-xl-sft"), _handler("acestep-v15-xl-turbo")
    app.state.handler, app.state.handler2, app.state.handler3 = h1, h2, None
    app.state._config_path, app.state._config_path2, app.state._config_path3 = "acestep-v15-xl-sft", "acestep-v15-xl-turbo", ""
    app.state._initialized, app.state._initialized2 = True, True
    app.state.dit_residency = residency
    app.state.executor = ThreadPoolExecutor(max_workers=1)
    if residency is not None:
        residency.active = h1
        residency.park(h2)
    register_residency_routes(app, verify_api_key=lambda: None, wrap_response=lambda data, code=200, error=None: {"data": data, "code": code, "error": error},
                              get_model_name=lambda p: p.split("/")[-1])
    return app


class ResidencyRoutesTests(unittest.TestCase):
    def test_status_and_activate(self):
        app = _app(DitResidency())
        client = TestClient(app)
        r = client.get("/v1/residency").json()["data"]
        self.assertEqual(r["mode"], "exclusive")
        self.assertEqual(r["active"], "acestep-v15-xl-sft")
        self.assertEqual([s["on_gpu"] for s in r["slots"]], [True, False])
        r = client.post("/v1/residency/activate", json={"model": "acestep-v15-xl-turbo"}).json()
        self.assertEqual(r["data"]["active"], "acestep-v15-xl-turbo")
        self.assertEqual([s["on_gpu"] for s in r["data"]["slots"]], [False, True])

    def test_shared_mode_rejects_activate(self):
        client = TestClient(_app(None))
        self.assertEqual(client.get("/v1/residency").json()["data"]["mode"], "shared")
        self.assertEqual(client.post("/v1/residency/activate", json={"slot": 2}).json()["code"], 400)


if __name__ == "__main__":
    unittest.main()
