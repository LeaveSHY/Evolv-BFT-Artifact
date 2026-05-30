import json
import tempfile
import unittest
from pathlib import Path

from generate_config import write_runtime_manifests


class WriteRuntimeManifestsTest(unittest.TestCase):
    def make_manifest(self) -> dict:
        return {
            "version": 1,
            "nodes_by_id": {
                0: {
                    "id": 0,
                    "peer_id": "peer0",
                    "public_key_hex": "pub0",
                    "private_key_hex": "priv0",
                    "vrf_public_key_hex": "vrfpub0",
                    "vrf_private_key_hex": "vrfpriv0",
                },
                1: {
                    "id": 1,
                    "peer_id": "peer1",
                    "public_key_hex": "pub1",
                    "private_key_hex": "priv1",
                    "vrf_public_key_hex": "vrfpub1",
                    "vrf_private_key_hex": "vrfpriv1",
                },
            },
        }

    def make_peers(self) -> list[dict]:
        return [
            {"node_id": 0, "multiaddr": "/dns/node-0/tcp/8080/p2p/peer0"},
            {"node_id": 1, "multiaddr": "/dns/node-1/tcp/8080/p2p/peer1"},
        ]

    def test_strips_private_keys_from_cluster_manifest(self) -> None:
        manifest = self.make_manifest()
        peers = self.make_peers()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cluster_path = write_runtime_manifests(Path(tmp_dir), manifest, peers)
            cluster_manifest = json.loads(cluster_path.read_text(encoding="utf-8"))

        self.assertEqual(len(cluster_manifest["nodes"]), 2)
        for node in cluster_manifest["nodes"]:
            self.assertNotIn("private_key_hex", node)
            self.assertNotIn("vrf_private_key_hex", node)
            self.assertTrue(node["p2p_multiaddr"].startswith("/dns/"))

    def test_keeps_only_local_private_keys(self) -> None:
        manifest = self.make_manifest()
        peers = self.make_peers()

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            write_runtime_manifests(out_dir, manifest, peers)

            node0_manifest = json.loads((out_dir / "manifests" / "node-0-manifest.json").read_text(encoding="utf-8"))
            node1_manifest = json.loads((out_dir / "manifests" / "node-1-manifest.json").read_text(encoding="utf-8"))

        node0_nodes = {node["id"]: node for node in node0_manifest["nodes"]}
        node1_nodes = {node["id"]: node for node in node1_manifest["nodes"]}

        self.assertEqual(node0_nodes[0]["private_key_hex"], "priv0")
        self.assertEqual(node0_nodes[0]["vrf_private_key_hex"], "vrfpriv0")
        self.assertNotIn("private_key_hex", node0_nodes[1])
        self.assertNotIn("vrf_private_key_hex", node0_nodes[1])

        self.assertEqual(node1_nodes[1]["private_key_hex"], "priv1")
        self.assertEqual(node1_nodes[1]["vrf_private_key_hex"], "vrfpriv1")
        self.assertNotIn("private_key_hex", node1_nodes[0])
        self.assertNotIn("vrf_private_key_hex", node1_nodes[0])

    def test_rejects_missing_requested_node(self) -> None:
        manifest = self.make_manifest()
        del manifest["nodes_by_id"][1]
        peers = self.make_peers()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "missing requested node 1"):
                write_runtime_manifests(Path(tmp_dir), manifest, peers)

    def test_rejects_missing_peer_id(self) -> None:
        manifest = self.make_manifest()
        manifest["nodes_by_id"][1]["peer_id"] = ""
        peers = self.make_peers()
        peers[1]["multiaddr"] = ""

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "peer_id"):
                write_runtime_manifests(Path(tmp_dir), manifest, peers)

    def test_rejects_missing_signing_key_material(self) -> None:
        manifest = self.make_manifest()
        manifest["nodes_by_id"][1]["public_key_hex"] = ""
        peers = self.make_peers()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "public_key_hex"):
                write_runtime_manifests(Path(tmp_dir), manifest, peers)

    def test_rejects_missing_vrf_material(self) -> None:
        manifest = self.make_manifest()
        manifest["nodes_by_id"][1]["vrf_private_key_hex"] = ""
        peers = self.make_peers()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "vrf_private_key_hex"):
                write_runtime_manifests(Path(tmp_dir), manifest, peers)


if __name__ == "__main__":
    unittest.main()
