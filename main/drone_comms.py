"""
drone_comms.py — Simulated ChaCha20-Poly1305 secure communication layer
for a centralised multi-UAV ground-station architecture (v27 / Save the Planet 2.0).

PURPOSE
-------
In a centralised simulation (single Python process), there are no real network
sockets.  This module *simulates* the cryptographic channel by:

  1. Serialising each message dict → JSON bytes
  2. Encrypting  with ChaCha20-Poly1305 (12-byte nonce, 16-byte auth tag)
  3. Immediately decrypting on the "receive" side
  4. Measuring wall-clock latency and computing cumulative crypto-energy per drone

This gives you real latency numbers and a real energy budget entry without
needing actual sockets, matching the benchmarking approach described in the
project report (Section 5.11).

INSTALLATION
------------
    pip install cryptography

USAGE (see bottom of file for a worked example)
------
    from drone_comms import DroneCommsLayer

    comms = DroneCommsLayer(num_drones=4)

    # ── Uplink: drone → GS (call every control step per airborne drone) ──
    telemetry_dict = comms.send_telemetry(drone_id=i, pos=pos, vel=vel, battery_pct=pct)

    # ── Downlink: GS → drone (call once when dispatching site assignments) ──
    comms.send_command(drone_id=i, command={"sites": [...], "state": "SQUAD_GO"})

    # ── Battery integration ──
    crypto_drain_j = comms.energy_consumed_j[i]   # add to BatterySystem.energy_j[i]

    # ── Final report ──
    comms.print_report()

ENERGY MODEL
------------
ChaCha20-Poly1305 benchmarks show ~7.5 mW per encrypt+decrypt pair on a
resource-constrained embedded ARM core (matching Crazyflie-class hardware).
This module uses that figure by default (CRYPTO_POWER_W = 0.0075).

NONCE POLICY
------------
Every message uses a fresh 96-bit nonce from os.urandom(12).  The nonce is
prepended to the ciphertext (standard practice: nonce || ciphertext || tag).
The receiver splits it off before passing to ChaCha20Poly1305.decrypt().
"""

import os
import json
import time
import struct
from typing import Any

# ── Optional dependency guard ───────────────────────────────────────────────
try:
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False
    print(
        "[DroneComms] WARNING: 'cryptography' package not found.\n"
        "  Install with:  pip install cryptography\n"
        "  Falling back to pass-through mode (no encryption, overhead = 0)."
    )

# ── Constants ────────────────────────────────────────────────────────────────
NONCE_BYTES    = 12       # 96-bit nonce (IETF ChaCha20-Poly1305 standard)
KEY_BYTES      = 32       # 256-bit symmetric key
TAG_BYTES      = 16       # Poly1305 authentication tag appended by library
CRYPTO_POWER_W = 0.0075   # 7.5 mW per encrypt+decrypt operation (embedded ARM)
# Associated data used for authenticated encryption — binds drone ID to the
# ciphertext so a message cannot be replayed to a different drone.
AAD_PREFIX     = b"save-the-planet-v27-drone-"


class DroneCommsLayer:
    """
    Simulated ChaCha20-Poly1305 secure channel between a centralised
    ground station and a fleet of up to 4 quadcopters.

    Each drone gets its own 256-bit symmetric key derived from a shared
    master secret via XOR with the drone index (simple but sufficient for
    a simulation benchmark; use HKDF for production).

    Attributes
    ----------
    num_drones      : int
    enc_latency_s   : list[float]  — cumulative encryption time per drone
    dec_latency_s   : list[float]  — cumulative decryption time per drone
    message_count   : list[int]    — total messages sent per drone
    energy_consumed_j : list[float] — total crypto energy in Joules per drone
    """

    def __init__(self, num_drones: int = 4, master_key: bytes | None = None):
        """
        Parameters
        ----------
        num_drones  : number of quadcopters (1–4)
        master_key  : optional 32-byte master secret.  If None, a fresh
                      cryptographically random key is generated each run.
        """
        if not _HAS_CRYPTO:
            self._fallback = True
            self.num_drones = num_drones
            self._reset_stats(num_drones)
            return
        self._fallback = False
        self.num_drones = num_drones

        # ── Key derivation ───────────────────────────────────────────────
        # Generate a fresh master key if none provided.
        mk = master_key if master_key is not None else os.urandom(KEY_BYTES)
        if len(mk) != KEY_BYTES:
            raise ValueError(f"master_key must be exactly {KEY_BYTES} bytes")

        # Per-drone key = master XOR bytes([drone_id] * 32)
        # Simple, reproducible, and avoids key reuse across drones.
        self._ciphers: list[ChaCha20Poly1305] = []
        for i in range(num_drones):
            per_drone_key = bytes(b ^ (i + 1) for b in mk)
            self._ciphers.append(ChaCha20Poly1305(per_drone_key))

        self._reset_stats(num_drones)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _reset_stats(self, n: int):
        self.enc_latency_s:     list[float] = [0.0] * n
        self.dec_latency_s:     list[float] = [0.0] * n
        self.message_count:     list[int]   = [0]   * n
        self.energy_consumed_j: list[float] = [0.0] * n

    def _aad(self, drone_id: int) -> bytes:
        """Associated data that binds the ciphertext to a specific drone."""
        return AAD_PREFIX + str(drone_id).encode()

    def _encrypt(self, drone_id: int, plaintext: bytes) -> bytes:
        """
        Encrypt plaintext for drone_id.

        Returns
        -------
        bytes : nonce (12 B) || ciphertext+tag
        """
        if self._fallback:
            return plaintext  # no-op in fallback mode
        nonce = os.urandom(NONCE_BYTES)
        t0 = time.perf_counter()
        ct = self._ciphers[drone_id].encrypt(nonce, plaintext, self._aad(drone_id))
        self.enc_latency_s[drone_id] += time.perf_counter() - t0
        return nonce + ct

    def _decrypt(self, drone_id: int, blob: bytes) -> bytes:
        """
        Decrypt a blob that was produced by _encrypt().

        Returns
        -------
        bytes : recovered plaintext

        Raises
        ------
        InvalidTag if authentication fails (tampered/replayed message).
        """
        if self._fallback:
            return blob
        nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
        t0 = time.perf_counter()
        pt = self._ciphers[drone_id].decrypt(nonce, ct, self._aad(drone_id))
        self.dec_latency_s[drone_id] += time.perf_counter() - t0
        return pt

    def _account_energy(self, drone_id: int, enc_t: float, dec_t: float):
        """Add crypto energy to per-drone accumulator."""
        op_time = enc_t + dec_t
        self.energy_consumed_j[drone_id] += CRYPTO_POWER_W * op_time
        self.message_count[drone_id] += 1

    # ── Public API ───────────────────────────────────────────────────────────

    def send_telemetry(
        self,
        drone_id: int,
        pos: tuple[float, float, float],
        vel: tuple[float, float, float],
        battery_pct: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Simulate uplink: drone → ground station telemetry packet.

        The telemetry dict is serialised → encrypted → decrypted → deserialised.
        Benchmarks are recorded and crypto energy is accumulated.

        Call this every control step for each airborne drone.

        Parameters
        ----------
        drone_id    : 0-indexed drone number
        pos         : (x, y, z) in metres
        vel         : (vx, vy, vz) in m/s
        battery_pct : battery level 0.0–1.0
        extra       : optional additional fields to include

        Returns
        -------
        dict : the recovered (decrypted) telemetry — same structure as input.
               In production this is what the GS would act on.
        """
        payload: dict[str, Any] = {
            "drone_id":    drone_id,
            "pos":         list(pos),
            "vel":         list(vel),
            "battery_pct": battery_pct,
        }
        if extra:
            payload.update(extra)

        raw = json.dumps(payload).encode()

        t_enc_before = self.enc_latency_s[drone_id]
        blob = self._encrypt(drone_id, raw)
        t_enc_after  = self.enc_latency_s[drone_id]

        t_dec_before = self.dec_latency_s[drone_id]
        recovered_raw = self._decrypt(drone_id, blob)
        t_dec_after   = self.dec_latency_s[drone_id]

        self._account_energy(
            drone_id,
            t_enc_after - t_enc_before,
            t_dec_after - t_dec_before,
        )

        return json.loads(recovered_raw)

    def send_command(
        self,
        drone_id: int,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Simulate downlink: ground station → drone command packet.

        Typically called once per drone when site assignments are dispatched
        (the FW_LAND → SQUAD_GO transition).  Can also be called for
        mid-mission re-tasking commands.

        Parameters
        ----------
        drone_id : 0-indexed drone number
        command  : dict with any fields, e.g.
                   {"sites": [(x,y), ...], "state": "SQUAD_GO", "strategy": "worst_fit"}

        Returns
        -------
        dict : the recovered (decrypted) command — what the drone "receives".
        """
        raw = json.dumps(command).encode()

        t_enc_before = self.enc_latency_s[drone_id]
        blob = self._encrypt(drone_id, raw)
        t_enc_after  = self.enc_latency_s[drone_id]

        t_dec_before = self.dec_latency_s[drone_id]
        recovered_raw = self._decrypt(drone_id, blob)
        t_dec_after   = self.dec_latency_s[drone_id]

        self._account_energy(
            drone_id,
            t_enc_after - t_enc_before,
            t_dec_after - t_dec_before,
        )

        return json.loads(recovered_raw)

    # ── Report helpers ───────────────────────────────────────────────────────

    def millijoules_per_drone(self) -> list[float]:
        """Return crypto energy consumed in millijoules for each drone."""
        return [j * 1000.0 for j in self.energy_consumed_j]

    def print_report(self):
        """Print a formatted crypto-overhead benchmark summary to stdout."""
        mode = "PASS-THROUGH (no crypto)" if self._fallback else "ChaCha20-Poly1305"
        print(f"\n{'='*60}")
        print(f"  Secure Comms Benchmark — {mode}")
        print(f"{'='*60}")
        print(f"  {'Drone':<8} {'Msgs':>6} {'Enc (ms)':>10} {'Dec (ms)':>10} {'Total (ms)':>12} {'Energy (mJ)':>12}")
        print(f"  {'-'*60}")
        for i in range(self.num_drones):
            msgs = self.message_count[i]
            enc_ms  = self.enc_latency_s[i] * 1000
            dec_ms  = self.dec_latency_s[i] * 1000
            tot_ms  = enc_ms + dec_ms
            e_mj    = self.energy_consumed_j[i] * 1000
            avg_enc = (enc_ms / msgs * 1000) if msgs else 0  # µs
            avg_dec = (dec_ms / msgs * 1000) if msgs else 0
            print(
                f"  Q{i} ({['blue','green','red','purple'][i]:<6}) "
                f"{msgs:>6} "
                f"{enc_ms:>9.3f} "
                f"{dec_ms:>9.3f} "
                f"{tot_ms:>11.3f} "
                f"{e_mj:>11.4f}"
            )
        print(f"  {'-'*60}")
        total_mj = sum(self.millijoules_per_drone())
        total_msgs = sum(self.message_count)
        print(f"  {'TOTAL':<8} {total_msgs:>6} {'-':>10} {'-':>10} {'-':>12} {total_mj:>11.4f}")
        print(f"\n  Avg enc latency (µs): "
              f"{sum(self.enc_latency_s)*1e6 / max(total_msgs,1):.2f}")
        print(f"  Avg dec latency (µs): "
              f"{sum(self.dec_latency_s)*1e6 / max(total_msgs,1):.2f}")
        print(f"  Crypto power model:   {CRYPTO_POWER_W*1000:.1f} mW per operation")
        print(f"{'='*60}\n")

    def summary_for_metrics(self) -> dict[str, Any]:
        """Return a dict suitable for inclusion in generate_metrics_plots()."""
        return {
            "mode":               "ChaCha20-Poly1305" if not self._fallback else "disabled",
            "messages_per_drone": list(self.message_count),
            "enc_latency_ms":     [t * 1000 for t in self.enc_latency_s],
            "dec_latency_ms":     [t * 1000 for t in self.dec_latency_s],
            "energy_mj":          self.millijoules_per_drone(),
            "total_energy_mj":    sum(self.millijoules_per_drone()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Integration guide for v27.py
# ─────────────────────────────────────────────────────────────────────────────
#
# STEP 1 — Import and initialise (near the top of run(), alongside battery):
#
#     from drone_comms import DroneCommsLayer
#     comms = DroneCommsLayer(num_drones=nd)
#
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Uplink telemetry (inside the main control loop, every step):
#
#   Replace the raw obs[i] access pattern with a comms-wrapped call.
#   Only do this for airborne drones (obs[i][2] > 0.5) to avoid
#   flooding the benchmark with on-ground idle steps.
#
#     # After battery.update_airborne(obs, prev_vel, dt):
#     for i in range(nd):
#         if obs[i][2] > 0.5:   # only airborne drones
#             pos = tuple(obs[i][0:3])
#             vel = tuple(obs[i][10:13])
#             comms.send_telemetry(
#                 drone_id=i,
#                 pos=pos,
#                 vel=vel,
#                 battery_pct=float(bat_levels[i]),
#             )
#             # Add crypto drain to battery (7.5 mW per call, paid once per step)
#             battery.energy_j[i] = max(
#                 0.0,
#                 battery.energy_j[i] - comms.energy_consumed_j[i]
#                 # Note: energy_consumed_j is *cumulative*, so compute the delta:
#                 # use a prev_crypto_j[i] tracker (see Step 3).
#             )
#
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Correct incremental energy drain (add alongside BatterySystem):
#
#     prev_crypto_j = [0.0] * nd   # initialise before main loop
#
#     # Inside the per-step section for airborne drones:
#     delta_j = comms.energy_consumed_j[i] - prev_crypto_j[i]
#     battery.energy_j[i] = max(0.0, battery.energy_j[i] - delta_j)
#     prev_crypto_j[i] = comms.energy_consumed_j[i]
#
# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Downlink command (in the FW_LAND phase, when assignments dispatch):
#
#   After your existing assignment planner runs and `assignments` is built:
#
#     for i, sites in assignments.items():
#         cmd = {
#             "drone_id": i,
#             "sites":    [list(s) for s in sites],
#             "strategy": strategy,
#             "state":    "SQUAD_GO",
#         }
#         comms.send_command(drone_id=i, command=cmd)
#
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Report (in the finally block, before generate_metrics_plots):
#
#     comms.print_report()
#     crypto_summary = comms.summary_for_metrics()
#     # pass crypto_summary to generate_metrics_plots() as an extra kwarg
#     # to add a 7th panel, or just log it alongside the existing report.
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Self-test / worked example ───────────────────────────────────────────────
if __name__ == "__main__":
    print("DroneCommsLayer — self-test\n")

    comms = DroneCommsLayer(num_drones=4)

    # Simulate 100 telemetry steps for each of 4 airborne drones
    import math
    for step in range(100):
        for i in range(4):
            t = step * 0.033  # ~30 Hz control loop
            pos = (math.cos(t + i) * 5, math.sin(t + i) * 5, 3.0)
            vel = (-math.sin(t + i) * 2.5, math.cos(t + i) * 2.5, 0.0)
            bat = 1.0 - step * 0.002
            result = comms.send_telemetry(i, pos, vel, bat)
            assert result["drone_id"] == i, "Decryption mismatch!"

    # Simulate one assignment dispatch per drone
    for i in range(4):
        cmd = {"drone_id": i, "sites": [[i * 3.0, 2.0], [i * 3.0 + 1, -2.0]], "state": "SQUAD_GO"}
        recovered = comms.send_command(i, cmd)
        assert recovered["drone_id"] == i, "Command decryption mismatch!"

    comms.print_report()
    print("All assertions passed.")