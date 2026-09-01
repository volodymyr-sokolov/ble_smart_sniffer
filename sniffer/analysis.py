"""Live per-address analysis: clustering, baselines, anomaly scoring, interference.

Everything here is incremental.  Nothing re-reads the whole session, because the
GUI updates at 20-30 Hz and packets arrive in bursts; a per-packet cost that
depends on how long the capture has been running would become the bottleneck of
a long session.

The scoring deliberately reports *which* feature fired, not only how far out the
packet was.  An operator looking at a red row needs to know that it was the PA
ramp and the multipath profile that moved, not merely that a number crossed 0.8.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_VECTOR_KEYS, PacketFeatures

# Features whose absolute value is meaningless without a disciplined reference.
# With no GPSDO these are dropped from the scoring vector rather than silently
# contributing receiver drift to a spoofing decision.
UNCALIBRATED_UNSAFE = ("cfo_ppm", "drift_hz", "drift_rate", "symbol_clock_ppm")


def _finite(v: np.ndarray) -> np.ndarray:
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def one_dimensional_split(values: np.ndarray, min_fraction: float = 0.15
                          ) -> tuple[float, float, float, int]:
    """Best two-mode split of a single feature: (sigma, low mean, high mean, n_low).

    Returns the separation in pooled standard deviations.  This exists because
    a distance in the whitened 14-dimensional feature space is the wrong test
    for "is there a second radio here": the one feature that separates gets
    averaged together with thirteen that do not, and the evidence is diluted
    below any usable threshold.

    Measured on a real impersonation -- an nRF52840 advertising a genuine
    device's address -- the full-vector separation came to 0.55 of the clusters'
    own spread, under the alert threshold, while the carrier offset alone was
    two disjoint modes 16 sigma apart with an empty 35 ppm gap between them.

    Both modes must hold at least `min_fraction` of the samples, so a handful of
    outliers cannot masquerade as a population.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 20:
        return 0.0, float("nan"), float("nan"), 0
    v = np.sort(v)
    k_min = max(int(n * min_fraction), 3)
    if n - k_min <= k_min:
        return 0.0, float("nan"), float("nan"), 0

    # Prefix sums give every candidate split's mean and variance in O(1).
    c1 = np.cumsum(v)
    c2 = np.cumsum(v * v)
    ks = np.arange(k_min, n - k_min + 1)
    n_lo = ks.astype(float)
    n_hi = float(n) - n_lo
    m_lo = c1[ks - 1] / n_lo
    m_hi = (c1[-1] - c1[ks - 1]) / n_hi
    var_lo = np.maximum(c2[ks - 1] / n_lo - m_lo**2, 0.0)
    var_hi = np.maximum((c2[-1] - c2[ks - 1]) / n_hi - m_hi**2, 0.0)
    pooled = np.sqrt((var_lo + var_hi) / 2.0)
    sigma = np.where(pooled > 1e-12, (m_hi - m_lo) / np.maximum(pooled, 1e-12), 0.0)
    best = int(np.argmax(sigma))
    return float(sigma[best]), float(m_lo[best]), float(m_hi[best]), int(ks[best])


# --------------------------------------------------------------------------
# streaming statistics
# --------------------------------------------------------------------------

class RunningStats:
    """Welford mean/variance over a feature vector, with a per-dimension count."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self.counts = np.zeros(dim)

    def update(self, x: np.ndarray) -> None:
        ok = np.isfinite(x)
        if not ok.any():
            return
        self.n += 1
        self.counts += ok
        delta = np.where(ok, x - self.mean, 0.0)
        self.mean += np.where(ok, delta / np.maximum(self.counts, 1), 0.0)
        delta2 = np.where(ok, x - self.mean, 0.0)
        self.m2 += delta * delta2

    @property
    def variance(self) -> np.ndarray:
        return np.where(self.counts > 1, self.m2 / np.maximum(self.counts - 1, 1), 0.0)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 0.0))


# --------------------------------------------------------------------------
# streaming k-means
# --------------------------------------------------------------------------

@dataclass
class Cluster:
    centroid: np.ndarray
    count: int = 0
    last_seen: float = 0.0
    # Running mean squared distance of members from the centroid: the cluster's
    # own spread, which is what any separation has to be judged against.
    mean_sq_radius: float = 0.0


class StreamingKMeans:
    """Online clustering of whitened feature vectors, per advertising address.

    Split-on-distance rather than fixed-k: the question is not "which of k
    groups is this" but "is there more than one radio behind this address", and
    that needs a model that can decide the answer is two.
    """

    def __init__(self, dim: int, max_clusters: int = 4, split_sigma: float = 1.30,
                 decay: float = 0.02) -> None:
        # `split_sigma` is in per-dimension sigma, so it means the same thing
        # regardless of how many features the vector carries.  1.30 was chosen
        # by measurement, not by taste: with global whitening and both radios
        # present, the per-dimension whitened distance between packets of the
        # same radio has a median of 0.99 and a 95th percentile of 1.46, while
        # between different radios the median is 1.71 with a 5th percentile of
        # 1.43.  Those distributions overlap for individual packets, so no
        # per-point threshold is clean -- but the cluster centroids still
        # converge to the two true means, and 1.30 is where a second cluster
        # forms for two radios and does not form for one.
        self.dim = dim
        self.max_clusters = max_clusters
        self.split_sigma = split_sigma
        self.decay = decay
        self.clusters: list[Cluster] = []

    def update(self, x: np.ndarray, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        x = _finite(x)
        if not self.clusters:
            self.clusters.append(Cluster(x.copy(), 1, now))
            return 0

        # Distances are per-dimension RMS, not raw norms.  A raw norm over a
        # 14-dimensional whitened vector has a typical length of sqrt(14) even
        # when every component is an ordinary one-sigma fluctuation, so a fixed
        # norm threshold splits a single device into three or four clusters and
        # then has nothing left to say when a second radio really does appear.
        root_dim = np.sqrt(self.dim)
        d = [float(np.linalg.norm(x - c.centroid)) / root_dim for c in self.clusters]
        i = int(np.argmin(d))
        # A point far from every centroid starts a new cluster, up to the cap.
        if d[i] > self.split_sigma and len(self.clusters) < self.max_clusters:
            self.clusters.append(Cluster(x.copy(), 1, now))
            return len(self.clusters) - 1

        c = self.clusters[i]
        c.count += 1
        rate = max(self.decay, 1.0 / c.count)
        c.centroid += rate * (x - c.centroid)
        c.mean_sq_radius += rate * (d[i] ** 2 - c.mean_sq_radius)
        c.last_seen = now
        return i

    def separation(self, min_count: int = 8) -> float:
        """How separated the two main clusters are, in units of their own spread.

        A raw centroid distance is not a usable alarm threshold.  Feature scatter
        differs enormously between devices and between signal levels, so a fixed
        distance fires constantly on a noisy device and never on a quiet one.
        Measured on air, a raw-distance test raised repeated "two radios" alerts
        for a single ordinary phone.

        This returns distance divided by the RMS radius of the two clusters --
        essentially a separation-to-spread ratio -- which is scale-free and means
        the same thing for every device.  Both clusters must also be genuinely
        populated: a handful of outliers is not a second transmitter.
        """
        live = [c for c in self.clusters if c.count >= min_count]
        if len(live) < 2:
            return 0.0
        live.sort(key=lambda c: -c.count)
        a, b = live[0], live[1]
        dist = float(np.linalg.norm(a.centroid - b.centroid)) / np.sqrt(self.dim)
        spread = float(np.sqrt(max(a.mean_sq_radius + b.mean_sq_radius, 1e-12)))
        return dist / spread

    def populous(self, min_count: int = 3) -> list[Cluster]:
        return [c for c in self.clusters if c.count >= min_count]


# --------------------------------------------------------------------------
# enrolled baselines
# --------------------------------------------------------------------------

@dataclass
class EnrollmentProgress:
    """Whether a baseline has seen enough variety to be trusted.

    A baseline collected in one sitting at one distance encodes the room as much
    as the radio.  Requiring several ambient temperatures and at least two
    distances is what stops the anomaly score from firing the first time the
    legitimate device is carried across the building.
    """

    packets: int = 0
    temperatures: set = field(default_factory=set)
    rssi_bands: set = field(default_factory=set)
    min_packets: int = 200
    min_temperatures: int = 3
    min_distances: int = 2

    def observe(self, temperature_c: float, rssi_dbfs: float) -> None:
        self.packets += 1
        if math.isfinite(temperature_c):
            self.temperatures.add(round(temperature_c))  # 1 C buckets
        if math.isfinite(rssi_dbfs):
            self.rssi_bands.add(int(rssi_dbfs // 6))  # ~6 dB ~ 2x distance

    @property
    def valid(self) -> bool:
        return (
            self.packets >= self.min_packets
            and len(self.temperatures) >= self.min_temperatures
            and len(self.rssi_bands) >= self.min_distances
        )

    def summary(self) -> dict:
        return {
            "packets": self.packets,
            "packets_required": self.min_packets,
            "temperatures": len(self.temperatures),
            "temperatures_required": self.min_temperatures,
            "distances": len(self.rssi_bands),
            "distances_required": self.min_distances,
            "valid": self.valid,
        }

    def blocking(self) -> list[str]:
        out = []
        if self.packets < self.min_packets:
            out.append(f"{self.packets}/{self.min_packets} packets")
        if len(self.temperatures) < self.min_temperatures:
            out.append(f"{len(self.temperatures)}/{self.min_temperatures} temperatures")
        if len(self.rssi_bands) < self.min_distances:
            out.append(f"{len(self.rssi_bands)}/{self.min_distances} distances")
        return out


class Baseline:
    """Enrolled reference statistics for one address, plus Mahalanobis scoring."""

    def __init__(self, address: str, keys: tuple[str, ...] = FEATURE_VECTOR_KEYS) -> None:
        self.address = address
        self.keys = keys
        self.stats = RunningStats(len(keys))
        self.progress = EnrollmentProgress()
        self.ramp_mean: np.ndarray | None = None
        self.delay_profile_mean: np.ndarray | None = None
        self._ramp_n = 0
        self._dp_n = 0

    def observe(self, feats: PacketFeatures, temperature_c: float) -> None:
        self.stats.update(feats.vector())
        self.progress.observe(temperature_c, feats.value("rssi_dbfs"))
        if feats.ramp_vector.size:
            v = feats.ramp_vector.astype(np.float64)
            self.ramp_mean = v.copy() if self.ramp_mean is None else (
                self.ramp_mean + (v - self.ramp_mean) / (self._ramp_n + 1)
            )
            self._ramp_n += 1
        if feats.delay_profile.size:
            v = feats.delay_profile.astype(np.float64)
            if self.delay_profile_mean is None or self.delay_profile_mean.size != v.size:
                self.delay_profile_mean = v.copy()
                self._dp_n = 1
            else:
                self.delay_profile_mean += (v - self.delay_profile_mean) / (self._dp_n + 1)
                self._dp_n += 1

    def score(self, feats: PacketFeatures, calibrated: bool) -> tuple[float, dict]:
        """Mahalanobis distance in whitened space, with per-feature contributions.

        Diagonal covariance only.  With a few hundred enrollment packets a full
        covariance over 14 dimensions is badly conditioned, and inverting it
        turns estimation noise into confident-looking anomalies.
        """
        if self.stats.n < 10:
            return float("nan"), {}
        x = feats.vector()
        mu = self.stats.mean
        sd = self.stats.std
        contrib: dict[str, float] = {}
        total = 0.0
        used = 0
        for i, key in enumerate(self.keys):
            if not calibrated and key in UNCALIBRATED_UNSAFE:
                continue
            if not np.isfinite(x[i]) or self.stats.counts[i] < 5:
                continue
            s = sd[i]
            if s <= 0 or not np.isfinite(s):
                continue
            z = (x[i] - mu[i]) / s
            contrib[key] = float(z)
            total += z * z
            used += 1
        if used == 0:
            return float("nan"), {}
        # Normalised so the score is a per-dimension RMS sigma, not a raw
        # distance that grows with however many features happened to be finite.
        return float(np.sqrt(total / used)), contrib

    def as_dict(self) -> dict:
        return {
            "address": self.address,
            "keys": list(self.keys),
            "n": self.stats.n,
            "mean": self.stats.mean.tolist(),
            "std": self.stats.std.tolist(),
            "counts": self.stats.counts.tolist(),
            "progress": self.progress.summary(),
            "ramp_mean": None if self.ramp_mean is None else self.ramp_mean.tolist(),
        }


# --------------------------------------------------------------------------
# per-address state
# --------------------------------------------------------------------------

@dataclass
class AddressState:
    address: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    count: int = 0
    rssi: deque = field(default_factory=lambda: deque(maxlen=256))
    intervals: deque = field(default_factory=lambda: deque(maxlen=256))
    last_timestamp_us: float = float("nan")
    delay_profile_ref: np.ndarray | None = None
    delay_dist: deque = field(default_factory=lambda: deque(maxlen=64))
    feature_history: dict = field(default_factory=dict)
    kmeans: StreamingKMeans | None = None
    cluster_counts: Counter = field(default_factory=Counter)
    event_window: deque = field(default_factory=lambda: deque(maxlen=8))

    @property
    def rssi_variance(self) -> float:
        return float(np.var(self.rssi)) if len(self.rssi) > 1 else float("nan")

    @property
    def adv_interval_ms(self) -> float:
        return float(np.median(self.intervals)) / 1000.0 if self.intervals else float("nan")

    @property
    def adv_interval_jitter_ms(self) -> float:
        if len(self.intervals) < 4:
            return float("nan")
        return float(np.std(self.intervals)) / 1000.0

    def adv_delay_uniformity(self) -> float:
        """KS-style statistic of advDelay against the uniform 0-10 ms model.

        BLE adds a uniform 0-10 ms dither to every advertising event.  A
        transmitter that reproduces an address but not that distribution --
        a software spoofer on a fixed timer, typically -- shows up here.
        Returns 0 for a perfect match, 1 for maximally non-uniform.
        """
        if len(self.intervals) < 20:
            return float("nan")
        d = np.asarray(self.intervals, dtype=float)
        base = np.min(d)
        delay = np.clip(d - base, 0.0, 10000.0) / 10000.0
        s = np.sort(delay)
        n = len(s)
        emp = np.arange(1, n + 1) / n
        return float(np.max(np.abs(s - emp)))


class LiveAnalyzer:
    """Running analysis across all addresses seen in the session."""

    def __init__(
        self,
        keys: tuple[str, ...] = FEATURE_VECTOR_KEYS,
        enroll_addresses: tuple[str, ...] = (),
        anomaly_threshold: float = 3.0,
        cluster_split_sigma: float = 1.30,
        cluster_separation_ratio: float = 0.9,
        bimodal_sigma: float = 6.0,
    ) -> None:
        self.keys = keys
        self.enroll_addresses = {a.upper() for a in enroll_addresses}
        self.anomaly_threshold = anomaly_threshold
        self.cluster_split_sigma = cluster_split_sigma
        self.cluster_separation_ratio = cluster_separation_ratio
        # A single unimodal population still yields about 2.5-3 sigma from an
        # optimally placed split, so the threshold has to clear that with room.
        self.bimodal_sigma = bimodal_sigma
        self.bimodal_keys = tuple(self.keys) + ("aoa_deg",)
        self.addresses: dict[str, AddressState] = {}
        self.baselines: dict[str, Baseline] = {}
        self.global_stats = RunningStats(len(keys))
        self.cluster_labels: dict[tuple[str, int], int] = {}
        self._next_label = 0

    # ------------------------------------------------------------------
    def enrolling(self, address: str) -> bool:
        return address.upper() in self.enroll_addresses

    def _whiten(self, v: np.ndarray) -> np.ndarray:
        sd = self.global_stats.std
        mu = self.global_stats.mean
        safe = np.where(sd > 1e-12, sd, 1.0)
        return _finite((v - mu) / safe)

    # ------------------------------------------------------------------
    def observe(self, rec) -> None:
        """Update all running state from one packet record, in place."""
        if rec.is_event or not rec.crc_ok or rec.features is None or not rec.adva:
            return

        addr = rec.adva
        st = self.addresses.get(addr)
        if st is None:
            st = AddressState(addr, first_seen=rec.wall_time)
            st.kmeans = StreamingKMeans(len(self.keys), split_sigma=self.cluster_split_sigma)
            self.addresses[addr] = st

        st.count += 1
        st.last_seen = rec.wall_time
        if math.isfinite(rec.rssi_dbfs):
            st.rssi.append(rec.rssi_dbfs)

        if math.isfinite(st.last_timestamp_us):
            gap = rec.timestamp_us - st.last_timestamp_us
            if 0 < gap < 11_000_000:  # ignore absurd gaps across retunes
                st.intervals.append(gap)
        st.last_timestamp_us = rec.timestamp_us

        vec = rec.features.vector()
        self.global_stats.update(vec)

        # --- clustering in whitened space -----------------------------
        w = self._whiten(vec)
        local = st.kmeans.update(w, rec.wall_time)
        key = (addr, local)
        if key not in self.cluster_labels:
            self.cluster_labels[key] = self._next_label
            self._next_label += 1
        rec.cluster_id = self.cluster_labels[key]
        st.cluster_counts[local] += 1

        # --- enrollment or scoring ------------------------------------
        if self.enrolling(addr):
            bl = self.baselines.setdefault(addr, Baseline(addr, self.keys))
            bl.observe(rec.features, rec.temperature_c)

        bl = self.baselines.get(addr)
        if bl is not None and bl.stats.n >= 10:
            score, contrib = bl.score(rec.features, rec.calibrated)
            rec.anomaly_score = score
            rec.anomaly_contributions = contrib

        rec.alerts = self._alerts(rec, st, bl)

    # ------------------------------------------------------------------
    def _alerts(self, rec, st: AddressState, bl: Baseline | None) -> list[str]:
        alerts: list[str] = []

        sep = st.kmeans.separation() if st.kmeans else 0.0
        if sep > self.cluster_separation_ratio and st.count >= 40:
            alerts.append(
                f"two separated feature clusters under {rec.adva} "
                f"(separation {sep:.1f}x their own spread)"
            )

        # Per-feature bimodality.  The full-vector test above answers "are these
        # packets far apart overall"; this one answers "does any single
        # measurement split into two populations", which is what a second radio
        # actually looks like and is far more sensitive.
        #
        # A carrier offset measured without a disciplined reference is not
        # comparable across sessions -- but the receiver's drift is common to
        # every device measured in the *same* session, so a split within one
        # session is still real evidence.  That is why the uncalibrated
        # exclusion list does not apply here.
        # Bearing is scanned for bimodality too, though it is not part of the
        # clustering vector: it is usually absent (single antenna), but when a
        # second antenna is fitted it is an independent detector -- two
        # transmitters sharing an address are rarely in the same direction, and
        # that holds even if their oscillators happen to match.
        for key in self.bimodal_keys:
            v = rec.features.value(key)
            if math.isfinite(v):
                hist = st.feature_history.setdefault(key, deque(maxlen=384))
                hist.append(v)
        if st.count >= 60 and st.count % 16 == 0:
            best = (0.0, "", 0.0, 0.0, 0)
            for key, hist in st.feature_history.items():
                if len(hist) < 40:
                    continue
                sigma, lo, hi, n_lo = one_dimensional_split(np.fromiter(hist, float))
                if sigma > best[0]:
                    best = (sigma, key, lo, hi, n_lo)
            if best[0] > self.bimodal_sigma:
                sigma, key, lo, hi, n_lo = best
                total = len(st.feature_history[key])
                alerts.append(
                    f"{key} under {rec.adva} splits into two populations "
                    f"{sigma:.1f} sigma apart ({lo:.3g} and {hi:.3g}; "
                    f"{n_lo} and {total - n_lo} packets) -- consistent with two "
                    f"transmitters sharing one address"
                )

        # duplicate address inside one advertising event (~20 ms window)
        st.event_window.append(rec.timestamp_us)
        if len(st.event_window) >= 2:
            recent = [t for t in st.event_window if rec.timestamp_us - t < 20_000]
            if len(recent) > 3:
                alerts.append(
                    f"{len(recent)} packets from {rec.adva} inside one advertising event"
                )

        if bl is not None and math.isfinite(rec.anomaly_score):
            if not bl.progress.valid:
                pass  # an incomplete baseline must not raise alerts
            elif rec.anomaly_score > self.anomaly_threshold:
                worst = sorted(
                    rec.anomaly_contributions.items(), key=lambda kv: -abs(kv[1])
                )[:3]
                detail = ", ".join(f"{k} {v:+.1f}s" for k, v in worst)
                alerts.append(
                    f"feature vector {rec.anomaly_score:.1f} sigma outside baseline ({detail})"
                )

        # Multipath step change for a device that should be stationary.
        #
        # The threshold has to be learned per address, not fixed.  An indoor
        # delay-power profile varies packet to packet by an amount that depends
        # entirely on the room, and a fixed L2 limit fires on every packet in a
        # reflective environment while missing a real change in an open one.
        # Measured on air, a fixed threshold of 1.5 produced 19 alerts in 14
        # seconds from ordinary stationary traffic -- which is worse than no
        # alert at all, because an operator learns to ignore it.
        if rec.features is not None and rec.features.delay_profile.size:
            prof = rec.features.delay_profile.astype(np.float64)
            if st.delay_profile_ref is not None and st.delay_profile_ref.size == prof.size:
                d = float(np.linalg.norm(prof - st.delay_profile_ref))
                st.delay_dist.append(d)
                if len(st.delay_dist) >= 32:
                    hist = np.asarray(st.delay_dist)
                    mu = float(np.median(hist))
                    # median absolute deviation: robust to the step we are
                    # trying to detect contaminating its own threshold
                    mad = float(np.median(np.abs(hist - mu))) * 1.4826
                    if mad > 1e-6 and d > mu + 6.0 * mad:
                        alerts.append(
                            f"multipath profile step change "
                            f"(L2 {d:.2f}, {(d - mu) / mad:.1f} MAD above this "
                            f"device's own baseline of {mu:.2f})"
                        )
                st.delay_profile_ref = 0.95 * st.delay_profile_ref + 0.05 * prof
            else:
                st.delay_profile_ref = prof.copy()

        return alerts

    # ------------------------------------------------------------------
    def enrollment_status(self) -> dict:
        return {a: b.progress.summary() for a, b in self.baselines.items()}

    def address_summary(self) -> list[dict]:
        out = []
        for a, st in self.addresses.items():
            out.append(
                {
                    "address": a,
                    "packets": st.count,
                    "rssi_mean": float(np.mean(st.rssi)) if st.rssi else float("nan"),
                    "rssi_variance": st.rssi_variance,
                    "adv_interval_ms": st.adv_interval_ms,
                    "adv_interval_jitter_ms": st.adv_interval_jitter_ms,
                    "adv_delay_uniformity": st.adv_delay_uniformity(),
                    "clusters": len(st.kmeans.populous()) if st.kmeans else 0,
                    "cluster_separation": st.kmeans.separation() if st.kmeans else 0.0,
                }
            )
        return sorted(out, key=lambda d: -d["packets"])


# --------------------------------------------------------------------------
# interference monitor
# --------------------------------------------------------------------------

INTERFERENCE_CLASSES = (
    "benign",
    "cw",
    "swept",
    "chirped",
    "wideband noise",
    "reactive",
    "wifi/coexistence",
)


@dataclass
class InterferenceReport:
    noise_floor_dbfs: float = float("nan")
    occupied_bandwidth_hz: float = float("nan")
    duty_cycle: float = 0.0
    spectral_kurtosis: float = float("nan")
    envelope_kurtosis: float = float("nan")
    classification: str = "benign"
    confidence: float = 0.0
    reaction_latency_us: float = float("nan")
    on_off_period_ms: float = float("nan")
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "noise_floor_dbfs": self.noise_floor_dbfs,
            "occupied_bandwidth_hz": self.occupied_bandwidth_hz,
            "duty_cycle": self.duty_cycle,
            "spectral_kurtosis": self.spectral_kurtosis,
            "envelope_kurtosis": self.envelope_kurtosis,
            "classification": self.classification,
            "confidence": self.confidence,
            "reaction_latency_us": self.reaction_latency_us,
            "on_off_period_ms": self.on_off_period_ms,
            "detail": self.detail,
        }


class InterferenceMonitor:
    """Characterises non-BLE energy on the channel, independently of decoding.

    Runs on the same blocks the demodulator sees but does not depend on it: the
    whole point is to describe a channel on which decoding is failing.
    """

    def __init__(self, sample_rate: float, history: int = 240) -> None:
        self.sample_rate = sample_rate
        self.noise_history: deque = deque(maxlen=history)
        self.duty_history: deque = deque(maxlen=history)
        self.bw_history: deque = deque(maxlen=history)
        self.last_report = InterferenceReport()
        self.preamble_times: deque = deque(maxlen=64)
        self.onset_times: deque = deque(maxlen=64)
        self.bit_error_positions: Counter = Counter()
        self.pdr_bins: dict = defaultdict(lambda: [0, 0])  # rssi bin -> [ok, total]

    # ------------------------------------------------------------------
    def observe_block(self, iq: np.ndarray, noise_floor_dbfs: float) -> InterferenceReport:
        rep = InterferenceReport(noise_floor_dbfs=noise_floor_dbfs)
        if iq.size < 1024:
            return rep

        env = np.abs(iq[:8192]).astype(np.float64)
        power = env**2
        # The occupancy threshold has to come from the tracked channel noise
        # floor, not from a quantile of this block.  A constant-envelope emitter
        # -- an unmodulated carrier being the obvious case -- has an almost
        # degenerate power distribution, so its own 20th percentile sits at its
        # own level and a self-referential threshold reports 0% duty for a
        # signal that is on continuously.
        if math.isfinite(noise_floor_dbfs):
            thresh = (10.0 ** (noise_floor_dbfs / 10.0)) * 6.0
        else:
            thresh = float(np.quantile(power, 0.2)) * 6.0
        on = power > thresh
        rep.duty_cycle = float(on.mean())

        # --- spectrum ---------------------------------------------------
        n = 4096 if iq.size >= 4096 else int(2 ** np.floor(np.log2(iq.size)))
        seg = iq[:n].astype(np.complex128) * np.hanning(n)
        spec = np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / self.sample_rate))
        total = spec.sum()
        if total > 0:
            # 99% occupied bandwidth
            csum = np.cumsum(spec) / total
            lo = freqs[np.searchsorted(csum, 0.005)]
            hi = freqs[min(np.searchsorted(csum, 0.995), n - 1)]
            rep.occupied_bandwidth_hz = float(hi - lo)
            p = spec / total
            # Spectral kurtosis separates a modulated carrier from noise-like
            # energy of the same total power.
            mu = p.mean()
            sd = p.std()
            rep.spectral_kurtosis = (
                float(np.mean((p - mu) ** 4) / (sd**4)) if sd > 0 else float("nan")
            )

        m = env.mean()
        s = env.std()
        rep.envelope_kurtosis = float(np.mean((env - m) ** 4) / s**4) if s > 0 else float("nan")

        self.noise_history.append(noise_floor_dbfs)
        self.duty_history.append(rep.duty_cycle)
        self.bw_history.append(rep.occupied_bandwidth_hz)

        rep.on_off_period_ms = self._periodicity(on)
        rep.classification, rep.confidence, rep.detail = self._classify(rep)
        rep.reaction_latency_us = self.reaction_latency()
        self.last_report = rep
        return rep

    # ------------------------------------------------------------------
    def _periodicity(self, on: np.ndarray) -> float:
        """Dominant on/off repetition period, via autocorrelation of the gate."""
        if on.size < 512 or on.mean() in (0.0, 1.0):
            return float("nan")
        x = on.astype(np.float64) - on.mean()
        n = int(2 ** np.ceil(np.log2(2 * x.size)))
        ac = np.fft.irfft(np.abs(np.fft.rfft(x, n)) ** 2)[: x.size]
        if ac[0] <= 0:
            return float("nan")
        ac = ac / ac[0]
        lo = max(int(20e-6 * self.sample_rate), 4)
        if lo >= ac.size:
            return float("nan")
        k = int(np.argmax(ac[lo:])) + lo
        if ac[k] < 0.25:
            return float("nan")
        return float(k / self.sample_rate * 1e3)

    def _classify(self, rep: InterferenceReport) -> tuple[str, float, str]:
        bw = rep.occupied_bandwidth_hz
        duty = rep.duty_cycle
        sk = rep.spectral_kurtosis

        if duty < 0.02:
            return "benign", 0.9, "channel essentially idle"
        if math.isfinite(self.reaction_latency()) and self.reaction_latency() < 50.0:
            return (
                "reactive",
                0.85,
                "energy onset within microseconds of preamble start -- "
                "consistent with a reactive emitter, not with coexistence",
            )
        if math.isfinite(bw) and bw < 200e3 and duty > 0.5:
            return "cw", 0.8, "narrowband, near-continuous"
        if math.isfinite(bw) and bw > 3e6 and math.isfinite(sk) and sk < 6:
            return "wideband noise", 0.7, "wide and noise-like (low spectral kurtosis)"
        if math.isfinite(rep.on_off_period_ms) and 0.5 < rep.on_off_period_ms < 20:
            return (
                "wifi/coexistence",
                0.6,
                f"bursty with a {rep.on_off_period_ms:.1f} ms period, "
                "typical of 802.11 or a microwave oven",
            )
        if len(self.bw_history) > 8:
            spread = float(np.nanstd(list(self.bw_history)))
            if spread > 500e3:
                return "swept", 0.6, "occupied bandwidth varying block to block"
        if math.isfinite(sk) and sk > 40:
            return "chirped", 0.5, "highly peaked spectrum moving between blocks"
        return "benign", 0.5, "no distinctive interference signature"

    # ------------------------------------------------------------------
    def note_preamble(self, t_us: float) -> None:
        self.preamble_times.append(t_us)

    def note_energy_onset(self, t_us: float) -> None:
        self.onset_times.append(t_us)

    def reaction_latency(self) -> float:
        """Median delay from preamble start to interfering-energy onset.

        This is the single most diagnostic measurement available here.  A jammer
        that listens and then keys within microseconds is reactive; ordinary
        coexistence traffic has no such relationship to our preambles and gives
        a latency distribution that is broad and unrelated.
        """
        if not self.preamble_times or not self.onset_times:
            return float("nan")
        pre = np.asarray(self.preamble_times)
        ons = np.asarray(self.onset_times)
        lat = []
        for t in ons:
            before = pre[pre <= t]
            if before.size:
                lat.append(t - before[-1])
        if len(lat) < 3:
            return float("nan")
        return float(np.median(lat))

    # ------------------------------------------------------------------
    def note_crc_failure(self, error_positions: np.ndarray) -> None:
        for p in np.asarray(error_positions).tolist():
            self.bit_error_positions[int(p)] += 1

    def note_delivery(self, rssi_dbfs: float, ok: bool) -> None:
        if not math.isfinite(rssi_dbfs):
            return
        b = int(rssi_dbfs // 3) * 3
        self.pdr_bins[b][1] += 1
        if ok:
            self.pdr_bins[b][0] += 1

    def pdr_vs_rssi(self) -> list[tuple[float, float, int]]:
        """(rssi bin, delivery ratio, count), ascending.

        A collapse at *high* received power is the signature of interference
        rather than distance, and is why this is plotted against RSSI instead of
        being reported as a single number.
        """
        out = []
        for b, (ok, total) in sorted(self.pdr_bins.items()):
            if total >= 3:
                out.append((float(b), ok / total, total))
        return out

    def bit_error_profile(self) -> tuple[np.ndarray, str]:
        """Histogram of bit-error positions over CRC-failed packets.

        A uniform spread points at broadband noise; errors bunched toward the
        tail point at a collision or an emitter that keys partway through.
        """
        if not self.bit_error_positions:
            return np.zeros(0), "no CRC failures recorded"
        top = max(self.bit_error_positions) + 1
        hist = np.zeros(top)
        for p, c in self.bit_error_positions.items():
            hist[p] = c
        if hist.sum() <= 0:
            return hist, "no CRC failures recorded"
        norm = hist / hist.sum()
        half = len(norm) // 2
        tail = norm[half:].sum()
        if tail > 0.7:
            verdict = "errors clustered late in the packet: collision or reactive emitter"
        elif tail < 0.3:
            verdict = "errors clustered early: likely sync or ramp related"
        else:
            verdict = "errors spread uniformly: consistent with broadband noise"
        return hist, verdict
