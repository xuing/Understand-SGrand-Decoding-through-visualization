import itertools
import os
from abc import ABC, abstractmethod
from venv import logger

import numpy as np

from codes import LinearBlockCode


class ChannelDecoder(ABC):
    """Abstract base class for all channel decoders."""

    def __init__(self, code: LinearBlockCode):
        self.code = code

    @abstractmethod
    def decode_to_codeword(self, received_vector: np.ndarray) -> np.ndarray:
        """Decode a received vector and return the estimated codeword."""
        pass


class SyndromeDecoder(ChannelDecoder):
    """Classical syndrome-based hard-decision decoder."""

    def __init__(self, code: LinearBlockCode):
        super().__init__(code)
        self.syndrome_map = {}

        # Zero syndrome entry
        m = code.H.shape[0]
        zero_s = tuple(np.zeros(m, dtype=int))
        self.syndrome_map[zero_s] = np.zeros(code.n, dtype=int)

        # Build syndrome → error lookup table up to weight t
        for w in range(1, code.t + 1):
            for idxs in itertools.combinations(range(code.n), w):
                e = np.zeros(code.n, dtype=int)
                e[list(idxs)] = 1
                s = tuple(e @ code.H.T % 2)
                if s not in self.syndrome_map:
                    self.syndrome_map[s] = e

    def decode_to_codeword(self, y: np.ndarray) -> np.ndarray:
        """Return corrected codeword if syndrome is known; otherwise return y."""
        s = tuple(y @ self.code.H.T % 2)
        if s in self.syndrome_map:
            return (y + self.syndrome_map[s]) % 2
        return y  # no correction possible


class MLDecoder(ChannelDecoder):
    def __init__(self, code: LinearBlockCode):
        super().__init__(code)
        # Precompute all codewords and their BPSK mapping
        self.codewords = self._gen_codewords()  # shape: (2^k, n)
        self.bpsk_codewords = 1 - 2 * self.codewords  # 0→+1, 1→-1

    def _gen_codewords(self) -> np.ndarray:
        all_u = np.array(list(itertools.product((0, 1), repeat=self.code.k)), dtype=int)  # (2^k, k)
        return (all_u @ self.code.G) % 2  # (2^k, n)

    def decode_to_codeword(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)

        if y.ndim != 1 or y.shape[0] != self.code.n:
            raise ValueError(
                f"MLDecoder.decode_to_codeword: y shape {y.shape} incompatible with n={self.code.n}."
            )

        # Soft or hard metric
        if y.dtype.kind == "f":
            # Soft ML: Euclidean metric
            y_row = y.reshape(1, -1)
            dists = np.sum((self.bpsk_codewords - y_row) ** 2, axis=1)
        else:
            # Hard ML: Hamming metric
            y_int = y.astype(int).reshape(1, -1)
            dists = np.sum(np.abs(self.codewords - y_int), axis=1)

        # Select the codeword with minimum distance
        idx = int(np.argmin(dists))
        c_hat = self.codewords[idx].copy()

        # Update query count
        self.code.queries = int(2 ** self.code.k)
        return c_hat


class GRANDDecoder(ChannelDecoder):
    """
    Hard-decision GRAND
    Enumerates error patterns by increasing Hamming weight.
    """

    def __init__(self, code: LinearBlockCode, max_weight: int = None):
        super().__init__(code)
        self.max_weight = max_weight if max_weight is not None else code.n

    def decode_to_codeword(self, y: np.ndarray):
        y = np.asarray(y)

        # Soft input (AWGN / LLR / ±1): use sign-based hard decision
        if y.dtype.kind == "f":
            y = (y < 0).astype(int)
        else:
            # Assume BSC-style 0/1 output; normalize to {0,1}
            y = (y.astype(int) & 1)

        # Check weight-0 error first
        if self.code.is_codeword(y):
            return y

        # Enumerate error patterns up to max_weight
        for w in range(1, self.max_weight + 1):
            # For each weight w, iterate through all combinations of error pattern positions.
            for idxs in itertools.combinations(range(self.code.n), w):
                c_test = y.copy()
                c_test[list(idxs)] ^= 1
                if self.code.is_codeword(c_test):
                    return c_test
        return y


class SGRANDDecoder(ChannelDecoder):
    """
    Soft Maximum Likelihood Decoding using GRAND (SGRAND-AB), approximating the algorithm in the literature.

    Main features:
    - Use soft input y to compute reliability order (OEI).
    - Use priority queue to enumerate error patterns in likelihood order.
    - Track number of queries and optionally stop at max_queries.
    """

    def __init__(self, code: LinearBlockCode, max_queries: int = None):
        super().__init__(code)
        self.max_queries = int(max_queries) if max_queries is not None else int(1e9)

    # ---------- metric: squared Euclidean distance ----------
    def calc_euclidean_distance(self, y: np.ndarray, theta_y: np.ndarray, e: np.ndarray) -> float:
        """
        Calculate Squared Euclidean Distance.

        Mapping:
            bits = (theta_y - e) mod 2
            x = 1 - 2*bits   (BPSK mapping)
        For BPSK over AWGN, minimizing this distance is equivalent to maximizing log-likelihood.
        """
        bits = (theta_y - e) & 1
        x = 1.0 - 2.0 * bits
        return np.sum((x - y) ** 2)

    # ---------- OEI ----------
    def compute_OEI(self, y: np.ndarray) -> np.ndarray:
        """Sort positions by |y| ascending (least reliable first)."""
        reliability = np.abs(y)
        return np.argsort(reliability)

    def expand_children(self, e: np.ndarray, oei: np.ndarray) -> List[np.ndarray]:
        """
        Expand children according to Algorithm 2 / Lemma III.2.

        Notes:
        - The second child must be built from e1 (after adding i_{j*+1}),
          not from the original e.
        """
        n = self.code.n

        ones = np.where(e[oei] == 1)[0]
        if len(ones) == 0:
            j_star = 0
        else:
            j_star = ones[-1] + 1  # convert to 1-based

        children: List[np.ndarray] = []

        if j_star < n:
            # First child: flip position i_{j*+1}
            e1 = e.copy()
            flip_index = oei[j_star]
            e1[flip_index] ^= 1
            children.append(e1.copy())

            # Second child: turn off previous bit i_{j*}
            if j_star > 0:
                e2 = e1.copy()
                off_index = oei[j_star - 1]
                e2[off_index] ^= 1
                children.append(e2)

        return children

    # ---------- main decoding ----------
    def decode_to_codeword(self, y: np.ndarray):
        # Ensure float for metric computation
        y = y.astype(float)
        # Hard decision as initial θ(y)
        theta_y = (y < 0).astype(int)

        # Reset query counter in code object
        self.code.queries = 0

        # Step 1: compute Ordered Error Index (OEI)
        oei = self.compute_OEI(y)

        # Step 2: priority queue initialization
        # Heap elements: (metric, e)
        S: List[Tuple[float, np.ndarray]] = []

        e0 = np.zeros(self.code.n, dtype=int)
        metric_e0 = self.calc_euclidean_distance(y, theta_y, e0)
        heapq.heappush(S, (metric_e0, e0))

        # Step 3: main search loop
        while S and self.code.queries < self.max_queries:
            _, e = heapq.heappop(S)

            # Test candidate codeword
            c_test = (theta_y - e) & 1
            if self.code.is_codeword(c_test):
                return c_test

            # Expand children according to Lemma III.2
            for child_e in self.expand_children(e, oei):
                metric_child = self.calc_euclidean_distance(y, theta_y, child_e)
                heapq.heappush(S, (metric_child, child_e))

        # Termination: erasure (no codeword found within max_queries)
        print("SGRANDDecoder: Maximum queries reached; decoding erasure.")
        return theta_y


#  For tracing version

import heapq
import json
import time
from typing import List, Tuple


def default_converter(o):
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.floating, np.float64, np.float32)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


class SGRANDDecoder_Trace(ChannelDecoder):
    def __init__(self, code: LinearBlockCode, max_queries: int = None, save_trace: bool = True):
        super().__init__(code)
        self.max_queries = int(max_queries) if max_queries is not None else int(1e9)
        self.save_trace = save_trace

    def calc_euclidean_distance(self, y: np.ndarray, theta_y: np.ndarray, e: np.ndarray) -> float:
        """
            Calculate Squared Euclidean Distance
            Mathematically equivalent to maximizing Log-Likelihood for AWGN.
            Ref: Lecture Notes Eq 9.7 "Minimizing the Euclidean distance".
        """
        bits = (theta_y - e) & 1
        x = 1.0 - 2.0 * bits
        return np.sum((x - y) ** 2)

    def compute_OEI(self, y: np.ndarray) -> np.ndarray:
        return np.argsort(np.abs(y))

    def expand_children(self, e: np.ndarray, oei: np.ndarray) -> List[np.ndarray]:
        n = self.code.n
        ones = np.where(e[oei] == 1)[0]
        j_star = ones[-1] + 1 if len(ones) > 0 else 0
        children = []

        if j_star < n:
            # Child 1: Append
            e1 = e.copy()
            e1[oei[j_star]] ^= 1
            children.append(e1)
            # Child 2: Shift
            if j_star > 0:
                e2 = e1.copy()
                e2[oei[j_star - 1]] ^= 1
                children.append(e2)
        return children

    def decode_to_codeword(self, y: np.ndarray):
        y = y.astype(float)
        theta_y = (y < 0).astype(int)
        oei = self.compute_OEI(y)
        self.code.queries = 0

        # --- Init Trace ---
        trace_data = None
        if self.save_trace:
            code_name = getattr(self.code, "name", "UnknownCode").replace(" ", "_")
            trace_data = {
                "meta": {
                    "name": code_name,
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                    "n": int(self.code.n), "k": int(self.code.k)
                },
                "y": y, "oei": oei, "steps": []
            }

        # --- Init S (Priority Queue) ---
        S = []
        counter = 0
        node_counter = 0
        visited = set()

        e0 = np.zeros(self.code.n, dtype=int)
        m0 = self.calc_euclidean_distance(y, theta_y, e0)

        root_id = "node_0"
        # Item: (-metric, push_order, error_vec, node_id, parent_id)
        heapq.heappush(S, (m0, counter, e0, root_id, None))
        visited.add(tuple(e0))
        counter += 1

        decoded_codeword = theta_y

        # --- Main Loop ---
        while S and self.code.queries < self.max_queries:

            # 1. Pop current best candidate (Wait to record!)
            metric, _, e, curr_id, parent_id = heapq.heappop(S)

            # 2. Check validity
            c_test = (theta_y - e) & 1
            is_valid = self.code.is_codeword(c_test)

            # 3. Expand & Push Children (Updating S)
            if not is_valid:  # Optimization: If valid, strictly we stop, so children don't matter for outcome
                children = self.expand_children(e, oei)
                for child_e in children:
                    k = tuple(child_e)
                    if k in visited: continue
                    visited.add(k)

                    node_counter += 1
                    child_id = f"node_{node_counter}"
                    metric_child = self.calc_euclidean_distance(y, theta_y, child_e)

                    heapq.heappush(S, (metric_child, counter, child_e, child_id, curr_id))
                    counter += 1

            # 4. [Trace] Snapshot S AFTER update (Matches Table I Column S)
            # The current 'e' is gone. New children are in.
            if self.save_trace:
                # Note: 'S' is a heap, so we must sort it to see the true priority order
                top_S = sorted(S, key=lambda x: x[0])

                trace_data["steps"].append({
                    "step": self.code.queries + 1,
                    "id": curr_id,
                    "parent": parent_id,
                    "metric": round(metric, 4),
                    "error_indices": np.where(e == 1)[0].tolist(),
                    "is_valid": bool(is_valid),
                    "queue": [
                        {
                            "metric": round(m, 4),
                            "err": np.where(err == 1)[0].tolist(),
                            "id": nid,
                            "parent": pid
                        }
                        # order: metric, order, error_vec, node_id, parent_id
                        for m, _, err, nid, pid in top_S
                    ]
                })

            if is_valid:
                decoded_codeword = c_test
                break

        else:
            print("SGRAND: Max queries reached.")

        # --- Save ---
        if self.save_trace and trace_data:
            filename = f"trace_{trace_data['meta']['name']}_queries{self.code.queries}_{trace_data['meta']['timestamp']}.json"
            try:
                with open(os.path.join("traces", filename), "w") as f:
                    json.dump(trace_data, f, default=default_converter)
                print(f"Trace saved: {filename}")
            except Exception as e:
                print(f"Error saving trace: {e}")

        return decoded_codeword

