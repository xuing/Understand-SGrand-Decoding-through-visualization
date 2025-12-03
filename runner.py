from dataclasses import dataclass
from typing import Callable, List, Optional, TypeVar
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np

from channel import Channel, BSCChannel
from codes import LinearBlockCode
from decoder import ChannelDecoder


# parallelized function to run a single SNR point
def _run_single_snr_point(
        idx: int,
        ebn0_db: float,
        code: LinearBlockCode,
        decoder_cls: type["ChannelDecoder"],
        channel_factory: Callable[[float], Channel],
        target_block_errors: int,
        max_blocks: int,
        base_seed: int | None
):
    """
    供子进程调用的独立函数。
    """
    # 在子进程中创建独立的信道实例
    channel = channel_factory(ebn0_db)

    # 简单的种子策略：基准种子 + 索引
    seed = (base_seed + idx) if base_seed is not None else None

    # 调用原有的蒙特卡洛函数
    stats = monte_carlo_until_errors(
        code=code,
        channel=channel,
        decoder_cls=decoder_cls,
        target_block_errors=target_block_errors,
        max_blocks=max_blocks,
        seed=seed,
        log_every=0  # 子进程中关闭详细打印
    )

    # 返回索引和关键数据
    return idx, stats.wer, stats.ber, stats.avg_queries, stats.n_blocks


# ============================================================
# 1. Data structures
# ============================================================

@dataclass
class BlockResult:
    """Result of a single transmission/decoding block."""
    u: np.ndarray
    c: np.ndarray
    y: np.ndarray
    u_hat: np.ndarray
    c_hat: np.ndarray
    block_error: bool
    bit_errors: int
    n_queries: int | None


@dataclass
class MonteCarloStats:
    """Aggregate Monte Carlo statistics for a fixed channel setting."""
    n_blocks: int
    n_block_errors: int = 0
    n_bit_errors: int = 0
    queries: list[int] | None = None

    def __post_init__(self):
        if self.queries is None:
            self.queries = []

    @property
    def wer(self) -> float:
        """Block error rate = WER."""
        return self.n_block_errors / self.n_blocks if self.n_blocks > 0 else 0.0

    @property
    def ber(self) -> float:
        """Bit error rate."""
        total_bits = self.n_blocks  # 这里注意：通常需要除以 k，请确保外部或此处逻辑一致
        # 根据原逻辑，total_bits 这里似乎只是 block 数，实际计算BER时要除以k
        # 但在MonteCarlo中 n_bit_errors 是累加的位错误。
        # 此处属性仅作简单的归一化，具体由外部处理或保持原样。
        # 为了不破坏逻辑，保持原样，但在 sweep 中会做处理。
        return self.n_bit_errors / total_bits if total_bits > 0 else 0.0

    @property
    def avg_queries(self) -> float:
        """Average number of codeword-membership tests per block."""
        return float(np.mean(self.queries)) if self.queries else 0.0


@dataclass
class SweepResult:
    """Statistics for a decoder when sweeping a scalar parameter (e.g., Eb/N0)."""
    param_values: np.ndarray  # e.g. Eb/N0 [dB]
    param_name: str  # e.g. "Eb/N0 [dB]"
    label: str  # curve label (code + decoder)
    bler: np.ndarray
    ber: np.ndarray
    avg_queries: np.ndarray


TDecoder = TypeVar("TDecoder", bound=ChannelDecoder)


# ============================================================
# 2. Single-block simulation
# ============================================================

def simulate_once(
        code: LinearBlockCode,
        channel: Channel,
        decoder: ChannelDecoder,
        u: np.ndarray,
) -> BlockResult:
    """
    Encode -> transmit -> decode for a single block.
    """
    # Reset query counter
    code.queries = 0

    # Encode
    c = code.encode(u)

    # Transmit through channel
    y = channel.transmit(c)

    # Decode
    c_hat = decoder.decode_to_codeword(y)
    c_hat = np.asarray(c_hat, dtype=int)

    # Extract estimated information bits
    u_hat = code.extract_info(c_hat)
    u_hat = np.asarray(u_hat, dtype=int)

    # Error statistics
    bit_errs = int(np.sum(u != u_hat))
    block_error = (bit_errs > 0)

    # Query count
    n_queries = int(code.queries) if getattr(code, "queries", 0) > 0 else None

    return BlockResult(
        u=u, c=c, y=y, u_hat=u_hat, c_hat=c_hat,
        block_error=block_error, bit_errors=bit_errs, n_queries=n_queries,
    )


def run_single(
        code: LinearBlockCode,
        channel: Channel,
        decoder_cls: type[TDecoder] | list[type[TDecoder]],
        u: Optional[np.ndarray] = None,
        verbose: bool = True,
):
    """
    Run a single example block, optionally using multiple decoders on the
    same information vector and the same realized noise.

    Parameters
    ----------
    code : LinearBlockCode
        Code under test.
    channel : Channel
        Channel instance (e.g., BSCChannel, BIAWGNChannel).
    decoder_cls : type[ChannelDecoder] or list[type[ChannelDecoder]]
        Single decoder class or a list of decoder classes to be applied
        to the same (u, y) pair.
    u : np.ndarray | None
        Information bits. If None, a random vector of length k is drawn.
    verbose : bool
        If True, pretty-print the block, channel output, and decoder
        results.

    Returns
    -------
    BlockResult or dict[str, BlockResult]
        If a single decoder class is given, returns its BlockResult
        (for backward compatibility). If a list of decoder classes is
        given, returns a dict mapping decoder class names to their
        BlockResult objects.
    """
    # Draw or reuse information bits
    if u is None:
        u = np.random.randint(0, 2, size=code.k, dtype=int)

    # Normalize decoder_cls to a list of classes
    if isinstance(decoder_cls, (list, tuple)):
        decoder_classes: list[type[TDecoder]] = list(decoder_cls)
    else:
        decoder_classes = [decoder_cls]

    # ------------------------------------------------------------
    # Common encode + transmit: same noise for all decoders
    # ------------------------------------------------------------
    # Encode once
    c = code.encode(u)

    # Transmit once
    y = channel.transmit(c)

    # Helper: pretty-print a bit vector, highlighting positions
    # where vec != ref in red (ANSI escape), if enable_color=True.
    def _format_bits_with_diff(
        ref_vec: np.ndarray,
        vec: np.ndarray,
        enable_color: bool = True,
    ) -> str:
        ref_bits = np.asarray(ref_vec, dtype=int)
        bits = np.asarray(vec, dtype=int)

        tokens: list[str] = []
        for rb, vb in zip(ref_bits, bits):
            if enable_color and (rb != vb):
                tokens.append(f"\033[31m{vb}\033[0m")  # red
            else:
                tokens.append(str(vb))
        return "[" + " ".join(tokens) + "]"

    # Run each decoder on the same (u, c, y)
    results: dict[str, BlockResult] = {}

    for dec_cls in decoder_classes:
        decoder = dec_cls(code)

        # Reset query counter for this decoder run (if present)
        if hasattr(code, "queries"):
            code.queries = 0

        # Decode
        c_hat = decoder.decode_to_codeword(y)
        c_hat = np.asarray(c_hat, dtype=int)

        # Extract estimated information bits
        u_hat = code.extract_info(c_hat)
        u_hat = np.asarray(u_hat, dtype=int)

        # Error statistics (still computed for BlockResult, but not printed)
        bit_errs = int(np.sum(u != u_hat))
        block_error = (bit_errs > 0)

        # Query count (if maintained by the code object)
        n_queries = int(code.queries) if getattr(code, "queries", 0) > 0 else None

        results[dec_cls.__name__] = BlockResult(
            u=u,
            c=c,
            y=y,
            u_hat=u_hat,
            c_hat=c_hat,
            block_error=block_error,
            bit_errors=bit_errs,
            n_queries=n_queries,
        )

    if verbose:
        # Header
        print("\n" + "=" * 80)
        if len(decoder_classes) == 1:
            only = decoder_classes[0]
            print(f">>> Single run: code={code.name}, "
                  f"decoder={only.__name__}, "
                  f"channel={type(channel).__name__}")
        else:
            decoder_names = ", ".join(cls.__name__ for cls in decoder_classes)
            print(f">>> Single run (multi-decoder): code={code.name}, "
                  f"decoders=[{decoder_names}], "
                  f"channel={type(channel).__name__}")
        print("-" * 80)

        # Common u, c
        print(f"u ({code.k:2d}): {u}")
        print(f"c ({code.n:2d}): {c}")

        # y：对于 BSCChannel，用红色标出与 c 不同的位置；其它通道保持原样
        if y.dtype.kind == "f":
            # Soft values: use numpy string formatting
            y_str = np.array2string(
                y, precision=2, suppress_small=True, separator=", "
            )
            print(f"y        : {y_str}")
        else:
            is_bsc = (type(channel).__name__ == "BSCChannel")
            y_str = _format_bits_with_diff(c, y, enable_color=is_bsc)
            print(f"y        : {y_str}")

        # Per-decoder results
        for dec_cls in decoder_classes:
            r = results[dec_cls.__name__]
            print("-" * 80)
            print(f"decoder  : {dec_cls.__name__}")
            # c_hat：总是用红色标出与 c 不同的部分，方便对比码字错误位置
            c_hat_str = _format_bits_with_diff(c, r.c_hat, enable_color=True)
            print(f"c_hat    : {c_hat_str}")
            print(f"u_hat    : {r.u_hat}")
            print(f"block OK : {not r.block_error}")
            if r.n_queries is not None:
                print(f"queries  : {r.n_queries}")

        print("=" * 80)

    # Backward-compatible return:
    if len(decoder_classes) == 1:
        # Return the single BlockResult as before
        return next(iter(results.values()))
    # For multiple decoders, return the full dict
    return results



# ============================================================
# 3. Monte Carlo (until target errors)
# ============================================================

def monte_carlo_until_errors(
        code: LinearBlockCode,
        channel: Channel,
        decoder_cls: type[TDecoder],
        target_block_errors: int,
        max_blocks: int,
        seed: Optional[int] = None,
        log_every: int = 0,
) -> MonteCarloStats:
    """
    Monte Carlo simulation loop.
    """
    if seed is not None:
        np.random.seed(seed)

    stats = MonteCarloStats(n_blocks=0)

    for blk in range(1, max_blocks + 1):
        decoder = decoder_cls(code)
        u = code.generate_message()

        result = simulate_once(code, channel, decoder, u)

        stats.n_blocks += 1
        stats.n_bit_errors += result.bit_errors
        if result.block_error:
            stats.n_block_errors += 1
        if result.n_queries is not None:
            stats.queries.append(result.n_queries)

        if log_every and blk % log_every == 0:
            print(
                f"[MC] blocks={blk}, "
                f"errors={stats.n_block_errors}, "
                f"current WER={stats.wer:.3e}"
            )

        if stats.n_block_errors >= target_block_errors:
            break

    return stats


# ============================================================
# 4. Sweep Eb/N0 (PARALLELIZED)
# ============================================================

def sweep_ebn0_until_errors(
        code: LinearBlockCode,
        decoder_cls: type[TDecoder],
        ebn0_min_db: float,
        ebn0_max_db: float,
        ebn0_step_db: float,
        channel_factory: Callable[[float], Channel],
        target_block_errors: int,
        max_blocks_per_snr: int,
        label: str,
        seed: Optional[int] = None,
        log_each_snr: bool = True,
        max_workers: int | None = None
) -> SweepResult:
    """
    Parallelized Eb/N0 sweep using ProcessPoolExecutor.
    """
    ebn0_values = np.arange(
        ebn0_min_db,
        ebn0_max_db + 1e-9,
        ebn0_step_db,
        dtype=float,
    )

    n_points = len(ebn0_values)
    bler_list = [0.0] * n_points
    ber_list = [0.0] * n_points
    avg_queries_list = [0.0] * n_points

    print(f"\n=== Starting Parallel Sweep: {label} ({n_points} points) ===")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, ebn0_db in enumerate(ebn0_values):
            futures.append(
                executor.submit(
                    _run_single_snr_point,
                    idx, ebn0_db, code, decoder_cls,
                    channel_factory, target_block_errors, max_blocks_per_snr, seed
                )
            )

        for future in as_completed(futures):
            idx, wer, raw_ber_stat, avg_q, n_blocks = future.result()

            bler_list[idx] = wer

            ber_list[idx] = raw_ber_stat / code.k

            avg_queries_list[idx] = avg_q

            ebn0_val = ebn0_values[idx]
            if log_each_snr:
                print(
                    f"[{label}] Eb/N0={ebn0_val:5.2f} dB | WER={wer:.3e} | BER={ber_list[idx]:.3e} | Blocks={n_blocks}")


    return SweepResult(
        param_values=ebn0_values,
        param_name="Eb/N0 [dB]",
        label=label,
        bler=np.array(bler_list, dtype=float),
        ber=np.array(ber_list, dtype=float),
        avg_queries=np.array(avg_queries_list, dtype=float),
    )

def sweep_bsc_until_errors(
        code: LinearBlockCode,
        decoder_cls: type[ChannelDecoder],
        p_min: float,
        p_max: float,
        p_step: float,
        target_block_errors: int,
        max_blocks_per_p: int,
        label: str,
        seed: int | None = None,
        log_each_p: bool = True,
) -> SweepResult:

    p_values = np.arange(p_min, p_max + 1e-12, p_step, dtype=float)

    bler_list = []
    ber_list = []
    avg_queries_list = []

    print(f"\n=== Starting BSC Sweep: {label} ({len(p_values)} points) ===")

    for idx, p in enumerate(p_values):
        if log_each_p:
            print(f"\n--- p = {p:.4f} ---")

        channel = BSCChannel(p)
        stats = monte_carlo_until_errors(
            code=code,
            channel=channel,
            decoder_cls=decoder_cls,
            target_block_errors=target_block_errors,
            max_blocks=max_blocks_per_p,
            seed=(seed + idx if seed else None),
            log_every=1000
        )

        # compute WER / BER / avg queries
        bler = stats.wer
        ber  = stats.n_bit_errors / (stats.n_blocks * code.k)
        avg_q = stats.avg_queries

        bler_list.append(bler)
        ber_list.append(ber)
        avg_queries_list.append(avg_q)

        if log_each_p:
            print(f"[{label}] p={p:.4f} | WER={bler:.3e} | BER={ber:.3e} | avgQ={avg_q:.1f} | Blocks={stats.n_blocks}")

    return SweepResult(
        param_values=p_values,
        param_name="BSC Crossover Probability p",
        label=label,
        bler=np.array(bler_list),
        ber=np.array(ber_list),
        avg_queries=np.array(avg_queries_list),
    )


# ============================================================
# 5. Plotting helpers
# ============================================================

def plot_wer_vs_ebn0(results: List[SweepResult], title: Optional[str] = None):
    fig, ax = plt.subplots()
    for res in results:
        ax.plot(res.param_values, res.bler, marker="o", label=res.label)
    ax.set_xlabel(results[0].param_name)
    ax.set_ylabel("Word Error Rate (WER)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    if title: ax.set_title(title)
    return fig, ax


def plot_queries_vs_ebn0(results: List[SweepResult], title: Optional[str] = None):
    fig, ax = plt.subplots()
    for res in results:
        y_vals = np.where(res.avg_queries <= 0, np.nan, res.avg_queries)
        if "ML" in res.label:
            ax.plot(res.param_values, y_vals, "--", color="0.5", label=f"{res.label} (baseline)")
        else:
            ax.plot(res.param_values, y_vals, marker="s", label=res.label)
    ax.set_xlabel(results[0].param_name)
    ax.set_ylabel("Avg Queries")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    if title: ax.set_title(title)
    return fig, ax


def plot_queries_vs_wer(results: List[SweepResult], title: Optional[str] = None):
    fig, ax = plt.subplots()
    for res in results:
        if "ML" in res.label: continue
        x, y = np.array(res.bler, float), np.array(res.avg_queries, float)
        mask = (x > 0) & (y > 0)
        if not np.any(mask): continue
        # sort by WER for cleaner line
        x, y = x[mask], y[mask]
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker="^", label=res.label)
    ax.set_xlabel("WER")
    ax.set_ylabel("Avg Queries")
    ax.set_xscale("log");
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    if title: ax.set_title(title)
    return fig, ax