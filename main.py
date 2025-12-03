import matplotlib.pyplot as plt
from pyparsing import LineStart

from channel import BSCChannel, BIAWGNChannel
from codes import Golay24, Golay23, ExtendedHamming84, LinearBlockCode
from decoder import SGRANDDecoder, MLDecoder, SyndromeDecoder, GRANDDecoder, SGRANDDecoder_Trace
from runner import run_single, plot_wer_vs_ebn0, plot_queries_vs_ebn0, plot_queries_vs_wer, sweep_bsc_until_errors
from runner import sweep_ebn0_until_errors


def single_for_all():
    # Prepare code
    golay24 = Golay24()
    golay23 = Golay23()
    hamming = ExtendedHamming84()

    codes = [golay24, golay23, hamming]

    bsc_1 = BSCChannel(p=0.1)

    for code in codes:
        # SyndromeDecoder on BSC
        run_single(code=code, channel=bsc_1, decoder_cls=SyndromeDecoder)
        # MLDecoder on BSC
        run_single(code=code, channel=bsc_1, decoder_cls=MLDecoder)
        # HardGRANDDecoder on BSC
        run_single(code=code, channel=bsc_1, decoder_cls=GRANDDecoder)

    bi_awgn_1 = BIAWGNChannel(snr_db=-10)
    for code in codes:
        # MLDecoder on AWGN
        run_single(code=code, channel=bi_awgn_1, decoder_cls=MLDecoder)
        # OrderedGRANDDecoder on AWGN
        run_single(code=code, channel=bi_awgn_1, decoder_cls=GRANDDecoder)
        # SGRANDDecoder on AWGN
        run_single(code=code, channel=bi_awgn_1, decoder_cls=SGRANDDecoder)


def awgn_channel_factory(snr_db: float):
    return BIAWGNChannel(snr_db=snr_db)


def run_awgn_monte_carlo():
    code = Golay23()

    decoder_specs = [
        ("GRAND", GRANDDecoder),
        ("SGRAND", SGRANDDecoder),
        ("ML", MLDecoder),
    ]

    ebn0_min_db = -3.0
    ebn0_max_db = 3.0
    ebn0_step_db = 0.5

    target_block_errors = 100  # aim for ~50–100 errors
    max_blocks_per_snr = 200_000_0  # safety cap
    base_seed = 3773

    results = []
    for idx, (name, dec_cls) in enumerate(decoder_specs):
        label = f"{name}"
        res = sweep_ebn0_until_errors(
            code=code,
            decoder_cls=dec_cls,
            ebn0_min_db=ebn0_min_db,
            ebn0_max_db=ebn0_max_db,
            ebn0_step_db=ebn0_step_db,
            channel_factory=awgn_channel_factory,
            target_block_errors=target_block_errors,
            max_blocks_per_snr=max_blocks_per_snr,
            label=label,
            seed=base_seed + idx * 1000,
            log_each_snr=True,
        )
        results.append(res)

    fig1, _ = plot_wer_vs_ebn0(
        results,
        title=f"{code.name} WER over BI-AWGN"
    )

    fig2, _ = plot_queries_vs_ebn0(
        results,
        title=f"{code.name} average number of queries"
    )

    fig3, _ = plot_queries_vs_wer(
        results,
        title=f"{code.name} average queries vs WER"
    )

    plt.show()


def run_bsc_monte_carlo():
    code = Golay24()  # or Golay24 / Hamming84

    decoder_specs = [
        ("Syndrome", SyndromeDecoder),
        ("ML", MLDecoder),
        ("GRAND", GRANDDecoder),
    ]

    p_min, p_max, p_step = 0.005, 0.12, 0.005
    target_block_errors = 100
    max_blocks_per_p = 200_000
    base_seed = 1337

    results = []

    for idx, (name, dec_cls) in enumerate(decoder_specs):
        label = name
        res = sweep_bsc_until_errors(
            code=code,
            decoder_cls=dec_cls,
            p_min=p_min,
            p_max=p_max,
            p_step=p_step,
            target_block_errors=target_block_errors,
            max_blocks_per_p=max_blocks_per_p,
            label=label,
            seed=base_seed + idx * 100,
            log_each_p=True,
        )
        results.append(res)

    # plotting
    plot_wer_vs_ebn0(results, title=f"{code.name} — BSC WER vs p")
    plot_queries_vs_ebn0(results, title=f"{code.name} — Queries vs p")
    plot_queries_vs_wer(results, title=f"{code.name} — Avg Queries vs WER")
    plt.show()


def generate_SGRAND_trace(code: LinearBlockCode, snr_db: float):
    bi_awgn_1 = BIAWGNChannel(snr_db=snr_db)

    run_single(code=code, channel=bi_awgn_1, decoder_cls=SGRANDDecoder_Trace)


if __name__ == "__main__":
    print("=== GRAND (Guessing Random Additive Noise Decoding) Simulation Framework ===\n")

    single_for_all()

    # run_awgn_monte_carlo()
    #
    # run_bsc_monte_carlo()

    generate_SGRAND_trace(code=ExtendedHamming84(), snr_db=-1.0)
