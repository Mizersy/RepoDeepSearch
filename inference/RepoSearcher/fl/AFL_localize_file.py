# import argparse
# import concurrent.futures
# import json
# import os
# from multiprocessing import Lock, Manager

# from datasets import load_dataset
# from tqdm import tqdm

from RepoSearcher.fl.AFL import AFL
from RepoSearcher.util.preprocess_data import (
    check_contains_valid_loc,
    filter_none_python,
    filter_out_test_files,
    get_repo_structure,
)
from RepoSearcher.util.utils import load_existing_instance_ids, load_jsonl, setup_logger, load_jsonl_partial

MAX_RETRIES = 5

def localize_instance(
    bug, args, swe_bench_data, start_file_locs, existing_instance_ids, write_lock=None
):
    instance_id = bug["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    if args.target_id is not None:
        if args.target_id != bug["instance_id"]:
            return

    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {bug['instance_id']}")
        return

    structure = get_repo_structure(
        instance_id, bug["repo"], bug["base_commit"], "playground"
    )

    logger.info(f"================ localize {instance_id} ================")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]

    filter_none_python(structure)  # some basic filtering steps
    filter_out_test_files(structure)

    found_files = []

    # file level localization
    if args.file_level:
        fl = AFL(
            instance_id,
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
        )
        if args.use_cosil:
            found_files, additional_artifact_loc_file, file_traj, file_loc_trajs = fl.file_localize_with_cosil(
                mock=args.mock
            )
        elif args.native_function_call:
            found_files, additional_artifact_loc_file, file_traj, file_loc_trajs = fl.file_localize_with_native_function_call(
                mock=args.mock
            )
        else:
            found_files, additional_artifact_loc_file, file_traj, file_loc_trajs = fl.file_localize_with_g(
                mock=args.mock
            )
    else:
        # assume start_file is provided
        for locs in start_file_locs:
            if locs["instance_id"] == instance_id:
                found_files = locs["found_files"]
                break

        if len(found_files) == 0:
            return


    if write_lock is not None:
        write_lock.acquire()
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "found_files": found_files,
                    "file_loc_trajs": file_loc_trajs,
                }
            )
            + "\n"
        )
    if write_lock is not None:
        write_lock.release()


def localize(args):
    if args.dataset.endswith(".jsonl"):
        # swe_bench_data = load_jsonl(args.dataset)
        # swe_bench_data = swe_bench_data[:5000]
        swe_bench_data = load_jsonl_partial(args.dataset, 6000)
    elif args.dataset == "SWE-Gym/SWE-Gym":
        swe_bench_data = load_dataset(args.dataset, split="train")
    else:
        swe_bench_data = load_dataset(args.dataset, split="test")
    start_file_locs = load_jsonl(args.start_file) if args.start_file else None
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )

    if args.num_threads == 1:
        for bug in tqdm(swe_bench_data, colour="MAGENTA"):
            localize_instance(
                bug, args, swe_bench_data, start_file_locs, existing_instance_ids
            )
    else:
        with Manager() as manager:
            write_lock = manager.Lock()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_threads
            ) as executor:
                futures = [
                    executor.submit(
                        localize_instance,
                        bug,
                        args,
                        swe_bench_data,
                        start_file_locs,
                        existing_instance_ids,
                        write_lock,
                    )
                    for bug in swe_bench_data
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(swe_bench_data),
                    colour="MAGENTA",
                ):
                    future.result()


def merge(args):
    """Merge predicted locations."""
    start_file_locs = load_jsonl(args.start_file)

    def merge_locs(sample_found_locs: list[dict]):
        merged_found_locs = {}
        for locs in sample_found_locs:
            for fn, file_found_locs in locs.items():
                if isinstance(file_found_locs, str) and file_found_locs.strip():
                    merged_found_locs.setdefault(fn, [""])[0] += "\n" + file_found_locs
                elif "\n".join(file_found_locs).strip():
                    merged_found_locs.setdefault(fn, [""])[0] += "\n" + "\n".join(
                        file_found_locs
                    )
        return merged_found_locs

    # Dump each location sample.
    for st_id in range(args.num_samples):
        en_id = st_id
        merged_locs = []
        for locs in start_file_locs:
            merged_found_locs = []
            if "found_edit_locs" in locs and len(locs["found_edit_locs"]):
                merged_found_locs = merge_locs(
                    locs["found_edit_locs"][st_id : st_id + 1]
                )
            merged_locs.append({**locs, "found_edit_locs": merged_found_locs})
        with open(
            f"{args.output_folder}/loc_merged_{st_id}-{en_id}_outputs.jsonl", "w"
        ) as f:
            for data in merged_locs:
                f.write(json.dumps(data) + "\n")


def check_valid_args(args):
    assert (
        not os.path.exists(args.output_file) or args.skip_existing
    ), "Output file already exists and not set to skip existing localizations"

    assert not (
        args.file_level and args.start_file
    ), "Cannot use both file_level and start_file"

    assert not (
        args.file_level and args.fine_grain_line_level and not args.related_level
    ), "Cannot use both file_level and fine_grain_line_level without related_level"

    assert not (
        (not args.file_level) and (not args.start_file)
    ), "Must use either file_level or start_file"

    # assert (not "deepseek" in args.model) or (
    #     args.backend == "deepseek"
    # ), "Must specify `--backend deepseek` if using a DeepSeek model"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument(
        "--start_file",
        type=str,
        help="""previous output file to start with to reduce
        the work, should use in combination without --file_level""",
    )
    parser.add_argument("--file_level", action="store_true")
    parser.add_argument("--related_level", action="store_true")
    parser.add_argument("--fine_grain_line_level", action="store_true")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--compress_assign", action="store_true")
    parser.add_argument("--compress_assign_total_lines", type=int, default=30)
    parser.add_argument("--compress_assign_prefix_lines", type=int, default=10)
    parser.add_argument("--compress_assign_suffix_lines", type=int, default=10)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--no_line_number", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument("--related_level_separate_file", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--keep_old_order", action="store_true")
    parser.add_argument("--irrelevant", action="store_true")
    parser.add_argument("--direct_edit_loc", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip localization of instance id's which already contain a localization in the output file.",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "deepseek", "anthropic", "claude", "Anomy"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="Current supported dataset for evaluation",
    )
    parser.add_argument(
        "--use_cosil",
        action="store_true",
        help="Use CoSIL's file localization prompt.",
    )
    parser.add_argument(
        "--native_function_call",
        action="store_true",
        help="Use CoSIL's file localization prompt.",
    )

    args = parser.parse_args()
    args.output_file = os.path.join(args.output_folder, args.output_file)
    check_valid_args(args)

    os.makedirs(os.path.join(args.output_folder, "localization_logs"), exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.merge:
        merge(args)
    else:
        localize(args)


if __name__ == "__main__":
    main()
