"""Search discrete hyperparameters using validation scores."""

import copy
import itertools
import random
from pathlib import Path

if __package__:
    from .train import ROOT, fit, make_parser, positive_int, prepare, read_yaml, save_run
else:
    from train import ROOT, fit, make_parser, positive_int, prepare, read_yaml, save_run


def candidates(base, space, trials, seed):
    fields = [(section, key, values)
              for section, options in space.items() for key, values in options.items()]
    for section, key, values in fields:
        if section not in {"model", "training"} or key not in base[section]:
            raise ValueError(f"Unknown search parameter: {section}.{key}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Expected a nonempty candidate list for {section}.{key}")
    if not fields:
        raise ValueError("Search configuration has no candidate parameters")
    baseline = tuple(base[section][key] for section, key, _ in fields)
    combinations = list(dict.fromkeys(itertools.product(*(values for _, _, values in fields))))
    if baseline in combinations:
        combinations.remove(baseline)
    rng = random.Random(seed)
    yield copy.deepcopy(base)
    for values in rng.sample(combinations, min(trials - 1, len(combinations))):
        config = copy.deepcopy(base)
        for (section, key, _), value in zip(fields, values):
            config[section][key] = value
        yield config


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--search-config", type=Path, help="Override the search YAML path")
    parser.add_argument("--trials", type=positive_int, default=10,
                        help="Maximum configurations to try, including the fixed baseline")
    args = parser.parse_args()
    config, splits = prepare(args)
    path = args.search_config or ROOT / "search_configs" / f"{args.dataset}.yaml"
    search = read_yaml(path)
    space = search.get(args.model, search.get("space"))
    if space is None:
        raise ValueError(f"No search space for {args.model} in {path}")
    records = []
    best_score = float("-inf")
    for trial, candidate in enumerate(candidates(config, space, args.trials, args.seed), start=1):
        fitted = fit(args, candidate, splits)
        best = fitted[2]
        score = best["validation"]["mean"]
        records.append({"trial": trial, "config": candidate, "best_epoch": best["epoch"],
                        "validation": best["validation"]})
        print(f"Trial {trial}: validation={score:.6f}", flush=True)
        if score > best_score:
            best_score = score
            winner = (trial, candidate, fitted)
        del fitted
    trial, candidate, fitted = winner
    save_run(args, candidate, splits, fitted,
             {"search_config": str(path.resolve()), "best_trial": trial, "trials": records})


if __name__ == "__main__":
    main()
