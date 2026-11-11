"""Same official Full/S2/Prefix scaffold with explicit single-attempt greedy.

The official n_critic_runs default causes later attempts to change temperature
0 to 0.1. Select its public --n-critic-runs 1 option for all three arms. No
evaluator, prompt, retry implementation, model output, or scoring code changes.
"""
import sys

import run_official_retry as bootstrap

load_official = bootstrap.load_official
install_overlay = bootstrap.install_overlay


def single_attempt_plan(plan):
    argv = list(plan['argv'])
    if any(value.split('=', 1)[0] == '--n-critic-runs' for value in argv):
        raise ValueError('Existing explicit critic-runs configuration cannot be silently overridden')
    return dict(plan, argv=[*argv, '--n-critic-runs', '1'], n_critic_runs=1,
                single_attempt_greedy=True, sampling_protocol='single_attempt_greedy',
                official_cli_configuration_only=True,
                critic_attempt_policy='one official critic attempt; no later temperature escalation',
                same_policy_for_arms=['Full', 'S2', 'Prefix'])


def main(argv=None):
    supplied = sys.argv[1:] if argv is None else argv
    if any(value.split('=', 1)[0] == '--n-critic-runs' for value in supplied):
        raise ValueError('This entry fixes n_critic_runs=1 explicitly; conflicting/duplicate CLI is not accepted')
    official = load_official()
    original = official.build_plan
    def build_plan(args):
        return single_attempt_plan(original(args))
    official.build_plan = build_plan
    try:
        return official.main(argv)
    finally:
        if official.build_plan is not build_plan:
            raise RuntimeError('Official build_plan overlay ownership changed')
        official.build_plan = original


if __name__ == '__main__':
    main()
