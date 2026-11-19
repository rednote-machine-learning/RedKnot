"""Explicit independent fixed-ratio arms; no automatic length/quality tuning."""
import math

ARM_FRACTIONS = {'Cone15': .15, 'Fixed15Opt': .15}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_packed_arm(method):
    method = method[:-4] if method.endswith('.max') else method
    return method if method in ARM_FRACTIONS else None


def fraction_for_method(method):
    arm = canonical_packed_arm(method)
    if arm is not None:
        return ARM_FRACTIONS[arm]
    require(method == 'Full', 'Unknown fixed comparison arm')
    return .20


def budget_admission(documents, total, fraction):
    require(fraction in tuple(ARM_FRACTIONS.values()), 'Only approved Cone15 fifteen percent body budget')
    require(type(total) is int and total > 0 and documents, 'Cacheable original documents required')
    previous, body, tails = 0, 0, 0
    for doc in documents:
        start, end = doc['start'], doc['end']
        require(type(start) is int and type(end) is int and 0 <= previous <= start < end <= total
            and end-start >= 256, 'Invalid complete cacheable document')
        previous = end; body += end-start; tails += 128
    budget = math.ceil(body*fraction)
    require(budget >= tails, 'Requested body budget cannot contain all mandatory tail128 positions')
    return dict(body_tokens=body, new_tokens=total-body, body_fraction=fraction,
        body_budget_tokens=budget, mandatory_body_tail_tokens=tails,
        ranked_body_tokens=budget-tails, total_selected_tokens=budget+total-body,
        tails_truncated=False, new_tokens_counted_outside_body_budget=True)
