"""Explicit independent fixed-ratio arms and RedKnot V2 policy budget admission."""
import math

ARM_FRACTIONS = {'Cone15': .15, 'Fixed15Opt': .15}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_packed_arm(method):
    method = method[:-4] if method.endswith('.max') else method
    return method if method in ARM_FRACTIONS else None


def fraction_for_method(method, policy=None):
    arm = canonical_packed_arm(method)
    if arm is not None:
        if policy is not None and getattr(policy, 'is_v2', False):
            return float(policy.fraction)
        return ARM_FRACTIONS[arm]
    require(method == 'Full', 'Unknown fixed comparison arm')
    return .20


def budget_admission(documents, total, fraction, *, policy=None):
    v2 = policy is not None and getattr(policy, 'is_v2', False)
    if v2:
        approved = (float(policy.fraction), float(policy.stage1_fraction))
        require(any(abs(float(fraction) - x) < 1e-9 for x in approved) and 0 < float(fraction) <= 1,
                f'Unapproved RedKnot V2 body fraction: {fraction}')
        head_width = int(policy.head_width)
        require(0 <= head_width <= 128, f'Invalid Online_Head width: {head_width}')
    else:
        require(fraction in tuple(ARM_FRACTIONS.values()), 'Only approved Cone15 fifteen percent body budget')
        head_width = 0
    require(type(total) is int and total > 0 and documents, 'Cacheable original documents required')
    previous, body, tails = 0, 0, 0
    for doc in documents:
        start, end = doc['start'], doc['end']
        require(type(start) is int and type(end) is int and 0 <= previous <= start < end <= total
            and end-start >= 256, 'Invalid complete cacheable document')
        previous = end; body += end-start; tails += 128
    requested = math.ceil(body*fraction)
    if v2:
        budget = max(requested, tails)
        floor_used = requested < tails
    else:
        require(requested >= tails, 'Requested body budget cannot contain all mandatory tail128 positions')
        budget = requested
        floor_used = False
    result = dict(body_tokens=body, new_tokens=total-body, body_fraction=float(fraction),
        body_budget_tokens=budget, mandatory_body_tail_tokens=tails,
        ranked_body_tokens=budget-tails, total_selected_tokens=budget+total-body,
        tails_truncated=False, new_tokens_counted_outside_body_budget=True)
    if v2:
        result.update(
            requested_body_budget_tokens=requested,
            budget_floor_used=floor_used,
            head_width=head_width,
            online_head_union_additive=head_width > 0,
            two_stage_l14=bool(policy.two_stage_l14),
            stage1_fraction=float(policy.stage1_fraction),
        )
    return result
