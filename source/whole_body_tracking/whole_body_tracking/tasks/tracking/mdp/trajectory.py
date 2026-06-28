from __future__ import annotations

import torch


def ball_circle_min_distance(
    ball_pos_xy: torch.Tensor,
    ball_vel_xy: torch.Tensor,
    anchor_xy: torch.Tensor,
    anchor_vel_xy: torch.Tensor,
    horizon: float = 0.5,
) -> torch.Tensor:
    """Minimum predicted ball-anchor distance over the next `horizon` seconds."""
    dv = ball_vel_xy - anchor_vel_xy
    dp = ball_pos_xy - anchor_xy
    v2 = (dv**2).sum(-1)
    dpv = (dp * dv).sum(-1)
    dp2 = (dp**2).sum(-1)

    t_star = torch.where(v2 > 1e-6, -dpv / v2.clamp(min=1e-6), torch.zeros_like(v2))
    t_clamp = t_star.clamp(0.0, horizon)
    min_dist2 = (v2 * t_clamp**2 + 2.0 * dpv * t_clamp + dp2).clamp_min(0.0)
    return torch.sqrt(min_dist2)


def ball_enters_circle(
    ball_pos_xy: torch.Tensor,
    ball_vel_xy: torch.Tensor,
    anchor_xy: torch.Tensor,
    anchor_vel_xy: torch.Tensor,
    radius: float = 0.5,
    horizon: float = 0.5,
) -> torch.Tensor:
    """Return whether a linear ball trajectory passes within `radius` of an anchor."""
    min_dist = ball_circle_min_distance(
        ball_pos_xy,
        ball_vel_xy,
        anchor_xy,
        anchor_vel_xy,
        horizon=horizon,
    )
    return min_dist < radius
