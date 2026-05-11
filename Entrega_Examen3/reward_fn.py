"""Funciones de recompensa para TORCS.

ESTE ARCHIVO ES CLAVE PARA EL EXAMEN.
Aquí es donde ustedes pueden experimentar con pesos y castigos y justificar
cómo cambia el comportamiento del agente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math


@dataclass
class RewardWeights:
    target_speed_kmh: float = 150.0
    min_reward_speed_kmh: float = 15.0

    # --- Velocidad ---
    # La recompensa de velocidad se escala hacia abajo en curvas para que
    # frenar no sea tan costoso para el agente.
    progress_weight: float = 8.0
    fast_bonus_weight: float = 2.0
    curve_speed_scale_min: float = 0.35

    # --- Supervivencia ---
    # Bono pequeño por cada paso que el coche permanece en pista y avanza.
    survival_bonus_weight: float = 1.5

    # --- Centrado ---
    centering_bonus_weight: float = 6.0
    center_bonus_radius: float = 0.80
    center_inner_radius: float = 0.35

    # --- Suavidad de volante ---
    smoothness_bonus_weight: float = 2.5
    smoothness_delta_threshold: float = 0.12

    # --- Frenado en curva ---
    curve_brake_bonus_weight: float = 5.0

    # --- Frenado anticipado (antes de entrar a curva) ---
    # Premia frenar cuando la curva se aproxima, no solo cuando ya está en ella.
    anticipation_brake_weight: float = 4.0

    # --- Dirección correcta en curva ---
    correct_direction_weight: float = 4.0
    correct_direction_min_curvature: float = 0.08

    # --- Alineación en curva ---
    # Bono por mantener el ángulo alineado con la pista dentro de una curva.
    curve_alignment_bonus_weight: float = 3.0

    # --- Penalización por exceso de velocidad en curva cerrada ---
    curve_overspeed_penalty: float = 3.0
    curve_overspeed_threshold_kmh: float = 90.0

    # --- Penalizaciones existentes ---
    lateral_penalty_weight: float = 2.0
    angle_penalty_weight: float = 3.5
    center_penalty_weight: float = 5.0
    edge_penalty_weight: float = 10.0
    low_speed_penalty: float = 8.0
    stopped_penalty: float = 12.0
    wrong_way_penalty: float = 220.0
    offtrack_penalty: float = 180.0
    collision_penalty: float = 260.0
    damage_delta_weight: float = 6.0
    control_penalty_weight: float = 1.5
    progress_delta_weight: float = 35.0   # subido de 20 → 35
    steer_oscillation_penalty_weight: float = 2.0
    straight_steer_penalty_weight: float = 5.0
    curve_bonus_weight: float = 5.0


DEFAULT_WEIGHTS = RewardWeights()


def _normalized_speed_to_kmh(speed_value: float) -> float:
    return speed_value * 300.0 if abs(speed_value) <= 3.0 else speed_value


def _normalized_angle_to_rad(angle_value: float) -> float:
    return angle_value * math.pi if abs(angle_value) <= 1.25 else angle_value


def compute_reward(obs: Dict, action: Dict, info: Dict, weights: RewardWeights = DEFAULT_WEIGHTS) -> float:
    """Recompensa con señales positivas explícitas y penalizaciones de seguridad.

    Señales positivas:
    1. Velocidad contextual  — menor valor en curvas para que frenar no cueste.
    2. Supervivencia         — bono por cada paso en pista avanzando.
    3. Centrado              — bono por mantenerse cerca del eje.
    4. Suavidad              — bono por volante estable en rectas.
    5. Frenado anticipado    — bono por frenar antes de entrar a curva.
    6. Frenado en curva      — bono por frenar dentro de curva.
    7. Dirección correcta    — bono por girar hacia el interior de la curva.
    8. Alineación en curva   — bono por ángulo alineado con la pista.
    9. Progreso              — bono por avanzar en la pista (progress_delta).

    Penalizaciones de seguridad:
    - Exceso de velocidad en curva cerrada.
    - Posición lateral, ángulo, borde, control simultáneo, oscilación.
    - Sentido contrario, salida de pista, colisión.
    """
    speed_x_kmh = _normalized_speed_to_kmh(float(obs.get("speedX", 0.0)))
    speed_y_kmh = _normalized_speed_to_kmh(float(obs.get("speedY", 0.0)))
    angle_rad = _normalized_angle_to_rad(float(obs.get("angle", 0.0)))
    track_pos = float(obs.get("trackPos", 0.0))

    track_sensors = list(obs.get("track", [1.0] * 19))[:19]
    if len(track_sensors) < 19:
        track_sensors += [1.0] * (19 - len(track_sensors))
    forward_arc = sum(track_sensors[6:13]) / 7.0
    in_curve = forward_arc < 0.72
    curve_depth = max(0.0, (0.72 - forward_arc) / 0.42)  # 0 suave .. 1 cerrada

    # track_curvature > 0 → curva izquierda → steer correcto es negativo
    # track_curvature < 0 → curva derecha  → steer correcto es positivo
    track_ahead_left = sum(track_sensors[6:9]) / 3.0
    track_ahead_right = sum(track_sensors[10:13]) / 3.0
    track_curvature = track_ahead_right - track_ahead_left

    alignment = math.cos(angle_rad)
    forward_speed_kmh = speed_x_kmh * alignment
    lateral_speed_kmh = abs(speed_x_kmh * math.sin(angle_rad)) + abs(speed_y_kmh)
    target_speed = max(weights.target_speed_kmh, weights.min_reward_speed_kmh)

    offtrack = bool(info.get("offtrack", abs(track_pos) > 1.0))
    collision = bool(info.get("collision", False))
    damage_delta = float(info.get("damage_delta", 0.0))
    progress_delta = float(info.get("progress_delta", 0.0))
    wrong_way = bool(
        info.get(
            "wrong_way",
            speed_x_kmh < -1.0 or alignment < 0.0 or progress_delta < -0.01,
        )
    )

    steer = abs(float(action.get("steer", 0.0)))
    steer_signed = float(action.get("steer", 0.0))
    accel = float(action.get("accel", 0.0))
    brake = float(action.get("brake", 0.0))
    steer_delta_abs = float(info.get("steer_delta_abs", 0.0))
    speed_ratio = min(abs(speed_x_kmh) / target_speed, 1.5)

    reward = 0.0

    # ------------------------------------------------------------------ #
    #  1. VELOCIDAD CONTEXTUAL                                             #
    # ------------------------------------------------------------------ #
    curve_speed_scale = 1.0 - (1.0 - weights.curve_speed_scale_min) * curve_depth

    if not offtrack and not wrong_way and forward_speed_kmh >= weights.min_reward_speed_kmh:
        useful_speed = min(forward_speed_kmh, target_speed)
        reward += weights.progress_weight * (useful_speed / target_speed) * curve_speed_scale
        reward += weights.fast_bonus_weight * max(
            0.0,
            (useful_speed - 0.55 * target_speed) / (0.45 * target_speed),
        ) * curve_speed_scale
    else:
        slow_gap = max(0.0, weights.min_reward_speed_kmh - max(0.0, forward_speed_kmh))
        reward -= weights.low_speed_penalty * (slow_gap / weights.min_reward_speed_kmh)

    if speed_x_kmh < 3.0:
        reward -= weights.stopped_penalty

    # ------------------------------------------------------------------ #
    #  2. SUPERVIVENCIA                                                    #
    #     Bono plano por cada paso en pista. Incentiva episodios largos.   #
    # ------------------------------------------------------------------ #
    if not offtrack and not wrong_way and forward_speed_kmh > 10.0:
        reward += weights.survival_bonus_weight

    # ------------------------------------------------------------------ #
    #  3. CENTRADO                                                         #
    # ------------------------------------------------------------------ #
    if not offtrack and not wrong_way and forward_speed_kmh > 10.0:
        abs_pos = abs(track_pos)
        if abs_pos <= weights.center_inner_radius:
            inner_ratio = 1.0 - (abs_pos / weights.center_inner_radius) * 0.2
            reward += weights.centering_bonus_weight * inner_ratio
        elif abs_pos <= weights.center_bonus_radius:
            outer_ratio = (weights.center_bonus_radius - abs_pos) / (weights.center_bonus_radius - weights.center_inner_radius)
            reward += weights.centering_bonus_weight * 0.8 * outer_ratio

    # ------------------------------------------------------------------ #
    #  4. SUAVIDAD DE VOLANTE (solo en rectas)                             #
    # ------------------------------------------------------------------ #
    on_straight = forward_arc > 0.85 and abs(track_pos) < 0.45 and abs(angle_rad) < 0.15
    if on_straight and forward_speed_kmh > 40.0:
        if steer_delta_abs < weights.smoothness_delta_threshold:
            smoothness = 1.0 - steer_delta_abs / weights.smoothness_delta_threshold
            reward += weights.smoothness_bonus_weight * smoothness

    # ------------------------------------------------------------------ #
    #  5. FRENADO ANTICIPADO                                               #
    #     Premia frenar cuando la curva se aproxima (antes de entrar).     #
    # ------------------------------------------------------------------ #
    approaching_curve = 0.55 < forward_arc < 0.85 and forward_speed_kmh > 80.0
    if approaching_curve and brake > 0.15 and not offtrack and not wrong_way:
        anticipation = (0.85 - forward_arc) / 0.30  # 0 lejos .. 1 cerca
        reward += weights.anticipation_brake_weight * min(brake, 1.0) * anticipation

    # ------------------------------------------------------------------ #
    #  6. FRENADO EN CURVA                                                 #
    # ------------------------------------------------------------------ #
    if in_curve and brake > 0.15 and not offtrack and not wrong_way:
        reward += weights.curve_brake_bonus_weight * min(brake, 1.0) * (0.4 + 0.6 * curve_depth)

    # ------------------------------------------------------------------ #
    #  7. DIRECCIÓN CORRECTA EN CURVA                                      #
    # ------------------------------------------------------------------ #
    if (
        in_curve
        and abs(track_curvature) > weights.correct_direction_min_curvature
        and abs(steer_signed) > 0.05
        and not offtrack
        and not wrong_way
        and forward_speed_kmh > 15.0
    ):
        turning_correctly = steer_signed * track_curvature < 0
        if turning_correctly:
            intensity = min(abs(steer_signed), 1.0)
            reward += weights.correct_direction_weight * intensity * (0.3 + 0.7 * curve_depth)
        else:
            reward -= weights.correct_direction_weight * 0.4 * min(abs(steer_signed), 1.0)

    # ------------------------------------------------------------------ #
    #  8. ALINEACIÓN EN CURVA                                              #
    #     Bono por mantener el ángulo cercano a 0 dentro de una curva.    #
    # ------------------------------------------------------------------ #
    if in_curve and not offtrack and not wrong_way and forward_speed_kmh > 15.0:
        if alignment > 0.92:
            align_bonus = (alignment - 0.92) / 0.08  # 0..1
            reward += weights.curve_alignment_bonus_weight * align_bonus

    # ------------------------------------------------------------------ #
    #  Penalizaciones                                                      #
    # ------------------------------------------------------------------ #
    reward -= weights.lateral_penalty_weight * min(lateral_speed_kmh / 50.0, 2.0)
    reward -= weights.angle_penalty_weight * max(0.0, 1.0 - alignment)
    reward -= weights.center_penalty_weight * (abs(track_pos) ** 2) * (1.0 + speed_ratio)

    if abs(track_pos) > 0.75:
        reward -= weights.edge_penalty_weight * (abs(track_pos) - 0.75) * (1.0 + speed_ratio)

    # Penalización por exceso de velocidad en curva cerrada
    if forward_arc < 0.55 and forward_speed_kmh > weights.curve_overspeed_threshold_kmh:
        overspeed = (forward_speed_kmh - weights.curve_overspeed_threshold_kmh) / 60.0
        reward -= weights.curve_overspeed_penalty * min(overspeed, 1.5) * curve_depth

    reward -= weights.control_penalty_weight * max(0.0, accel + brake - 1.0)
    if speed_x_kmh > 120.0 and steer > 0.8:
        reward -= weights.control_penalty_weight * (steer - 0.8) * 4.0

    reward -= weights.steer_oscillation_penalty_weight * min(steer_delta_abs, 1.5)
    if abs(track_pos) < 0.45 and abs(angle_rad) < 0.15 and forward_speed_kmh > 40.0:
        reward -= weights.steer_oscillation_penalty_weight * 0.5 * min(steer_delta_abs, 1.0)
        reward -= weights.straight_steer_penalty_weight * (steer ** 2)

    # 9. PROGRESO EN PISTA
    reward += weights.progress_delta_weight * max(-0.03, min(progress_delta, 0.2))

    if wrong_way:
        reward -= weights.wrong_way_penalty + 0.25 * abs(speed_x_kmh)

    if offtrack:
        reward -= weights.offtrack_penalty + 20.0 * max(0.0, abs(track_pos) - 1.0)

    if collision or damage_delta > 0:
        reward -= weights.collision_penalty + weights.damage_delta_weight * max(0.0, damage_delta)

    # Bono por navegar curvas correctamente — alineado, centrado y avanzando.
    if (
        in_curve
        and not offtrack
        and not wrong_way
        and abs(track_pos) < 0.60
        and forward_speed_kmh > 20.0
    ):
        reward += weights.curve_bonus_weight * (0.5 + 0.5 * curve_depth) * min(alignment, 1.0)

    return float(reward)
