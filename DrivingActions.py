from config import *
 
def motor_to_action(speedA, speedB, eps=5.0, min_speed=5.0, motor_a_is_left=True):
    """
    Convertit les vitesses moteurs en action discrète.

    Hypothèse :
    - speedA = moteur gauche
    - speedB = moteur droit

    À vérifier sur ta voiture :
    si la voiture tourne à gauche quand speedB > speedA, l'hypothèse est correcte.
    """

    speedA = float(speedA)
    speedB = float(speedB)

    # Stop
    if abs(speedA) < min_speed and abs(speedB) < min_speed:
        return "stop"

    # Avance / recule
    if abs(speedA - speedB) < eps:
        if speedA > min_speed and speedB > min_speed:
            return "forward"
        elif speedA < -min_speed and speedB < -min_speed:
            return "backward"
        else:
            return "stop"

    # Différentiel gauche/droite
    if motor_a_is_left:
        if speedB > speedA + eps:
            return "left"
        elif speedA > speedB + eps:
            return "right"
    else:
        if speedA > speedB + eps:
            return "left"
        elif speedB > speedA + eps:
            return "right"

    return "other"