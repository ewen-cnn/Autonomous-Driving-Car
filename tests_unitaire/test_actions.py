from actions import classify_direction, motor_to_action, normalize_model_action, row_to_model_action


def test_normalize_model_action_maps_fine_grained_left_to_left():
    assert normalize_model_action("sharp_left") == "left"


def test_normalize_model_action_ignores_stop():
    assert normalize_model_action("stop") is None


def test_motor_to_action_detects_forward():
    assert motor_to_action(25, 25) == "forward"


def test_motor_to_action_detects_left_turn():
    assert motor_to_action(10, 30) == "left"


def test_classify_direction_from_gpio_row():
    row = {
        "speedA": 20,
        "speedB": 35,
        "GPIO1": 1,
        "GPIO2": 0,
        "GPIO3": 0,
        "GPIO4": 1,
    }
    assert classify_direction(row) == "light_left"
    assert row_to_model_action(row) == "left"
