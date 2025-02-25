import pygame
from typing import Dict
from dataclasses import dataclass


@dataclass
class ActionState:
    ended: bool
    value: int


def handle_keyboard_input(
    event_key_to_action: Dict[str, int]
) -> ActionState:
    print("Handling keyboard input.")
    event = pygame.event.wait()
    print(event.type)
    if event.type == pygame.QUIT:
        print("User closed the window.")
        return ActionState(ended=True, action=0)
    if event.type == pygame.KEYDOWN:
        print("Keydown event.", event.key)
        if event.key in event_key_to_action:
            return ActionState(ended=False, value=event_key_to_action[event.key])
    return ActionState(ended=False, value=-1)
