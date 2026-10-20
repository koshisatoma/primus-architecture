"""Conservative indirect-request candidates; never executes suggested actions."""


class Pragmatics:
    INDIRECT = {
        'cold': 'close_window',
        'hot': 'open_window',
        'dark': 'turn_on_light',
        'thirsty': 'offer_water',
        'hungry': 'offer_food',
        'heavy': 'help_carry',
    }

    def __init__(self):
        self.last_repair = None

    def interpret_indirect(self, text):
        normalized = ' '.join(text.casefold().split())
        if normalized.endswith('?'):
            return None
        for cue, action in self.INDIRECT.items():
            if cue in normalized:
                return action
        return None

    def repair(self, confusion_cue, contrast):
        self.last_repair = contrast
        return f'Not the {confusion_cue} one — the {contrast} one.'

    @staticmethod
    def register(bond, agreeableness):
        if bond > 0.7:
            return 'intimate-casual'
        if agreeableness > 0.7:
            return 'polite'
        return 'neutral'
