

class Cell:
    """
    A cell is a point in space that can have a state and a spawn time.
    The spawn time is the iteration time at which the cell was created.
    The state is the current state of the cell.
    The next state is the state the cell will have at the next iteration.
    """
    def __init__(self, state=0, spawn_default_value=-1):
        self.spawn_default_value = spawn_default_value
        self.state = 0
        self.next_state = 0
        self.spawn_time = spawn_default_value
        self.set_next_state(state, it_time=0)
        self.update_state()

    def get_state(self):
        return self.state

    def get_next_state(self):
        return self.next_state

    def get_spawn(self):
        return self.spawn_time

    def set_next_state(self, new_state, it_time):
        self.next_state = new_state
        # So, if new_state is True or > 0, but other types might have weird interactions
        if new_state and self.get_spawn() == -1:
            self.spawn_time = it_time
        if not new_state:
            self.spawn_time = self.spawn_default_value

    def update_state(self):
        self.state = self.get_next_state()