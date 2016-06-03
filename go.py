import re
from collections import namedtuple
import itertools

N = 9
W = N + 1
# Represent a board as a string, with '.' empty, 'X' to play, 'O' other player.
# Whitespace is used as a border (to avoid IndexError when computing neighbors)

# A Coordinate `c` is an int: an index into the board.
# A Move is a (Coordinate c | None).

# Human-readable move notation: columns go from A to T left to right
# rows go from 1 to 19 from bottom to top.
#  ABCD
#  ____\n
# 4....\n
# 3....\n
# 2....\n
# 1....

COLUMNS = 'ABCDEFGHJKLMNOPQRST'

EMPTY_BOARD = '\n'.join(
    [' ' * N] + 
    ['.' * N for i in range(N)] + 
    [' ' * W])

SWAP_COLORS = str.maketrans('XO', 'OX')

def load_board(string):
    string = re.sub(r'[^OX\.#]+', '', string)
    assert len(string) == N ** 2, "Board to load didn't have right dimensions"
    return '\n'.join([' ' * N] + [string[k*N:(k+1)*N] for k in range(N)] + [' ' * W])

def parse_coords(s):
    if s == 'pass':
        return None
    s = s.upper()
    col = COLUMNS.index(s[0])
    rows_from_top = N - int(s[1:])
    return W + (W * rows_from_top) + col

def neighbors(c):
    return [c+1, c-1, c+W, c-W]

def place_stone(board, color, c):
    return board[:c] + color + board[c+1:]

def capture_stones(board, stones):
    b = bytearray(board, encoding='ascii')
    for s in stones:
        b[s] = ord('.')
    return str(b, encoding='ascii')

def flood_fill(board, c):
    'From a starting coordinate c, flood-fill the board with a #'
    b = bytearray(board, encoding='ascii')
    color = b[c]
    entire_group = [c]
    frontier = [c]
    while frontier:
        current = frontier.pop()
        b[current] = ord('#')
        for n in neighbors(current):
            if b[n] == color:
                frontier.append(n)
                entire_group.append(n)
    return str(b, encoding='ascii'), set(entire_group)

def find_neighbors(color, board, stones):
    'Find all neighbors of a set of stones of a given color'
    potential_neighbors = set(itertools.chain(*(neighbors(s) for s in stones)))
    return {c for c in potential_neighbors if board[c] == color}

def find_liberties(board, stones):
    'Given a board and a set of stones, find liberties of those stones'
    return find_neighbors('.', board, stones)

class Group(namedtuple('Group', 'stones liberties')):
    '''
    stones: a set of Coordinates belonging to this group
    liberties: a set of Coordinates that are empty and adjacent to this group.
    '''
    pass


def deduce_groups(board):
    'Given a board, return a 2-tuple; a list of groups for each player'
    def find_groups(board, color):
        groups = []
        while color in board:
            c = board.index(color)
            board, stones = flood_fill(board, c)
            liberties = find_liberties(board, stones)
            groups.append(Group(stones=stones, liberties=liberties))
        return groups

    return find_groups(board, 'X'), find_groups(board, 'O')

def update_groups(board, existing_X_groups, existing_O_groups, c):
    '''
    When a move is played, update the list of groups and their liberties.
    This means possibly appending the new move to a group, creating a new 1-stone group, or merging existing groups.
    The new move should be of color X.
    The board should represent the state after the move has been played at `c`.
    '''
    assert board[c] == 'X'

    updated_X_groups, groups_to_merge = [], []
    for g in existing_X_groups:
        if c in g.liberties:
            groups_to_merge.append(g)
        else:
            updated_X_groups.append(g)

    new_stones = set([c])
    new_liberties = set(n for n in neighbors(c) if board[n] == '.')
    for g in groups_to_merge:
        new_stones = new_stones | g.stones
        new_liberties = new_liberties | g.liberties
    new_liberties = new_liberties - set([c])
    updated_X_groups.append(Group(stones=new_stones, liberties=new_liberties))

    updated_O_groups = []
    for g in existing_O_groups:
        if c in g.liberties:
            updated_O_groups.append(Group(stones=g.stones, liberties=g.liberties - {c}))
        else:
            updated_O_groups.append(g)

    return updated_X_groups, updated_O_groups

class Position(namedtuple('Position', 'board n caps groups ko')):
    '''
    board: a string representation of the board
    n: an int representing moves played so far
    caps: a (int, int) tuple of captures; caps[0] is the person to play.
    groups: a (list(Group), list(Group)) tuple of lists of Groups; groups[0] represents the groups of the person to play.
    ko: a Move
    '''
    @staticmethod
    def initial_state():
        return Position(EMPTY_BOARD, n=0, caps=(0, 0), groups=(set(), set()), ko=None)

    def pass_move(self):
        return Position(self.board.translate(SWAP_COLORS), self.n+1, (self.caps[1], self.caps[0]), (self.groups[1], self.groups[0]), None)

    def play_move(self, c):
        if c is None:
            return self.pass_move()
        if c == self.ko:
            return None

        working_board = place_stone(self.board, 'X', c)
        new_X_groups, new_O_groups = update_groups(working_board, self.groups[0], self.groups[1], c)

        # process opponent's captures first, then your own suicides.
        # As stones are removed, liberty counts become inaccurate.
        O_captures = set()
        X_suicides = set()
        surviving_O_groups = []
        surviving_X_groups = []
        for group in new_O_groups:
            if not group.liberties:
                O_captures |= group.stones
                working_board = capture_stones(working_board, group.stones)
            else:
                surviving_O_groups.append(group)

        if O_captures:
            coords_with_updates = find_neighbors('X', working_board, O_captures)
            final_O_groups = surviving_O_groups
            final_X_groups = [g if not (g.stones & coords_with_updates)
                else Group(stones=g.stones, liberties=find_liberties(working_board, g.stones))
                for g in new_X_groups]
        else:
            for group in new_X_groups:
                if not group.liberties:
                    X_suicides |= group.stones
                    working_board = capture_stones(working_board, group.stones)
                else:
                    surviving_X_groups.append(group)

            coords_with_updates = find_neighbors('O', working_board, X_suicides)
            final_X_groups = surviving_X_groups
            final_O_groups = [g if not (g.stones & coords_with_updates)
                else Group(stones=g.stones, liberties=find_liberties(working_board, g.stones))
                for g in new_O_groups]

        return Position(
            board=working_board.translate(SWAP_COLORS),
            n=self.n + 1,
            caps=(self.caps[1] + len(X_suicides), self.caps[0] + len(O_captures)),
            groups=(final_O_groups, final_X_groups),
            ko=None
        )