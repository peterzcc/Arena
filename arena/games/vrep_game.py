__author__ = 'flyers'

import logging
from .game import Game
from arena.games.vrep import vrep

logger = logging.getLogger(__name__)

# _dirname = os.path.dirname(os.path.realpath(__file__))
# _default_rom_path = os.path.join(_dirname, "roms", "remoteApi.so")

class VREPGame(Game):
