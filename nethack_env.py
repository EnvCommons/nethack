import gymnasium as gym
import nle  # noqa: F401 - registers NetHack environments
import numpy as np
from typing import List
from pydantic import BaseModel

from openreward.environments import (
    Environment, JSONObject, ToolOutput, tool, TextBlock,
)
from nle import nethack

MAX_EPISODE_STEPS = 1_000_000
NUM_TASKS = 1000

HUNGER_MAP = {
    0: "Satiated",
    1: "Not Hungry",
    2: "Hungry",
    3: "Weak",
    4: "Fainting",
}

ENCUMBRANCE_MAP = {
    0: "Unencumbered",
    1: "Burdened",
    2: "Stressed",
    3: "Strained",
    4: "Overtaxed",
    5: "Overloaded",
}


def _build_action_lookup():
    """Build mappings between keystroke strings and NLE action indices.

    Returns:
        keystroke_to_index: dict mapping keystroke string -> index into nethack.ACTIONS
        index_to_info: dict mapping index -> (keystroke_string, action_name, class_name)
    """
    actions = nethack.ACTIONS
    keystroke_to_index = {}
    index_to_info = {}

    for idx, action in enumerate(actions):
        val = int(action)
        name = action.name if hasattr(action, "name") else str(action)
        cls = type(action).__name__

        if val == 13:
            keystroke = "enter"
        elif val == 27:
            keystroke = "escape"
        elif val < 32 and val > 0:
            keystroke = f"ctrl+{chr(val + 96)}"
        elif val == 32:
            keystroke = "space"
        elif 33 <= val <= 126:
            keystroke = chr(val)
        elif val >= 0x80:
            inner = val & 0x7F
            if 32 <= inner < 127:
                keystroke = f"meta+{chr(inner)}"
            else:
                keystroke = f"meta+{inner}"
        else:
            keystroke = f"char({val})"

        # First occurrence wins for duplicate keystrokes (e.g. + is SEESPELLS and PLUS)
        if keystroke not in keystroke_to_index:
            keystroke_to_index[keystroke] = idx
        index_to_info[idx] = (keystroke, name, cls)

    return keystroke_to_index, index_to_info


KEYSTROKE_TO_INDEX, INDEX_TO_INFO = _build_action_lookup()

COMMANDS_REFERENCE = """NETHACK COMMAND REFERENCE

MOVEMENT (single step):
  k - North          j - South          h - West           l - East
  y - Northwest      u - Northeast      b - Southwest      n - Southeast

MOVEMENT (far, until interrupted):
  K - North (far)    J - South (far)    H - West (far)     L - East (far)
  Y - NW (far)       U - NE (far)       B - SW (far)       N - SE (far)

MOVEMENT (rush, stop at interesting things):
  g - Rush prefix (then direction)      G - Rush far prefix (then direction)

STAIRS & WAITING:
  < - Go up stairs   > - Go down stairs . - Wait/search/rest

ITEMS:
  , - Pick up         d - Drop            D - Drop multiple
  e - Eat             q - Quaff potion    r - Read scroll
  w - Wield weapon    W - Wear armor      T - Take off armor
  P - Put on ring/amulet                  R - Remove ring/amulet
  a - Apply/use tool  z - Zap wand        t - Throw
  f - Fire from quiver                    Q - Select quiver ammo
  A - Take off all

COMBAT:
  F - Force fight (then direction)
  ctrl+d - Kick (then direction)

INFORMATION:
  i - Inventory       I - Inventory by type
  : - Look here       ; - Far look (then direction or cursor)
  / - What is (identify symbol)
  \\ - Show discoveries
  @ - Toggle autopickup
  ctrl+x - Show attributes
  V - History          v - Version

INTERACTION:
  o - Open door/container               c - Close door
  s - Search for hidden doors/traps     p - Pay shopkeeper
  C - Name/call a monster or object

MAGIC:
  Z - Cast spell      meta+e - Enhance skills

PRAYER & RELIGION:
  meta+p - Pray       meta+o - Offer sacrifice
  meta+t - Turn undead

DUNGEON INTERACTION:
  meta+d - Dip item   meta+f - Force lock
  meta+j - Jump       meta+l - Loot container
  meta+r - Rub item   meta+s - Sit down
  meta+u - Untrap     meta+m - Use monster ability

EQUIPMENT:
  x - Swap weapons    X - Two-weapon combat
  E - Engrave         meta+i - Invoke artifact

MOVEMENT MODES:
  m - Move without picking up           M - Move far without picking up
  _ - Travel to location

VIEWING INVENTORY SUBSETS:
  ) - Show weapon     [ - Show armor     = - Show rings
  " - Show amulet     ( - Show tools     + - Show spells
  $ - Show gold       * - Show all equipment
  ^ - Show traps      ` - Show discovered items

MENU / PROMPTS:
  enter - Confirm / advance --More-- prompt
  space - Dismiss / page through menu
  escape - Cancel current action
  # - Extended command (then type command name)

SPECIAL (ctrl+key):
  ctrl+a - Redo last command    ctrl+d - Kick
  ctrl+o - Overview             ctrl+p - Previous messages
  ctrl+r - Redraw screen        ctrl+t - Teleport (if able)

SPECIAL (meta+key):
  meta+a - Adjust inventory letters     meta+c - Chat
  meta+C - Show conduct                 meta+q - Quit
  meta+v - Version (extended)           meta+w - Wipe face
  meta+A - Annotate level               meta+R - Ride mount
  meta+T - Tip container                meta+? - Extended command list

LETTERS & NUMBERS (for menus/item selection):
  0-9 - Enter quantity or select from numbered lists
  a-z, A-Z - Select items in inventory/menu prompts
  ' - Select previous choice

NOTE: When prompted for a direction, use a movement key (k/j/h/l/y/u/b/n).
When prompted to choose an item, send the item's inventory letter.
When you see --More--, send "enter" to continue.
When you see (end), send "space" to dismiss."""


class TaskSpec(BaseModel):
    id: str
    seed: int


class StepParams(BaseModel, extra="forbid"):
    keystroke: str


class CommandsParams(BaseModel, extra="forbid"):
    pass


class NetHackEnvironment(Environment):

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.env = None
        self.current_obs = None
        self.current_score = 0
        self.step_count = 0
        self.game_over = False

    @staticmethod
    def _make_env():
        """Create a NetHack gym environment with the full action space."""
        return gym.make(
            "NetHackScore-v0",
            actions=nethack.ACTIONS,
            allow_all_yn_questions=True,
            allow_all_modes=True,
            max_episode_steps=MAX_EPISODE_STEPS,
            penalty_step=0.0,
            penalty_time=0.0,
            observation_keys=(
                "glyphs", "chars", "colors", "specials",
                "blstats", "message",
                "inv_glyphs", "inv_strs", "inv_letters", "inv_oclasses",
                "tty_chars", "tty_colors", "tty_cursor", "misc",
            ),
        )

    async def setup(self):
        self.env = self._make_env()
        obs, _info = self.env.reset(seed=self.config.seed)
        self.current_obs = obs
        self.current_score = int(obs["blstats"][9])
        self.step_count = 0
        self.game_over = False

    async def teardown(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return [
                {"id": f"nethack_{i}", "seed": i}
                for i in range(NUM_TASKS)
            ]
        return []

    @staticmethod
    def _format_tty(tty_chars: np.ndarray) -> str:
        lines = []
        for row in range(tty_chars.shape[0]):
            line = "".join(chr(c) if 32 <= c < 127 else " " for c in tty_chars[row])
            lines.append(line.rstrip())
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    @staticmethod
    def _format_blstats(blstats: np.ndarray) -> str:
        hunger = HUNGER_MAP.get(int(blstats[21]), f"Unknown({blstats[21]})")
        encumbrance = ENCUMBRANCE_MAP.get(int(blstats[22]), f"Unknown({blstats[22]})")
        return (
            f"HP: {blstats[10]}/{blstats[11]} | "
            f"Pw: {blstats[14]}/{blstats[15]} | "
            f"AC: {blstats[16]} | "
            f"XP: {blstats[18]}({blstats[19]}) | "
            f"Dlvl: {blstats[12]} | "
            f"Gold: {blstats[13]} | "
            f"Score: {blstats[9]} | "
            f"T: {blstats[20]} | "
            f"Str: {blstats[3]} Dex: {blstats[4]} Con: {blstats[5]} "
            f"Int: {blstats[6]} Wis: {blstats[7]} Cha: {blstats[8]} | "
            f"Hunger: {hunger} | "
            f"Encumbrance: {encumbrance}"
        )

    @staticmethod
    def _format_message(message: np.ndarray) -> str:
        msg_bytes = bytes(message)
        return msg_bytes.split(b"\x00")[0].decode("ascii", errors="replace").strip()

    def _format_observation(self, obs: dict) -> str:
        parts = []
        message = self._format_message(obs["message"])
        if message:
            parts.append(f"Message: {message}")
        parts.append("Screen:")
        parts.append("```")
        parts.append(self._format_tty(obs["tty_chars"]))
        parts.append("```")
        parts.append(f"Stats: {self._format_blstats(obs['blstats'])}")
        return "\n".join(parts)

    async def get_prompt(self) -> List[TextBlock]:
        observation_text = self._format_observation(self.current_obs)
        prompt = f"""You are playing NetHack, a roguelike dungeon exploration game. Your goal is to descend through the Dungeons of Doom, retrieve the Amulet of Yendor, and ascend to offer it to your deity. Maximize your in-game score.

HOW TO READ THE SCREEN:
- @ is your character
- . is floor, # is corridor, | and - are walls
- < is stairs up, > is stairs down
- Letters represent monsters (d=dog, k=kobold, D=dragon, etc.)
- Symbols represent items: ! potion, ? scroll, [ armor, ) weapon, = ring, / wand, % food, * gem, ( tool
- The bottom lines show your stats

HOW TO PLAY:
Use the `step` tool with a keystroke to perform actions. Common keystrokes:
- Movement: k(north) j(south) h(west) l(east) y(NW) u(NE) b(SW) n(SE)
- Far movement: K J H L Y U B N (move until interrupted)
- Stairs: <(go up) >(go down)
- Items: ,(pickup) d(drop) e(eat) q(quaff) r(read) w(wield) W(wear) T(take off) a(apply) z(zap)
- Combat: F(fight direction) ctrl+d(kick)
- Info: i(inventory) :(look here) ;(far look) /(what is symbol)
- Misc: s(search) o(open) c(close) .(wait/search) #(extended command)
- Menu/prompts: enter(confirm/more) space(dismiss) escape(cancel)

Use the `commands` tool at any time to see the full list of available keystrokes.

When you see "--More--" in a message, send "enter" to continue.
When prompted with a direction, send a movement key (k/j/h/l/y/u/b/n).
When prompted to select an item, send the letter of the item.

CURRENT GAME STATE:
{observation_text}

Begin playing. Think carefully about your strategy before each action."""
        return [TextBlock(text=prompt)]

    @tool
    async def step(self, params: StepParams) -> ToolOutput:
        """Send a keystroke to NetHack and receive the updated game state.
        Use single characters like 'k' (north), 'e' (eat), ',' (pickup).
        For control keys use 'ctrl+d' (kick). For meta keys use 'meta+j' (jump).
        Use 'enter' for --More-- prompts and 'space' to dismiss menus."""
        if self.game_over:
            return ToolOutput(
                blocks=[TextBlock(text="The game is already over.")],
                metadata={"error": "game_finished"},
                reward=0.0,
                finished=True,
            )

        keystroke = params.keystroke.strip()
        action_index = KEYSTROKE_TO_INDEX.get(keystroke)

        if action_index is None:
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"Unknown keystroke: '{keystroke}'. "
                    "Use the `commands` tool to see valid keystrokes. "
                    "Common: k(north) j(south) h(west) l(east) e(eat) ,(pickup) >(descend) <(ascend)"
                )],
                metadata={"error": "invalid_keystroke", "keystroke": keystroke},
                reward=0.0,
                finished=False,
            )

        obs, reward, terminated, truncated, info = self.env.step(action_index)
        self.step_count += 1
        self.current_obs = obs

        new_score = int(obs["blstats"][9])
        score_delta = new_score - self.current_score
        self.current_score = new_score

        finished = terminated or truncated
        if finished:
            self.game_over = True

        obs_text = self._format_observation(obs)
        display_parts = [obs_text]

        if score_delta > 0:
            display_parts.append(f"\n[Score: {new_score} (+{score_delta})]")
        elif score_delta < 0:
            display_parts.append(f"\n[Score: {new_score} ({score_delta})]")

        if finished:
            if terminated:
                display_parts.append(f"\n=== GAME OVER === Final Score: {new_score}")
            else:
                display_parts.append(f"\n=== MAX STEPS REACHED === Final Score: {new_score}")

        display_text = "\n".join(display_parts)

        return ToolOutput(
            blocks=[TextBlock(text=display_text)],
            metadata={
                "score": new_score,
                "score_delta": score_delta,
                "step": self.step_count,
                "hp": int(obs["blstats"][10]),
                "hp_max": int(obs["blstats"][11]),
                "depth": int(obs["blstats"][12]),
                "time": int(obs["blstats"][20]),
                "reward": float(reward),
            },
            reward=float(reward),
            finished=finished,
        )

    @tool
    async def commands(self, params: CommandsParams) -> ToolOutput:
        """Show a reference of all available NetHack keystroke commands."""
        return ToolOutput(
            blocks=[TextBlock(text=COMMANDS_REFERENCE)],
            metadata={},
            reward=0.0,
            finished=False,
        )
