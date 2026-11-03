# NetHack

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/NetHack)

## Description

NetHack is an environment for evaluating agents on the classic roguelike dungeon exploration game [NetHack](https://www.nethack.org/), built on the [NetHack Learning Environment (NLE)](https://github.com/NetHack-LE/nle). Agents explore procedurally generated dungeons by sending individual keystrokes and receiving ASCII terminal screen observations along with structured player statistics. The goal is to maximize the in-game score through strategic exploration, combat, item management, and dungeon navigation.

## Capabilities

- Strategic exploration of procedurally generated dungeons
- Combat decision-making against diverse monsters
- Inventory and resource management under constraints
- Long-horizon sequential decision-making under uncertainty
- Reading and interpreting ASCII game displays

## Compute Requirements

4 GB RAM, 2 CPUs. NLE compiles from source and requires cmake and build tools in the Docker image.

## License

[MIT](https://opensource.org/licenses/MIT). NetHack and NLE are distributed under the [NetHack General Public License](https://github.com/NetHack-LE/nle/blob/main/LICENSE).

## Tasks

There are 1,000 tasks in the `train` split (seeds 0-999).

Each task corresponds to a unique random seed that generates a distinct dungeon layout, starting character, and item placement. All tasks use the full NetHack action space (121 actions). Since every task is a procedurally generated dungeon with no structural difference between them, a single split is used.

## Reward Structure

This is a dense, verifiable reward environment. After each keystroke action, the agent receives the raw reward from the NLE engine, which is the in-game score delta (change in score since the previous step). Score increases from exploration, combat, item identification, gold collection, and dungeon descent.

We do not use LLM graders for this environment.

## Data

No external data is required. Game state is generated procedurally by the NetHack engine using seeded randomness. Data is stored on the OpenReward platform.

## Tools

Agents are given two tools:

- **`step(keystroke)`**: Send a keystroke to the game and receive the updated terminal screen, player stats, and game messages. Accepts single characters (`k`, `,`, `<`), control keys (`ctrl+d`), meta keys (`meta+j`), and special keys (`enter`, `space`, `escape`).
- **`commands()`**: Display a reference of all available NetHack keystroke commands grouped by category.

## Time Horizon

NetHack is a multi-turn environment with up to 1,000,000 steps per episode. Each step corresponds to one keystroke action. Episodes end on character death, ascension, or reaching the step limit.

## Other Environment Requirements

There are no further environment requirements; NetHack works out of the box without any secrets or API keys.

## Safety

Agents interact only with a roguelike game simulation. The environment has no access to external systems, the internet, or sensitive data.

## Citations

```bibtex
@inproceedings{kuettler2020nethack,
  author    = {Heinrich K{\"{u}}ttler and Nantas Nardelli and
               Alexander H. Miller and Roberta Raileanu and
               Marco Selvatici and Edward Grefenstette and
               Tim Rockt{\"{a}}schel},
  title     = {{The NetHack Learning Environment}},
  booktitle = {Proceedings of the Conference on Neural Information
               Processing Systems (NeurIPS)},
  year      = {2020},
}
```
