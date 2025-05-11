from gymnasium.envs.registration import register

register(
    id="AlignHole-v0",
    entry_point="envs.AlignHole_v0:AlignHoleEnv",
)
