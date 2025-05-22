from stable_baselines3.ddpg import DDPG
from test_script import test_ddpg_agent

if __name__ == '__main__':
    # model = DDPG('MlpPolicy', 'LunarLanderContinuous-v3', verbose=1)
    # model.learn(total_timesteps=100000)
    # model.save("ddpg_lunar")

    test_ddpg_agent('ddpg_lunar', 5, render_mode='human')
    test_ddpg_agent('ddpg_lunar', 100)
