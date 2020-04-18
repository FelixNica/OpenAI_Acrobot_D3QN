from dueling_dqn_keras import Agent
import numpy as np
import gym
from utils import plotLearning
import time

if __name__ == '__main__':
    env = gym.make('Acrobot-v1')

    agent = Agent(env_dims=[6], n_actions=3, max_frames=64, dims=512)
    agent.load_model()
    agent.set_learning_rate(lr=0.000001)

    n_games = 100
    scores = []
    epsilon = 1
    eps_history = []
    total_steps = 0
    time_ = time.time()

    for i in range(n_games):
        done = False
        observation = env.reset()
        agent.reset_state(observation)
        score = 0
        step_counter = 0
        step_times = []
        infer_times = []
        learn_times = []

        epsilon = epsilon - 1/((i+10)*4)
        if epsilon <= 0.009:
            epsilon = 0.009
        #agent.set_epsilon(epsilon)
        agent.set_epsilon(0)

        while not done:
            start_time = time.time()
            action = agent.choose_action()
            infer_time = time.time() - start_time
            observation, reward, done, info = env.step(action)
            score += reward
            agent.observe(observation, action, reward, done)

            learn_time = time.time()
            if i >= 2000:
                agent.learn()
            learn_time = time.time() - learn_time

            env.render()
            time.sleep(0.015)
            step_counter += 1
            step_time = time.time() - start_time
            step_times.append(step_time)
            infer_times.append(infer_time)
            learn_times.append(learn_time)

        total_steps += step_counter
        avg_step_time = np.mean(step_times)
        avg_infer_time = np.mean(infer_times)
        avg_learn_time = np.mean(learn_times)
        game_time = np.sum(step_times)

        scores.append(score)

        eps_history.append(epsilon)
        avg_score = np.mean(scores[-100:])


        print('game:', i)
        print('steps:', step_counter)
        print('total steps:', total_steps)
        print('score: %.2f' % score)
        print('average score: %.2f' % avg_score)
        print('epsilon', epsilon)
        print('average step time: %.5f' % avg_step_time, 'seconds')
        print('agent infer time: %.5f' % avg_infer_time, 'seconds')
        print('agent learn time: %.5f' % avg_learn_time, 'seconds')
        print('game time: %.5f' % game_time, 'seconds')
        print('total time: %.2f' % ((time.time() - time_) / 60), 'minutes')
        print(80 * '- ')

    filename = 'acrobot_tf2.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)
    #agent.save_model()



