import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
import numpy as np


def build_dqn(lr, n_actions, input_dims, dims):
    inputs = keras.layers.Input(shape=input_dims)

    x = keras.layers.Dense(dims, activation='relu')(inputs)
    x = keras.layers.Dense(dims, activation='relu')(x)
    x = keras.layers.Dense(dims, activation='relu')(x)
    x = keras.layers.Dense(dims, activation='relu')(x)
    x = keras.layers.Dense(dims, activation='relu')(x)

    V = keras.layers.Dense(1, activation=None)(x)
    A = keras.layers.Dense(n_actions, activation=None)(x)

    Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

    model = Model(inputs=inputs, outputs=Q)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    model.summary()

    return model


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones


class FrameBuffer:
    def __init__(self, max_frames, env_dims):
        self.nr_dims = env_dims[0]
        self.nr_frames = max_frames
        self.frame_memory = np.zeros((self.nr_frames * self.nr_dims), dtype=np.float32)
        self.frame_counter = 0

    def get_input_shape(self):
        return self.frame_memory.shape

    def store(self, frame):
        self.frame_memory = np.roll(self.frame_memory, self.nr_dims)
        self.frame_memory[:self.nr_dims] = frame

    def reset(self, initial_observation=None):
        self.frame_memory = np.zeros((self.nr_frames * self.nr_dims), dtype=np.float32)
        if initial_observation is not None:
            self.store(initial_observation)

    def get_state(self):
        return self.frame_memory


class Agent():
    def __init__(self, env_dims, n_actions, batch_size=64,
                 max_frames=64, mem_size=100000,
                 epsilon=0, gamma=0.99, lr=0.0001, replace=100,
                 dims=512, file_name='dueling_dqn.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.file_name = file_name
        self.replace = replace
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.learn_step_counter = 0

        self.frames = FrameBuffer(max_frames, env_dims)
        self.input_dims = self.frames.get_input_shape()
        self.memory = ReplayBuffer(mem_size, self.input_dims)

        self.q_eval = build_dqn(lr, n_actions, self.input_dims, dims)
        self.q_next = build_dqn(lr, n_actions, self.input_dims, dims)

    def reset_state(self, initial_observation=None):
        self.frames.reset(initial_observation)

    def set_epsilon(self, epsilon):
        if epsilon >= 0:
            self.epsilon = epsilon
        else:
            self.epsilon = 0

    def set_learning_rate(self, lr):
        print('lr was', K.get_value(self.q_eval.optimizer.lr))
        K.set_value(self.q_eval.optimizer.lr, lr)
        print('lr is', K.get_value(self.q_eval.optimizer.lr))

    def observe(self, new_observation, action, reward, done):
        state = self.frames.get_state()
        self.frames.store(new_observation)
        new_state = self.frames.get_state()

        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([self.frames.get_state()])
            actions = self.q_eval(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # changed remainder to 1...
        # for some reason, 0 -> target net does not init
        if self.learn_step_counter % self.replace == 1:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = tf.math.reduce_max(self.q_next(states_), axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)

        # improve on my solution!
        for idx, terminal in enumerate(dones):
            if terminal:
                q_next[idx] = 0.0
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx]

        self.q_eval.train_on_batch(states, q_target)
        self.learn_step_counter += 1

    def save_model(self):
        self.q_eval.save(self.file_name)

    def load_model(self):
        self.q_eval = load_model(self.file_name)
        self.q_next.set_weights(self.q_eval.get_weights())
