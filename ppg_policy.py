# from pyvirtualdisplay import Display
# disp = Display(backend="xvfb",size=(1920,1080)).start()

# from Aircraft_Landing_Problem_gym_3000 import ALP_gym
# from stable_baselines import PPO2,DQN,ACER
# from ppo2 import PPO2
from ppg import PPG
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback,StopTrainingOnRewardThreshold, EvalCallback, StopTrainingOnRewardThreshold
# from stable_baselines.acer.policies import MlpPolicy, CnnPolicy
# from stable_baselines.ppo2 import CnnPolicy
from stable_baselines.common.policies import MlpPolicy,CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack,SubprocVecEnv
from stable_baselines.common.policies import ActorCriticPolicy, register_policy
from stable_baselines.common.tf_layers import linear,conv_to_fc,ortho_init,conv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
import tensorflow as tf
import numpy as np
# from tensorflow.keras.layers import Attention,Add,Multiply,GlobalAveragePooling2D
# env = ALP_gym()
# from layer import augmented_conv2d
# from attn_augconv import augmented_conv2d
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


def nature_cnn(scaled_images,n=0,n_neurons_final=512, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, f'c{n}_1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, f'c{n}_2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, f'c{n}_3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, f'fc{n}_1', n_hidden=n_neurons_final, init_scale=np.sqrt(2)))


class PPG_CNN(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=True, **kwargs):
        super(PPG_CNN, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            #========================[POLICY NETWORK]==================================#
            extracted_features = nature_cnn(self.processed_obs,n=0, **kwargs)
            pi_h = extracted_features
            pi_latent = pi_h
            vf_aux = tf.layers.dense(extracted_features, 1, name='vf1')

            #===========================[VALUE NETWORK]===============================================#
            vf_h = nature_cnn(self.processed_obs,n=1,n_neurons_final=64, **kwargs)
            value_fn = tf.layers.dense(vf_h, 1, name='vf2')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

            

        self.value_aux_fn = vf_aux
        self.value_flat_aux = self.value_aux_fn[:, 0]

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
    
    def value_aux(self,obs):
        return self.sess.run(self.value_flat_aux, {self.obs_ph: obs})

if __name__ == "__main__":
    env = VecFrameStack(make_atari_env("PongNoFrameskip-v4",num_env=1,seed=0),4)
    envs = VecFrameStack(make_atari_env("PongNoFrameskip-v4",num_env=5,seed=0),4)

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./checkpoints/',name_prefix='PPO2')
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=25, verbose=1)
    eval_callback = EvalCallback(env, best_model_save_path='./checkpoints/best/',log_path='./board_logs/', \
                                eval_freq=5000,deterministic=True, render=False, callback_on_new_best=callback_on_best)

    # envs = VecFrameStack(make_vec_env(ALP_gym,n_envs=5),4)
    model = PPG(PPG_CNN, envs, verbose=0,gamma=0.99, tensorboard_log="./board_logs")
    # model = PPO2.load("./checkpoints/best/best_model",env = envs, tensorboard_log="./board_logs/")
    # model = PPO2.load("PPO",env = envs, tensorboard_log="./board_logs/")
    model.learn(int(5e6))
    # model.save("PPO_Attention")
